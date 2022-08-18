import math

import numpy as np
import dpctl
import pyopencl

from sklearn_numba_dpex.kmeans.kernels import (
    make_initialize_to_zeros_1dim_float32_kernel,
    make_initialize_to_zeros_2dim_float32_kernel,
    make_initialize_to_zeros_3dim_float32_kernel,
    make_copyto_kernel,
    make_broadcast_division_kernel,
    make_center_shift_kernel,
    make_half_l2_norm_dim0_kernel,
    make_sum_reduction_1dim_kernel,
    make_sum_reduction_2dim_kernel,
    make_sum_reduction_3dim_kernel,
    make_fused_fixed_window_kernel,
    make_assignment_fixed_window_kernel,
)


def _check_power_of_2(e):
    if e != 2 ** (math.log2(e)):
        raise RuntimeError(f"Expected a power of 2, got {e}")
    return e


class KMeansDriver:
    def __init__(
        self,
        global_mem_cache_size=None,
        preferred_work_group_size_multiple=None,
        work_group_size_multiplier=None,
        centroids_window_width_multiplier=None,
        centroids_window_height_ratio_multiplier=None,
        centroids_private_copies_max_cache_occupancy=None,
        device=None,
    ):
        dpctl_device = dpctl.SyclDevice(device)

        # TODO: when dpctl.SyclDevice also exposes relevant attributes,
        # remove the dependency to opencl and only use dpctl.
        cl_device = next(
            device
            for platform in pyopencl.get_platforms()
            for device in platform.get_devices()
            if device.name == dpctl_device.name
        )

        self.global_mem_cache_size = (
            global_mem_cache_size or cl_device.global_mem_cache_size
        )

        self.preferred_work_group_size_multiple = _check_power_of_2(
            preferred_work_group_size_multiple
            or cl_device.preferred_work_group_size_multiple
        )

        self.work_group_size_multiplier = _check_power_of_2(
            work_group_size_multiplier
            or (
                cl_device.max_work_group_size // self.preferred_work_group_size_multiple
            )
        )

        self.centroids_window_width_multiplier = _check_power_of_2(
            centroids_window_width_multiplier or 1
        )

        self.centroids_window_height_ratio_multiplier = (
            centroids_window_height_ratio_multiplier or 2
        )

        self.centroids_private_copies_max_cache_occupancy = (
            centroids_private_copies_max_cache_occupancy or 0.7
        )

        self.device = device

    def __call__(
        self,
        X,
        sample_weight,
        centers_init,
        max_iter=300,
        verbose=False,
        x_squared_norms=None,
        tol=1e-4,
        n_threads=1,
    ):
        """Inspired by the "Fused Fixed" strategy exposed in:
        Kruliš, M., & Kratochvíl, M. (2020, August). Detailed analysis and
        optimization of CUDA K-means algorithm. In 49th International
        Conference on Parallel Processing-ICPP (pp. 1-11).
        """
        work_group_size = (
            self.work_group_size_multiplier * self.preferred_work_group_size_multiple
        )

        n_features = X.shape[1]
        n_samples = X.shape[0]
        n_clusters = centers_init.shape[0]

        reset_inertia = make_initialize_to_zeros_1dim_float32_kernel(
            n_samples=n_samples, work_group_size=work_group_size
        )

        copyto = make_copyto_kernel(
            n_clusters, n_features, work_group_size=work_group_size
        )

        broadcast_division = make_broadcast_division_kernel(
            n_samples=n_clusters,
            n_features=n_features,
            work_group_size=work_group_size,
        )

        get_center_shifts = make_center_shift_kernel(
            n_clusters, n_features, work_group_size=work_group_size
        )

        half_l2_norm = make_half_l2_norm_dim0_kernel(
            n_clusters, n_features, work_group_size=work_group_size
        )

        # NB: assumes work_group_size is a power of two
        reduce_inertia = make_sum_reduction_1dim_kernel(
            n_samples, work_group_size=work_group_size, device=self.device
        )

        reduce_center_shifts = make_sum_reduction_1dim_kernel(
            n_clusters, work_group_size=work_group_size, device=self.device
        )

        (
            nb_centroids_private_copies,
            fixed_window_fused_kernel,
        ) = make_fused_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            preferred_work_group_size_multiple=self.preferred_work_group_size_multiple,
            global_mem_cache_size=self.global_mem_cache_size,
            centroids_window_width_multiplier=self.centroids_window_width_multiplier,
            centroids_window_height_ratio_multiplier=self.centroids_window_height_ratio_multiplier,
            centroids_private_copies_max_cache_occupancy=self.centroids_private_copies_max_cache_occupancy,
            work_group_size=work_group_size,
        )

        reset_centroid_counts_private_copies = (
            make_initialize_to_zeros_2dim_float32_kernel(
                n_samples=n_clusters,
                n_features=nb_centroids_private_copies,
                work_group_size=work_group_size,
            )
        )

        reset_centroids_private_copies = make_initialize_to_zeros_3dim_float32_kernel(
            dim0=nb_centroids_private_copies,
            dim1=n_features,
            dim2=n_clusters,
            work_group_size=work_group_size,
        )

        reduce_centroid_counts_private_copies = make_sum_reduction_2dim_kernel(
            n_samples=n_clusters,
            n_features=nb_centroids_private_copies,
            work_group_size=work_group_size,
        )

        reduce_centroids_private_copies = make_sum_reduction_3dim_kernel(
            dim0=nb_centroids_private_copies,
            dim1=n_features,
            dim2=n_clusters,
            work_group_size=work_group_size,
        )

        fixed_window_assignment_kernel = make_assignment_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            preferred_work_group_size_multiple=self.preferred_work_group_size_multiple,
            centroids_window_width_multiplier=self.centroids_window_width_multiplier,
            centroids_window_height_ratio_multiplier=self.centroids_window_height_ratio_multiplier,
            work_group_size=work_group_size,
        )

        X_t = dpctl.tensor.from_numpy(X.T, device=self.device)
        centroids_t = dpctl.tensor.from_numpy(centers_init.T, device=self.device)
        centroids_t_copy_array = dpctl.tensor.empty_like(
            centroids_t, device=self.device
        )
        best_centroids_t = dpctl.tensor.empty_like(
            centroids_t_copy_array, device=self.device
        )
        centroids_half_l2_norm = dpctl.tensor.empty(
            n_clusters, dtype=np.float32, device=self.device
        )
        centroid_counts = dpctl.tensor.empty(
            n_clusters, dtype=np.float32, device=self.device
        )
        center_shifts = dpctl.tensor.empty(
            n_clusters, dtype=np.float32, device=self.device
        )
        inertia = dpctl.tensor.empty(n_samples, dtype=np.float32, device=self.device)
        assignments_idx = dpctl.tensor.empty(
            n_samples, dtype=np.uint32, device=self.device
        )

        centroids_t_private_copies = dpctl.tensor.empty(
            (nb_centroids_private_copies, n_features, n_clusters),
            dtype=np.float32,
            device=self.device,
        )
        centroid_counts_private_copies = dpctl.tensor.empty(
            (nb_centroids_private_copies, n_clusters),
            dtype=np.float32,
            device=self.device,
        )

        n_iteration = 0
        center_shifts_sum = np.inf
        best_inertia = np.inf
        copyto(centroids_t, best_centroids_t)

        while (n_iteration < max_iter) and (center_shifts_sum >= tol):
            half_l2_norm(centroids_t, centroids_half_l2_norm)

            reset_inertia(inertia)
            reset_centroid_counts_private_copies(centroid_counts_private_copies)
            reset_centroids_private_copies(centroids_t_private_copies)

            # TODO: implement special case where only one copy is needed
            fixed_window_fused_kernel(
                X_t,
                centroids_t,
                centroids_half_l2_norm,
                inertia,
                centroids_t_private_copies,
                centroid_counts_private_copies,
            )

            reduce_centroid_counts_private_copies(
                centroid_counts_private_copies, centroid_counts
            )
            reduce_centroids_private_copies(
                centroids_t_private_copies, centroids_t_copy_array
            )

            broadcast_division(centroids_t_copy_array, centroid_counts)

            get_center_shifts(centroids_t, centroids_t_copy_array, center_shifts)

            center_shifts_sum = dpctl.tensor.asnumpy(
                reduce_center_shifts(center_shifts)
            )[0]

            inertia_sum = dpctl.tensor.asnumpy(reduce_inertia(inertia))[0]

            if inertia_sum < best_inertia:
                best_inertia = inertia_sum
                copyto(centroids_t, best_centroids_t)

            centroids_t, centroids_t_copy_array = (
                centroids_t_copy_array,
                centroids_t,
            )

            n_iteration += 1

        half_l2_norm(best_centroids_t, centroids_half_l2_norm)
        reset_inertia(inertia)

        fixed_window_assignment_kernel(
            X_t,
            best_centroids_t,
            centroids_half_l2_norm,
            inertia,
            assignments_idx,
        )

        inertia_sum = dpctl.tensor.asnumpy(reduce_inertia(inertia))[0]

        return (
            dpctl.tensor.asnumpy(assignments_idx),
            inertia_sum,
            np.ascontiguousarray(dpctl.tensor.asnumpy(best_centroids_t.T)),
            n_iteration,
        )
