import math
import warnings
from functools import lru_cache

import numpy as np
import dpctl
from sklearn.exceptions import NotSupportedByEngineError, DataConversionWarning

from sklearn_numba_dpex.utils._device import _DeviceParams

from sklearn_numba_dpex.kmeans.kernels import (
    make_lloyd_single_step_fixed_window_kernel,
    make_compute_labels_inertia_fixed_window_kernel,
    make_label_assignment_fixed_window_kernel,
    make_compute_inertia_fixed_window_kernel,
    make_compute_euclidean_distances_fixed_window_kernel,
    make_centroid_shifts_kernel,
    make_reduce_centroid_data_kernel,
    make_initialize_to_zeros_2d_kernel,
    make_initialize_to_zeros_3d_kernel,
    make_broadcast_division_1d_2d_kernel,
    make_half_l2_norm_2d_axis0_kernel,
    make_sum_reduction_1d_kernel,
)


def _check_power_of_2(e):
    if e != 2 ** (math.log2(e)):
        raise ValueError(f"Expected a power of 2, got {e}")
    return e


@lru_cache
class KMeansDriver:
    """GPU optimized implementation of Lloyd's k-means.

    The current implementation is called "fused fixed", it consists in a sliding window
    of fixed size on the value of the centroids that work items use to accumulate the
    distances of a given sample to each centroids. It is followed, in a same kernel, by
    an update of new centroids in a context of memory privatization to avoid a too high
    cost of atomic operations.

    This class instantiates into a callable that mimics the interface of sklearn's
    private function `_kmeans_single_lloyd` .

    Parameters
    ----------
    preferred_work_group_size_multiple : int
        The kernels will use this value to optimize the distribution of work items. If
        None, it is automatically fetched with pyopencl if possible, else a default of
        64 is applied. It is required to be a power of two.

    work_group_size_multiplier : int
        The size of groups of work items used to execute the kernels is defined as
        `work_group_size_multiplier * preferred_work_group_size_multiple` and, if None,
        is chosen to be equal to the value of max_work_group_size which is fetched
        with dpctl. It is required to be a power of two.

    centroids_window_width_multiplier : int
        The width of the window over the centroids is defined as
        `centroids_window_width_multiplier x preferred_work_group_size_multiple`. The
        higher the value, the higher the cost in shared memory. If None, will default
        to 1. It is required to be a power of two.

    centroids_window_height : int
        The height of the window, counted as a number of features. The higher the
        value, the higher the cost in shared memory. If None, will default to 16. It is
        required to be a power of two.

    global_mem_cache_size : int
        Size in bytes of the size of the global memory cache size. If None, the value
        will be automatically fetched with dpctl. It is used to estimate the maximum
        number of copies of the array of centroids that can be used for privatization.

    centroids_private_copies_max_cache_occupancy : float
        A maximum fraction of global_mem_cache_size that is allowed to be expected for
        use when estimating the maximum number of copies that can be used for
        privatization. If None, will default to 0.7.

    device: str
        A valid sycl device filter.

    X_layout: str
        'F' or 'C'. If None, will default to 'F'.

    dtype: np.float32 or np.float64
        The floating point precision that the kernels should use. If None, will adapt
        to the dtype of the data, else, will cast the data to the appropriate dtype.

    Notes
    -----
    The implementation has been extensively inspired by the "Fused Fixed" strategy
    exposed in [1]_, along with its reference implementatino by the same authors [2]_,
    and the reader can also refer to the complementary slide deck [3]_  with schemas
    that intuitively explain the main computation.

    .. [1] Kruliš, M., & Kratochvíl, M. (2020, August). Detailed analysis and
        optimization of CUDA K-means algorithm. In 49th International
        Conference on Parallel Processing-ICPP (pp. 1-11).

    .. [2] https://github.com/krulis-martin/cuda-kmeans

    .. [3] https://jnamaral.github.io/icpp20/slides/Krulis_Detailed.pdf

    """

    def __init__(
        self,
        preferred_work_group_size_multiple=None,
        work_group_size_multiplier=None,
        centroids_window_width_multiplier=None,
        centroids_window_height=None,
        global_mem_cache_size=None,
        centroids_private_copies_max_cache_occupancy=None,
        device=None,
        X_layout=None,
        dtype=None,
    ):
        dpctl_device = dpctl.SyclDevice(device)
        device_params = _DeviceParams(dpctl_device)

        # TODO: set the best possible defaults for all the parameters based on an
        # exhaustive grid search.

        self.global_mem_cache_size = (
            global_mem_cache_size or device_params.global_mem_cache_size
        )

        self.preferred_work_group_size_multiple = _check_power_of_2(
            preferred_work_group_size_multiple
            or device_params.preferred_work_group_size_multiple
        )

        # So far the best default value has been empirically found to be 2 so we use
        # this default.
        # TODO: when it's available in dpctl, use the `max_group_size` attribute
        # exposed by the kernel instead ?
        self.work_group_size_multiplier = _check_power_of_2(
            work_group_size_multiplier or 2
        )

        self.centroids_window_width_multiplier = _check_power_of_2(
            centroids_window_width_multiplier or 1
        )

        self.centroids_window_height = _check_power_of_2(centroids_window_height or 16)

        self.centroids_private_copies_max_cache_occupancy = (
            centroids_private_copies_max_cache_occupancy or 0.7
        )

        self.device = dpctl_device

        # FIXME: "C" is not available at the time (raises a ValueError).
        self.X_layout = X_layout or "F"

        self.has_aspect_fp64 = device_params.has_aspect_fp64

        self.dtype = dtype
        if dtype is not None:
            dtype = np.dtype(dtype).type
            if (dtype != np.float32) and (dtype != np.float64):
                raise ValueError(f"Valid types are float64, float32, but got f{dtype}")
            self.dtype = dtype
            if (self.dtype == np.float64) and not self.has_aspect_fp64:
                raise NotSupportedByEngineError(
                    f"Computations with precision f{self.dtype} has been explicitly "
                    f"requested to the KMeans driver but the device {dpctl_device.name} does not "
                    f"support it."
                )

    def lloyd(
        self,
        X,
        sample_weight,
        centers_init,
        max_iter=300,
        verbose=False,
        tol=1e-4,
    ):
        """This call is expected to accept the same inputs than sklearn's private
        _kmeans_single_lloyd and produce the same outputs.
        """
        (
            assignments_idx,
            inertia,
            best_centroids,
            n_iteration,
            output_dtype,
        ) = self._lloyd(
            X,
            sample_weight,
            centers_init,
            max_iter,
            verbose,
            tol,
        )

        # TODO: explore leveraging dpnp to benefit from USM to avoid moving
        # centroids back and forth between device and host memory in case
        # a subsequent `.predict` call is requested on the same GPU later.
        return (
            dpctl.tensor.asnumpy(assignments_idx).astype(np.int32),
            inertia,
            # XXX: having a C-contiguous centroid array is expected in sklearn in some
            # unit test and by the cython engine.
            np.ascontiguousarray(
                dpctl.tensor.asnumpy(best_centroids).astype(output_dtype)
            ),
            n_iteration,
        )

    def _lloyd(
        self,
        X,
        sample_weight,
        centers_init,
        max_iter=300,
        verbose=False,
        tol=1e-4,
    ):

        (
            X,
            sample_weight,
            cluster_centers,
            compute_dtype,
            output_dtype,
            work_group_size,
            n_features,
            n_samples,
            n_clusters,
        ) = self._check_inputs(X, sample_weight, centers_init)

        # Create a set of kernels
        (
            n_centroids_private_copies,
            fused_lloyd_fixed_window_single_step_kernel,
        ) = make_lloyd_single_step_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            with_sample_weight=sample_weight is not None,
            return_inertia=bool(verbose),
            compute_exact_inertia=True,
            preferred_work_group_size_multiple=self.preferred_work_group_size_multiple,
            global_mem_cache_size=self.global_mem_cache_size,
            centroids_window_width_multiplier=self.centroids_window_width_multiplier,
            centroids_window_height=self.centroids_window_height,
            centroids_private_copies_max_cache_occupancy=self.centroids_private_copies_max_cache_occupancy,
            work_group_size=work_group_size,
            dtype=compute_dtype,
        )

        fixed_window_assignment_kernel = make_compute_labels_inertia_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            with_sample_weight=sample_weight is not None,
            preferred_work_group_size_multiple=self.preferred_work_group_size_multiple,
            centroids_window_width_multiplier=self.centroids_window_width_multiplier,
            centroids_window_height=self.centroids_window_height,
            work_group_size=work_group_size,
            dtype=compute_dtype,
        )

        reset_cluster_sizes_private_copies_kernel = make_initialize_to_zeros_2d_kernel(
            size0=n_centroids_private_copies,
            size1=n_clusters,
            work_group_size=work_group_size,
            dtype=compute_dtype,
        )

        reset_centroids_private_copies_kernel = make_initialize_to_zeros_3d_kernel(
            size0=n_centroids_private_copies,
            size1=n_features,
            size2=n_clusters,
            work_group_size=work_group_size,
            dtype=compute_dtype,
        )

        broadcast_division_kernel = make_broadcast_division_1d_2d_kernel(
            size0=n_features,
            size1=n_clusters,
            work_group_size=work_group_size,
        )

        compute_centroid_shifts_kernel = make_centroid_shifts_kernel(
            n_clusters=n_clusters,
            n_features=n_features,
            work_group_size=work_group_size,
            dtype=compute_dtype,
        )

        half_l2_norm_kernel = make_half_l2_norm_2d_axis0_kernel(
            size0=n_features,
            size1=n_clusters,
            work_group_size=work_group_size,
            dtype=compute_dtype,
        )

        reduce_inertia_kernel = make_sum_reduction_1d_kernel(
            size=n_samples,
            work_group_size=work_group_size,
            device=self.device,
            dtype=compute_dtype,
        )

        reduce_centroid_shifts_kernel = make_sum_reduction_1d_kernel(
            size=n_clusters,
            work_group_size=work_group_size,
            device=self.device,
            dtype=compute_dtype,
        )

        reduce_centroid_data_kernel = make_reduce_centroid_data_kernel(
            n_centroids_private_copies=n_centroids_private_copies,
            n_features=n_features,
            n_clusters=n_clusters,
            work_group_size=work_group_size,
            dtype=compute_dtype,
        )

        X_t, sample_weight, centroids_t = self._load_transposed_data_to_device(
            X,
            sample_weight,
            cluster_centers,
        )

        # Allocate the necessary memory in the device global memory
        new_centroids_t = dpctl.tensor.empty_like(centroids_t, device=self.device)

        centroids_half_l2_norm = dpctl.tensor.empty(
            n_clusters, dtype=compute_dtype, device=self.device
        )
        cluster_sizes = dpctl.tensor.empty(
            n_clusters, dtype=compute_dtype, device=self.device
        )
        centroid_shifts = dpctl.tensor.empty(
            n_clusters, dtype=compute_dtype, device=self.device
        )
        per_sample_inertia = dpctl.tensor.empty(
            n_samples, dtype=compute_dtype, device=self.device
        )
        assignments_idx = dpctl.tensor.empty(
            n_samples, dtype=np.uint32, device=self.device
        )
        new_centroids_t_private_copies = dpctl.tensor.empty(
            (n_centroids_private_copies, n_features, n_clusters),
            dtype=compute_dtype,
            device=self.device,
        )
        cluster_sizes_private_copies = dpctl.tensor.empty(
            (n_centroids_private_copies, n_clusters),
            dtype=compute_dtype,
            device=self.device,
        )
        empty_clusters_list = dpctl.tensor.empty(
            n_clusters, dtype=np.uint32, device=self.device
        )

        # nb_empty_clusters_ is a scalar handled in kernels via a one-element array.
        nb_empty_clusters = dpctl.tensor.empty(1, dtype=np.int32, device=self.device)

        # The loop
        n_iteration = 0
        centroid_shifts_sum = np.inf

        # TODO: Investigate possible speedup with a custom dpctl queue with a custom
        # DAG of events and a final single "wait"
        while (n_iteration < max_iter) and (centroid_shifts_sum > tol):
            half_l2_norm_kernel(centroids_t, centroids_half_l2_norm)

            reset_cluster_sizes_private_copies_kernel(cluster_sizes_private_copies)
            reset_centroids_private_copies_kernel(new_centroids_t_private_copies)
            nb_empty_clusters[0] = np.int32(0)

            # TODO: implement special case where only one copy is needed
            fused_lloyd_fixed_window_single_step_kernel(
                X_t,
                sample_weight,
                centroids_t,
                centroids_half_l2_norm,
                per_sample_inertia,
                new_centroids_t_private_copies,
                cluster_sizes_private_copies,
            )

            reduce_centroid_data_kernel(
                cluster_sizes_private_copies,
                new_centroids_t_private_copies,
                cluster_sizes,
                new_centroids_t,
                empty_clusters_list,
                nb_empty_clusters,
            )

            nb_empty_clusters_ = dpctl.tensor.asnumpy(nb_empty_clusters)[0]
            if nb_empty_clusters_ > 0:
                # TODO: instead of this warning, set the centroid for empty clusters
                # to new points carefully chosen in the dataset. (mimic scikit-learn
                # behavior where the points with the highest inertia are chosen)
                warnings.warn("Found an empty cluster", RuntimeWarning)

            broadcast_division_kernel(new_centroids_t, cluster_sizes)

            compute_centroid_shifts_kernel(
                centroids_t, new_centroids_t, centroid_shifts
            )

            centroid_shifts_sum, *_ = dpctl.tensor.asnumpy(
                reduce_centroid_shifts_kernel(centroid_shifts)
            )

            if verbose:
                # ???: verbosity comes at the cost of performance since it triggers
                # computing exact inertia at each iteration. Shouldn't this be
                # documented ?
                inertia, *_ = dpctl.tensor.asnumpy(
                    reduce_inertia_kernel(per_sample_inertia)
                )
                print(f"Iteration {n_iteration}, inertia {inertia}")

            # ???: unlike sklearn, sklearn_intelex checks that pseudo_inertia decreases
            # and keep an additional copy of centroids that is updated only if the
            # value of the pseudo_inertia is smaller than all the past values.
            #
            # (the check should not be needed because we have theoritical guarantee
            # that inertia decreases, if it doesn't it should only be because of
            # rounding errors ?)
            #
            # To this purpose this code could be inserted here:
            #
            # if pseudo_inertia < best_pseudo_inertia:
            #     best_pseudo_inertia = pseudo_inertia
            #     copyto_kernel(centroids_t, best_centroids_t)
            #
            # Note that what that is saved as "best centroid" is the array before the
            # update, to which the pseudo inertia that is computed at this iteration
            # refers. For this reason, this strategy is not compatible with sklearn
            # unit tests, that consider new_centroids_t (the array after the update)
            # to be the best centroids at  each iteration.

            centroids_t, new_centroids_t = (
                new_centroids_t,
                centroids_t,
            )

            n_iteration += 1

        if verbose:
            converged_at = n_iteration - 1
            if centroid_shifts_sum == 0:  # NB: possible if tol = 0
                print(f"Converged at iteration {converged_at}: strict convergence.")

            elif centroid_shifts_sum <= tol:
                print(
                    f"Converged at iteration {converged_at}: center shift "
                    f"{centroid_shifts_sum} within tolerance {tol}."
                )

        # Finally, run an assignment kernel to compute the assignments to the best
        # centroids found, along with the exact inertia.
        half_l2_norm_kernel(centroids_t, centroids_half_l2_norm)

        fixed_window_assignment_kernel(
            X_t,
            sample_weight,
            centroids_t,
            centroids_half_l2_norm,
            per_sample_inertia,
            assignments_idx,
        )

        # inertia = per_sample_inertia.sum()
        inertia = dpctl.tensor.asnumpy(reduce_inertia_kernel(per_sample_inertia))
        # inertia is now a 1-sized numpy array, we transform it into a scalar:
        inertia = inertia[0]

        return (
            assignments_idx,
            inertia,
            centroids_t.T,
            n_iteration,
            output_dtype,
        )

    def get_labels(
        self,
        X,
        sample_weight,
        centers,
    ):
        labels = self._get_labels(X, centers)
        return dpctl.tensor.asnumpy(labels).astype(np.int32)

    def _get_labels(
        self,
        X,
        centers,
    ):
        """This call is expected to accept the same inputs than
        `sklearn.cluster._kmeans._labels_inertia` while solely computing
        the samples' labels, hence returning (labels, None).
        """
        (
            X,
            _,
            centers,
            compute_dtype,
            _,
            work_group_size,
            n_features,
            n_samples,
            n_clusters,
        ) = self._check_inputs(
            X,
            sample_weight=None,
            cluster_centers=centers,
        )

        # Create a set of kernels
        label_assignment_fixed_window_kernel = make_label_assignment_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            preferred_work_group_size_multiple=self.preferred_work_group_size_multiple,
            centroids_window_width_multiplier=self.centroids_window_width_multiplier,
            centroids_window_height=self.centroids_window_height,
            work_group_size=work_group_size,
            dtype=compute_dtype,
        )

        half_l2_norm_kernel = make_half_l2_norm_2d_axis0_kernel(
            size0=n_features,
            size1=n_clusters,
            work_group_size=work_group_size,
            dtype=compute_dtype,
        )

        X_t, _, centroids_t = self._load_transposed_data_to_device(X, None, centers)

        centroids_half_l2_norm = dpctl.tensor.empty(
            n_clusters, dtype=compute_dtype, device=self.device
        )
        assignments_idx = dpctl.tensor.empty(
            n_samples, dtype=np.uint32, device=self.device
        )

        half_l2_norm_kernel(centroids_t, centroids_half_l2_norm)

        label_assignment_fixed_window_kernel(
            X_t,
            centroids_t,
            centroids_half_l2_norm,
            assignments_idx,
        )

        return assignments_idx

    def get_inertia(
        self,
        X,
        sample_weight,
        centers,
    ):
        """This call is expected to accept the same inputs than
        `sklearn.cluster._kmeans._labels_inertia` while solely computing
        the samples' inertia, hence returning (None, inertia).
        """
        (
            X,
            sample_weight,
            centers,
            compute_dtype,
            output_dtype,
            work_group_size,
            n_features,
            n_samples,
            n_clusters,
        ) = self._check_inputs(X, sample_weight, centers)

        # Create a set of kernels
        compute_inertia_fixed_window_kernel = make_compute_inertia_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            with_sample_weight=sample_weight is not None,
            preferred_work_group_size_multiple=self.preferred_work_group_size_multiple,
            centroids_window_width_multiplier=self.centroids_window_width_multiplier,
            centroids_window_height=self.centroids_window_height,
            work_group_size=work_group_size,
            dtype=compute_dtype,
        )

        half_l2_norm_kernel = make_half_l2_norm_2d_axis0_kernel(
            size0=n_features,
            size1=n_clusters,
            work_group_size=work_group_size,
            dtype=compute_dtype,
        )

        reduce_inertia_kernel = make_sum_reduction_1d_kernel(
            size=n_samples,
            work_group_size=work_group_size,
            device=self.device,
            dtype=compute_dtype,
        )

        X_t, sample_weight, centroids_t = self._load_transposed_data_to_device(
            X, sample_weight, centers
        )

        centroids_half_l2_norm = dpctl.tensor.empty(
            n_clusters, dtype=compute_dtype, device=self.device
        )
        per_sample_inertia = dpctl.tensor.empty(
            n_samples, dtype=compute_dtype, device=self.device
        )

        half_l2_norm_kernel(centroids_t, centroids_half_l2_norm)

        compute_inertia_fixed_window_kernel(
            X_t,
            sample_weight,
            centroids_t,
            centroids_half_l2_norm,
            per_sample_inertia,
        )

        # inertia = per_sample_inertia.sum()
        inertia = dpctl.tensor.asnumpy(reduce_inertia_kernel(per_sample_inertia))
        # inertia is now a 1-sized numpy array, we transform it into a scalar:
        inertia = inertia.astype(output_dtype)[0]

        return inertia

    def get_euclidean_distances(self, X, Y):

        euclidean_distances, output_dtype = self._get_euclidean_distances(
            X,
            Y,
        )

        return dpctl.tensor.asnumpy(euclidean_distances).astype(output_dtype)

    def _get_euclidean_distances(self, X, Y):
        (
            X,
            _,
            Y,
            compute_dtype,
            output_dtype,
            work_group_size,
            n_features,
            n_samples,
            n_clusters,
        ) = self._check_inputs(
            X,
            sample_weight=None,
            cluster_centers=Y,
        )

        label_assignment_fixed_window_kernel = make_compute_euclidean_distances_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            preferred_work_group_size_multiple=self.preferred_work_group_size_multiple,
            centroids_window_width_multiplier=self.centroids_window_width_multiplier,
            centroids_window_height=self.centroids_window_height,
            work_group_size=work_group_size,
            dtype=compute_dtype,
        )

        half_l2_norm_kernel = make_half_l2_norm_2d_axis0_kernel(
            size0=n_features,
            size1=n_clusters,
            work_group_size=work_group_size,
            dtype=compute_dtype,
        )

        X_t, _, Y_t = self._load_transposed_data_to_device(X, None, Y)

        Y_half_l2_norm = dpctl.tensor.empty(
            n_clusters, dtype=compute_dtype, device=self.device
        )
        euclidean_distances_t = dpctl.tensor.empty(
            (n_clusters, n_samples), dtype=compute_dtype, device=self.device
        )

        half_l2_norm_kernel(Y_t, Y_half_l2_norm)

        label_assignment_fixed_window_kernel(
            X_t,
            Y_t,
            Y_half_l2_norm,
            euclidean_distances_t,
        )
        return euclidean_distances_t.T, output_dtype()

    def _set_dtype(self, X, sample_weight, centers_init):
        input_dtype = output_dtype = np.dtype(X.dtype).type
        compute_dtype = np.dtype(self.dtype or input_dtype).type
        copy = True
        if (compute_dtype != np.float32) and (compute_dtype != np.float64):
            text = (
                f"KMeans has been set to compute with type {compute_dtype} but only "
                f"the types float32 and float64 are supported. The computations and "
                f"outputs will default back to float32 type."
            )
            output_dtype = compute_dtype = np.float32
        elif (compute_dtype == np.float64) and not self.has_aspect_fp64:
            text = (
                f"KMeans is set to compute with type {compute_dtype} but this type is "
                f"not supported by the device {self.device.name}. The computations "
                f"will default back to float32 type."
            )
            compute_dtype = np.float32
        elif compute_dtype != X.dtype:
            text = (
                f"KMeans is set to run with dtype {compute_dtype} but the data has "
                f"been submitted with type {X.dtype}."
            )

        else:
            copy = False

        if copy:
            text += (
                f" A copy of the data casted to type {compute_dtype} will be created. "
                f"To save memory and suppress this warning, ensure that the dtype of "
                f"the input data matches the dtype required for computations."
            )
            warnings.warn(text, DataConversionWarning)
            # TODO: instead of triggering a copy on the host side, we could use the
            # dtype to allocate a shared USM buffer and fill it with casted values from
            # X. In this case we should only warn when:
            #     (dtype == np.float64) and not self.has_aspect_fp64
            # The other cases would not trigger any additional memory copies.
            X = X.astype(compute_dtype)

        centers_init_dtype = centers_init.dtype
        if centers_init.dtype != compute_dtype:
            warnings.warn(
                f"The centers have been passed with type {centers_init_dtype} but "
                f"type {compute_dtype} is expected. A copy will be created with the "
                f"correct type {compute_dtype}. Ensure that the centers are passed "
                f"with the correct dtype to save memory and suppress this warning.",
                DataConversionWarning,
            )
            centers_init = centers_init.astype(compute_dtype)

        if sample_weight is not None and sample_weight.dtype != compute_dtype:
            warnings.warn(
                f"sample_weight has been passed with type {sample_weight.dtype} but "
                f"type {compute_dtype} is expected. A copy will be created with the "
                f"correct type {compute_dtype}. Ensure that sample_weight is passed "
                f"with the correct dtype to save memory and suppress this warning.",
                DataConversionWarning,
            )
            sample_weight = sample_weight.astype(compute_dtype)

        return X, sample_weight, centers_init, compute_dtype, output_dtype

    def _check_inputs(self, X, sample_weight, cluster_centers):

        if sample_weight is not None and not (sample_weight != sample_weight[0]).any():
            sample_weight = None

        (
            X,
            sample_weight,
            cluster_centers,
            compute_dtype,
            output_dtype,
        ) = self._set_dtype(X, sample_weight, cluster_centers)

        work_group_size = (
            self.work_group_size_multiplier * self.preferred_work_group_size_multiple
        )

        n_features = X.shape[1]
        n_samples = X.shape[0]
        n_clusters = cluster_centers.shape[0]
        return (
            X,
            sample_weight,
            cluster_centers,
            compute_dtype,
            output_dtype,
            work_group_size,
            n_features,
            n_samples,
            n_clusters,
        )

    def _load_transposed_data_to_device(self, X, sample_weight, cluster_centers):
        # Transfer the input data to device memory,
        # TODO: let the user pass directly dpctl.tensor or dpnp arrays to avoid copies.
        if self.X_layout == "C":
            # TODO: support the C layout and benchmark it and default to it if
            # performances are better
            raise ValueError("C layout is currently not supported.")
            X_t = dpctl.tensor.from_numpy(X, device=self.device).T
            assert (
                X_t.strides[0] == 1
            )  # Fortran memory layout, equivalent to C layout on transposed
        elif self.X_layout == "F":
            X_t = dpctl.tensor.from_numpy(X.T, device=self.device)
            assert (
                X_t.strides[1] == 1
            )  # C memory layout, equivalent to Fortran layout on transposed
        else:
            raise ValueError(
                f"Expected X_layout to be equal to 'C' or 'F', but got {self.X_layout} ."
            )
        return (
            X_t,
            dpctl.tensor.from_numpy(sample_weight, device=self.device)
            if sample_weight is not None
            else dpctl.tensor.empty(1, dtype=X.dtype, device=self.device),
            dpctl.tensor.from_numpy(cluster_centers.T, device=self.device),
        )
