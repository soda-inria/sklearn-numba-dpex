import numpy as np
import dpctl
from sklearn_numba_dpex.kmeans.kernels import (
    make_initialize_to_zeros_kernel_1_float32,
    make_initialize_to_zeros_kernel_2_float32,
    make_initialize_to_zeros_kernel_3_float32,
    make_copyto_kernel,
    make_broadcast_division_kernel,
    make_center_shift_kernel,
    make_half_l2_norm_kernel_dim0,
    make_sum_reduction_kernel_1,
    make_sum_reduction_kernel_2,
    make_sum_reduction_kernel_3,
    make_fused_kernel_fixed_window,
    make_assignment_kernel_fixed_window,
)


# class KMeansDriver:


#     def __init__(
#         # work_group_size="auto",
#         # preferred_work_group_size_multiple="auto",
#         # work_group_size_multiplier="auto",
#         # work_group_size_multiplier_for_cluster_window_length="auto",
#         # ratio_multiplier_for_window_height="auto",
#         # global_memory_cache_size="auto",
#         # work_group_size_multiple_for_cluster_window_length="auto",
#         # window_height
#         ):
#     """
#     parameters:
#         preferred work group size (used for privatization strategy), power of 2
#         width of the window (number of clusters), multiple of preferred work group size
#             should be a power of 2
#         work group sized, multiple of preferred work grup size, power of 2
#         height of the window (number of dims), should be a multiple of the ratio
#             (work group size / width)

#     """

#     pass


def kmeans(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    x_squared_norms=None,
    tol=1e-4,
    n_threads=1,
    device=None,
):

    # NB: all parameters given when instanciating the kernels can impact performance,
    # could be benchmarked, and can be set with more generality regarding the device
    # specs using tools to fetch device information (like pyopencl)

    # warp_size could be retrieved with e.g pyopencl, maybe dpctl ?
    # in doubt, use a high number that could be a multiple of the real warp
    # size (which is usually a power of 2) rather than a low number
    WARP_SIZE = 64
    # cache size (in bytes) can also be retrieved
    L2_CACHE_SIZE = 1572864
    work_group_size = 4 * WARP_SIZE

    dim = X.shape[1]
    n = X.shape[0]
    n_clusters = centers_init.shape[0]

    reset_inertia = make_initialize_to_zeros_kernel_1_float32(
        n=n, work_group_size=work_group_size
    )

    copyto = make_copyto_kernel(
        n_clusters, dim, work_group_size=work_group_size
    )

    broadcast_division = make_broadcast_division_kernel(
        n=n_clusters, dim=dim, work_group_size=work_group_size
    )

    get_center_shifts = make_center_shift_kernel(
        n_clusters, dim, work_group_size=work_group_size
    )

    half_l2_norm = make_half_l2_norm_kernel_dim0(
        n_clusters, dim, work_group_size=work_group_size
    )

    # NB: assumes work_group_size is a power of two
    reduce_inertia = make_sum_reduction_kernel_1(
        n, work_group_size=work_group_size, device=device
    )

    reduce_center_shifts = make_sum_reduction_kernel_1(
        n_clusters, work_group_size=work_group_size, device=device
    )

    (
        nb_centroids_private_copies,
        fused_kernel_fixed_window,
    ) = make_fused_kernel_fixed_window(
        n,
        dim,
        n_clusters,
        warp_size=WARP_SIZE,
        l2_cache_size=L2_CACHE_SIZE,
        # to benchmark
        # biggest possible values supported by shared memory and private registry ?
        window_length_multiple=1,
        # to benchmark
        # biggest possible value supported by shared memory and by the device
        cluster_window_per_thread_group=4,
        number_of_load_iter=4,
    )

    reset_centroid_counts_private_copies = (
        make_initialize_to_zeros_kernel_2_float32(
            n=n_clusters,
            dim=nb_centroids_private_copies,
            work_group_size=work_group_size,
        )
    )

    reset_centroids_private_copies = make_initialize_to_zeros_kernel_3_float32(
        dim0=nb_centroids_private_copies,
        dim1=dim,
        dim2=n_clusters,
        work_group_size=work_group_size,
    )

    reduce_centroid_counts_private_copies = make_sum_reduction_kernel_2(
        n=n_clusters,
        dim=nb_centroids_private_copies,
        work_group_size=work_group_size,
    )

    reduce_centroids_private_copies = make_sum_reduction_kernel_3(
        dim0=nb_centroids_private_copies,
        dim1=dim,
        dim2=n_clusters,
        work_group_size=work_group_size,
    )

    assignment_kernel_fixed_window = make_assignment_kernel_fixed_window(
        n,
        dim,
        n_clusters,
        warp_size=WARP_SIZE,
        window_length_multiple=1,
        cluster_window_per_thread_group=4,
        number_of_load_iter=4,
    )

    # The dpex kernel expects X to be shaped as (n_features, n_samples) as the
    # usual (n_samples, n_features). Furthermore, using a Fortran memory layout
    # is actually more L1 cache friendly on the GPU and results in a small
    # performance increase (e.g. 15% faster).
    X = dpctl.tensor.from_numpy(X, device=device).T
    centroids = dpctl.tensor.from_numpy(
        np.ascontiguousarray(centers_init).T, device=device
    )
    centroids_copy_array = dpctl.tensor.empty_like(centroids, device=device)
    best_centroids = dpctl.tensor.empty_like(centroids, device=device)
    centroids_half_l2_norm = dpctl.tensor.empty(
        n_clusters, dtype=np.float32, device=device
    )
    centroid_counts = dpctl.tensor.empty(
        n_clusters, dtype=np.float32, device=device
    )
    center_shifts = dpctl.tensor.empty(
        n_clusters, dtype=np.float32, device=device
    )
    inertia = dpctl.tensor.empty(n, dtype=np.float32, device=device)
    assignments_idx = dpctl.tensor.empty(n, dtype=np.uint32, device=device)

    centroids_private_copies = dpctl.tensor.empty(
        (nb_centroids_private_copies, dim, n_clusters),
        dtype=np.float32,
        device=device,
    )
    centroid_counts_private_copies = dpctl.tensor.empty(
        (nb_centroids_private_copies, n_clusters),
        dtype=np.float32,
        device=device,
    )

    n_iteration = 0
    center_shifts_sum = np.inf
    best_inertia = np.inf
    copyto(centroids, best_centroids)

    while (n_iteration < max_iter) and (center_shifts_sum >= tol):
        half_l2_norm(centroids, centroids_half_l2_norm)

        reset_inertia(inertia)
        reset_centroid_counts_private_copies(centroid_counts_private_copies)
        reset_centroids_private_copies(centroids_private_copies)

        # TODO: implement special case where only one copy is needed
        fused_kernel_fixed_window(
            X,
            centroids,
            centroids_half_l2_norm,
            inertia,
            centroids_private_copies,
            centroid_counts_private_copies,
        )

        reduce_centroid_counts_private_copies(
            centroid_counts_private_copies, centroid_counts
        )
        reduce_centroids_private_copies(
            centroids_private_copies, centroids_copy_array
        )

        broadcast_division(centroids_copy_array, centroid_counts)

        get_center_shifts(centroids, centroids_copy_array, center_shifts)

        center_shifts_sum = dpctl.tensor.asnumpy(
            reduce_center_shifts(center_shifts)
        )[0]

        inertia_sum = dpctl.tensor.asnumpy(reduce_inertia(inertia))[0]

        if inertia_sum < best_inertia:
            best_inertia = inertia_sum
            copyto(centroids, best_centroids)

        centroids, centroids_copy_array = centroids_copy_array, centroids

        n_iteration += 1

    half_l2_norm(best_centroids, centroids_half_l2_norm)
    reset_inertia(inertia)

    assignment_kernel_fixed_window(
        X,
        best_centroids,
        centroids_half_l2_norm,
        inertia,
        assignments_idx,
    )

    inertia_sum = dpctl.tensor.asnumpy(reduce_inertia(inertia))[0]

    return (
        dpctl.tensor.asnumpy(assignments_idx),
        inertia_sum,
        np.ascontiguousarray(dpctl.tensor.asnumpy(best_centroids.T)),
        n_iteration,
    )


def kmeans_cpu(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    x_squared_norms=None,
    tol=1e-4,
    n_threads=1,
):
    return kmeans(
        X,
        sample_weight,
        centers_init,
        max_iter,
        verbose,
        x_squared_norms,
        tol,
        n_threads,
        device="cpu",
    )


def kmeans_gpu(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    x_squared_norms=None,
    tol=1e-4,
    n_threads=1,
):
    return kmeans(
        X,
        sample_weight,
        centers_init,
        max_iter,
        verbose,
        x_squared_norms,
        tol,
        n_threads,
        device="gpu",
    )
