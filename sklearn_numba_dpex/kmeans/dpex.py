import numpy as np
import dpctl
from sklearn_numba_dpex.kmeans.kernels.kmeans_numba_dpex_fused import (
    get_initialize_to_zeros_kernel_1_float32,
    get_initialize_to_zeros_kernel_2_float32,
    get_initialize_to_zeros_kernel_3_float32,
    get_copyto_kernel,
    get_broadcast_division_kernel,
    get_center_shift_kernel,
    get_half_l2_norm_kernel_dim0,
    get_sum_reduction_kernel_1,
    get_sum_reduction_kernel_2,
    get_sum_reduction_kernel_3,
    get_fused_kernel_fixed_window,
    get_assignment_kernel_fixed_window,
)


def kmeans_fused(
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
    DEFAULT_THREAD_GROUP_SIZE = 4 * WARP_SIZE

    dim = X.shape[1]
    n = X.shape[0]
    n_clusters = centers_init.shape[0]

    reset_inertia = get_initialize_to_zeros_kernel_1_float32(
        n=n, thread_group_size=DEFAULT_THREAD_GROUP_SIZE
    )

    copyto = get_copyto_kernel(
        n_clusters, dim, thread_group_size=DEFAULT_THREAD_GROUP_SIZE
    )

    broadcast_division = get_broadcast_division_kernel(
        n=n_clusters, dim=dim, thread_group_size=DEFAULT_THREAD_GROUP_SIZE
    )

    get_center_shifts = get_center_shift_kernel(
        n_clusters, dim, thread_group_size=DEFAULT_THREAD_GROUP_SIZE
    )

    half_l2_norm = get_half_l2_norm_kernel_dim0(
        n_clusters, dim, thread_group_size=DEFAULT_THREAD_GROUP_SIZE
    )

    # NB: assumes thread_group_size is a power of two
    reduce_inertia = get_sum_reduction_kernel_1(
        n, thread_group_size=DEFAULT_THREAD_GROUP_SIZE, device=device
    )

    reduce_center_shifts = get_sum_reduction_kernel_1(
        n_clusters, thread_group_size=DEFAULT_THREAD_GROUP_SIZE, device=device
    )

    (
        nb_centroids_private_copies,
        fused_kernel_fixed_window,
    ) = get_fused_kernel_fixed_window(
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
        get_initialize_to_zeros_kernel_2_float32(
            n=n_clusters,
            dim=nb_centroids_private_copies,
            thread_group_size=DEFAULT_THREAD_GROUP_SIZE,
        )
    )

    reset_centroids_private_copies = get_initialize_to_zeros_kernel_3_float32(
        dim0=nb_centroids_private_copies,
        dim1=dim,
        dim2=n_clusters,
        thread_group_size=DEFAULT_THREAD_GROUP_SIZE,
    )

    reduce_centroid_counts_private_copies = get_sum_reduction_kernel_2(
        n=n_clusters,
        dim=nb_centroids_private_copies,
        thread_group_size=DEFAULT_THREAD_GROUP_SIZE,
    )

    reduce_centroids_private_copies = get_sum_reduction_kernel_3(
        dim0=nb_centroids_private_copies,
        dim1=dim,
        dim2=n_clusters,
        thread_group_size=DEFAULT_THREAD_GROUP_SIZE,
    )

    assignment_kernel_fixed_window = get_assignment_kernel_fixed_window(
        n,
        dim,
        n_clusters,
        warp_size=WARP_SIZE,
        window_length_multiple=1,
        cluster_window_per_thread_group=4,
        number_of_load_iter=4,
    )

    X = dpctl.tensor.from_numpy(np.ascontiguousarray(X).T, device=device)
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


def kmeans_fused_cpu(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    x_squared_norms=None,
    tol=1e-4,
    n_threads=1,
):
    return kmeans_fused(
        X,
        sample_weight,
        centers_init,
        max_iter=300,
        verbose=False,
        x_squared_norms=None,
        tol=1e-4,
        n_threads=1,
        device="cpu",
    )


def kmeans_fused_gpu(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    x_squared_norms=None,
    tol=1e-4,
    n_threads=1,
):
    return kmeans_fused(
        X,
        sample_weight,
        centers_init,
        max_iter=300,
        verbose=False,
        x_squared_norms=None,
        tol=1e-4,
        n_threads=1,
        device="gpu",
    )
