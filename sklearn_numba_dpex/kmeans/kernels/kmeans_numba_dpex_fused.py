import numba_dpex as dpex
from numba import float32, int32

import dpctl
import numpy as np
import math

from sklearn_numba_dpex.utils import (
    cached_kernel_factory,
)


@cached_kernel_factory
def get_initialize_to_zeros_kernel_1_int32(n, thread_group_size):

    n_threads = math.ceil(n / thread_group_size) * thread_group_size
    int32zero = int32(0)

    @dpex.kernel
    def initialize_to_zeros(x):
        thread_i = dpex.get_global_id(0)

        if thread_i >= n:
            return

        x[thread_i] = int32zero

    return initialize_to_zeros[n_threads, thread_group_size]


@cached_kernel_factory
def get_initialize_to_zeros_kernel_1_float32(n, thread_group_size):

    n_threads = math.ceil(n / thread_group_size) * thread_group_size
    f32zero = float32(0)

    @dpex.kernel
    def initialize_to_zeros(x):
        thread_i = dpex.get_global_id(0)

        if thread_i >= n:
            return

        x[thread_i] = f32zero

    return initialize_to_zeros[n_threads, thread_group_size]


@cached_kernel_factory
def get_initialize_to_zeros_kernel_2_float32(n, dim, thread_group_size):

    nb_items = n * dim
    n_threads = math.ceil(nb_items / thread_group_size) * thread_group_size
    f32zero = float32(0.0)

    @dpex.kernel
    def initialize_to_zeros(x):
        thread_i = dpex.get_global_id(0)

        if thread_i >= nb_items:
            return

        row = thread_i // n
        col = thread_i % n
        x[row, col] = f32zero

    return initialize_to_zeros[n_threads, thread_group_size]


@cached_kernel_factory
def get_initialize_to_zeros_kernel_2_float32(n, dim, thread_group_size):

    nb_items = n * dim
    n_threads = math.ceil(nb_items / thread_group_size) * thread_group_size
    f32zero = float32(0.0)

    @dpex.kernel
    def initialize_to_zeros(x):
        thread_i = dpex.get_global_id(0)

        if thread_i >= nb_items:
            return

        row = thread_i // n
        col = thread_i % n
        x[row, col] = f32zero

    return initialize_to_zeros[n_threads, thread_group_size]


@cached_kernel_factory
def get_initialize_to_zeros_kernel_3_float32(
    dim0, dim1, dim2, thread_group_size
):

    nb_items = dim0 * dim1 * dim2
    stride0 = dim1 * dim2
    n_threads = math.ceil(nb_items / thread_group_size) * thread_group_size
    f32zero = float32(0.0)

    @dpex.kernel
    def initialize_to_zeros(x):
        thread_i = dpex.get_global_id(0)

        if thread_i >= nb_items:
            return

        d0 = thread_i // stride0
        stride0_id = thread_i % stride0
        d1 = stride0_id // dim2
        d2 = stride0_id % dim2
        x[d0, d1, d2] = f32zero

    return initialize_to_zeros[n_threads, thread_group_size]


@cached_kernel_factory
def get_copyto_kernel(n, dim, thread_group_size):
    n_threads = math.ceil(n / thread_group_size) * thread_group_size

    @dpex.kernel
    def copyto_kernel(X, Y):
        x = dpex.get_global_id(0)

        if x >= n:
            return

        for d in range(dim):
            Y[d, x] = X[d, x]

    return copyto_kernel[n_threads, thread_group_size]


@cached_kernel_factory
def get_broadcast_division_kernel(n, dim, thread_group_size):
    n_threads = math.ceil(n / thread_group_size) * thread_group_size

    @dpex.kernel
    def broadcast_division_kernel(X, v):
        x = dpex.get_global_id(0)

        if x >= n:
            return

        divisor = v[x]

        for d in range(dim):
            X[d, x] = X[d, x] / divisor

    return broadcast_division_kernel[n_threads, thread_group_size]


@cached_kernel_factory
def get_center_shift_kernel(n, dim, thread_group_size):
    n_threads = math.ceil(n / thread_group_size) * thread_group_size
    f32zero = float32(0.0)

    @dpex.kernel
    def center_shift_kernel(previous_center, center, center_shift):
        x = dpex.get_global_id(0)

        if x >= n:
            return

        tmp = f32zero

        for d in range(dim):
            center_diff = previous_center[d, x] - center[d, x]
            tmp += center_diff * center_diff

        center_shift[x] = tmp

    return center_shift_kernel[n_threads, thread_group_size]


@cached_kernel_factory
def get_half_l2_norm_kernel_dim0(n, dim, thread_group_size):
    n_threads = math.ceil(n / thread_group_size) * thread_group_size
    f32zero = float32(0.0)

    @dpex.kernel
    def half_l2_norm_kernel_dim0(X, result):
        x = dpex.get_global_id(0)

        if x >= n:
            return

        l2_norm = f32zero

        for d in range(dim):
            item = X[d, x]
            l2_norm += item * item

        result[x] = l2_norm / 2

    return half_l2_norm_kernel_dim0[n_threads, thread_group_size]


@cached_kernel_factory
def get_sum_reduction_kernel_1(n, thread_group_size):
    local_nb_iterations = math.floor(math.log2(thread_group_size))
    f32zero = float32(0.0)

    @dpex.kernel
    def sum_reduction_kernel(v, w):
        group = dpex.get_group_id(0)
        thread = dpex.get_local_id(0)
        first_thread = thread == 0

        n = v.shape[0]

        shm = dpex.local.array(thread_group_size, dtype=float32)

        group_start = group * thread_group_size * 2
        thread_operand_1 = group_start + thread
        thread_operand_2 = group_start + thread_group_size + thread

        if thread_operand_1 >= n:
            shm[thread] = f32zero
        elif thread_operand_2 >= n:
            shm[thread] = v[thread_operand_1]
        else:
            shm[thread] = v[thread_operand_1] + v[thread_operand_2]

        dpex.barrier()
        current_size = thread_group_size
        for i in range(local_nb_iterations):
            current_size = current_size // 2
            if thread < current_size:
                shm[thread] += shm[thread + current_size]

            dpex.barrier()

        if first_thread:
            w[group] = shm[0]

    _steps_data = []
    n_groups = n
    while n_groups > 1:
        n_groups = math.ceil(n_groups / (2 * thread_group_size))
        n_threads = n_groups * thread_group_size
        _steps_data.append(
            (
                sum_reduction_kernel[n_threads, thread_group_size],
                dpctl.tensor.empty(n_groups, dtype=np.float32),
            )
        )

    def sum_reduction(v):
        for sum_fn, w in _steps_data:
            sum_fn(v, w)
            v = w
        return v

    return sum_reduction


@cached_kernel_factory
def get_sum_reduction_kernel_2(n, dim, thread_group_size):
    n_threads = math.ceil(n / thread_group_size) * thread_group_size
    f32zero = float32(0.0)

    @dpex.kernel
    def sum_reduction_kernel(v, w):
        thread = dpex.get_global_id(0)
        if thread >= n:
            return
        tmp = f32zero
        for d in range(dim):
            tmp += v[d, thread]
        w[thread] = tmp

    return sum_reduction_kernel[n_threads, thread_group_size]


@cached_kernel_factory
def get_sum_reduction_kernel_3(dim0, dim1, dim2, thread_group_size):
    nb_groups_for_dim2 = math.ceil(dim2 / thread_group_size)
    nb_threads_for_dim2 = nb_groups_for_dim2 * thread_group_size
    n_threads = nb_threads_for_dim2 * dim1
    f32zero = float32(0.0)

    @dpex.kernel
    def sum_reduction_kernel(v, w):
        group = dpex.get_group_id(0)
        thread = dpex.get_local_id(0)
        d1 = group // nb_groups_for_dim2
        d2 = ((group % nb_groups_for_dim2) * thread_group_size) + thread
        if d2 >= dim2:
            return
        tmp = f32zero
        for d in range(dim0):
            tmp += v[d, d1, d2]
        w[d1, d2] = tmp

    return sum_reduction_kernel[n_threads, thread_group_size]


@cached_kernel_factory
def get_fused_kernel_fixed_window(
    n,
    dim,
    n_clusters,
    warp_size,
    l2_cache_size,
    window_length_multiple,
    cluster_window_per_thread_group,
    number_of_load_iter,
):
    r = warp_size * window_length_multiple
    thread_group_size = r * cluster_window_per_thread_group
    h = number_of_load_iter * cluster_window_per_thread_group

    n_cluster_groups = math.ceil(n_clusters / r)
    n_threads = (math.ceil(n / thread_group_size)) * (thread_group_size)
    n_dim_windows = math.ceil(dim / h)
    window_shm_shape = (h, r)

    inf = float32(math.inf)
    f32zero = float32(0.0)
    f32one = float32(1.0)

    nb_cluster_items = n_clusters * (dim + 1)
    nb_cluster_bytes = 4 * nb_cluster_items
    l2_cache_size_allocation = 0.9
    nb_centroids_private_copies = int(
        (l2_cache_size * l2_cache_size_allocation) // nb_cluster_bytes
    )

    @dpex.kernel
    def fused_kernel_fixed_window(
        X,
        current_centroids,
        centroids_half_l2_norm,
        inertia,
        centroids_private_copies,
        centroid_counts_private_copies,
    ):
        global_thread = dpex.get_global_id(0)
        local_thread = dpex.get_local_id(0)

        window_shm = dpex.local.array(shape=window_shm_shape, dtype=float32)
        local_centroids_half_l2_norm = dpex.local.array(shape=r, dtype=float32)
        partial_scores = dpex.private.array(shape=r, dtype=float32)

        first_centroid_global_idx = 0

        min_idx = 0
        min_score = inf

        window_col = local_thread % r
        window_row_offset = local_thread // r

        for _0 in range(n_cluster_groups):

            for i in range(r):
                partial_scores[i] = f32zero

            half_l2_norm_idx = first_centroid_global_idx + local_thread
            if local_thread < r:
                if half_l2_norm_idx < n_clusters:
                    l2norm = centroids_half_l2_norm[half_l2_norm_idx]
                else:
                    l2norm = inf
                local_centroids_half_l2_norm[local_thread] = l2norm

            global_window_col = first_centroid_global_idx + window_col

            first_dim_global_idx = 0

            for _1 in range(n_dim_windows):
                load_first_dim_local_idx = 0
                for load_iter in range(number_of_load_iter):
                    window_row = load_first_dim_local_idx + window_row_offset
                    global_window_row = first_dim_global_idx + window_row

                    if (global_window_row < dim) and (
                        global_window_col < n_clusters
                    ):
                        item = current_centroids[
                            global_window_row, global_window_col
                        ]
                    else:
                        item = f32zero

                    window_shm[window_row, window_col] = item

                    load_first_dim_local_idx += cluster_window_per_thread_group

                dpex.barrier()

                for d in range(h):
                    current_dim_global_idx = d + first_dim_global_idx
                    if (current_dim_global_idx < dim) and (global_thread < n):
                        # performance for the line thereafter relies on L1 cache
                        X_feature = X[current_dim_global_idx, global_thread]
                    else:
                        X_feature = f32zero
                    for i in range(r):
                        centroid_feature = window_shm[d, i]
                        partial_scores[i] += centroid_feature * X_feature

                dpex.barrier()

                first_dim_global_idx += h

            for i in range(r):
                current_score = (
                    local_centroids_half_l2_norm[i] - partial_scores[i]
                )
                if current_score < min_score:
                    min_score = current_score
                    min_idx = first_centroid_global_idx + i

            dpex.barrier()

            first_centroid_global_idx += r

        if global_thread >= n:
            return

        inertia[global_thread] = min_score

        warp_id = global_thread // warp_size
        dpex.atomic.add(
            centroid_counts_private_copies,
            (warp_id % nb_centroids_private_copies, min_idx),
            f32one,
        )
        for d in range(dim):
            dpex.atomic.add(
                centroids_private_copies,
                ((warp_id + d) % nb_centroids_private_copies, d, min_idx),
                X[d, global_thread],
            )

    return (
        nb_centroids_private_copies,
        fused_kernel_fixed_window[n_threads, thread_group_size],
    )


@cached_kernel_factory
def get_assignment_kernel_fixed_window(
    n,
    dim,
    n_clusters,
    warp_size,
    window_length_multiple,
    cluster_window_per_thread_group,
    number_of_load_iter,
):
    r = warp_size * window_length_multiple
    thread_group_size = r * cluster_window_per_thread_group
    h = number_of_load_iter * cluster_window_per_thread_group

    n_cluster_groups = math.ceil(n_clusters / r)
    n_threads = (math.ceil(n / thread_group_size)) * (thread_group_size)
    n_dim_windows = math.ceil(dim / h)
    window_shm_shape = (h, r)

    inf = float32(math.inf)
    f32zero = float32(0.0)
    f32two = float32(2)

    @dpex.kernel
    def assigment_kernel_fixed_window(
        X,
        current_centroids,
        centroids_half_l2_norm,
        inertia,
        assignments_idx,
    ):
        global_thread = dpex.get_global_id(0)
        local_thread = dpex.get_local_id(0)

        window_shm = dpex.local.array(shape=window_shm_shape, dtype=float32)
        local_centroids_half_l2_norm = dpex.local.array(shape=r, dtype=float32)
        partial_scores = dpex.private.array(shape=r, dtype=float32)

        first_centroid_global_idx = 0

        min_idx = 0
        min_score = inf

        X_l2_norm = f32zero

        window_col = local_thread % r
        window_row_offset = local_thread // r

        for _0 in range(n_cluster_groups):

            for i in range(r):
                partial_scores[i] = f32zero

            half_l2_norm_idx = first_centroid_global_idx + local_thread
            if local_thread < r:
                if half_l2_norm_idx < n_clusters:
                    l2norm = centroids_half_l2_norm[half_l2_norm_idx]
                else:
                    l2norm = inf
                local_centroids_half_l2_norm[local_thread] = l2norm

            global_window_col = first_centroid_global_idx + window_col

            first_dim_global_idx = 0

            for _1 in range(n_dim_windows):
                load_first_dim_local_idx = 0
                for load_iter in range(number_of_load_iter):
                    window_row = load_first_dim_local_idx + window_row_offset
                    global_window_row = first_dim_global_idx + window_row

                    if (global_window_row < dim) and (
                        global_window_col < n_clusters
                    ):
                        item = current_centroids[
                            global_window_row, global_window_col
                        ]
                    else:
                        item = f32zero

                    window_shm[window_row, window_col] = item

                    load_first_dim_local_idx += cluster_window_per_thread_group

                dpex.barrier()

                for d in range(h):
                    current_dim_global_idx = d + first_dim_global_idx
                    if (current_dim_global_idx < dim) and (global_thread < n):
                        # performance for the line thereafter relies on L1 cache
                        X_feature = X[current_dim_global_idx, global_thread]
                    else:
                        X_feature = f32zero
                    X_l2_norm += X_feature * X_feature
                    for i in range(r):
                        centroid_feature = window_shm[d, i]
                        partial_scores[i] += centroid_feature * X_feature

                dpex.barrier()

                first_dim_global_idx += h

            for i in range(r):
                current_score = (
                    local_centroids_half_l2_norm[i] - partial_scores[i]
                )
                if current_score < min_score:
                    min_score = current_score
                    min_idx = first_centroid_global_idx + i

            dpex.barrier()

            first_centroid_global_idx += r

        if global_thread >= n:
            return

        assignments_idx[global_thread] = min_idx
        inertia[global_thread] = X_l2_norm + (f32two * min_score)

    return assigment_kernel_fixed_window[n_threads, thread_group_size]
