import numba_dpex as dpex
from numba import float32, int32

import dpctl
import numpy as np
import math

from sklearn_numba_dpex.utils import (
    cached_kernel_factory,
)


@cached_kernel_factory
def make_initialize_to_zeros_1dim_int32_kernel(n_samples, work_group_size):

    global_size = math.ceil(n_samples / work_group_size) * work_group_size
    int32zero = int32(0)

    @dpex.kernel
    def initialize_to_zeros(x):
        sample_idx = dpex.get_global_id(0)

        if sample_idx >= n_samples:
            return

        x[sample_idx] = int32zero

    return initialize_to_zeros[global_size, work_group_size]


@cached_kernel_factory
def make_initialize_to_zeros_1dim_float32_kernel(n_samples, work_group_size):

    global_size = math.ceil(n_samples / work_group_size) * work_group_size
    f32zero = float32(0)

    @dpex.kernel
    def initialize_to_zeros(x):
        sample_idx = dpex.get_global_id(0)

        if sample_idx >= n_samples:
            return

        x[sample_idx] = f32zero

    return initialize_to_zeros[global_size, work_group_size]


@cached_kernel_factory
def make_initialize_to_zeros_2dim_float32_kernel(
    n_samples, n_features, work_group_size
):

    nb_items = n_samples * n_features
    global_size = math.ceil(nb_items / work_group_size) * work_group_size
    f32zero = float32(0.0)

    @dpex.kernel
    def initialize_to_zeros(x):
        item_idx = dpex.get_global_id(0)

        if item_idx >= nb_items:
            return

        i = item_idx // n_samples
        j = item_idx % n_samples
        x[i, j] = f32zero

    return initialize_to_zeros[global_size, work_group_size]


@cached_kernel_factory
def make_initialize_to_zeros_2dim_float32_kernel(
    n_samples, n_features, work_group_size
):

    nb_items = n_samples * n_features
    global_size = math.ceil(nb_items / work_group_size) * work_group_size
    f32zero = float32(0.0)

    @dpex.kernel
    def initialize_to_zeros(x):
        item_idx = dpex.get_global_id(0)

        if item_idx >= nb_items:
            return

        i = item_idx // n_samples
        j = item_idx % n_samples
        x[i, j] = f32zero

    return initialize_to_zeros[global_size, work_group_size]


@cached_kernel_factory
def make_initialize_to_zeros_3dim_float32_kernel(
    dim0, dim1, dim2, work_group_size
):

    nb_items = dim0 * dim1 * dim2
    stride0 = dim1 * dim2
    global_size = math.ceil(nb_items / work_group_size) * work_group_size
    f32zero = float32(0.0)

    @dpex.kernel
    def initialize_to_zeros(x):
        item_idx = dpex.get_global_id(0)

        if item_idx >= nb_items:
            return

        i = item_idx // stride0
        stride0_id = item_idx % stride0
        j = stride0_id // dim2
        k = stride0_id % dim2
        x[i, j, k] = f32zero

    return initialize_to_zeros[global_size, work_group_size]


@cached_kernel_factory
def make_copyto_kernel(n_samples, n_features, work_group_size):
    global_size = math.ceil(n_samples / work_group_size) * work_group_size

    @dpex.kernel
    def copyto_kernel(X, Y):
        sample_idx = dpex.get_global_id(0)

        if sample_idx >= n_samples:
            return

        for feature_idx in range(n_features):
            Y[feature_idx, sample_idx] = X[feature_idx, sample_idx]

    return copyto_kernel[global_size, work_group_size]


@cached_kernel_factory
def make_broadcast_division_kernel(n_samples, n_features, work_group_size):
    global_size = math.ceil(n_samples / work_group_size) * work_group_size

    @dpex.kernel
    def broadcast_division(X, v):
        sample_idx = dpex.get_global_id(0)

        if sample_idx >= n_samples:
            return

        divisor = v[sample_idx]

        for feature_idx in range(n_features):
            X[feature_idx, sample_idx] = X[feature_idx, sample_idx] / divisor

    return broadcast_division[global_size, work_group_size]


@cached_kernel_factory
def make_center_shift_kernel(n_samples, n_features, work_group_size):
    global_size = math.ceil(n_samples / work_group_size) * work_group_size
    f32zero = float32(0.0)

    @dpex.kernel
    def center_shift(previous_center, center, center_shift):
        sample_idx = dpex.get_global_id(0)

        if sample_idx >= n_samples:
            return

        tmp = f32zero

        for feature_idx in range(n_features):
            center_diff = (
                previous_center[feature_idx, sample_idx]
                - center[feature_idx, sample_idx]
            )
            tmp += center_diff * center_diff

        center_shift[sample_idx] = tmp

    return center_shift[global_size, work_group_size]


@cached_kernel_factory
def make_half_l2_norm_dim0_kernel(n_samples, n_features, work_group_size):
    global_size = math.ceil(n_samples / work_group_size) * work_group_size
    f32zero = float32(0.0)

    @dpex.kernel
    def half_l2_norm(X, result):
        sample_idx = dpex.get_global_id(0)

        if sample_idx >= n_samples:
            return

        l2_norm = f32zero

        for feature_idx in range(n_features):
            item = X[feature_idx, sample_idx]
            l2_norm += item * item

        result[sample_idx] = l2_norm / 2

    return half_l2_norm[global_size, work_group_size]


@cached_kernel_factory
def make_sum_reduction_1dim_kernel(n_samples, work_group_size, device):
    local_nb_iterations = math.floor(math.log2(work_group_size))
    f32zero = float32(0.0)

    @dpex.kernel
    def sum_reduction_kernel(v, w):
        group_id = dpex.get_group_id(0)
        local_work_id = dpex.get_local_id(0)
        first_work_id = local_work_id == 0

        n_samples = v.shape[0]

        shm = dpex.local.array(work_group_size, dtype=float32)

        first_sample_idx = group_id * work_group_size * 2
        augend_idx = first_sample_idx + local_work_id
        addend_idx = first_sample_idx + work_group_size + local_work_id

        if augend_idx >= n_samples:
            shm[local_work_id] = f32zero
        elif addend_idx >= n_samples:
            shm[local_work_id] = v[augend_idx]
        else:
            shm[local_work_id] = v[augend_idx] + v[addend_idx]

        dpex.barrier()
        current_size = work_group_size
        for i in range(local_nb_iterations):
            current_size = current_size // 2
            if local_work_id < current_size:
                shm[local_work_id] += shm[local_work_id + current_size]

            dpex.barrier()

        if first_work_id:
            w[group_id] = shm[0]

    _steps_data = []
    n_groups = n_samples
    while n_groups > 1:
        n_groups = math.ceil(n_groups / (2 * work_group_size))
        global_size = n_groups * work_group_size
        _steps_data.append(
            (
                sum_reduction_kernel[global_size, work_group_size],
                dpctl.tensor.empty(n_groups, dtype=np.float32, device=device),
            )
        )

    def sum_reduction(v):
        for sum_fn, w in _steps_data:
            sum_fn(v, w)
            v = w
        return v

    return sum_reduction


@cached_kernel_factory
def make_sum_reduction_2dim_kernel(n_samples, n_features, work_group_size):
    global_size = math.ceil(n_samples / work_group_size) * work_group_size
    f32zero = float32(0.0)

    @dpex.kernel
    def sum_reduction_kernel(v, w):
        sample_idx = dpex.get_global_id(0)
        if sample_idx >= n_samples:
            return
        tmp = f32zero
        for feature_idx in range(n_features):
            tmp += v[feature_idx, sample_idx]
        w[sample_idx] = tmp

    return sum_reduction_kernel[global_size, work_group_size]


@cached_kernel_factory
def make_sum_reduction_3dim_kernel(dim0, dim1, dim2, work_group_size):
    nb_groups_for_dim2 = math.ceil(dim2 / work_group_size)
    nb_threads_for_dim2 = nb_groups_for_dim2 * work_group_size
    global_size = nb_threads_for_dim2 * dim1
    f32zero = float32(0.0)

    @dpex.kernel
    def sum_reduction_kernel(v, w):
        group = dpex.get_group_id(0)
        thread = dpex.get_local_id(0)
        d1 = group // nb_groups_for_dim2
        d2 = ((group % nb_groups_for_dim2) * work_group_size) + thread
        if d2 >= dim2:
            return
        tmp = f32zero
        for d in range(dim0):
            tmp += v[d, d1, d2]
        w[d1, d2] = tmp

    return sum_reduction_kernel[global_size, work_group_size]


@cached_kernel_factory
def make_fused_fixed_window_kernel(
    n_samples,
    n_features,
    n_clusters,
    preferred_work_group_size_multiple,
    global_mem_cache_size,
    centroids_window_width_multiplier,
    centroids_window_height_ratio_multiplier,
    centroids_private_copies_max_cache_occupancy,
    work_group_size,
):
    window_nb_centroids = (
        preferred_work_group_size_multiple * centroids_window_width_multiplier
    )
    nb_of_window_features_per_work_group = (
        work_group_size // window_nb_centroids
    )
    window_nb_features = (
        centroids_window_height_ratio_multiplier
        * nb_of_window_features_per_work_group
    )

    nb_windows_per_feature = math.ceil(n_clusters / window_nb_centroids)
    global_size = (math.ceil(n_samples / work_group_size)) * (work_group_size)
    nb_windows_per_centroid = math.ceil(n_features / window_nb_features)
    centroids_window_shape = (window_nb_features, (window_nb_centroids + 1))

    inf = float32(math.inf)
    f32zero = float32(0.0)
    f32one = float32(1.0)

    nb_cluster_items = n_clusters * (n_features + 1)
    nb_cluster_bytes = 4 * nb_cluster_items
    nb_centroids_private_copies = int(
        (global_mem_cache_size * centroids_private_copies_max_cache_occupancy)
        // nb_cluster_bytes
    )

    @dpex.kernel
    def fused_kmeans(
        X_t,
        current_centroids_t,
        centroids_half_l2_norm,
        inertia,
        centroids_t_private_copies,
        centroid_counts_private_copies,
    ):
        sample_idx = dpex.get_global_id(0)
        local_work_id = dpex.get_local_id(0)

        centroids_window = dpex.local.array(
            shape=centroids_window_shape, dtype=float32
        )
        centroids_window_half_l2_norm = dpex.local.array(
            shape=window_nb_centroids, dtype=float32
        )
        partial_scores = dpex.private.array(
            shape=window_nb_centroids, dtype=float32
        )

        first_centroid_idx = 0

        min_idx = 0
        min_score = inf

        window_loading_centroid_idx = local_work_id % window_nb_centroids
        window_loading_feature_offset = local_work_id // window_nb_centroids

        for _0 in range(nb_windows_per_feature):

            for i in range(window_nb_centroids):
                partial_scores[i] = f32zero

            half_l2_norm_loading_idx = first_centroid_idx + local_work_id
            if local_work_id < window_nb_centroids:
                if half_l2_norm_loading_idx < n_clusters:
                    l2norm = centroids_half_l2_norm[half_l2_norm_loading_idx]
                else:
                    l2norm = inf
                centroids_window_half_l2_norm[local_work_id] = l2norm

            loading_centroid_idx = (
                first_centroid_idx + window_loading_centroid_idx
            )

            first_feature_idx = 0

            for _1 in range(nb_windows_per_centroid):

                centroid_window_first_loading_feature_idx = 0

                for _2 in range(centroids_window_height_ratio_multiplier):
                    window_loading_feature_idx = (
                        centroid_window_first_loading_feature_idx
                        + window_loading_feature_offset
                    )
                    loading_feature_idx = (
                        first_feature_idx + window_loading_feature_idx
                    )

                    if (loading_feature_idx < n_features) and (
                        loading_centroid_idx < n_clusters
                    ):
                        value = current_centroids_t[
                            loading_feature_idx, loading_centroid_idx
                        ]
                    else:
                        value = f32zero

                    centroids_window[
                        window_loading_feature_idx, window_loading_centroid_idx
                    ] = value

                    centroid_window_first_loading_feature_idx += (
                        nb_of_window_features_per_work_group
                    )

                dpex.barrier()

                for window_feature_idx in range(window_nb_features):
                    feature_idx = window_feature_idx + first_feature_idx
                    if (feature_idx < n_features) and (sample_idx < n_samples):
                        # performance for the line thereafter relies on L1 cache
                        X_value = X_t[feature_idx, sample_idx]
                    else:
                        X_value = f32zero
                    for window_centroid_idx in range(window_nb_centroids):
                        centroid_value = centroids_window[
                            window_feature_idx, window_centroid_idx
                        ]
                        partial_scores[window_centroid_idx] += (
                            centroid_value * X_value
                        )

                dpex.barrier()

                first_feature_idx += window_nb_features

            for i in range(window_nb_centroids):
                current_score = (
                    centroids_window_half_l2_norm[i] - partial_scores[i]
                )
                if current_score < min_score:
                    min_score = current_score
                    min_idx = first_centroid_idx + i

            dpex.barrier()

            first_centroid_idx += window_nb_centroids

        if sample_idx >= n_samples:
            return

        inertia[sample_idx] = min_score

        privatization_idx = (
            sample_idx // preferred_work_group_size_multiple
        ) % nb_centroids_private_copies
        dpex.atomic.add(
            centroid_counts_private_copies,
            (privatization_idx, min_idx),
            f32one,
        )

        for feature_idx in range(n_features):
            dpex.atomic.add(
                centroids_t_private_copies,
                # ((privatization_idx + feature_idx) % nb_centroids_private_copies, feature_idx, min_idx),
                (privatization_idx, feature_idx, min_idx),
                X_t[feature_idx, sample_idx],
            )

    return (
        nb_centroids_private_copies,
        fused_kmeans[global_size, work_group_size],
    )


@cached_kernel_factory
def make_assignment_fixed_window_kernel(
    n_samples,
    n_features,
    n_clusters,
    preferred_work_group_size_multiple,
    centroids_window_width_multiplier,
    centroids_window_height_ratio_multiplier,
    work_group_size,
):
    window_nb_centroids = (
        preferred_work_group_size_multiple * centroids_window_width_multiplier
    )
    nb_of_window_features_per_work_group = (
        work_group_size // window_nb_centroids
    )
    window_nb_features = (
        centroids_window_height_ratio_multiplier
        * nb_of_window_features_per_work_group
    )

    nb_windows_per_feature = math.ceil(n_clusters / window_nb_centroids)
    global_size = (math.ceil(n_samples / work_group_size)) * (work_group_size)
    nb_windows_per_centroid = math.ceil(n_features / window_nb_features)
    centroids_window_shape = (window_nb_features, (window_nb_centroids + 1))

    inf = float32(math.inf)
    f32zero = float32(0.0)
    f32two = float32(2)

    @dpex.kernel
    def assignment(
        X_t,
        current_centroids_t,
        centroids_half_l2_norm,
        inertia,
        assignments_idx,
    ):
        sample_idx = dpex.get_global_id(0)
        local_work_id = dpex.get_local_id(0)

        centroids_window = dpex.local.array(
            shape=centroids_window_shape, dtype=float32
        )
        centroids_window_half_l2_norm = dpex.local.array(
            shape=window_nb_centroids, dtype=float32
        )
        partial_scores = dpex.private.array(
            shape=window_nb_centroids, dtype=float32
        )

        first_centroid_idx = 0

        min_idx = 0
        min_score = inf

        X_l2_norm = f32zero

        window_loading_centroid_idx = local_work_id % window_nb_centroids
        window_loading_feature_offset = local_work_id // window_nb_centroids

        for _0 in range(nb_windows_per_feature):

            for i in range(window_nb_centroids):
                partial_scores[i] = f32zero

            half_l2_norm_loading_idx = first_centroid_idx + local_work_id
            if local_work_id < window_nb_centroids:
                if half_l2_norm_loading_idx < n_clusters:
                    l2norm = centroids_half_l2_norm[half_l2_norm_loading_idx]
                else:
                    l2norm = inf
                centroids_window_half_l2_norm[local_work_id] = l2norm

            loading_centroid_idx = (
                first_centroid_idx + window_loading_centroid_idx
            )

            first_feature_idx = 0

            for _1 in range(nb_windows_per_centroid):

                centroid_window_first_loading_feature_idx = 0

                for _2 in range(centroids_window_height_ratio_multiplier):
                    window_loading_feature_idx = (
                        centroid_window_first_loading_feature_idx
                        + window_loading_feature_offset
                    )
                    loading_feature_idx = (
                        first_feature_idx + window_loading_feature_idx
                    )

                    if (loading_feature_idx < n_features) and (
                        loading_centroid_idx < n_clusters
                    ):
                        value = current_centroids_t[
                            loading_feature_idx, loading_centroid_idx
                        ]
                    else:
                        value = f32zero

                    centroids_window[
                        window_loading_feature_idx, window_loading_centroid_idx
                    ] = value

                    centroid_window_first_loading_feature_idx += (
                        nb_of_window_features_per_work_group
                    )

                dpex.barrier()

                for window_feature_idx in range(window_nb_features):
                    feature_idx = window_feature_idx + first_feature_idx
                    if (feature_idx < n_features) and (sample_idx < n_samples):
                        # performance for the line thereafter relies on L1 cache
                        X_value = X_t[feature_idx, sample_idx]
                    else:
                        X_value = f32zero
                    if _0 == 0:
                        X_l2_norm += X_value * X_value
                    for window_centroid_idx in range(window_nb_centroids):
                        centroid_value = centroids_window[
                            window_feature_idx, window_centroid_idx
                        ]
                        partial_scores[window_centroid_idx] += (
                            centroid_value * X_value
                        )

                dpex.barrier()

                first_feature_idx += window_nb_features

            for i in range(window_nb_centroids):
                current_score = (
                    centroids_window_half_l2_norm[i] - partial_scores[i]
                )
                if current_score < min_score:
                    min_score = current_score
                    min_idx = first_centroid_idx + i

            dpex.barrier()

            first_centroid_idx += window_nb_centroids

        if sample_idx >= n_samples:
            return

        assignments_idx[sample_idx] = min_idx
        inertia[sample_idx] = X_l2_norm + (f32two * min_score)

    return assignment[global_size, work_group_size]
