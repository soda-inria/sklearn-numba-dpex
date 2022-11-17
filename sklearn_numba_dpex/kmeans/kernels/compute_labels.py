import math
from functools import lru_cache

import numpy as np
import numba_dpex as dpex

from ._base_kmeans_kernel_funcs import (
    make_pairwise_ops_base_kernel_funcs,
    make_update_closest_centroid_kernel_func,
)

# NB: refer to the definition of the main lloyd function for a more comprehensive
# inline commenting of the kernel.


@lru_cache
def make_label_assignment_fixed_window_kernel(
    n_samples,
    n_features,
    n_clusters,
    preferred_work_group_size_multiple,
    centroids_window_width_multiplier,
    centroids_window_height,
    work_group_size,
    dtype,
):

    window_n_centroids = (
        preferred_work_group_size_multiple * centroids_window_width_multiplier
    )

    (
        initialize_window_of_centroids,
        load_window_of_centroids_and_features,
        accumulate_dot_products,
    ) = make_pairwise_ops_base_kernel_funcs(
        n_samples,
        n_features,
        n_clusters,
        centroids_window_height,
        window_n_centroids,
        ops="product",
        dtype=dtype,
        work_group_size=work_group_size,
        initialize_window_of_centroids_half_l2_norms=True,
    )

    update_closest_centroid = make_update_closest_centroid_kernel_func(
        n_clusters, window_n_centroids
    )

    n_windows_for_centroids = math.ceil(n_clusters / window_n_centroids)
    n_windows_for_features = math.ceil(n_features / centroids_window_height)
    last_centroid_window_idx = n_windows_for_centroids - 1
    last_feature_window_idx = n_windows_for_features - 1

    centroids_window_shape = (centroids_window_height, (window_n_centroids + 1))

    inf = dtype(math.inf)
    zero_idx = np.int64(0)

    @dpex.kernel
    # fmt: off
    def assignment(
        X_t,                      # IN READ-ONLY   (n_features, n_samples)
        centroids_t,              # IN READ-ONLY   (n_features, n_clusters)
        centroids_half_l2_norm,   # IN             (n_clusters,)
        assignments_idx,          # OUT            (n_samples,)
    ):
    # fmt: on

        sample_idx = dpex.get_global_id(zero_idx)
        local_work_id = dpex.get_local_id(zero_idx)

        centroids_window = dpex.local.array(shape=centroids_window_shape, dtype=dtype)
        window_of_centroids_half_l2_norms = dpex.local.array(
            shape=window_n_centroids, dtype=dtype
        )
        dot_products = dpex.private.array(shape=window_n_centroids, dtype=dtype)

        first_centroid_idx = zero_idx

        min_idx = zero_idx
        min_sample_pseudo_inertia = inf

        window_loading_centroid_idx = local_work_id % window_n_centroids
        window_loading_feature_offset = local_work_id // window_n_centroids

        for _0 in range(n_windows_for_centroids):
            is_last_centroid_window = _0 == last_centroid_window_idx

            initialize_window_of_centroids(
                local_work_id,
                first_centroid_idx,
                centroids_half_l2_norm,
                window_of_centroids_half_l2_norms,
                dot_products,
                is_last_centroid_window
            )

            loading_centroid_idx = first_centroid_idx + window_loading_centroid_idx

            first_feature_idx = zero_idx

            for _1 in range(n_windows_for_features):
                is_last_feature_window = _1 == last_feature_window_idx
                load_window_of_centroids_and_features(
                    first_feature_idx,
                    loading_centroid_idx,
                    window_loading_centroid_idx,
                    window_loading_feature_offset,
                    centroids_t,
                    centroids_window,
                    is_last_feature_window
                )

                dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

                accumulate_dot_products(
                    sample_idx,
                    first_feature_idx,
                    X_t,
                    centroids_window,
                    dot_products,
                    is_last_feature_window,
                    is_last_centroid_window
                )

                dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

                first_feature_idx += centroids_window_height

            min_idx, min_sample_pseudo_inertia = update_closest_centroid(
                first_centroid_idx,
                min_idx,
                min_sample_pseudo_inertia,
                window_of_centroids_half_l2_norms,
                dot_products,
                is_last_centroid_window
            )

            dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

            first_centroid_idx += window_n_centroids

        # No update step, only store min_idx in the output array
        if sample_idx >= n_samples:
            return

        assignments_idx[sample_idx] = min_idx

    global_size = (math.ceil(n_samples / work_group_size)) * (work_group_size)
    return assignment[global_size, work_group_size]
