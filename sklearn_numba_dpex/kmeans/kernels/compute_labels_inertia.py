import math
from functools import lru_cache

import numba_dpex as dpex

from ._base_kmeans_kernel_funcs import (
    _make_initialize_window_kernel_funcs,
    _make_update_closest_centroid_kernel_func,
    _make_accumulate_dot_products_kernel_func,
)

# NB: refer to the definition of the main lloyd function for a more comprehensive
# inline commenting of the kernel.


@lru_cache
def make_compute_labels_inertia_fixed_window_kernel(
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
        _initialize_window_of_centroids,
        _load_window_of_centroids_and_features,
    ) = _make_initialize_window_kernel_funcs(
        n_clusters,
        n_features,
        work_group_size,
        window_n_centroids,
        centroids_window_height,
        dtype,
    )

    _update_closest_centroid = _make_update_closest_centroid_kernel_func(
        window_n_centroids
    )

    _accumulate_dot_products = _make_accumulate_dot_products_kernel_func(
        n_samples,
        n_features,
        centroids_window_height,
        window_n_centroids,
        with_X_l2_norm=False,
        dtype=dtype,
    )

    _accumulate_dot_products_and_X_l2_norm = _make_accumulate_dot_products_kernel_func(
        n_samples,
        n_features,
        centroids_window_height,
        window_n_centroids,
        with_X_l2_norm=True,
        dtype=dtype,
    )

    n_windows_per_feature = math.ceil(n_clusters / window_n_centroids)
    n_windows_per_centroid = math.ceil(n_features / centroids_window_height)

    centroids_window_shape = (centroids_window_height, (window_n_centroids + 1))

    inf = dtype(math.inf)
    two = dtype(2)

    @dpex.kernel
    # fmt: off
    def labels_inertia(
        X_t,                      # IN READ-ONLY   (n_features, n_samples)
        centroids_t,              # IN             (n_features, n_clusters)
        centroids_half_l2_norm,   # IN             (n_clusters,)
        per_sample_inertia,       # OUT            (n_samples,)
        assignments_idx,          # OUT            (n_samples,)
    ):
    # fmt: on
        """
        This kernel is very similar to the fused fixed kernel, with a few
        differences:
            - the closest cluster to the current sample is stored into the array
            assignments_idx instead of being discarded
            - The exact inertia is computed, instead of pseudo-inertia
            - the update step is skipped
        """

        sample_idx = dpex.get_global_id(0)
        local_work_id = dpex.get_local_id(0)

        centroids_window = dpex.local.array(shape=centroids_window_shape, dtype=dtype)
        centroids_window_half_l2_norm = dpex.local.array(
            shape=window_n_centroids, dtype=dtype
        )
        dot_products = dpex.private.array(shape=window_n_centroids, dtype=dtype)

        first_centroid_idx = 0

        min_idx = 0
        min_sample_pseudo_inertia = inf

        window_loading_centroid_idx = local_work_id % window_n_centroids
        window_loading_feature_offset = local_work_id // window_n_centroids

        for _0 in range(n_windows_per_feature):
            _initialize_window_of_centroids(
                local_work_id,
                first_centroid_idx,
                centroids_half_l2_norm,
                centroids_window_half_l2_norm,
                dot_products,
            )

            loading_centroid_idx = first_centroid_idx + window_loading_centroid_idx

            first_feature_idx = 0

            for _1 in range(n_windows_per_centroid):
                _load_window_of_centroids_and_features(
                    first_feature_idx,
                    loading_centroid_idx,
                    window_loading_centroid_idx,
                    window_loading_feature_offset,
                    centroids_t,
                    centroids_window,
                )

                dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

                # Loop on all features in the current window and accumulate the dot
                # products
                if _0 == 0:
                    X_l2_norm = _accumulate_dot_products_and_X_l2_norm(
                        sample_idx,
                        first_feature_idx,
                        X_t,
                        centroids_window,
                        dot_products,
                    )
                else:
                    _accumulate_dot_products(
                        sample_idx,
                        first_feature_idx,
                        X_t,
                        centroids_window,
                        dot_products,
                    )

                dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

                first_feature_idx += centroids_window_height

            min_idx, min_sample_pseudo_inertia = _update_closest_centroid(
                first_centroid_idx,
                min_idx,
                min_sample_pseudo_inertia,
                centroids_window_half_l2_norm,
                dot_products,
            )

            dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

            first_centroid_idx += window_n_centroids

        # No update step, only store min_idx in the output array
        if sample_idx >= n_samples:
            return

        assignments_idx[sample_idx] = min_idx
        per_sample_inertia[sample_idx] = X_l2_norm + (two * min_sample_pseudo_inertia)

    global_size = (math.ceil(n_samples / work_group_size)) * (work_group_size)
    return labels_inertia[global_size, work_group_size]
