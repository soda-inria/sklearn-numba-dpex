import math

import numpy as np
import numba_dpex as dpex


def _make_initialize_window_kernel_funcs(
    n_clusters,
    n_features,
    work_group_size,
    window_n_centroids,
    window_n_features,
    dtype,
    initialize_window_of_centroids_half_l2_norms=False,
):
    zero = dtype(0.0)
    zero_idx = np.int64(0)
    inf = dtype(math.inf)

    n_window_features_per_work_group = work_group_size // window_n_centroids

    centroids_window_height_ratio_multiplier = (
        window_n_features // n_window_features_per_work_group
    )

    @dpex.func
    # fmt: off
    def _load_window_of_centroids_and_features(
        first_feature_idx,              # PARAM
        loading_centroid_idx,           # PARAM
        window_loading_centroid_idx,    # PARAM
        window_loading_feature_offset,  # PARAM
        current_centroids_t,            # IN
        centroids_window,               # OUT
    ):
    # fmt: on
        centroid_window_first_loading_feature_idx = zero_idx

        # The work items in the work group cooperatively load the values in
        # shared memory. At each iteration, the work item loads one value and
        # adjacent work items load adjacent values.
        for _2 in range(centroids_window_height_ratio_multiplier):
            window_loading_feature_idx = (
                centroid_window_first_loading_feature_idx
                + window_loading_feature_offset
            )
            loading_feature_idx = first_feature_idx + window_loading_feature_idx

            if (loading_feature_idx < n_features) and (
                loading_centroid_idx < n_clusters
            ):
                value = current_centroids_t[loading_feature_idx, loading_centroid_idx]
            else:
                value = zero

            centroids_window[
                window_loading_feature_idx, window_loading_centroid_idx
            ] = value

            centroid_window_first_loading_feature_idx += (
                n_window_features_per_work_group
            )

    @dpex.func
    def _initialize_results(results):
        # Initialize the partial pseudo inertia dot product for each
        # of the window_n_centroids centroids in the window.
        for i in range(window_n_centroids):
            results[i] = zero

    if not initialize_window_of_centroids_half_l2_norms:
        return _initialize_results, _load_window_of_centroids_and_features

    @dpex.func
    # fmt: off
    def _initialize_window_of_centroids(
        local_work_id,                  # PARAM
        first_centroid_idx,             # PARAM
        centroids_half_l2_norm,         # IN
        window_of_centroids_half_l2_norms,  # OUT
        results,                        # OUT
    ):
    # fmt: on
        _initialize_results(results)

        # The first `window_n_centroids` work items cooperate on loading the
        # values of centroids_half_l2_norm relevant to current window. Each work
        # item loads one single value.
        half_l2_norm_loading_idx = first_centroid_idx + local_work_id
        if local_work_id < window_n_centroids:
            if half_l2_norm_loading_idx < n_clusters:
                l2norm = centroids_half_l2_norm[half_l2_norm_loading_idx]
            else:
                l2norm = inf
            window_of_centroids_half_l2_norms[local_work_id] = l2norm

    return _initialize_window_of_centroids, _load_window_of_centroids_and_features


def _make_accumulate_sum_of_ops_kernel_func(
    n_samples, n_features, window_n_features, window_n_centroids, ops, dtype
):

    zero = dtype(0.0)

    accumulate_dot_product = ops == "product"
    accumulate_squared_diff = ops == "squared_diff"

    if not accumulate_dot_product and not accumulate_squared_diff:
        raise ValueError(
            f'Expected ops to take values "product" or "squared_diff", got "{ops}" '
            "instead."
        )

    @dpex.func
    # fmt: off
    def _accumulate_sum_of_ops(
        sample_idx,          # PARAM
        first_feature_idx,   # PARAM
        X_t,                 # IN
        centroids_window,    # IN
        result,              # OUT
    ):
    # fmt: on
        for window_feature_idx in range(window_n_features):

            feature_idx = window_feature_idx + first_feature_idx
            if (feature_idx < n_features) and (sample_idx < n_samples):
                # performance for the line thereafter relies on L1 cache
                X_value = X_t[feature_idx, sample_idx]
            else:
                X_value = zero

            # For this given feature, loop on all centroids in the current
            # window and accumulate the partial results
            for window_centroid_idx in range(window_n_centroids):
                centroid_value = centroids_window[window_feature_idx, window_centroid_idx]
                if accumulate_dot_product:
                    result[window_centroid_idx] += centroid_value * X_value
                else:
                    diff = centroid_value - X_value
                    result[window_centroid_idx] += diff * diff

    return _accumulate_sum_of_ops


def _make_update_closest_centroid_kernel_func(window_n_centroids):
    @dpex.func
    # fmt: off
    def _update_closest_centroid(
        first_centroid_idx,             # PARAM
        min_idx,                        # PARAM
        min_sample_pseudo_inertia,      # PARAM
        window_of_centroids_half_l2_norms,  # IN
        dot_products,                   # IN
    ):
    # fmt: on
        for i in range(window_n_centroids):
            current_sample_pseudo_inertia = (
                window_of_centroids_half_l2_norms[i] - dot_products[i]
            )
            if current_sample_pseudo_inertia < min_sample_pseudo_inertia:
                min_sample_pseudo_inertia = current_sample_pseudo_inertia
                min_idx = first_centroid_idx + i
        return min_idx, min_sample_pseudo_inertia

    return _update_closest_centroid
