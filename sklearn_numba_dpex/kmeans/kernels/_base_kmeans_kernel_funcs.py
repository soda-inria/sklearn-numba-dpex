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
):
    zero = dtype(0.0)
    zero_idx = np.int64(0)
    inf = dtype(math.inf)

    @dpex.func
    # fmt: off
    def _initialize_window_of_centroids(
        local_work_id,                  # PARAM
        first_centroid_idx,             # PARAM
        centroids_half_l2_norm,         # IN
        centroids_window_half_l2_norm,  # OUT
        dot_products,                   # OUT
    ):
    # fmt: on
        # Initialize the partial pseudo inertia dot product for each 
        # of the window_n_centroids centroids in the window.
        for i in range(window_n_centroids):
            dot_products[i] = zero

        # The first `window_n_centroids` work items cooperate on loading the
        # values of centroids_half_l2_norm relevant to current window. Each work
        # item loads one single value.
        half_l2_norm_loading_idx = first_centroid_idx + local_work_id
        if local_work_id < window_n_centroids:
            if half_l2_norm_loading_idx < n_clusters:
                l2norm = centroids_half_l2_norm[half_l2_norm_loading_idx]
            else:
                l2norm = inf
            centroids_window_half_l2_norm[local_work_id] = l2norm

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

    return _initialize_window_of_centroids, _load_window_of_centroids_and_features


def _make_accumulate_dot_products_kernel_func(
    n_samples, n_features, window_n_features, window_n_centroids, with_X_l2_norm, dtype
):
    # TODO: this factorization is disappointing but I don't think numba.func can support
    # more than this level of abstraction. I tried using a const bool to swtich
    # X_l2_norm computation on and off coupled with explicitly passing the `signature`
    # argument, but to no result (the JIT will fail).

    zero = dtype(0.0)

    @dpex.func
    def _get_X_value(sample_idx, window_feature_idx, first_feature_idx, X_t):
        feature_idx = window_feature_idx + first_feature_idx
        if (feature_idx < n_features) and (sample_idx < n_samples):
            # performance for the line thereafter relies on L1 cache
            return X_t[feature_idx, sample_idx]
        else:
            return zero

    @dpex.func
    # fmt: off
    def _accumulate_one_feature(
        X_value,              # PARAM
        window_feature_idx,   # PARAM
        centroids_window,     # IN
        dot_products          # OUT
    ):
    # fmt: on
        # For this given feature, loop on all centroids in the current
        # window and accumulate the dot products
        for window_centroid_idx in range(window_n_centroids):
            centroid_value = centroids_window[window_feature_idx, window_centroid_idx]
            dot_products[window_centroid_idx] += centroid_value * X_value

    if with_X_l2_norm:

        @dpex.func
        # fmt: off
        def _accumulate_dot_products(
            sample_idx,          # PARAM
            first_feature_idx,   # PARAM
            X_t,                 # IN
            centroids_window,    # IN
            dot_products,        # OUT
        ):
        # fmt: on
            X_l2_norm = zero

            # Loop on all features in the current window and accumulate the dot
            # products
            for window_feature_idx in range(window_n_features):
                X_value = _get_X_value(
                    sample_idx, window_feature_idx, first_feature_idx, X_t
                )
                _accumulate_one_feature(
                    X_value, window_feature_idx, centroids_window, dot_products
                )
                X_l2_norm += X_value * X_value

            return X_l2_norm

    else:

        @dpex.func
        # fmt: off
        def _accumulate_dot_products(
            sample_idx,          # PARAM
            first_feature_idx,   # PARAM
            X_t,                 # IN
            centroids_window,    # IN
            dot_products,        # OUT
        ):
        # fmt: on

            for window_feature_idx in range(window_n_features):
                X_value = _get_X_value(
                    sample_idx, window_feature_idx, first_feature_idx, X_t
                )
                _accumulate_one_feature(
                    X_value, window_feature_idx, centroids_window, dot_products
                )

    return _accumulate_dot_products


def _make_update_closest_centroid_kernel_func(window_n_centroids):
    @dpex.func
    # fmt: off
    def _update_closest_centroid(
        first_centroid_idx,             # PARAM
        min_idx,                        # PARAM
        min_sample_pseudo_inertia,      # PARAM
        centroids_window_half_l2_norm,  # IN
        dot_products,                   # IN
    ):
    # fmt: on
        for i in range(window_n_centroids):
            current_sample_pseudo_inertia = (
                centroids_window_half_l2_norm[i] - dot_products[i]
            )
            if current_sample_pseudo_inertia < min_sample_pseudo_inertia:
                min_sample_pseudo_inertia = current_sample_pseudo_inertia
                min_idx = first_centroid_idx + i
        return min_idx, min_sample_pseudo_inertia

    return _update_closest_centroid
