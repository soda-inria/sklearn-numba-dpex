import numba_dpex as dpex


def make_pairwise_ops_base_kernel_funcs(
    n_samples,
    n_features,
    n_clusters,
    window_n_features,
    window_n_centroids,
    ops,
    dtype,
    initialize_window_of_centroids_half_l2_norms=False,
):
    # The kernel funcs in this file must behave differently depending on whether the
    # window over the array of centroids (which has a fixed size):
    #    - covers a full set of elements of the array (_full window_),
    #    - covers a partial set of elements (_last window_), which can happen for
    #      the last windows along each dimension if the size of the window along
    #      said dimensions does not divide the size of the array of centroids.
    #
    # Indeed, for such windows, the loops on the elements of the window must  be
    # interrupted early to not include compute that would correspond to out-of-bounds
    # elements in the array of centroids (which would, in this case, cause extra
    # compute time that, depending on the case, could noticeably hurt performance).

    # Unfortunately, `numba_dpex` does not offer tools that would enable at the same
    # time re-using the same kernel functions for the different cases, and passing the
    # relevant variables (here, `window_n_centroids`, `last_window_n_centroids`,
    # `window_n_features` and `last_window_n_features`), whose value is known at
    # compile time, as constant variables (which can benefit performance). This
    # function proposes a best effort at keeping things the most possibly factorized
    # and readable, while enabling passing the variables as compile-time constants.
    # It consists in instanciating a kernel functions for each different value the
    # variables can take, and adding conditional switches to use the appropriate
    # instance depending on the state of the main loop over the windows
    # (`is_last_centroid_window`, `is_last_feature_window`)

    kmeans_kernel_func_factory = _KMeansKernelFuncFactory(
        n_samples,
        n_features,
        n_clusters,
        ops,
        dtype,
        initialize_window_of_centroids_half_l2_norms,
    )

    last_window_n_centroids = n_clusters % window_n_centroids or window_n_centroids
    last_window_n_features = n_features % window_n_features or window_n_features

    make_initialize_window_kernel_func = (
        kmeans_kernel_func_factory.make_initialize_window_kernel_func
    )

    initialize_full_window_of_centroids = make_initialize_window_kernel_func(
        window_n_centroids
    )

    initialize_last_window_of_centroids = make_initialize_window_kernel_func(
        last_window_n_centroids
    )

    if initialize_window_of_centroids_half_l2_norms:

        @dpex.func
        def initialize_window_of_centroids(
            local_work_id,
            first_centroid_idx,
            centroids_half_l2_norm,
            is_last_centroid_window,
            # OUT
            window_of_centroids_half_l2_norms,
            dot_products,
        ):
            if is_last_centroid_window:
                initialize_last_window_of_centroids(
                    local_work_id,
                    first_centroid_idx,
                    centroids_half_l2_norm,
                    # OUT
                    window_of_centroids_half_l2_norms,
                    dot_products,
                )
            else:
                initialize_full_window_of_centroids(
                    local_work_id,
                    first_centroid_idx,
                    centroids_half_l2_norm,
                    # OUT
                    window_of_centroids_half_l2_norms,
                    dot_products,
                )

    else:

        @dpex.func
        def initialize_window_of_centroids(
            is_last_centroid_window,
            # OUT
            dot_products,
        ):
            if is_last_centroid_window:
                initialize_last_window_of_centroids(dot_products)
            else:
                initialize_full_window_of_centroids(dot_products)

    load_window_of_centroids_and_features = (
        kmeans_kernel_func_factory.make_load_window_kernel_func()
    )

    make_accumulate_sum_of_ops_kernel_func = (
        kmeans_kernel_func_factory.make_accumulate_sum_of_ops_kernel_func
    )

    accumulate_full_window_dot_products = make_accumulate_sum_of_ops_kernel_func(
        window_n_features, window_n_centroids
    )

    accumulate_last_centroid_window_dot_products = (
        make_accumulate_sum_of_ops_kernel_func(
            window_n_features, last_window_n_centroids
        )
    )

    accumulate_last_feature_window_dot_products = (
        make_accumulate_sum_of_ops_kernel_func(
            last_window_n_features, window_n_centroids
        )
    )

    accumulate_last_centroid_and_last_feature_window_dot_products = (
        make_accumulate_sum_of_ops_kernel_func(
            last_window_n_features, last_window_n_centroids
        )
    )

    @dpex.func
    def accumulate_dot_products(
        sample_idx,
        first_feature_idx,
        X_t,
        centroids_window,
        is_last_feature_window,
        is_last_centroid_window,
        # OUT
        dot_products,
    ):
        if is_last_feature_window and is_last_centroid_window:
            accumulate_last_centroid_and_last_feature_window_dot_products(
                sample_idx,
                first_feature_idx,
                X_t,
                centroids_window,
                # OUT
                dot_products,
            )
        elif is_last_feature_window:
            accumulate_last_feature_window_dot_products(
                sample_idx, first_feature_idx, X_t, centroids_window, dot_products
            )
        elif is_last_centroid_window:
            accumulate_last_centroid_window_dot_products(
                sample_idx, first_feature_idx, X_t, centroids_window, dot_products
            )
        else:
            accumulate_full_window_dot_products(
                sample_idx, first_feature_idx, X_t, centroids_window, dot_products
            )

    return (
        initialize_window_of_centroids,
        load_window_of_centroids_and_features,
        accumulate_dot_products,
    )


class _KMeansKernelFuncFactory:
    def __init__(
        self,
        n_samples,
        n_features,
        n_clusters,
        ops,
        dtype,
        initialize_window_of_centroids_half_l2_norms,
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_clusters = n_clusters

        self.accumulate_dot_product = ops == "product"
        self.accumulate_squared_diff = ops == "squared_diff"

        if not self.accumulate_dot_product and not self.accumulate_squared_diff:
            raise ValueError(
                f'Expected ops to take values "product" or "squared_diff", got "{ops}" '
                "instead."
            )

        self.dtype = dtype
        self.initialize_window_of_centroids_half_l2_norms = (
            initialize_window_of_centroids_half_l2_norms
        )

    def make_initialize_window_kernel_func(self, window_n_centroids):
        zero = self.dtype(0.0)

        @dpex.func
        def _initialize_results(results):
            # Initialize the partial pseudo inertia dot product for each
            # of the window_n_centroids centroids in the window.
            for i in range(window_n_centroids):
                results[i] = zero

        if not self.initialize_window_of_centroids_half_l2_norms:
            return _initialize_results

        @dpex.func
        # fmt: off
        def _initialize_window_of_centroids(
            local_work_id,                      # PARAM
            first_centroid_idx,                 # PARAM
            centroids_half_l2_norm,             # IN       (self.n_clusters,)
            window_of_centroids_half_l2_norms,  # OUT      (window_n_centroids,)
            results,                            # OUT      (window_n_centroids,)
        ):
            # fmt: on
            _initialize_results(results)

            # The first `window_n_centroids` work items cooperate on loading the
            # values of centroids_half_l2_norm relevant to current window. Each work
            # item loads one single value.
            if local_work_id < window_n_centroids:
                half_l2_norm_loading_idx = first_centroid_idx + local_work_id
                window_of_centroids_half_l2_norms[local_work_id] = (
                    centroids_half_l2_norm[half_l2_norm_loading_idx]
                )

        return _initialize_window_of_centroids

    def make_load_window_kernel_func(self):
        n_features = self.n_features
        n_clusters = self.n_clusters

        zero = self.dtype(0.0)

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
            # The work items in the work group cooperatively load the values in shared
            # memory. At each iteration, the work item loads one value and adjacent work
            # items load adjacent values.
            loading_feature_idx = first_feature_idx + window_loading_feature_offset

            # TODO: the following condition can be cheaper to compute if we know that
            # it's the last window or not.
            if (loading_feature_idx < n_features) and (
                loading_centroid_idx < n_clusters
             ):
                value = current_centroids_t[loading_feature_idx, loading_centroid_idx]
            else:
                # TODO: this branching is not necessary ?
                value = zero

            centroids_window[
                window_loading_feature_offset, window_loading_centroid_idx
            ] = value

        return _load_window_of_centroids_and_features

    def make_accumulate_sum_of_ops_kernel_func(
        self, window_n_features, window_n_centroids
    ):

        zero = self.dtype(0.0)
        n_samples = self.n_samples
        accumulate_dot_product = self.accumulate_dot_product

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
                # TODO: the following condition can be cheaper to compute if we know
                # that it's the last window or not. Also, it should be at the outside of
                # the loop.
                if sample_idx < n_samples:
                    # performance for the line thereafter relies on L1 cache
                    X_value = X_t[feature_idx, sample_idx]
                else:
                    X_value = zero

                # For this given feature, loop on all centroids in the current
                # window and accumulate the partial results
                for window_centroid_idx in range(window_n_centroids):
                    # TODO: here, reading the values in the sliding windows once then
                    # shifting within the private memory of the sub groups would be more
                    # efficient, but `numba_dpex` lacks the corresponding intrinsics.
                    # Those intrinsics are defined in the SYCL spec and exist in
                    # `numba.cuda` so there's hope it also happens for `numba_dpex`.
                    # see https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_shift_left_and_shift_right  # noqa
                    centroid_value = (
                        centroids_window[window_feature_idx, window_centroid_idx]
                    )
                    if accumulate_dot_product:
                        result[window_centroid_idx] += centroid_value * X_value
                    else:
                        diff = centroid_value - X_value
                        result[window_centroid_idx] += diff * diff

        return _accumulate_sum_of_ops


def make_update_closest_centroid_kernel_func(n_clusters, window_n_centroids):

    last_window_n_centroids = ((n_clusters - 1) % window_n_centroids) + 1

    update_closest_centroid_full = _make_update_closest_centroid_kernel_func(
        window_n_centroids
    )

    update_last_closest_centroid = _make_update_closest_centroid_kernel_func(
        last_window_n_centroids
    )

    @dpex.func
    def update_closest_centroid(
        first_centroid_idx,
        min_idx,
        min_sample_pseudo_inertia,
        window_of_centroids_half_l2_norms,
        is_last_centroid_window,
        dot_products,
    ):

        if is_last_centroid_window:
            return update_last_closest_centroid(
                first_centroid_idx,
                min_idx,
                min_sample_pseudo_inertia,
                window_of_centroids_half_l2_norms,
                dot_products,
            )
        else:
            return update_closest_centroid_full(
                first_centroid_idx,
                min_idx,
                min_sample_pseudo_inertia,
                window_of_centroids_half_l2_norms,
                dot_products,
            )

    return update_closest_centroid


def _make_update_closest_centroid_kernel_func(window_n_centroids):
    @dpex.func
    # fmt: off
    def update_closest_centroid(
        first_centroid_idx,                  # PARAM
        min_idx,                             # PARAM
        min_sample_pseudo_inertia,           # PARAM
        window_of_centroids_half_l2_norms,   # IN
        dot_products,                        # IN
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

    return update_closest_centroid
