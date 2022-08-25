import math
from functools import lru_cache

import numba_dpex as dpex
import dpctl

# General note on kernel implementation
#
# In many of those kernels, many values are defined outside of the definition of the
# kernel. This trick is aimed at forcing the JIT compiler to detect those values as
# constants. The compiler is supposed to unroll the loops that use constant values to 
# parameterize the number of iterations in the loop, which should reduce the execution
# time.
#
# That's why in some cases we prefer defining the shape of the input as a constant
# outside the kernel, rather than fetching it from within the kernel with the `.shape`
# attribute.
#
# The drawback is that it triggers the JIT compilation of a new kernel for each
# different input size, which could be a setback in some contexts but should not be 
# an issue for KMeans since the size of the inputs remain constants accross iterations 
# in the main loop.

# TODO: write unittests for each distinct kernel.

@lru_cache
def make_fused_fixed_window_kernel(
    n_samples,
    n_features,
    n_clusters,
    preferred_work_group_size_multiple,
    global_mem_cache_size,
    centroids_window_width_multiplier,
    centroids_window_height,
    centroids_private_copies_max_cache_occupancy,
    work_group_size,
    dtype,
):
    window_n_centroids = (
        preferred_work_group_size_multiple * centroids_window_width_multiplier
    )
    n_window_features_per_work_group = work_group_size // window_n_centroids

    centroids_window_height_ratio_multiplier = (
        centroids_window_height // n_window_features_per_work_group
    )

    window_n_features = centroids_window_height

    n_windows_per_feature = math.ceil(n_clusters / window_n_centroids)
    global_size = (math.ceil(n_samples / work_group_size)) * (work_group_size)
    n_windows_per_centroid = math.ceil(n_features / window_n_features)
    centroids_window_shape = (window_n_features, (window_n_centroids + 1))

    inf = dtype(math.inf)
    zero = dtype(0.0)
    one = dtype(1.0)

    n_cluster_items = n_clusters * (n_features + 1)
    n_cluster_bytes = 4 * n_cluster_items
    # TODO: control that this value is not higher than the number of sub-groups of size
    # preferred_work_group_size_multiple that can effectively run concurrently. We
    # should fetch this information and apply it here.
    n_centroids_private_copies = int(
        (global_mem_cache_size * centroids_private_copies_max_cache_occupancy)
        // n_cluster_bytes
    )

    # TODO: currently, constant memory is not supported by numba_dpex, but for read-only
    # inputs such as X_t it is generally regarded as faster. Once support is available
    # (NB: it's already supported by numba.cuda) X_t should be an input to the factory
    # rather than an input to the kernel.
    @dpex.kernel
    # fmt: off
    def fused_kmeans(
        X_t,                               # IN READ-ONLY   (n_features, n_samples)
        current_centroids_t,               # IN             (n_features, n_clusters)
        centroids_half_l2_norm,            # IN             (n_clusters,)
        per_sample_pseudo_inertia,         # OUT            (n_samples,)
        new_centroids_t_private_copies,    # OUT            (n_private_copies, n_features, n_clusters)
        cluster_sizes_private_copies,      # OUT            (n_private_copies, n_clusters)
    ):
    # fmt: on
        """One full iteration of LLoyd's k-means.

        The kernel is meant to be invoked on a 1D grid spanning the data samples of
        the training set in parallel.

        Each work-item will assign that sample to its nearest centroid and accumulate
        the feature values of the data samples into the new version of the centroid
        for the next iteration. The centroid assignment and centroid update steps are
        performed in a single fused kernel to avoid introducing a large intermediate
        label assignment array to be re-read from global memory before performing the
        centroid update step.

        To avoid atomic contention when performing the centroid updates concurrently
        for different data samples, we use private copies of the new centroid array in
        global memory. Those copies are meant to be re-reduced afterwards.

        The distance is the euclidean distance. Note that it is not necessary to
        compute the exact value to find the closest centroid. Indeed, minimizing
            |x-y|^2 = |x|^2 - 2<x.y> + |y|^2
        over y amounts to minimizing
            (1/2)y^2 - xy .
        Moreover the value (1/2)y^2 has been pre-computed in the array
        centroids_half_l2_norm to reduce the overall number of floating point
        operations in the kernel.
        """
        sample_idx = dpex.get_global_id(0)
        local_work_id = dpex.get_local_id(0)

        # This array in shared memory is used as a sliding array over values of
        # current_centroids_t. During each iteration in the inner loop, a new one is
        # loaded and used by all work items in the work group to compute partial
        # results. The array slides over the features in the outer loop, and over the
        # samples in the inner loop.
        centroids_window = dpex.local.array(shape=centroids_window_shape, dtype=dtype)

        # This array in shared memory is used as a sliding array over the centroids.
        # It contains values of centroids_half_l2_norm for each centroid in the sliding
        # centroids_window array. It is updated once per iteration in the outer loop.
        centroids_window_half_l2_norm = dpex.local.array(
            shape=window_n_centroids, dtype=dtype
        )

        # In the inner loop each work item accumulates in private memory the
        # dot product of the sample at the sample_idx relatively to each centroid
        # in the window.
        partial_dot_products = dpex.private.array(shape=window_n_centroids, dtype=dtype)

        first_centroid_idx = 0

        # The two variables that are initialized here will contain the result we seek,
        # i.e, at the end of the outer loop, it will be equal to the closest centroid
        # to the current sample and the corresponding pseudo inertia.
        min_idx = 0
        min_sample_pseudo_inertia = inf

        # Those variables are used in the inner loop during loading of the window of
        # centroids
        window_loading_centroid_idx = local_work_id % window_n_centroids
        window_loading_feature_offset = local_work_id // window_n_centroids

        # STEP 1: compute the closest centroid
        # Outer loop: iterate on successive windows of size window_n_centroids that
        # cover all centroids in current_centroids_t
        for _0 in range(n_windows_per_feature):

            # initialize the partial pseudo inertia for each of the window_n_centroids
            # centroids in the window
            for i in range(window_n_centroids):
                partial_dot_products[i] = zero

            # the `window_n_centroids` first work items cooperate on loading the
            # values of centroids_half_l2_norm relevant to current window. Each work
            # item loads one single value.
            half_l2_norm_loading_idx = first_centroid_idx + local_work_id
            if local_work_id < window_n_centroids:
                if half_l2_norm_loading_idx < n_clusters:
                    l2norm = centroids_half_l2_norm[half_l2_norm_loading_idx]
                else:
                    l2norm = inf
                centroids_window_half_l2_norm[local_work_id] = l2norm

            loading_centroid_idx = first_centroid_idx + window_loading_centroid_idx

            first_feature_idx = 0

            # Inner loop: interate on successive windows of size window_n_features
            # that cover all features for current given centroids
            for _1 in range(n_windows_per_centroid):

                centroid_window_first_loading_feature_idx = 0

                # The work items in the work group load cooperatively the values in
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
                        value = current_centroids_t[
                            loading_feature_idx, loading_centroid_idx
                        ]
                    else:
                        value = zero

                    centroids_window[
                        window_loading_feature_idx, window_loading_centroid_idx
                    ] = value

                    centroid_window_first_loading_feature_idx += (
                        n_window_features_per_work_group
                    )

                # Since other work items are responsible for loading the relevant data
                # for the next step, we need to wait for completion of all work items
                # before going forward
                dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

                # Loop on all features in the current window and accumulate the dot
                # products
                for window_feature_idx in range(window_n_features):
                    feature_idx = window_feature_idx + first_feature_idx
                    if (feature_idx < n_features) and (sample_idx < n_samples):
                        # performance for the line thereafter relies on L1 cache
                        X_value = X_t[feature_idx, sample_idx]
                    else:
                        X_value = zero
                    # For this given feature, loop on all centroids in the current
                    # window and accumulate the dot products
                    for window_centroid_idx in range(window_n_centroids):
                        centroid_value = centroids_window[
                            window_feature_idx, window_centroid_idx
                        ]
                        partial_dot_products[window_centroid_idx] += (
                            centroid_value * X_value
                        )

                # When the next iteration starts work items will overwrite shared memory
                # with new values, so before that we must wait for all reading
                # operations in the current iteration to be over for all work items.
                dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

                first_feature_idx += window_n_features

            # End of inner loop. The pseudo inertia is now computed for all centroids
            # in the window, we can coalesce it to the accumulation of the min pseudo
            # inertia for the current sample.
            for i in range(window_n_centroids):
                current_sample_pseudo_inertia = (
                    centroids_window_half_l2_norm[i] - partial_dot_products[i]
                )
                if current_sample_pseudo_inertia < min_sample_pseudo_inertia:
                    min_sample_pseudo_inertia = current_sample_pseudo_inertia
                    min_idx = first_centroid_idx + i

            # When the next iteration starts work items will overwrite shared memory
            # with new values, so before that we must wait for all reading
            # operations in the current iteration to be over for all work items.
            dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

            first_centroid_idx += window_n_centroids

        # End of outer loop. By now min_idx and min_sample_pseudo_inertia contains the expected
        # values.

        # STEP 2: update centroids.

        # Each work item updates n_features values in global memory for the centroid
        # at position min_idx. All work items across all work groups have read access to
        # global memory and may run similar update instructions at the same time. That
        # creates race conditions, so update operations need to be enclosed in atomic
        # operations that act like locks and will sequentialize updates when different
        # work items collide on a given value.

        # However there is a very significant performance cost to sequentialization,
        # which we mitigate with a strategy of "privatization" for reducing the
        # probability of collisions. The array of centroids is duplicated in global
        # memory as many time as possible and each sub-group of work items of size
        # `preferred_work_group_size_multiple` is assigned to a different duplicata and
        # update the values of this single duplicata.

        # The resulting copies of centroids updates will then need to be reduced to a
        # single array of centroids in a complementary kernel.

        # The privatization is more effective when there is a low number of centroid
        # values (equal to (n_clusters * n_features)) comparatively to the global
        # number of work items, i.e. when the probability of collision is high. At the
        # opposite end where the probability of collision is low, privatization might
        # be detrimental to performance and we might prefer simpler, faster code
        # with updates directly made into the final array of centroids.

        # The privatization strategy also applies to the updates of the centroid
        # counts.

        if sample_idx >= n_samples:
            return

        per_sample_pseudo_inertia[sample_idx] = min_sample_pseudo_inertia

        # each work item is assigned an array of centroids in a round robin manner
        privatization_idx = (
            sample_idx // preferred_work_group_size_multiple
        ) % n_centroids_private_copies

        dpex.atomic.add(
            cluster_sizes_private_copies,
            (privatization_idx, min_idx),
            one,
        )

        for feature_idx in range(n_features):
            dpex.atomic.add(
                new_centroids_t_private_copies,
                (privatization_idx, feature_idx, min_idx),
                X_t[feature_idx, sample_idx],
            )

    return (
        n_centroids_private_copies,
        fused_kmeans[global_size, work_group_size],
    )


@lru_cache
def make_assignment_fixed_window_kernel(
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
    n_window_features_per_work_group = work_group_size // window_n_centroids
    centroids_window_height_ratio_multiplier = (
        centroids_window_height // n_window_features_per_work_group
    )

    window_n_features = centroids_window_height

    n_windows_per_feature = math.ceil(n_clusters / window_n_centroids)
    global_size = (math.ceil(n_samples / work_group_size)) * (work_group_size)
    n_windows_per_centroid = math.ceil(n_features / window_n_features)
    centroids_window_shape = (window_n_features, (window_n_centroids + 1))

    inf = dtype(math.inf)
    zero = dtype(0.0)
    two = dtype(2)

    # TODO: factorize the fused fixed kernel and the assignment kernel using dpex.func
    # namespace. Ensure no impact on performance.
    # fmt: off
    @dpex.kernel
    def assignment(
        X_t,                      # IN READ-ONLY   (n_features, n_samples)
        current_centroids_t,      # IN             (n_features, n_clusters)
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
        partial_dot_products = dpex.private.array(shape=window_n_centroids, dtype=dtype)

        first_centroid_idx = 0

        min_idx = 0
        min_sample_pseudo_inertia = inf

        X_l2_norm = zero

        window_loading_centroid_idx = local_work_id % window_n_centroids
        window_loading_feature_offset = local_work_id // window_n_centroids

        for _0 in range(n_windows_per_feature):

            for i in range(window_n_centroids):
                partial_dot_products[i] = zero

            half_l2_norm_loading_idx = first_centroid_idx + local_work_id
            if local_work_id < window_n_centroids:
                if half_l2_norm_loading_idx < n_clusters:
                    l2norm = centroids_half_l2_norm[half_l2_norm_loading_idx]
                else:
                    l2norm = inf
                centroids_window_half_l2_norm[local_work_id] = l2norm

            loading_centroid_idx = first_centroid_idx + window_loading_centroid_idx

            first_feature_idx = 0

            for _1 in range(n_windows_per_centroid):

                centroid_window_first_loading_feature_idx = 0

                for _2 in range(centroids_window_height_ratio_multiplier):
                    window_loading_feature_idx = (
                        centroid_window_first_loading_feature_idx
                        + window_loading_feature_offset
                    )
                    loading_feature_idx = first_feature_idx + window_loading_feature_idx

                    if (loading_feature_idx < n_features) and (
                        loading_centroid_idx < n_clusters
                    ):
                        value = current_centroids_t[
                            loading_feature_idx, loading_centroid_idx
                        ]
                    else:
                        value = zero

                    centroids_window[
                        window_loading_feature_idx, window_loading_centroid_idx
                    ] = value

                    centroid_window_first_loading_feature_idx += (
                        n_window_features_per_work_group
                    )

                dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

                for window_feature_idx in range(window_n_features):
                    feature_idx = window_feature_idx + first_feature_idx
                    if (feature_idx < n_features) and (sample_idx < n_samples):
                        # performance for the line thereafter relies on L1 cache
                        X_value = X_t[feature_idx, sample_idx]
                    else:
                        X_value = zero
                    for window_centroid_idx in range(window_n_centroids):
                        centroid_value = centroids_window[
                            window_feature_idx, window_centroid_idx
                        ]
                        partial_dot_products[window_centroid_idx] += (
                            centroid_value * X_value
                        )
                    # The l2 norm of the current sample is needed to compute the exact
                    # value of the inertia. It is accumulated in the first iteration
                    # of the outer loop
                    if _0 == 0:
                        X_l2_norm += X_value * X_value

                dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

                first_feature_idx += window_n_features

            for i in range(window_n_centroids):
                current_sample_pseudo_inertia = (
                    centroids_window_half_l2_norm[i] - partial_dot_products[i]
                )
                if current_sample_pseudo_inertia < min_sample_pseudo_inertia:
                    min_sample_pseudo_inertia = current_sample_pseudo_inertia
                    min_idx = first_centroid_idx + i

            dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

            first_centroid_idx += window_n_centroids

        # No update step, only store min_idx in the output array
        if sample_idx >= n_samples:
            return

        assignments_idx[sample_idx] = min_idx
        per_sample_inertia[sample_idx] = X_l2_norm + (two * min_sample_pseudo_inertia)

    return assignment[global_size, work_group_size]


# TODO: all auxilliary kernels thereafter might be better optimized and we could
# benchmark alternative implementations for each of them, that could include
#    - using 2D or 3D grid of work groups and work items where applicable (e.g. in
# some of the kernels that take 2D or 3D data as input) rather than using 1D grid. When
# doing so, one should be especially careful about how the segments of adjacent work
# items of size preferred_work_group_size_multiple are dispatched especially regarding
# RW  operations in memory. A wrong dispatch strategy could slash memory bandwith and
# reduce performance. Using 2D or 3D grid correctly might on the other hand improve
# performance since it saves costly indexing operations (like //)
#    - investigate if flat 1D-like indexing also works for ND kernels, thus saving the
# need to compute coordinates for each dimension for element-wise operations.
#    - or using numba + dpnp to directly leverage kernels that are shipped in dpnp to 
# replace numpy methods.
# However, in light of our main goal that is bringing a GPU KMeans to scikit-learn, the 
# importance of those TODOs is currently seen as secondary, since the execution time of 
# those kernels is only a small fraction of the total execution time and the
# improvements that further optimizations can add will only be marginal. There is no
# doubt, though, that a lot could be learnt about kernel programming in the process.


@lru_cache
def make_centroid_shifts_kernel(n_clusters, n_features, work_group_size, dtype):
    global_size = math.ceil(n_clusters / work_group_size) * work_group_size
    zero = dtype(0.0)

    # Optimized for C-contiguous array and for
    # size1 >> preferred_work_group_size_multiple
    @dpex.kernel
    # fmt: off
    def centroid_shifts(
        centroids_t,        # IN    (n_features, n_clusters)
        new_centroids_t,    # IN    (n_features, n_clusters)
        centroid_shifts,    # OUT   (n_clusters,)
    ):
    # fmt: on
        sample_idx = dpex.get_global_id(0)

        if sample_idx >= n_clusters:
            return

        squared_centroid_diff = zero

        for feature_idx in range(n_features):
            center_diff = (
                centroids_t[feature_idx, sample_idx]
                - new_centroids_t[feature_idx, sample_idx]
            )
            squared_centroid_diff += center_diff * center_diff

        centroid_shifts[sample_idx] = squared_centroid_diff

    return centroid_shifts[global_size, work_group_size]


@lru_cache
def make_initialize_to_zeros_1d_kernel(size, work_group_size, dtype):

    global_size = math.ceil(size / work_group_size) * work_group_size
    zero = dtype(0)

    @dpex.kernel
    def initialize_to_zeros(data):
        item_idx = dpex.get_global_id(0)

        if item_idx >= size:
            return

        data[item_idx] = zero

    return initialize_to_zeros[global_size, work_group_size]


@lru_cache
def make_initialize_to_zeros_2d_kernel(size0, size1, work_group_size, dtype):

    n_items = size0 * size1
    global_size = math.ceil(n_items / work_group_size) * work_group_size
    zero = dtype(0.0)

    # Optimized for C-contiguous arrays
    @dpex.kernel
    def initialize_to_zeros(data):
        item_idx = dpex.get_global_id(0)

        if item_idx >= n_items:
            return

        row_idx = item_idx // size1
        col_idx = item_idx % size1
        data[row_idx, col_idx] = zero

    return initialize_to_zeros[global_size, work_group_size]


@lru_cache
def make_initialize_to_zeros_3d_kernel(size0, size1, size2, work_group_size, dtype):

    n_items = size0 * size1 * size2
    stride0 = size1 * size2
    global_size = math.ceil(n_items / work_group_size) * work_group_size
    zero = dtype(0.0)

    # Optimized for C-contiguous arrays
    @dpex.kernel
    def initialize_to_zeros(data):
        item_idx = dpex.get_global_id(0)

        if item_idx >= n_items:
            return

        i = item_idx // stride0
        stride0_idx = item_idx % stride0
        j = stride0_idx // size2
        k = stride0_idx % size2
        data[i, j, k] = zero

    return initialize_to_zeros[global_size, work_group_size]


@lru_cache
def make_copyto_2d_kernel(size0, size1, work_group_size):
    global_size = math.ceil(size1 / work_group_size) * work_group_size

    # Optimized for C-contiguous array and for
    # size1 >> preferred_work_group_size_multiple
    @dpex.kernel
    def copyto_kernel(input_array, target_array):
        col_idx = dpex.get_global_id(0)

        if col_idx >= size1:
            return

        for row_idx in range(size0):
            target_array[row_idx, col_idx] = input_array[row_idx, col_idx]

    return copyto_kernel[global_size, work_group_size]


@lru_cache
def make_broadcast_division_1d_2d_kernel(size0, size1, work_group_size):
    global_size = math.ceil(size1 / work_group_size) * work_group_size

    # NB: inplace. # Optimized for C-contiguous array and for
    # size1 >> preferred_work_group_size_multiple
    @dpex.kernel
    def broadcast_division(dividend_array, divisor_vector):
        col_idx = dpex.get_global_id(0)

        if col_idx >= size1:
            return

        divisor = divisor_vector[col_idx]

        for row_idx in range(size0):
            dividend_array[row_idx, col_idx] = (
                dividend_array[row_idx, col_idx] / divisor
            )

    return broadcast_division[global_size, work_group_size]


@lru_cache
def make_half_l2_norm_2d_axis0_kernel(size0, size1, work_group_size, dtype):
    global_size = math.ceil(size1 / work_group_size) * work_group_size
    zero = dtype(0.0)
    two = dtype(2.0)

    # Optimized for C-contiguous array and for
    # size1 >> preferred_work_group_size_multiple
    # fmt: off
    @dpex.kernel
    def half_l2_norm(
            data,     # IN        (size0, size1)
            result    # OUT       (size1,)
            ):    
    # fmt: on
        col_idx = dpex.get_global_id(0)

        if col_idx >= size1:
            return

        partial_l2_norm = zero

        for row_idx in range(size0):
            item = data[row_idx, col_idx]
            partial_l2_norm += item * item

        result[col_idx] = partial_l2_norm / two

    return half_l2_norm[global_size, work_group_size]


@lru_cache
def make_sum_reduction_1d_kernel(size, work_group_size, device, dtype):
    """numba_dpex does not provide tools such as `cuda.reduce` so we implement from
    scratch a reduction strategy. The strategy relies on the commutativity of the
    operation used for the reduction, thus allowing to reduce the input in any order.

    The strategy consists in performing local reductions in each work group using local
    memory where each work item combine two values, thus halving the number of values,
    and the number of active work items. At each iteration the work items are discarded
    in a bracket manner. The work items with the greatest ids are discarded first, and
    we rely on the fact that the remaining work items are adjacents to optimize the RW
    operations.

    Once the reduction is done in a work group the result is written in global memory,
    thus creating an intermediary result whose size is divided by 
    (2 * work_group_size). This is repeated as many time as needed until only one value 
    remains in global memory.

    NB: work_group_size is assumed to be a power of 2.
    """
    # Number of iteration in each execution of the kernel:
    local_n_iterations = math.floor(math.log2(work_group_size))

    zero = dtype(0.0)

    @dpex.kernel
    # fmt: off
    def partial_sum_reduction(
        summands,    # IN        (size,)
        result,      # OUT       (math.ceil(size / (2 * work_group_size))
    ):
    # fmt: on
        # NB: This kernel only perform a partial reduction
        group_id = dpex.get_group_id(0)
        local_work_id = dpex.get_local_id(0)
        first_work_id = local_work_id == 0

        size = summands.shape[0]

        local_data = dpex.local.array(work_group_size, dtype=dtype)

        first_value_idx = group_id * work_group_size * 2
        augend_idx = first_value_idx + local_work_id
        addend_idx = first_value_idx + work_group_size + local_work_id

        # Each work item reads two value in global memory and sum it into the local
        # memory
        if augend_idx >= size:
            local_data[local_work_id] = zero
        elif addend_idx >= size:
            local_data[local_work_id] = summands[augend_idx]
        else:
            local_data[local_work_id] = summands[augend_idx] + summands[addend_idx]

        dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)
        current_nb_work_items = work_group_size
        for i in range(local_n_iterations - 1):
            # We discard half of the remaining active work items at each iteration
            current_nb_work_items = current_nb_work_items // 2
            if local_work_id < current_nb_work_items:
                local_data[local_work_id] += local_data[
                    local_work_id + current_nb_work_items
                ]

            dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

        # At this point local_data[0] = local_data[1]  is equal to the sum of all
        # elements in summands that have been covered by the work group, we write it
        # into global memory
        if first_work_id:
            result[group_id] = local_data[0] + local_data[1]

    # As many partial reductions as necessary are chained until only one element
    # remains.
    kernels_and_empty_tensors_pairs = []
    n_groups = size
    # TODO: at some point, the cost of scheduling the kernel is more than the cost of
    # running the reduction iteration. At this point the loop should stop and then a
    # single work item should iterates one time on the remaining values to finish the
    # reduction.
    while n_groups > 1:
        n_groups = math.ceil(n_groups / (2 * work_group_size))
        global_size = n_groups * work_group_size
        kernel = partial_sum_reduction[global_size, work_group_size]
        partial_result = dpctl.tensor.empty(n_groups, dtype=dtype, device=device)
        kernels_and_empty_tensors_pairs.append((kernel, partial_result))

    def sum_reduction(summands):
        for kernel, partial_result in kernels_and_empty_tensors_pairs:
            kernel(summands, partial_result)
            summands = partial_result
        return partial_result

    return sum_reduction


@lru_cache
def make_sum_reduction_2d_axis0_kernel(size0, size1, work_group_size, dtype):
    global_size = math.ceil(size1 / work_group_size) * work_group_size
    zero = dtype(0.0)

    # Optimized for C-contiguous array and for
    # size1 >> preferred_work_group_size_multiple
    @dpex.kernel
    def sum_reduction(summands, result):
        col_idx = dpex.get_global_id(0)
        if col_idx >= size1:
            return
        partial_result = zero
        for row_idx in range(size0):
            partial_result += summands[row_idx, col_idx]
        result[col_idx] = partial_result

    return sum_reduction[global_size, work_group_size]


@lru_cache
def make_sum_reduction_3d_axis0_kernel(size0, size1, size2, work_group_size, dtype):
    n_work_groups_for_dim2 = math.ceil(size2 / work_group_size)
    n_work_items_for_dim2 = n_work_groups_for_dim2 * work_group_size
    global_size = n_work_items_for_dim2 * size1
    zero = dtype(0.0)

    # Optimized for C-contiguous array and for
    # (size1 * size2) >> preferred_work_group_size_multiple
    @dpex.kernel
    def sum_reduction(summands, result):
        group_idx = dpex.get_group_id(0)
        item_idx = dpex.get_local_id(0)
        j = group_idx // n_work_groups_for_dim2
        k = ((group_idx % n_work_groups_for_dim2) * work_group_size) + item_idx
        if k >= size2:
            return
        partial_result = zero
        for i in range(size0):
            partial_result += summands[i, j, k]
        result[j, k] = partial_result

    return sum_reduction[global_size, work_group_size]


