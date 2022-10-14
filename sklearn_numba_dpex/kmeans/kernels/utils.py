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

import math
from functools import lru_cache

import numpy as np
import dpctl
import numba_dpex as dpex


@lru_cache
def make_relocate_empty_clusters_kernel(
    n_relocated_clusters,
    n_features,
    n_selected_gt_threshold,
    work_group_size,
    dtype,
):
    n_work_groups_for_cluster = math.ceil(n_features / work_group_size)
    n_work_items_for_cluster = n_work_groups_for_cluster * work_group_size
    global_size = n_work_items_for_cluster * n_relocated_clusters

    n_selected_gt_threshold = n_selected_gt_threshold - 1
    zero = dtype(0.0)

    @dpex.kernel
    # fmt: off
    def relocate_empty_clusters(
        X_t,                        # IN READ-ONLY   (n_features, n_samples)
        sample_weight,              # IN READ-ONLY   (n_samples,)
        assignments_idx,            # IN             (n_samples,)
        samples_far_from_center,    # IN             (n_samples,)
        empty_clusters_list,        # IN             (n_clusters,)
        per_sample_inertia,         # INOUT          (n_samples,)
        centroids_t,                # INOUT          (n_features, n_clusters)
        cluster_sizes               # INOUT          (n_clusters,)
            ):
    # fmt: on
        """NB: because of how the array was created, samples_far_from_center values
        are located between (n_selected_gt_threshold - relocated_idx) and
        (relocated_idx) indices.
        """
        group_idx = dpex.get_group_id(0)
        item_idx = dpex.get_local_id(0)
        relocated_idx = group_idx // n_work_groups_for_cluster
        feature_idx = ((group_idx % n_work_groups_for_cluster) * work_group_size) + item_idx

        if feature_idx >= n_features:
            return

        relocated_cluster_idx = empty_clusters_list[relocated_idx]
        new_location_X_idx = samples_far_from_center[n_selected_gt_threshold - relocated_idx]
        new_location_previous_assignment = assignments_idx[new_location_X_idx]

        new_centroid_value = X_t[feature_idx, new_location_X_idx]
        new_location_weight = sample_weight[new_location_X_idx]
        X_centroid_addend = new_centroid_value * new_location_weight

        # Cancel the contribution to the updated centroids of the sample that was once
        # assigned to new_location_previous_assignment but is now assigned to the
        # cluster of the centroids that relocates to this sample
        dpex.atomic.sub(
            centroids_t,
            (feature_idx, new_location_previous_assignment),
            X_centroid_addend
            )

        # The relocated centroid has only one contribution now, which is the sample
        # to which it has been relocated to
        centroids_t[feature_idx, relocated_cluster_idx] = X_centroid_addend

        # Likewise, we update the weights in the clusters
        if feature_idx == zero:
            per_sample_inertia[new_location_X_idx] = zero
            dpex.atomic.sub(
                cluster_sizes,
                new_location_previous_assignment,
                new_location_weight
            )
            cluster_sizes[relocated_cluster_idx] = new_location_weight

    return relocate_empty_clusters[global_size, work_group_size]


@lru_cache
def make_select_samples_far_from_centroid_kernel(
    threshold, n_selected, n_samples, work_group_size
):
    global_size = math.ceil(n_selected / work_group_size) * work_group_size
    zero = np.int64(0)
    one = np.int32(1)
    max_n_selected_gt_threshold = np.int32(n_selected - 1)
    min_n_selected_eq_threshold = np.int32(2)
    max_n_selected_eq_threshold = np.int32(n_selected + 1)

    @dpex.kernel
    # fmt: off
    def select_samples_far_from_centroid(
            distance_to_centroid,           # IN       (n_samples,)
            selected_samples_idx,           # OUT      (n_samples,)
            n_selected_gt_threshold,        # OUT      (1, )
            n_selected_eq_threshold,        # OUT      (1, )
        ):
    # fmt: on
        """This kernel writes in selected_samples_idx the idx of the top n_selected
        items in distance_to_centroid with highest values.

        threshold is expected to have been pre-computed (by partitioning) such that
        there are at most (n_selected-1) values that are strictly greater than
        threshold, and at least n_selected values that are greater or equal than
        threshold.

        Because the exact number of values strictly equal to the threshold is not known
        and that the goal is to select the top n_selected greater items above threshold,
        we write indices of values strictly greater than threshold at the beginning of
        the array, and indices of values equal to threshold at the end of the array.
        """

        sample_idx = dpex.get_global_id(0)
        if sample_idx >= n_samples:
            return

        n_selected_gt_threshold_ = n_selected_gt_threshold[zero]
        n_selected_eq_threshold_ = n_selected_eq_threshold[zero]
        if ((n_selected_gt_threshold_ == max_n_selected_gt_threshold)
            and (n_selected_eq_threshold_ == min_n_selected_eq_threshold)):
            return

        distance_to_centroid_ = distance_to_centroid[sample_idx]
        if distance_to_centroid_ < threshold:
            return

        if distance_to_centroid_ > threshold:
            selected_idx = dpex.atomic.add(
                n_selected_gt_threshold,
                zero,
                one
                )
            selected_samples_idx[selected_idx] = sample_idx
            return

        if n_selected_eq_threshold_ == max_n_selected_eq_threshold:
            return

        selected_idx = -dpex.atomic.add(
            n_selected_eq_threshold,
            zero,
            one
            )
        selected_samples_idx[selected_idx] = sample_idx

    return select_samples_far_from_centroid[global_size, work_group_size]


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
def make_reduce_centroid_data_kernel(
    n_centroids_private_copies,
    n_features,
    n_clusters,
    work_group_size,
    dtype,
):

    n_work_groups_for_clusters = math.ceil(n_clusters / work_group_size)
    n_work_items_for_clusters = n_work_groups_for_clusters * work_group_size
    global_size = n_work_items_for_clusters * n_features
    zero = dtype(0.0)
    i_one = np.int32(1)
    l_zero = np.int64(0)

    # Optimized for C-contiguous array and assuming
    # n_features * n_clusters >> preferred_work_group_size_multiple
    @dpex.kernel
    # fmt: off
    def reduce_centroid_data(
        cluster_sizes_private_copies,  # IN      (n_copies, n_clusters)
        centroids_t_private_copies,    # IN      (n_copies, n_features, n_clusters)
        cluster_sizes,                 # OUT     (n_clusters,)
        centroids_t,                   # OUT     (n_features, n_clusters)
        empty_clusters_list,           # OUT     (n_clusters,)
        n_empty_clusters,              # OUT     (1,)
    ):
    # fmt: on

        group_idx = dpex.get_group_id(0)
        item_idx = dpex.get_local_id(0)
        feature_idx = group_idx // n_work_groups_for_clusters
        cluster_idx = (
            (group_idx % n_work_groups_for_clusters) * work_group_size
        ) + item_idx
        if cluster_idx >= n_clusters:
            return

        # reduce the centroid values
        sum_ = zero
        for copy_idx in range(n_centroids_private_copies):
            sum_ += centroids_t_private_copies[copy_idx, feature_idx, cluster_idx]
        centroids_t[feature_idx, cluster_idx] = sum_

        # reduce the cluster sizes
        if feature_idx == 0:
            sum_ = zero
            for copy_idx in range(n_centroids_private_copies):
                sum_ += cluster_sizes_private_copies[copy_idx, cluster_idx]
            cluster_sizes[cluster_idx] = sum_

            # register empty clusters
            if sum_ == zero:
                current_n_empty_clusters = dpex.atomic.add(
                    n_empty_clusters, l_zero, i_one
                )
                empty_clusters_list[current_n_empty_clusters] = cluster_idx

    return reduce_centroid_data[global_size, work_group_size]


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
    @dpex.kernel
    # fmt: off
    def half_l2_norm(
        data,    # IN        (size0, size1)
        result,  # OUT       (size1,)
    ):
    # fmt: on
        col_idx = dpex.get_global_id(0)

        if col_idx >= size1:
            return

        l2_norm = zero

        for row_idx in range(size0):
            item = data[row_idx, col_idx]
            l2_norm += item * item

        result[col_idx] = l2_norm / two

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
        result,      # OUT       (math.ceil(size / (2 * work_group_size),)
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
        current_n_work_items = work_group_size
        for i in range(local_n_iterations - 1):
            # We discard half of the remaining active work items at each iteration
            current_n_work_items = current_n_work_items // 2
            if local_work_id < current_n_work_items:
                local_data[local_work_id] += local_data[
                    local_work_id + current_n_work_items
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
        result = dpctl.tensor.empty(n_groups, dtype=dtype, device=device)
        kernels_and_empty_tensors_pairs.append((kernel, result))

    def sum_reduction(summands):
        for kernel, result in kernels_and_empty_tensors_pairs:
            kernel(summands, result)
            summands = result
        return result

    return sum_reduction
