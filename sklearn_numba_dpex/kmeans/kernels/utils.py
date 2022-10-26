import math
from functools import lru_cache

import numpy as np
import numba_dpex as dpex


zero_idx = np.int64(0)


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
        group_idx = dpex.get_group_id(zero_idx)
        item_idx = dpex.get_local_id(zero_idx)
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


        # Cancel the contribution to new_location_previous_assignment, the previously 
        # updated centroids of the sample. This contribution will be assigned to the 
        # centroid of the clusters that relocates to this sample.
        dpex.atomic.sub(
            centroids_t,
            (feature_idx, new_location_previous_assignment),
            X_centroid_addend
            )

        # The relocated centroid has only one contribution now, which is the one of the
        # sample the cluster has been relocated to.
        centroids_t[feature_idx, relocated_cluster_idx] = X_centroid_addend

        # Likewise, we update the weights in the clusters. This is done once using 
        # `feature_idx`'s value.
        if feature_idx == zero_idx:
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
    n_selected, n_samples, work_group_size
):
    global_size = math.ceil(n_samples / work_group_size) * work_group_size
    one_incr = np.int32(1)
    max_n_selected_gt_threshold = np.int32(n_selected - 1)
    min_n_selected_eq_threshold = np.int32(2)
    max_n_selected_eq_threshold = np.int32(n_selected + 1)

    @dpex.kernel
    # fmt: off
    def select_samples_far_from_centroid(
            distance_to_centroid,           # IN       (n_samples,)
            threshold,                      # IN       (n_samples,)
            selected_samples_idx,           # OUT      (n_samples,)
            n_selected_gt_threshold,        # OUT      (1,)
            n_selected_eq_threshold,        # OUT      (1,)
        ):
    # fmt: on
        """This kernel writes in selected_samples_idx the idx of the top n_selected
        items in distance_to_centroid with highest values.

        threshold is expected to have been pre-computed (by partitioning) such that
        there are at most `n_selected-1` values that are strictly greater than
        threshold, and at least n_selected values that are greater or equal than
        threshold.

        Because the exact number of values strictly equal to the threshold is not known
        and that the goal is to select the top n_selected greater items above threshold,
        we write indices of values strictly greater than threshold at the beginning of
        the selected_samples_idx, and indices of values equal to threshold at the end
        of the selected_samples_idx array.
        """
        threshold_ = threshold[0]

        sample_idx = dpex.get_global_id(zero_idx)
        if sample_idx >= n_samples:
            return

        n_selected_gt_threshold_ = n_selected_gt_threshold[zero_idx]
        n_selected_eq_threshold_ = n_selected_eq_threshold[zero_idx]
        if ((n_selected_gt_threshold_ == max_n_selected_gt_threshold)
            and (n_selected_eq_threshold_ >= min_n_selected_eq_threshold)):
            return

        distance_to_centroid_ = distance_to_centroid[sample_idx]
        if distance_to_centroid_ < threshold_:
            return

        if distance_to_centroid_ > threshold_:
            selected_idx = dpex.atomic.add(
                n_selected_gt_threshold,
                zero_idx,
                one_incr,
            )
            selected_samples_idx[selected_idx] = sample_idx
            return

        if n_selected_eq_threshold_ >= max_n_selected_eq_threshold:
            return

        selected_idx = -dpex.atomic.add(
            n_selected_eq_threshold,
            zero_idx,
            one_incr,
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
        sample_idx = dpex.get_global_id(zero_idx)

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
    one_incr = np.int32(1)

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

        group_idx = dpex.get_group_id(zero_idx)
        item_idx = dpex.get_local_id(zero_idx)
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
        if feature_idx == zero_idx:
            sum_ = zero
            for copy_idx in range(n_centroids_private_copies):
                sum_ += cluster_sizes_private_copies[copy_idx, cluster_idx]
            cluster_sizes[cluster_idx] = sum_

            # register empty clusters
            if sum_ == zero:
                current_n_empty_clusters = dpex.atomic.add(
                    n_empty_clusters, zero_idx, one_incr
                )
                empty_clusters_list[current_n_empty_clusters] = cluster_idx

    return reduce_centroid_data[global_size, work_group_size]
