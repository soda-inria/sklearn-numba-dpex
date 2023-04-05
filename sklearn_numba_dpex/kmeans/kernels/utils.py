import math
from functools import lru_cache

import dpctl.tensor as dpt
import numba_dpex as dpex
import numpy as np

zero_idx = np.int64(0)


@lru_cache
def make_relocate_empty_clusters_kernel(
    n_relocated_clusters, n_features, work_group_size, dtype
):
    n_work_groups_for_cluster = math.ceil(n_features / work_group_size)
    n_work_items_for_cluster = n_work_groups_for_cluster * work_group_size
    global_size = n_work_items_for_cluster * n_relocated_clusters

    zero = dtype(0.0)

    @dpex.kernel
    # fmt: off
    def relocate_empty_clusters(
        X_t,                        # IN READ-ONLY   (n_features, n_samples)
        sample_weight,              # IN READ-ONLY   (n_samples,)
        assignments_idx,            # IN             (n_samples,)
        samples_far_from_center,    # IN             (n_relocated_clusters,)
        empty_clusters_list,        # IN             (n_clusters,)
        per_sample_inertia,         # INOUT          (n_samples,)
        centroids_t,                # INOUT          (n_features, n_clusters)
        cluster_sizes               # INOUT          (n_clusters,)
    ):
        # fmt: on
        group_idx = dpex.get_group_id(zero_idx)
        item_idx = dpex.get_local_id(zero_idx)
        relocated_idx = group_idx // n_work_groups_for_cluster
        feature_idx = (
            ((group_idx % n_work_groups_for_cluster) * work_group_size) + item_idx
        )

        if feature_idx >= n_features:
            return

        relocated_cluster_idx = empty_clusters_list[relocated_idx]
        new_location_X_idx = samples_far_from_center[relocated_idx]
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
    n_centroid_items = n_features * n_clusters
    n_sums = n_centroid_items + n_clusters  # n additional sums for cluster_sizes
    global_size = math.ceil(n_sums / work_group_size) * work_group_size

    zero = dtype(0.0)
    one_incr = np.int32(1)

    # Optimized for C-contiguous array and assuming
    # n_features * n_clusters >> preferred_work_group_size_multiple
    @dpex.kernel
    # fmt: off
    def _reduce_centroid_data_kernel(
        cluster_sizes_private_copies,            # IN      (n_copies, n_clusters)
        centroids_t_private_copies_reshaped,     # IN      (n_copies, n_features * n_clusters)  # noqa
        cluster_sizes,                           # OUT     (n_clusters,)
        centroids_t_flattened,                   # OUT     (n_features * n_clusters,)           # noqa
        empty_clusters_list,                     # OUT     (n_clusters,)
        n_empty_clusters,                        # OUT     (1,)
    ):
        # fmt: on
        item_idx = dpex.get_global_id(zero_idx)
        if item_idx >= n_sums:
            return

        sum_ = zero

        # reduce the centroid values
        if item_idx < n_centroid_items:
            for copy_idx in range(n_centroids_private_copies):
                sum_ += centroids_t_private_copies_reshaped[copy_idx, item_idx]
            centroids_t_flattened[item_idx] = sum_
            return

        cluster_idx = item_idx - n_centroid_items
        for copy_idx in range(n_centroids_private_copies):
            sum_ += cluster_sizes_private_copies[copy_idx, cluster_idx]
        cluster_sizes[cluster_idx] = sum_

        if sum_ > 0:
            return

        # register empty clusters
        if sum_ == zero:
            current_n_empty_clusters = dpex.atomic.add(
                n_empty_clusters, zero_idx, one_incr
            )
            empty_clusters_list[current_n_empty_clusters] = cluster_idx

    reduce_centroid_data_kernel = _reduce_centroid_data_kernel[
        global_size, work_group_size
    ]

    def reduce_centroid_data(
        cluster_sizes_private_copies,
        centroids_t_private_copies,
        cluster_sizes,
        centroids_t,
        empty_clusters_list,
        n_empty_clusters,
    ):
        centroids_t_private_copies_reshaped = dpt.reshape(
            centroids_t_private_copies,
            (n_centroids_private_copies, n_features * n_clusters),
        )
        centroids_t_flattened = dpt.asarray(dpt.reshape(centroids_t, (-1,)))
        reduce_centroid_data_kernel(
            cluster_sizes_private_copies,
            centroids_t_private_copies_reshaped,
            cluster_sizes,
            centroids_t_flattened,
            empty_clusters_list,
            n_empty_clusters,
        )

    return reduce_centroid_data


@lru_cache
def make_is_same_clustering_kernel(n_samples, n_clusters, work_group_size, device):
    # TODO: are there possible optimizations for this kernel ?
    # - fusing the two kernels (It would require a lock ? There's a risk of concurrency
    # issues among threads that try to write different values into `mapping` and
    # threads that read `mapping`, all this at the same time. Tools in `dpex.atomic`
    # seems to not be enough to overcome that at the moment.)
    # - early stop

    def is_same_clustering(labels1, labels2):
        mapping = dpt.empty(sh=(n_clusters,), dtype=np.int32, device=device)
        result = dpt.asarray([1], dtype=np.int32, device=device)
        _build_mapping[global_size, work_group_size](labels1, labels2, mapping)
        _is_same_clustering[global_size, work_group_size](
            labels1, labels2, mapping, result
        )
        return bool(result[0])

    @dpex.kernel
    # fmt: off
    def _build_mapping(
        labels1,            # IN      (n_samples,)
        labels2,            # IN      (n_samples,)
        mapping,            # BUFFER  (n_clusters,)
    ):
        # fmt: on
        sample_idx = dpex.get_global_id(zero_idx)
        if sample_idx >= n_samples:
            return

        label1 = labels1[sample_idx]
        label2 = labels2[sample_idx]
        mapping[label1] = label2

    @dpex.kernel
    # fmt: off
    def _is_same_clustering(
        labels1,
        labels2,
        mapping,
        result
    ):
        # fmt: on
        """`result` is expected to be an empty array with dtype int32 of size (1,)
        initialized with value 1.
        """
        sample_idx = dpex.get_global_id(zero_idx)
        if sample_idx >= n_samples:
            return

        if mapping[labels1[sample_idx]] != labels2[sample_idx]:
            result[zero_idx] = zero_idx

    global_size = math.ceil(n_samples / work_group_size) * work_group_size

    return is_same_clustering


@lru_cache
def make_get_nb_distinct_clusters_kernel(
    n_samples, n_clusters, work_group_size, device
):
    one_incr = np.int32(1)

    @dpex.kernel
    def get_nb_distinct_clusters(labels, clusters_seen, nb_distinct_clusters):
        sample_idx = dpex.get_global_id(zero_idx)

        if sample_idx >= n_samples:
            return

        label = labels[sample_idx]

        if clusters_seen[label] > zero_idx:
            return

        previous_value = dpex.atomic.add(clusters_seen, label, one_incr)
        if previous_value == zero_idx:
            dpex.atomic.add(nb_distinct_clusters, zero_idx, one_incr)

    global_size = math.ceil(n_samples / work_group_size) * work_group_size
    return get_nb_distinct_clusters[global_size, work_group_size]
