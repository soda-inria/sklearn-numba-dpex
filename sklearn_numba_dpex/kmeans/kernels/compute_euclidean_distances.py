import math
from functools import lru_cache

import numba_dpex as dpex
import numpy as np

from ._base_kmeans_kernel_funcs import make_pairwise_ops_base_kernel_funcs

# NB: refer to the definition of the main lloyd function for a more comprehensive
# inline commenting of the kernel.


@lru_cache
def make_compute_euclidean_distances_fixed_window_kernel(
    n_samples, n_features, n_clusters, sub_group_size, work_group_size, dtype, device
):

    window_n_centroids = sub_group_size
    centroids_window_width = window_n_centroids + 1

    if work_group_size == "max":
        if device.has_aspect_cpu:
            work_group_size = 2 ** (
                math.floor(
                    math.log2(
                        device.local_mem_size
                        / (centroids_window_width * np.dtype(dtype).itemsize)
                    )
                )
            )
        else:
            work_group_size = device.max_work_group_size

    centroids_window_height = work_group_size // sub_group_size

    if centroids_window_height * sub_group_size != work_group_size:
        raise ValueError(
            "Expected work_group_size to be a multiple of sub_group_size but got "
            f"sub_group_size={sub_group_size} and work_group_size={work_group_size}"
        )

    (
        initialize_window_of_centroids,
        load_window_of_centroids_and_features,
        accumulate_sq_distances,
    ) = make_pairwise_ops_base_kernel_funcs(
        n_samples,
        n_features,
        n_clusters,
        centroids_window_height,
        window_n_centroids,
        ops="squared_diff",
        dtype=dtype,
        initialize_window_of_centroids_half_l2_norms=False,
    )

    n_windows_for_centroids = math.ceil(n_clusters / window_n_centroids)
    n_windows_for_features = math.ceil(n_features / centroids_window_height)
    last_centroid_window_idx = n_windows_for_centroids - 1
    last_feature_window_idx = n_windows_for_features - 1

    centroids_window_shape = (centroids_window_height, centroids_window_width)

    zero_idx = np.int64(0)

    @dpex.kernel
    # fmt: off
    def compute_distances(
        X_t,                      # IN READ-ONLY   (n_features, n_samples)
        current_centroids_t,      # IN READ-ONLY   (n_features, n_clusters)
        euclidean_distances_t,    # OUT            (n_clusters, n_samples)
    ):
        # fmt: on

        sample_idx = dpex.get_global_id(zero_idx)
        local_work_id = dpex.get_local_id(zero_idx)

        centroids_window = dpex.local.array(shape=centroids_window_shape, dtype=dtype)

        sq_distances = dpex.private.array(shape=window_n_centroids, dtype=dtype)

        first_centroid_idx = zero_idx

        window_loading_centroid_idx = local_work_id % window_n_centroids
        window_loading_feature_offset = local_work_id // window_n_centroids

        for centroid_window_idx in range(n_windows_for_centroids):
            is_last_centroid_window = centroid_window_idx == last_centroid_window_idx
            initialize_window_of_centroids(is_last_centroid_window, sq_distances)

            loading_centroid_idx = first_centroid_idx + window_loading_centroid_idx

            first_feature_idx = zero_idx

            for feature_window_idx in range(n_windows_for_features):
                is_last_feature_window = feature_window_idx == last_feature_window_idx
                load_window_of_centroids_and_features(
                    first_feature_idx,
                    loading_centroid_idx,
                    window_loading_centroid_idx,
                    window_loading_feature_offset,
                    current_centroids_t,
                    centroids_window,
                )

                dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)
                accumulate_sq_distances(
                    sample_idx,
                    first_feature_idx,
                    X_t,
                    centroids_window,
                    is_last_feature_window,
                    is_last_centroid_window,
                    sq_distances
                )

                first_feature_idx += centroids_window_height

                dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

            if sample_idx < n_samples:
                for i in range(window_n_centroids):
                    centroid_idx = first_centroid_idx + i
                    if centroid_idx < n_clusters:
                        euclidean_distances_t[first_centroid_idx + i, sample_idx] = (
                            math.sqrt(sq_distances[i])
                        )

            first_centroid_idx += window_n_centroids

            dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

    global_size = (math.ceil(n_samples / work_group_size)) * (work_group_size)
    return compute_distances[global_size, work_group_size]
