import math
from functools import lru_cache

import numba_dpex as dpex
import numpy as np

from sklearn_numba_dpex.common._utils import _check_max_work_group_size

from ._base_kmeans_kernel_funcs import make_pairwise_ops_base_kernel_funcs

# NB: refer to the definition of the main lloyd function for a more comprehensive
# inline commenting of the kernel.


@lru_cache
def make_compute_euclidean_distances_fixed_window_kernel(
    n_samples, n_features, n_clusters, sub_group_size, work_group_size, dtype, device
):

    window_n_centroids = sub_group_size

    input_work_group_size = work_group_size
    work_group_size = _check_max_work_group_size(
        work_group_size, device, window_n_centroids * np.dtype(dtype).itemsize
    )

    centroids_window_height = work_group_size // sub_group_size

    if (work_group_size == input_work_group_size) and (
        (centroids_window_height * sub_group_size) != work_group_size
    ):
        raise ValueError(
            "Expected work_group_size to be a multiple of sub_group_size but got "
            f"sub_group_size={sub_group_size} and work_group_size={work_group_size}"
        )

    work_group_shape = (window_n_centroids, centroids_window_height)

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

    centroids_window_shape = (centroids_window_height, window_n_centroids)

    zero_idx = np.int64(0)
    one_idx = np.int64(1)

    @dpex.kernel
    # fmt: off
    def compute_distances(
        X_t,                      # IN READ-ONLY   (n_features, n_samples)
        current_centroids_t,      # IN READ-ONLY   (n_features, n_clusters)
        euclidean_distances_t,    # OUT            (n_clusters, n_samples)
    ):
        # fmt: on

        centroids_window = dpex.local.array(shape=centroids_window_shape, dtype=dtype)

        sq_distances = dpex.private.array(shape=window_n_centroids, dtype=dtype)

        first_centroid_idx = zero_idx

        local_col_idx = dpex.get_local_id(zero_idx)

        window_loading_feature_offset = dpex.get_local_id(one_idx)
        window_loading_centroid_idx = local_col_idx

        sample_idx = (
            (dpex.get_global_id(one_idx) * sub_group_size)
            + local_col_idx
        )

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

                dpex.barrier(dpex.LOCAL_MEM_FENCE)
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

                dpex.barrier(dpex.LOCAL_MEM_FENCE)

            _save_distance(
                sample_idx,
                first_centroid_idx,
                sq_distances,
                # OUT
                euclidean_distances_t
            )

            first_centroid_idx += window_n_centroids

            dpex.barrier(dpex.LOCAL_MEM_FENCE)

    # HACK 906: see sklearn_numba_dpex.patches.tests.test_patches.test_need_to_workaround_numba_dpex_906  # noqa
    @dpex.func
    # fmt: off
    def _save_distance(
        sample_idx,                 # PARAM
        first_centroid_idx,         # PARAM
        sq_distances,               # IN
        euclidean_distances_t       # OUT
    ):
        # fmt: on
        if sample_idx >= n_samples:
            return

        for i in range(window_n_centroids):
            centroid_idx = first_centroid_idx + i

            # ?
            if centroid_idx < n_clusters:
                euclidean_distances_t[centroid_idx, sample_idx] = (
                    math.sqrt(sq_distances[i])
                )

    n_windows_for_sample = math.ceil(n_samples / window_n_centroids)

    global_size = (
        window_n_centroids,
        math.ceil(n_windows_for_sample / centroids_window_height)
        * centroids_window_height,
    )
    return compute_distances[global_size, work_group_shape]
