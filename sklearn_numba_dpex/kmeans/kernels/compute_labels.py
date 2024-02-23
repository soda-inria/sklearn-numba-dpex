import math
from functools import lru_cache

import numba_dpex as dpex
import numpy as np
from numba_dpex.kernel_api import NdRange

from sklearn_numba_dpex.common._utils import _check_max_work_group_size

from ._base_kmeans_kernel_funcs import (
    make_pairwise_ops_base_kernel_funcs,
    make_update_closest_centroid_kernel_func,
)

# NB: refer to the definition of the main lloyd function for a more comprehensive
# inline commenting of the kernel.


@lru_cache
def make_label_assignment_fixed_window_kernel(
    n_samples, n_features, n_clusters, sub_group_size, work_group_size, dtype, device
):
    window_n_centroids = sub_group_size

    dtype_itemsize = np.dtype(dtype).itemsize
    input_work_group_size = work_group_size
    work_group_size = _check_max_work_group_size(
        work_group_size,
        device,
        required_local_memory_per_item=dtype_itemsize,
        required_memory_constant=window_n_centroids * dtype_itemsize,
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
        load_window_of_centroids_and_features,
        accumulate_dot_products,
        initialize_window_half_l2_norm,
    ) = make_pairwise_ops_base_kernel_funcs(
        n_samples,
        n_features,
        n_clusters,
        centroids_window_height,
        window_n_centroids,
        ops="product",
        dtype=dtype,
        initialize_window_of_centroids_half_l2_norms=True,
    )

    update_closest_centroid = make_update_closest_centroid_kernel_func(
        n_clusters, window_n_centroids
    )

    n_windows_for_centroids = math.ceil(n_clusters / window_n_centroids)
    n_windows_for_features = math.ceil(n_features / centroids_window_height)
    last_centroid_window_idx = n_windows_for_centroids - 1
    last_feature_window_idx = n_windows_for_features - 1

    centroids_window_shape = (centroids_window_height, window_n_centroids)

    inf = dtype(math.inf)
    zero_idx = np.int64(0)
    one_idx = np.int64(1)

    @dpex.kernel
    # fmt: off
    def assignment(
        X_t,                      # IN READ-ONLY   (n_features, n_samples)
        centroids_t,              # IN READ-ONLY   (n_features, n_clusters)
        centroids_half_l2_norm,   # IN             (n_clusters,)
        assignments_idx,          # OUT            (n_samples,)
    ):
        # fmt: on
        local_row_idx = dpex.get_local_id(one_idx)
        local_col_idx = dpex.get_local_id(zero_idx)
        sample_idx = (
            (dpex.get_global_id(one_idx) * sub_group_size)
            + local_col_idx
        )

        centroids_window = dpex.local.array(shape=centroids_window_shape, dtype=dtype)
        window_of_centroids_half_l2_norms = dpex.local.array(
            shape=window_n_centroids, dtype=dtype
        )
        dot_products = dpex.private.array(shape=window_n_centroids, dtype=dtype)

        first_centroid_idx = zero_idx

        min_idx = zero_idx
        min_sample_pseudo_inertia = inf

        window_loading_centroid_idx = local_col_idx
        window_loading_feature_offset = local_row_idx

        for centroid_window_idx in range(n_windows_for_centroids):
            is_last_centroid_window = centroid_window_idx == last_centroid_window_idx

            initialize_window_half_l2_norm(
                local_row_idx,
                local_col_idx,
                first_centroid_idx,
                centroids_half_l2_norm,
                is_last_centroid_window,
                window_of_centroids_half_l2_norms,
            )

            loading_centroid_idx = first_centroid_idx + window_loading_centroid_idx

            first_feature_idx = zero_idx

            for feature_windiw_idx in range(n_windows_for_features):
                is_last_feature_window = feature_windiw_idx == last_feature_window_idx
                load_window_of_centroids_and_features(
                    first_feature_idx,
                    loading_centroid_idx,
                    window_loading_centroid_idx,
                    window_loading_feature_offset,
                    centroids_t,
                    centroids_window,
                )

                dpex.barrier(dpex.LOCAL_MEM_FENCE)

                accumulate_dot_products(
                    sample_idx,
                    first_feature_idx,
                    X_t,
                    centroids_window,
                    is_last_feature_window,
                    is_last_centroid_window,
                    dot_products,

                )

                first_feature_idx += centroids_window_height

                dpex.barrier(dpex.LOCAL_MEM_FENCE)

            min_idx, min_sample_pseudo_inertia = update_closest_centroid(
                first_centroid_idx,
                min_idx,
                min_sample_pseudo_inertia,
                window_of_centroids_half_l2_norms,
                is_last_centroid_window,
                dot_products,
            )

            first_centroid_idx += window_n_centroids

            dpex.barrier(dpex.LOCAL_MEM_FENCE)

        _setitem_if(
            sample_idx < n_samples,
            sample_idx,
            min_idx,
            # OUT
            assignments_idx,
        )

    # HACK 906: see sklearn_numba_dpex.patches.tests.test_patches.test_need_to_workaround_numba_dpex_906  # noqa
    @dpex.func
    def _setitem_if(condition, index, value, array):
        if condition:
            array[index] = value
        return condition

    n_windows_for_sample = math.ceil(n_samples / window_n_centroids)

    global_size = (
        window_n_centroids,
        math.ceil(n_windows_for_sample / centroids_window_height)
        * centroids_window_height,
    )

    def kernel_call(*args):
        dpex.call_kernel(assignment, NdRange(global_size, work_group_shape), *args)

    return kernel_call
