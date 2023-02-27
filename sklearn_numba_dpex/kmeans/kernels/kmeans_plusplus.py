import math
from functools import lru_cache

import numba_dpex as dpex
import numpy as np

from sklearn_numba_dpex.common._utils import _check_max_work_group_size
from sklearn_numba_dpex.common.random import make_rand_uniform_kernel_func

from ._base_kmeans_kernel_funcs import make_pairwise_ops_base_kernel_funcs

# NB: refer to the definition of the main lloyd function for a more comprehensive
# inline commenting of the kernel.


@lru_cache
def make_kmeansplusplus_init_kernel(
    n_samples,
    n_features,
    work_group_size,
    dtype,
):

    zero_idx = np.int64(0)
    zero_init = dtype(0.0)

    @dpex.kernel
    # fmt: off
    def kmeansplusplus_init(
        X_t,                      # IN READ-ONLY   (n_features, n_samples)
        sample_weight,            # IN READ-ONLY   (n_samples,)
        centers_t,                # OUT            (n_features, n_clusters)
        center_indices,           # OUT            (n_clusters,)
        closest_dist_sq,          # OUT            (n_samples,)
    ):
        # fmt: on
        sample_idx = dpex.get_global_id(zero_idx)
        if sample_idx >= n_samples:
            return

        starting_center_id_ = center_indices[zero_idx]

        sq_distance = zero_init
        for feature_idx in range(n_features):
            diff = X_t[feature_idx, sample_idx] - X_t[feature_idx, starting_center_id_]
            sq_distance += diff * diff

        sq_distance *= sample_weight[sample_idx]
        closest_dist_sq[sample_idx] = sq_distance

        if sample_idx > zero_idx:
            return

        for feature_idx in range(n_features):
            centers_t[feature_idx, zero_idx] = X_t[feature_idx, starting_center_id_]

    global_size = (math.ceil(n_samples / work_group_size)) * (work_group_size)
    return kmeansplusplus_init[global_size, work_group_size]


@lru_cache
def make_sample_center_candidates_kernel(
    n_samples,
    n_local_trials,
    work_group_size,
    dtype,
):

    rand_uniform_kernel_func = make_rand_uniform_kernel_func(np.dtype(dtype))

    zero_idx = np.int64(0)
    one_incr = np.int32(1)
    one_decr = np.int32(-1)
    zero_init = dtype(0.0)
    max_candidate_id = np.int32(n_samples - 1)

    @dpex.kernel
    # fmt: off
    def sample_center_candidates(
        closest_dist_sq,          # IN             (n_features, n_samples)
        total_potential,          # IN             (1,)
        random_state,             # INOUT          (n_local_trials, 2)
        candidates_id,            # OUT            (n_local_trials,)
    ):
        # fmt: on
        local_trial_idx = dpex.get_global_id(zero_idx)
        if local_trial_idx >= n_local_trials:
            return
        random_value = (rand_uniform_kernel_func(random_state, local_trial_idx)
                        * total_potential[zero_idx])

        cumulative_potential = zero_init
        candidate_id = one_decr
        while (
                (random_value > cumulative_potential) and
                (candidate_id < max_candidate_id)
        ):
            candidate_id += one_incr
            cumulative_potential += closest_dist_sq[candidate_id]
        candidates_id[local_trial_idx] = candidate_id

    global_size = (math.ceil(n_local_trials / work_group_size)) * work_group_size
    return sample_center_candidates[global_size, work_group_size]


@lru_cache
def make_kmeansplusplus_single_step_fixed_window_kernel(
    n_samples, n_features, n_candidates, sub_group_size, work_group_size, dtype, device
):

    window_n_candidates = sub_group_size

    input_work_group_size = work_group_size
    work_group_size = _check_max_work_group_size(
        work_group_size, device, required_local_memory_per_item=np.dtype(dtype).itemsize
    )

    candidates_window_height = work_group_size // sub_group_size

    if (work_group_size == input_work_group_size) and (
        (candidates_window_height * sub_group_size) != work_group_size
    ):
        raise ValueError(
            "Expected work_group_size to be a multiple of sub_group_size but got "
            f"sub_group_size={sub_group_size} and work_group_size={work_group_size}"
        )

    work_group_shape = (candidates_window_height, window_n_candidates)

    (
        initialize_window_of_candidates,
        load_window_of_candidates_and_features,
        accumulate_sq_distances,
    ) = make_pairwise_ops_base_kernel_funcs(
        n_samples,
        n_features,
        n_samples,
        candidates_window_height,
        window_n_candidates,
        ops="squared_diff",
        dtype=dtype,
        initialize_window_of_centroids_half_l2_norms=False,
    )

    n_windows_for_candidates = math.ceil(n_candidates / window_n_candidates)
    n_windows_for_features = math.ceil(n_features / candidates_window_height)
    last_candidate_window_idx = n_windows_for_candidates - 1
    last_feature_window_idx = n_windows_for_features - 1

    zero_idx = np.int64(0)
    one_idx = np.int64(1)

    @dpex.kernel
    # fmt: off
    def kmeansplusplus_single_step(
        X_t,                               # IN READ-ONLY   (n_features, n_samples)
        sample_weight,                     # IN READ-ONLY   (n_samples,)
        candidates_ids,                    # IN             (n_candidates,)
        closest_dist_sq,                   # IN             (n_samples,)
        sq_distances_t,                    # OUT            (n_candidates, n_samples)
    ):
        # fmt: on

        candidates_window = dpex.local.array(shape=work_group_shape, dtype=dtype)

        sq_distances = dpex.private.array(shape=window_n_candidates, dtype=dtype)

        first_candidate_idx = zero_idx

        local_col_idx = dpex.get_local_id(one_idx)

        window_loading_feature_offset = dpex.get_local_id(zero_idx)
        window_loading_candidate_idx = local_col_idx

        sample_idx = (
            (dpex.get_global_id(zero_idx) * sub_group_size)
            + local_col_idx
        )

        for candidate_window_idx in range(n_windows_for_candidates):
            is_last_candidate_window = candidate_window_idx == last_candidate_window_idx
            initialize_window_of_candidates(is_last_candidate_window, sq_distances)

            loading_candidate_idx = first_candidate_idx + window_loading_candidate_idx
            if loading_candidate_idx < n_candidates:
                loading_candidate_idx = candidates_ids[loading_candidate_idx]
            else:
                loading_candidate_idx = n_samples

            first_feature_idx = zero_idx

            for feature_window_idx in range(n_windows_for_features):
                is_last_feature_window = feature_window_idx == last_feature_window_idx
                load_window_of_candidates_and_features(
                    first_feature_idx,
                    loading_candidate_idx,
                    window_loading_candidate_idx,
                    window_loading_feature_offset,
                    X_t,
                    candidates_window,
                )

                dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

                accumulate_sq_distances(
                    sample_idx,
                    first_feature_idx,
                    X_t,
                    candidates_window,
                    is_last_feature_window,
                    is_last_candidate_window,
                    sq_distances
                )

                first_feature_idx += candidates_window_height

                dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

            _save_sq_distances(
                sample_idx,
                first_candidate_idx,
                sq_distances,
                sample_weight,
                closest_dist_sq,
                # OUT
                sq_distances_t
            )

            first_candidate_idx += window_n_candidates

            dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

    # HACK 906: see sklearn_numba_dpex.patches.tests.test_patches.test_hack_906
    @dpex.func
    # fmt: off
    def _save_sq_distances(
        sample_idx,             # PARAM
        first_candidate_idx,    # PARAM
        sq_distances,           # IN
        sample_weight,          # IN
        closest_dist_sq,        # IN
        sq_distances_t,         # OUT
    ):
        # fmt: on
        if sample_idx >= n_samples:
            return

        sample_weight_ = sample_weight[sample_idx]
        closest_dist_sq_ = closest_dist_sq[sample_idx]
        for i in range(window_n_candidates):
            candidate_idx = first_candidate_idx + i
            if candidate_idx < n_candidates:
                sq_distance_i = min(sq_distances[i] * sample_weight_, closest_dist_sq_)
                sq_distances_t[first_candidate_idx + i, sample_idx] = sq_distance_i

    n_windows_for_samples = math.ceil(n_samples / window_n_candidates)

    global_size = (
        math.ceil(n_windows_for_samples / candidates_window_height)
        * candidates_window_height,
        window_n_candidates,
    )
    return kmeansplusplus_single_step[global_size, work_group_shape]
