import math
from functools import lru_cache

import numba_dpex as dpex
import numpy as np

from sklearn_numba_dpex.common._utils import _check_max_work_group_size

from ._base_kmeans_kernel_funcs import (
    make_pairwise_ops_base_kernel_funcs,
    make_update_closest_centroid_kernel_func,
)

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
def make_lloyd_single_step_fixed_window_kernel(
    n_samples,
    n_features,
    n_clusters,
    return_assignments,
    check_strict_convergence,
    sub_group_size,
    centroids_private_copies_max_cache_occupancy,
    work_group_size,
    dtype,
    device,
):
    # The height of the window on centroids (or, equivalently, the number of features
    # in the window), and the width (number of centroids in the window), are chosen
    # such that:
    #   - the width is equal to the `sub_group_size`, which is the minimal size that
    #     optimizes IO patterns by having items in each sub group read contiguous
    #     memory slots.  This configuration can optimize IO bandwidth by triggering
    #     coalescence of the read operations of each item in the sub group. Higher
    #     sizes would increase the pressure on private memory (since the size of the
    #     private `dot_products` array is equal to the width of the window) with no
    #     particular gain to expect.
    #   - the height is chosen such that the window counts `work_group_size` items.
    #     When a window is cooperatively loaded into shared memory by a work group,
    #     each work item is thus tasked with loading one and only one item. The number
    #     of items is: `centroids_window_height * window_n_centroids`, i.e.
    #     `centroids_window_height * sub_group_size`
    window_n_centroids = sub_group_size

    dtype_itemsize = np.dtype(dtype).itemsize
    input_work_group_size = work_group_size
    work_group_size = _check_max_work_group_size(
        work_group_size,
        device,
        required_local_memory_per_item=dtype_itemsize,
        required_memory_constant=sub_group_size * dtype_itemsize,
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
        accumulate_dot_products,
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

    n_subgroups = math.ceil(n_samples / window_n_centroids)

    # NB: for more details about the privatization strategy the following variables
    # refer too, please read the inline commenting that address it in the kernel
    # definition.

    # Each set of `sub_group_size` consecutive work items is assigned one private
    # copy, and several such sets can be assigned to the same private copy. Thus, at
    # most `n_subgroups` private copies are needed.
    # Moreover, collisions can only occur between sub groups that execute concurrently.
    # Thus, at most `nb_concurrent_sub_groups` private copies are needed.
    # TODO: `nb_concurrent_sub_groups` is considered equal to
    # `device.max_compute_units`. We're not sure that this is the correct
    # read of the device specs. Confirm or fix once it's made clearer. Suggested reads
    # that highlight complexity of the execution model:
    # - https://github.com/IntelPython/dpctl/issues/1033
    # - https://stackoverflow.com/a/6490897
    n_centroids_private_copies = int(min(n_subgroups, device.max_compute_units))

    # Safety check for edge case where `n_centroids_private_copies` equals 0 because
    # `n_samples` is null.
    n_centroids_private_copies = max(n_centroids_private_copies, 1)

    zero_idx = np.int64(0)
    one_idx = np.int64(1)
    zero_as_uint32 = np.uint32(0)
    inf = dtype(math.inf)

    # TODO: currently, constant memory is not supported by numba_dpex, but for read-only
    # inputs such as X_t it is generally regarded as faster. Once support is available
    # (NB: it's already supported by numba.cuda) X_t should be an input to the factory
    # rather than an input to the kernel.
    # XXX: parts of the kernels are factorized using `dpex.func` namespace that allow
    # defining device functions that can be used within `dpex.kernel` definitions.
    # Howver, `dpex.func` functions does not support dpex.barrier calls nor
    # creating local or private arrays. As a consequence, factorizing the kmeans kernels
    # remains a best effort and some code patternsd remain duplicated, In particular
    # the following kernel definition contains a lot of inline comments but those
    # comments are not repeated in the similar patterns in the other kernels
    @dpex.kernel
    # fmt: off
    def fused_lloyd_single_step(
        X_t,                               # IN READ-ONLY   (n_features, n_samples)
        sample_weight,                     # IN READ-ONLY   (n_features,)
        current_centroids_t,               # IN             (n_features, n_clusters)
        centroids_half_l2_norm,            # IN             (n_clusters,)
        previous_assignments_idx,          # IN             (n_samples,)
        assignments_idx,                   # OUT            (n_samples,)
        strict_convergence_status,         # OUT            (1,)
        new_centroids_t_private_copies,    # OUT            (n_private_copies, n_features, n_clusters)  # noqa
        cluster_sizes_private_copies,      # OUT            (n_private_copies, n_clusters)  # noqa
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
            |x-c|^2 = |x|^2 - 2<x.c> + |c|^2
        over c for a given x amounts to minimizing
            (1/2)c^2 - <x.c> .
        Moreover the value (1/2)c^2 has been pre-computed in the array
        centroids_half_l2_norm to reduce the overall number of floating point
        operations in the kernel.
        """
        # NB: the axis in the following dpex calls are reversed, so the kernel further
        # reads like a SYCL kernel that maps 2D group size with a row-major order,
        # despite that `numba_dpex` chose to mimic the column-major order style of
        # mapping 2D group sizes in cuda.
        sub_group_idx = dpex.get_global_id(one_idx)
        local_row_idx = dpex.get_local_id(one_idx)
        local_col_idx = dpex.get_local_id(zero_idx)

        # Let's start by remapping the 2D grid of work items to a 1D grid that reflect
        # how contiguous work items address one contiguoue sample_idx:
        sample_idx = (sub_group_idx * sub_group_size) + local_col_idx
        # NB: The 2D work group shape makes it easier (and less expensive) to map
        # the local memory arrays to the array of centroids. Do not get confused by the
        # fact that this shape is unrelated to how the kernel is parallelized on the
        # samples, where each work item applies to one sample.

        # This array in shared memory is used as a sliding array over values of
        # current_centroids_t. During each iteration in the inner loop, a new one is
        # loaded and used by all work items in the work group to compute partial
        # results. The array slides over the features in the outer loop, and over the
        # samples in the inner loop.
        centroids_window = dpex.local.array(shape=centroids_window_shape, dtype=dtype)

        # This array in shared memory is used as a sliding array over the centroids.
        # It contains values of centroids_half_l2_norm for each centroid in the sliding
        # centroids_window array. It is updated once per iteration in the outer loop.
        window_of_centroids_half_l2_norms = dpex.local.array(
            shape=window_n_centroids, dtype=dtype
        )

        # In the inner loop each work item accumulates in private memory the
        # dot product of the sample at the sample_idx relatively to each centroid
        # in the window.
        dot_products = dpex.private.array(shape=window_n_centroids, dtype=dtype)

        first_centroid_idx = zero_idx

        # The two variables that are initialized here will contain the result we seek,
        # i.e, at the end of the outer loop, it will be equal to the closest centroid
        # to the current sample and the corresponding pseudo inertia.
        min_idx = zero_idx
        min_sample_pseudo_inertia = inf

        # Those variables are used in the inner loop during loading of the window of
        # centroidswork_group_shape
        window_loading_feature_offset = local_row_idx
        window_loading_centroid_idx = local_col_idx

        # STEP 1: compute the closest centroid
        # Outer loop: iterate on successive windows of size window_n_centroids that
        # cover all centroids in current_centroids_t

        # TODO: currently. `numba_dpex` does not try to unroll loops. We can expect
        # (following https://github.com/IntelPython/numba-dpex/issues/770) that loop
        # unrolling will work in a future release, like it already does in vanilla
        # `numba`. Note though, that `numba` cannot unroll nested loops, so won't
        # `numba_dpex`. To leverage loop unrolling, the following nested loop will
        # require to be un-nested.
        for centroid_window_idx in range(n_windows_for_centroids):
            # window_of_centroids_half_l2_norms and dot_products
            # are modified in place.
            is_last_centroid_window = centroid_window_idx == last_centroid_window_idx
            initialize_window_of_centroids(
                local_row_idx,
                local_col_idx,
                first_centroid_idx,
                centroids_half_l2_norm,
                is_last_centroid_window,
                # OUT
                window_of_centroids_half_l2_norms,
                dot_products,
            )

            loading_centroid_idx = first_centroid_idx + window_loading_centroid_idx

            first_feature_idx = zero_idx

            # Inner loop: interate on successive windows of size window_n_features
            # that cover all features for current given centroids
            for feature_window_idx in range(n_windows_for_features):
                # centroids_window is modified inplace
                is_last_feature_window = feature_window_idx == last_feature_window_idx
                load_window_of_centroids_and_features(
                    first_feature_idx,
                    loading_centroid_idx,
                    window_loading_centroid_idx,
                    window_loading_feature_offset,
                    current_centroids_t,
                    # OUT
                    centroids_window,
                )
                # Since other work items are responsible for loading the relevant data
                # for the next step, we need to wait for completion of all work items
                # before going forward
                dpex.barrier(dpex.LOCAL_MEM_FENCE)

                accumulate_dot_products(
                    sample_idx,
                    first_feature_idx,
                    X_t,
                    centroids_window,
                    is_last_feature_window,
                    is_last_centroid_window,
                    # OUT
                    dot_products
                )

                first_feature_idx += centroids_window_height

                # When the next iteration starts work items will overwrite shared memory
                # with new values, so before that we must wait for all reading
                # operations in the current iteration to be over for all work items.
                dpex.barrier(dpex.LOCAL_MEM_FENCE)

            # End of inner loop. The pseudo inertia is now computed for all centroids
            # in the window, we can coalesce it to the accumulation of the min pseudo
            # inertia for the current sample.
            min_idx, min_sample_pseudo_inertia = update_closest_centroid(
                first_centroid_idx,
                min_idx,
                min_sample_pseudo_inertia,
                window_of_centroids_half_l2_norms,
                is_last_centroid_window,
                dot_products,
            )

            first_centroid_idx += window_n_centroids

            # When the next iteration starts work items will overwrite shared memory
            # with new values, so before that we must wait for all reading
            # operations in the current iteration to be over for all work items.
            dpex.barrier(dpex.LOCAL_MEM_FENCE)

        # End of outer loop. By now min_idx and min_sample_pseudo_inertia
        # contains the expected values.

        _update_result_data(
            sample_idx,
            min_idx,
            sub_group_idx,
            X_t,
            sample_weight,
            previous_assignments_idx,
            # OUT
            assignments_idx,
            strict_convergence_status,
            cluster_sizes_private_copies,
            new_centroids_t_private_copies,
        )

    # HACK 906: see sklearn_numba_dpex.patches.tests.test_patches.test_need_to_workaround_numba_dpex_906  # noqa
    @dpex.func
    # fmt: off
    def _update_result_data(
        sample_idx,                         # PARAM
        min_idx,                            # PARAM
        sub_group_idx,                      # PARAM
        X_t,                                # IN
        sample_weight,                      # IN
        previous_assignments_idx,           # IN
        assignments_idx,                    # OUT
        strict_convergence_status,          # OUT
        cluster_sizes_private_copies,       # OUT
        new_centroids_t_private_copies,     # OUT
    ):
        # fmt: on

        # NB: this check can't be moved at the top at the kernel, because if a work item
        # exits early with a `return` it will never reach the barriers, thus causing a
        # deadlock. Early returns are only possible when there are no barriers within
        # the remaining set of instructions for the running kernel.
        if sample_idx >= n_samples:
            return

        if return_assignments:
            assignments_idx[sample_idx] = min_idx

        if check_strict_convergence:
            current_strict_convergence_status = strict_convergence_status[zero_idx]
            if (current_strict_convergence_status != zero_as_uint32):
                if (previous_assignments_idx[sample_idx] != min_idx):
                    strict_convergence_status[zero_idx] = zero_as_uint32

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
        # `sub_group_size` is assigned to a different duplicata and update the values
        # of this single duplicata.

        # The resulting copies of centroids updates will then need to be reduced to a
        # single array of centroids in a complementary kernel.

        # The privatization is more effective when there is a low number of centroid
        # values (equal to `n_clusters * n_features`) comparatively to the global
        # number of work items, i.e. when the probability of collision is high. At the
        # opposite end where the probability of collision is low, privatization might
        # be detrimental to performance and we might prefer simpler, faster code
        # with updates directly made into the final array of centroids.

        # The privatization strategy also applies to the updates of the centroid
        # counts.

        # each work item is assigned an array of centroids in a round robin manner
        privatization_idx = sub_group_idx % n_centroids_private_copies
        weight = sample_weight[sample_idx]

        dpex.atomic.add(
            cluster_sizes_private_copies,
            (privatization_idx, min_idx),
            weight
        )

        for feature_idx in range(n_features):
            dpex.atomic.add(
                new_centroids_t_private_copies,
                (privatization_idx, feature_idx, min_idx),
                X_t[feature_idx, sample_idx] * weight,
            )

    global_size = (
        window_n_centroids,
        math.ceil(n_subgroups / centroids_window_height) * centroids_window_height,
    )

    return (
        n_centroids_private_copies,
        fused_lloyd_single_step[global_size, work_group_shape],
    )
