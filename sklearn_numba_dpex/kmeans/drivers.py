import warnings
import math

import numpy as np
import dpctl
import dpctl.tensor as dpt
import dpnp

from sklearn_numba_dpex.common.random import (
    get_random_raw,
    create_xoroshiro128pp_states,
)

from sklearn_numba_dpex.common.kernels import (
    make_initialize_to_zeros_2d_kernel,
    make_initialize_to_zeros_3d_kernel,
    make_broadcast_division_1d_2d_axis0_kernel,
    make_broadcast_ops_1d_2d_axis1_kernel,
    make_half_l2_norm_2d_axis0_kernel,
    make_sum_reduction_2d_axis1_kernel,
    make_argmin_reduction_1d_kernel,
)

from sklearn_numba_dpex.kmeans.kernels import (
    make_lloyd_single_step_fixed_window_kernel,
    make_compute_euclidean_distances_fixed_window_kernel,
    make_label_assignment_fixed_window_kernel,
    make_compute_inertia_kernel,
    make_kmeansplusplus_init_kernel,
    make_sample_center_candidates_kernel,
    make_kmeansplusplus_single_step_fixed_window_kernel,
    make_relocate_empty_clusters_kernel,
    make_select_samples_far_from_centroid_kernel,
    make_centroid_shifts_kernel,
    make_reduce_centroid_data_kernel,
)


def lloyd(
    X_t,
    sample_weight,
    centroids_t,
    use_uniform_weights,
    max_iter=300,
    verbose=False,
    tol=1e-4,
):
    n_features, n_samples = X_t.shape
    n_clusters = centroids_t.shape[1]
    compute_dtype = X_t.dtype.type

    device = X_t.device.sycl_device
    max_work_group_size = device.max_work_group_size
    sub_group_size = min(device.sub_group_sizes)
    global_mem_cache_size = device.global_mem_cache_size
    centroids_private_copies_max_cache_occupancy = 0.7

    verbose = bool(verbose)

    # Create a set of kernels
    (
        n_centroids_private_copies,
        fused_lloyd_fixed_window_single_step_kernel,
    ) = make_lloyd_single_step_fixed_window_kernel(
        n_samples,
        n_features,
        n_clusters,
        return_assignments=bool(verbose),
        sub_group_size=sub_group_size,
        global_mem_cache_size=global_mem_cache_size,
        centroids_private_copies_max_cache_occupancy=centroids_private_copies_max_cache_occupancy,
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    assignment_fixed_window_kernel = make_label_assignment_fixed_window_kernel(
        n_samples,
        n_features,
        n_clusters,
        sub_group_size=sub_group_size,
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    compute_inertia_kernel = make_compute_inertia_kernel(
        n_samples, n_features, max_work_group_size, compute_dtype
    )

    reset_cluster_sizes_private_copies_kernel = make_initialize_to_zeros_2d_kernel(
        size0=n_centroids_private_copies,
        size1=n_clusters,
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    reset_centroids_private_copies_kernel = make_initialize_to_zeros_3d_kernel(
        size0=n_centroids_private_copies,
        size1=n_features,
        size2=n_clusters,
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    broadcast_division_kernel = make_broadcast_division_1d_2d_axis0_kernel(
        size0=n_features,
        size1=n_clusters,
        work_group_size=max_work_group_size,
    )

    compute_centroid_shifts_kernel = make_centroid_shifts_kernel(
        n_clusters=n_clusters,
        n_features=n_features,
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    half_l2_norm_kernel = make_half_l2_norm_2d_axis0_kernel(
        size0=n_features,
        size1=n_clusters,
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    reduce_inertia_kernel = make_sum_reduction_2d_axis1_kernel(
        size0=n_samples,
        size1=None,  # 1d reduction
        work_group_size=max_work_group_size,
        device=device,
        dtype=compute_dtype,
    )

    reduce_centroid_shifts_kernel = make_sum_reduction_2d_axis1_kernel(
        size0=n_clusters,
        size1=None,  # 1d reduction
        work_group_size=max_work_group_size,
        device=device,
        dtype=compute_dtype,
    )

    reduce_centroid_data_kernel = make_reduce_centroid_data_kernel(
        n_centroids_private_copies=n_centroids_private_copies,
        n_features=n_features,
        n_clusters=n_clusters,
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    # Allocate the necessary memory in the device global memory
    new_centroids_t = dpt.empty_like(centroids_t, device=device)
    centroids_half_l2_norm = dpt.empty(n_clusters, dtype=compute_dtype, device=device)
    cluster_sizes = dpt.empty(n_clusters, dtype=compute_dtype, device=device)
    centroid_shifts = dpt.empty(n_clusters, dtype=compute_dtype, device=device)
    # NB: the same buffer is used for those two arrays because it is never needed
    # to store those simultaneously in memory.
    sq_dist_to_nearest_centroid = per_sample_inertia = dpt.empty(
        n_samples, dtype=compute_dtype, device=device
    )
    assignments_idx = dpt.empty(n_samples, dtype=np.uint32, device=device)
    new_centroids_t_private_copies = dpt.empty(
        (n_centroids_private_copies, n_features, n_clusters),
        dtype=compute_dtype,
        device=device,
    )
    cluster_sizes_private_copies = dpt.empty(
        (n_centroids_private_copies, n_clusters),
        dtype=compute_dtype,
        device=device,
    )
    empty_clusters_list = dpt.empty(n_clusters, dtype=np.uint32, device=device)

    # n_empty_clusters_ is a scalar handled in kernels via a one-element array.
    n_empty_clusters = dpt.empty(1, dtype=np.int32, device=device)

    # The loop
    n_iteration = 0
    centroid_shifts_sum = np.inf

    # TODO: Investigate possible speedup with a custom dpctl queue with a custom
    # DAG of events and a final single "wait"
    while (n_iteration < max_iter) and (centroid_shifts_sum > tol):
        half_l2_norm_kernel(centroids_t, centroids_half_l2_norm)

        reset_cluster_sizes_private_copies_kernel(cluster_sizes_private_copies)
        reset_centroids_private_copies_kernel(new_centroids_t_private_copies)
        n_empty_clusters[0] = np.int32(0)

        # TODO: implement special case where only one copy is needed
        fused_lloyd_fixed_window_single_step_kernel(
            X_t,
            sample_weight,
            centroids_t,
            centroids_half_l2_norm,
            assignments_idx,
            new_centroids_t_private_copies,
            cluster_sizes_private_copies,
        )

        if verbose:
            # ???: verbosity comes at the cost of performance since it triggers
            # computing exact inertia at each iteration. Shouldn't this be
            # documented ?
            compute_inertia_kernel(
                X_t,
                sample_weight,
                new_centroids_t,
                assignments_idx,
                per_sample_inertia,
            )
            inertia, *_ = dpt.asnumpy(reduce_inertia_kernel(per_sample_inertia))
            print(f"Iteration {n_iteration}, inertia {inertia:5.3e}")

        reduce_centroid_data_kernel(
            cluster_sizes_private_copies,
            new_centroids_t_private_copies,
            cluster_sizes,
            new_centroids_t,
            empty_clusters_list,
            n_empty_clusters,
        )

        n_empty_clusters_ = int(n_empty_clusters[0])
        if n_empty_clusters_ > 0:
            # NB: empty cluster very rarely occurs, and it's more efficient to
            # compute inertia and labels only after occurrences have been detected
            # at the cost of an additional pass on data, rather than computing
            # inertia by default during the first pass on data in case there's an
            # empty cluster.

            # if verbose is True, then assignments to closest centroids already have
            # been computed in the main kernel
            if not verbose:
                assignment_fixed_window_kernel(
                    X_t, centroids_t, centroids_half_l2_norm, assignments_idx
                )

            # if verbose is True and if sample_weight is uniform, distances to
            # closest centroids already have been computed in the main kernel
            if not verbose or not use_uniform_weights:
                # Note that we intentionally we pass unit weights instead of
                # sample_weight so that per_sample_inertia will be updated to the
                # (unweighted) squared distance to the nearest centroid.
                compute_inertia_kernel(
                    X_t,
                    dpt.ones_like(sample_weight),
                    centroids_t,
                    assignments_idx,
                    sq_dist_to_nearest_centroid,
                )

            _relocate_empty_clusters(
                n_empty_clusters_,
                X_t,
                sample_weight,
                new_centroids_t,
                cluster_sizes,
                assignments_idx,
                empty_clusters_list,
                sq_dist_to_nearest_centroid,
                per_sample_inertia,
                max_work_group_size,
            )

        broadcast_division_kernel(new_centroids_t, cluster_sizes)

        compute_centroid_shifts_kernel(centroids_t, new_centroids_t, centroid_shifts)

        centroid_shifts_sum, *_ = reduce_centroid_shifts_kernel(centroid_shifts)

        # ???: unlike sklearn, sklearn_intelex checks that pseudo_inertia decreases
        # and keep an additional copy of centroids that is updated only if the
        # value of the pseudo_inertia is smaller than all the past values.
        #
        # (the check should not be needed because we have theoritical guarantee
        # that inertia decreases, if it doesn't it should only be because of
        # rounding errors ?)
        #
        # To this purpose this code could be inserted here:
        #
        # if pseudo_inertia < best_pseudo_inertia:
        #     best_pseudo_inertia = pseudo_inertia
        #     copyto_kernel(centroids_t, best_centroids_t)
        #
        # Note that what that is saved as "best centroid" is the array before the
        # update, to which the pseudo inertia that is computed at this iteration
        # refers. For this reason, this strategy is not compatible with sklearn
        # unit tests, that consider new_centroids_t (the array after the update)
        # to be the best centroids at each iteration.

        centroids_t, new_centroids_t = (new_centroids_t, centroids_t)

        n_iteration += 1

    if verbose:
        converged_at = n_iteration - 1
        if centroid_shifts_sum == 0:  # NB: possible if tol = 0
            print(f"Converged at iteration {converged_at}: strict convergence.")

        elif centroid_shifts_sum <= tol:
            print(
                f"Converged at iteration {converged_at}: center shift "
                f"{centroid_shifts_sum} within tolerance {tol}."
            )

    # Finally, run an assignment kernel to compute the assignments to the best
    # centroids found, along with the exact inertia.
    half_l2_norm_kernel(centroids_t, centroids_half_l2_norm)

    # NB: inertia and labels could be computed in a single fused kernel, however,
    # re-using the quantity Q = ((1/2)c^2 - <x.c>) that is computed in the
    # assignment kernel to compute distances to closest centroids to evaluate
    # |x|^2 - 2 * Q leads to numerical instability, we prefer evaluating the
    # expression |x-c|^2 which is stable but requires an additional pass on the
    # data.
    # See https://github.com/soda-inria/sklearn-numba-dpex/issues/28
    assignment_fixed_window_kernel(
        X_t, centroids_t, centroids_half_l2_norm, assignments_idx
    )

    compute_inertia_kernel(
        X_t, sample_weight, centroids_t, assignments_idx, per_sample_inertia
    )

    # inertia = per_sample_inertia.sum()
    inertia = dpt.asnumpy(reduce_inertia_kernel(per_sample_inertia))
    # inertia is now a 1-sized numpy array, we transform it into a scalar:
    inertia = inertia[0]

    return assignments_idx, inertia, centroids_t, n_iteration


def _relocate_empty_clusters(
    n_empty_clusters,
    X_t,
    sample_weight,
    centroids_t,
    cluster_sizes,
    assignments_idx,
    empty_clusters_list,
    sq_dist_to_nearest_centroid,
    per_sample_inertia,
    work_group_size,
):
    compute_dtype = X_t.dtype.type
    n_features, n_samples = X_t.shape
    device = X_t.device.sycl_device

    select_samples_far_from_centroid_kernel = (
        make_select_samples_far_from_centroid_kernel(
            n_empty_clusters, n_samples, work_group_size
        )
    )

    # NB: partition/argpartition kernels are hard to implement right, we use dpnp
    # implementation of `partition` and process to an additional pass on the data
    # to finish the argpartition.
    # ???: how does the dpnp GPU implementation of partition compare with
    # np.partition ?
    # TODO: if the performance compares well, we could also remove some of the
    # kernels in .kernels.utils and replace it with dpnp functions.
    kth = n_samples - n_empty_clusters
    threshold = dpnp.partition(
        dpnp.ndarray(
            shape=sq_dist_to_nearest_centroid.shape,
            buffer=sq_dist_to_nearest_centroid,
        ),
        kth=kth,
    ).get_array()[kth : (kth + 1)]

    samples_far_from_center = dpt.empty(n_samples, dtype=np.uint32, device=device)
    n_selected_gt_threshold = dpt.zeros(1, dtype=np.int32, device=device)
    n_selected_eq_threshold = dpt.ones(1, dtype=np.int32, device=device)
    select_samples_far_from_centroid_kernel(
        sq_dist_to_nearest_centroid,
        threshold,
        samples_far_from_center,
        n_selected_gt_threshold,
        n_selected_eq_threshold,
    )

    n_selected_gt_threshold_ = int(n_selected_gt_threshold[0])

    # Centroids of empty clusters are relocated to samples in X that are the
    # farthest from their respective centroids. new_centroids_t is updated
    # accordingly.
    relocate_empty_clusters_kernel = make_relocate_empty_clusters_kernel(
        n_empty_clusters,
        n_features,
        n_selected_gt_threshold_,
        work_group_size,
        compute_dtype,
    )

    relocate_empty_clusters_kernel(
        X_t,
        sample_weight,
        assignments_idx,
        samples_far_from_center,
        empty_clusters_list,
        per_sample_inertia,
        centroids_t,
        cluster_sizes,
    )


def prepare_data_for_lloyd(X_t, init, tol, copy_x):
    n_features, n_samples = X_t.shape
    compute_dtype = X_t.dtype.type

    device = X_t.device.sycl_device
    max_work_group_size = device.max_work_group_size

    sum_axis1_kernel = make_sum_reduction_2d_axis1_kernel(
        X_t.shape[0],
        X_t.shape[1],
        device.max_work_group_size,
        device=device,
        dtype=compute_dtype,
    )

    X_mean = (sum_axis1_kernel(X_t) / compute_dtype(n_samples))[:, 0]

    if (X_mean == 0).astype(int).sum() == len(X_mean):
        X_mean = None
    else:
        X_t = dpt.asarray(X_t, copy=copy_x)
        broadcast_X_minus_X_mean = make_broadcast_ops_1d_2d_axis1_kernel(
            n_features,
            n_samples,
            ops=_minus,
            work_group_size=max_work_group_size,
        )

        broadcast_X_minus_X_mean(X_t, X_mean)

        if isinstance(init, dpt.usm_ndarray):
            n_clusters = init.shape[1]
            broadcast_init_minus_X_mean = make_broadcast_ops_1d_2d_axis1_kernel(
                n_features,
                n_clusters,
                ops=_minus,
                work_group_size=max_work_group_size,
            )
            broadcast_init_minus_X_mean(init, X_mean)

    variance_kernel = make_sum_reduction_2d_axis1_kernel(
        n_features * n_samples,
        None,
        max_work_group_size,
        device=device,
        dtype=compute_dtype,
        fused_unary_func=_square,
    )
    variance = variance_kernel(dpt.reshape(X_t, -1)) / n_features
    tol = variance * tol

    return X_t, X_mean, init, tol


def _square(x):
    return x * x


def restore_data_after_lloyd(X_t, X_mean):
    n_features, n_samples = X_t.shape

    device = X_t.device.sycl_device
    max_work_group_size = device.max_work_group_size

    X_t = dpt.asarray(X_t, copy=False)
    broadcast_X_plus_X_mean = make_broadcast_ops_1d_2d_axis1_kernel(
        n_features,
        n_samples,
        ops=_plus,
        work_group_size=max_work_group_size,
    )
    broadcast_X_plus_X_mean(X_t, X_mean)


def _minus(x, y):
    return x - y


def _plus(x, y):
    return x + y


def get_labels_inertia(X_t, centroids_t, sample_weight, with_inertia):
    compute_dtype = X_t.dtype.type
    n_features, n_samples = X_t.shape
    n_clusters = centroids_t.shape[1]
    device = X_t.device.sycl_device
    max_work_group_size = device.max_work_group_size
    sub_group_size = min(device.sub_group_sizes)

    label_assignment_fixed_window_kernel = make_label_assignment_fixed_window_kernel(
        n_samples,
        n_features,
        n_clusters,
        sub_group_size=sub_group_size,
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    half_l2_norm_kernel = make_half_l2_norm_2d_axis0_kernel(
        size0=n_features,
        size1=n_clusters,
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    centroids_half_l2_norm = dpt.empty(n_clusters, dtype=compute_dtype, device=device)
    assignments_idx = dpt.empty(n_samples, dtype=np.uint32, device=device)

    half_l2_norm_kernel(centroids_t, centroids_half_l2_norm)

    label_assignment_fixed_window_kernel(
        X_t, centroids_t, centroids_half_l2_norm, assignments_idx
    )

    if not with_inertia:
        return assignments_idx, None

    compute_inertia_kernel = make_compute_inertia_kernel(
        n_samples, n_features, max_work_group_size, compute_dtype
    )

    reduce_inertia_kernel = make_sum_reduction_2d_axis1_kernel(
        size0=n_samples,
        size1=None,  # 1d reduction
        work_group_size=max_work_group_size,
        device=device,
        dtype=compute_dtype,
    )

    per_sample_inertia = dpt.empty(n_samples, dtype=compute_dtype, device=device)

    compute_inertia_kernel(
        X_t, sample_weight, centroids_t, assignments_idx, per_sample_inertia
    )

    # inertia = per_sample_inertia.sum()
    inertia = dpt.asnumpy(reduce_inertia_kernel(per_sample_inertia))

    return assignments_idx, inertia


def get_euclidean_distances(X_t, Y_t):
    compute_dtype = X_t.dtype.type
    n_features, n_samples = X_t.shape
    n_clusters = Y_t.shape[1]
    device = X_t.device.sycl_device
    max_work_group_size = device.max_work_group_size
    sub_group_size = min(device.sub_group_sizes)

    euclidean_distances_fixed_window_kernel = (
        make_compute_euclidean_distances_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            sub_group_size=sub_group_size,
            work_group_size=max_work_group_size,
            dtype=compute_dtype,
        )
    )

    euclidean_distances_t = dpt.empty(
        (n_clusters, n_samples), dtype=compute_dtype, device=device
    )

    euclidean_distances_fixed_window_kernel(X_t, Y_t, euclidean_distances_t)
    return euclidean_distances_t.T


def kmeans_plusplus(
    X_t,
    sample_weight,
    n_clusters,
    random_state,
):
    compute_dtype = X_t.dtype.type
    n_features, n_samples = X_t.shape
    device = X_t.device.sycl_device
    max_work_group_size = device.max_work_group_size
    sub_group_size = min(device.sub_group_sizes)

    # NB: the implementation differs from sklearn implementation with regards to
    # sample_weight, which is ignored in sklearn, but used here.
    # TODO: check that this implementation is correct when samples weights aren't
    # uniform.

    # Same retrial heuristic as scikit-learn (at least until <1.2)
    n_local_trials = 2 + int(np.log(n_clusters))

    # TODO: this block is also written in common.kernel.random, factorize ?
    from_cpu_to_device = False
    if not device.has_aspect_cpu:
        try:
            cpu_device = dpctl.SyclDevice("cpu")
            from_cpu_to_device = True
        except dpctl.SyclDeviceCreationError:
            warnings.warn(
                "No CPU found, falling back to the initialization of the k-means RNG "
                "on the default device."
            )

    kmeansplusplus_init_kernel = make_kmeansplusplus_init_kernel(
        n_samples,
        n_features,
        max_work_group_size,
        compute_dtype,
    )

    sample_center_candidates_kernel = make_sample_center_candidates_kernel(
        n_samples,
        n_local_trials,
        max_work_group_size,
        compute_dtype,
    )

    (
        kmeansplusplus_single_step_fixed_window_kernel
    ) = make_kmeansplusplus_single_step_fixed_window_kernel(
        n_samples,
        n_features,
        n_local_trials,
        sub_group_size,
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    select_best_candidate_kernel = make_argmin_reduction_1d_kernel(
        n_local_trials,
        max_work_group_size,
        device=device,
        dtype=compute_dtype,
    )

    reduce_potential_1d_kernel = make_sum_reduction_2d_axis1_kernel(
        size0=n_samples,
        size1=None,
        work_group_size=max_work_group_size,
        device=device,
        dtype=compute_dtype,
    )

    reduce_potential_2d_kernel = make_sum_reduction_2d_axis1_kernel(
        size0=n_local_trials,
        size1=n_samples,
        work_group_size=max_work_group_size,
        device=device,
        dtype=compute_dtype,
    )

    random_state = create_xoroshiro128pp_states(
        n_local_trials,
        seed=random_state,
        device=cpu_device if from_cpu_to_device else device,
    )
    if from_cpu_to_device:
        new_work_group_size = cpu_device.max_work_group_size
        new_global_size = (
            math.ceil(
                sample_center_candidates_kernel.global_size[0] / new_work_group_size
            )
            * new_work_group_size
        )
        sample_center_candidates_kernel = sample_center_candidates_kernel.configure(
            sycl_queue=random_state.sycl_queue,
            global_size=[new_global_size],
            local_size=[new_work_group_size],
        )

    centers_t = dpt.empty(
        sh=(n_features, n_clusters), dtype=compute_dtype, device=device
    )

    center_indices = dpt.full((n_clusters,), -1, dtype=np.int32)

    sq_distances_t = dpt.empty(
        sh=(n_local_trials, n_samples), dtype=compute_dtype, device=device
    )

    closest_dist_sq = dpt.empty(sh=(n_samples,), dtype=compute_dtype, device=device)

    candidate_ids = dpt.empty(sh=(n_local_trials,), dtype=np.int32, device=device)

    # Pick first center randomly
    starting_center_id = np.int32(get_random_raw(random_state) % n_samples)
    center_indices[0] = starting_center_id[0]

    # track index of point, initialize list of closest distances and calculate
    # current potential
    kmeansplusplus_init_kernel(
        X_t, sample_weight, centers_t, center_indices, closest_dist_sq
    )
    total_potential = reduce_potential_1d_kernel(closest_dist_sq)

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # First, let's sample indices of candidates using a empirical cumulative
        # density function built using the potential of the samples and squared
        # distances to each sample's closest centroids.

        # NB: this step consists in highly sequential instructions that can only
        # be parallelized in n_local_trials threads, it's up to 50x faster to run it
        # on CPU, and becomes the bottleneck if ran on GPU. Depending on hardware
        # and weight of data transfers, are there cases where it would be more
        # efficient to keep it on GPU ?
        # ???: would it be bad to sample the sample weight once outside the loop and
        # reuse it at each iteration ?
        if from_cpu_to_device:
            candidate_ids = candidate_ids.to_device(cpu_device)

            sample_center_candidates_kernel(
                closest_dist_sq.to_device(cpu_device),
                total_potential.to_device(cpu_device),
                random_state,
                candidate_ids,
            )
            candidate_ids = candidate_ids.to_device(device)
        else:
            sample_center_candidates_kernel(
                closest_dist_sq, total_potential, random_state, candidate_ids
            )

        # Now, for each (sample, candidate)-pair, compute the minimum between
        # their distance and the previous minimum.

        # XXX: at the cost of one additional pass on data, we could avoid storing
        # entirely distance_to_candidates_t in memory, and save
        # `dtype.nbytes * n_local_trials * n_sample` bytes in memory.
        # Which is better ?
        kmeansplusplus_single_step_fixed_window_kernel(
            X_t, sample_weight, candidate_ids, closest_dist_sq, sq_distances_t
        )

        candidate_potentials = reduce_potential_2d_kernel(sq_distances_t)[:, 0]
        best_candidate = select_best_candidate_kernel(candidate_potentials)[0]

        total_potential = candidate_potentials[best_candidate : (best_candidate + 1)]

        # Pick the c-th centroid and update the distance
        # to the closest centroid for each sample.
        closest_dist_sq = sq_distances_t[best_candidate, :]
        center_index = candidate_ids[best_candidate]
        centers_t[:, c] = X_t[:, center_index]
        center_indices[c] = center_index

    return centers_t, center_indices
