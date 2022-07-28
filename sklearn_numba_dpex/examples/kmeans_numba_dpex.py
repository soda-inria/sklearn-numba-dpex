# Reproduction of https://intel.github.io/scikit-learn-intelex/samples/kmeans.html
# install with: pip install scikit-learn-intelex "sklearn<1.1" pandas within the
# docker container

# NB: pass --cpus=x when opening the docker container to prevent
# crashing because of too high cpu leverage

from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

from numpy.random import default_rng
import numba_dppy as dpex
from numba import float32, uint32

import dpctl
import numpy as np
import math


# NB: must have rq <= p and rp <= q
def get_assignment_step_regs_kernel(rp, rq, p, q, n, dim, n_clusters):

    group_nb_centroids = q * rq
    group_nb_x_train = p * rp

    q_centroids_shm_dim = (group_nb_centroids, dim)
    p_x_train_shm_dim = (group_nb_x_train, dim)
    local_assignments_shm_dim = (p, rp, q)

    n_cluster_groups = math.ceil(n_clusters/(q * rq))
    n_data_groups = math.ceil(n/(p * rp))
    n_threads = (n_data_groups * p, n_cluster_groups * q)
    thread_group_size = (p, q)

    @dpex.kernel
    def assignment_step_regs(x_train, centroids, coarse_assignments_dist,
                             coarse_assignments_idx):

        p_q_group_j = dpex.get_group_id(1)

        p_q_thread_i = dpex.get_local_id(0)
        p_q_thread_j = dpex.get_local_id(1)

        thread_i = dpex.get_global_id(0)
        thread_j = dpex.get_global_id(1)

        first_x_train_global_idx = thread_i * rp
        first_x_train_local_idx = p_q_thread_i * rp
        first_centroid_global_idx = thread_j * rq
        first_centroid_local_idx = p_q_thread_j * rq

        q_centroids = dpex.local.array(shape=q_centroids_shm_dim, dtype=float32)
        p_x_train = dpex.local.array(shape=p_x_train_shm_dim, dtype=float32)
        local_assignments_dist = dpex.local.array(shape=local_assignments_shm_dim, dtype=float32)
        local_assignments_idx = dpex.local.array(shape=local_assignments_shm_dim, dtype=uint32)

        # TODO: factor in a dpex kernel function ? need to test limitations of
        # kernel functions before.
        # Can it enclose memory use ?
        # !!!: those copy are efficients (regarding coalesced reading in global
        # memory) only if dim > max(rp, rq), which is an okay assumption

        # here there is an improvement over the original paper since we use
        # p * rp threads for the copy instead of p threads, it holds as long as
        # q > rp, which we assume

        # initialize shared memory for x_train
        if p_q_thread_j < rp:
            x_train_global_idx = first_x_train_global_idx + p_q_thread_j
            if x_train_global_idx < n:
                x_train_local_idx = first_x_train_local_idx + p_q_thread_j
                for d in range(dim):
                    p_x_train[x_train_local_idx, d] = x_train[
                        x_train_global_idx, d]

        # here there is an improvement over the original paper since we use
        # q * rq threads for the copy instead of p threads, it holds as long as
        # p > rq, which we assume

        # initialize shared memory for centroids
        if p_q_thread_i < rq:
            centroid_global_idx = first_centroid_global_idx + p_q_thread_i
            if centroid_global_idx < n_clusters:
                centroid_local_idx = first_centroid_local_idx + p_q_thread_i
                for d in range(dim):
                    q_centroids[centroid_local_idx, d] = centroids[
                        centroid_global_idx, d]

        dpex.barrier()

        # Compute the closest centroids
        for k_p in range(rp):
            x_train_global_idx = first_x_train_global_idx + k_p
            if x_train_global_idx >= n:
                continue

            x_train_local_idx = first_x_train_local_idx + k_p

            current_best_dist = float32(math.inf)
            current_best_idx = uint32(0)

            for k_q in range(rq):
                centroid_global_idx = first_centroid_global_idx + k_q
                if centroid_global_idx >= n_clusters:
                    continue

                centroid_local_idx = first_centroid_local_idx + k_q

                dist = float32(0.0)
                for d in range(dim):
                    x_train_feature = p_x_train[x_train_local_idx, d]
                    centroid_feature = q_centroids[centroid_local_idx, d]
                    diff = centroid_feature - x_train_feature
                    dist += diff * diff
                smaller_dist = dist < current_best_dist
                current_best_dist = dist if smaller_dist else current_best_dist
                current_best_idx = uint32(k_q) if smaller_dist else current_best_idx

            local_assignments_dist[p_q_thread_i, k_p, p_q_thread_j] = current_best_dist
            local_assignments_idx[p_q_thread_i, k_p, p_q_thread_j] = current_best_idx + first_centroid_global_idx

        dpex.barrier()

        # here there is an improvement over the original paper since we use
        # p * rp threads for the copy instead of p threads, it holds as long as
        # q > rp, which we assume

        if p_q_thread_j >= rp:
            return

        if x_train_global_idx >= n:
            return

        min_dist = local_assignments_dist[p_q_thread_i, p_q_thread_j, 0]
        min_idx = local_assignments_idx[p_q_thread_i, p_q_thread_j, 0]
        for j in range(1, q):
            current_dist = local_assignments_dist[p_q_thread_i, p_q_thread_j, j]
            smaller_dist = current_dist < min_dist
            min_dist = current_dist if smaller_dist else min_dist
            min_idx = local_assignments_idx[p_q_thread_i, p_q_thread_j, j] if smaller_dist else min_idx

        coarse_assignments_dist[first_x_train_global_idx + p_q_thread_j, p_q_group_j] = min_dist
        coarse_assignments_idx[first_x_train_global_idx + p_q_thread_j, p_q_group_j] = min_idx

    return n_cluster_groups, assignment_step_regs[n_threads, thread_group_size]

def get_assignment_step_fixed_kernel(h, r, thread_group_size, n, dim, n_clusters):
    n_cluster_groups = math.ceil(n_clusters/r)
    n_threads = n * math.ceil(n_clusters / r)
    n_threads = math.ceil(n_threads / thread_group_size) * thread_group_size
    n_windows = math.ceil(dim/h)
    window_shm_shape = (r, h)
    window_elem_nb = (h * r)
    partial_sums_shm_shape = (thread_group_size, r)
    nb_window_entry_per_thread = math.ceil(r * h / thread_group_size)

    @dpex.kernel
    def assignment_step_fixed(x_train, centroids, coarse_assignments_dist,
                               coarse_assignments_idx):

        group = dpex.get_group_id(0)
        local_thread = dpex.get_local_id(0)

        x_train_global_idx = thread_group_size * (group // n_cluster_groups) + local_thread
        centroid_window = (group % n_cluster_groups)
        first_centroid_global_idx = centroid_window * r

        window = dpex.local.array(shape=window_shm_shape, dtype=float32)
        # NB: this should be in private memory but numba_dpex does not expose
        # functions to manage private memory so we'll use shared memory instead
        # possible slight performance grief but should be minor
        partial_sums = dpex.local.array(shape=partial_sums_shm_shape, dtype=float32)

        for window_k in range(n_windows):

            first_dim_global_idx = window_k * h

            # load memory
            for k in range(nb_window_entry_per_thread):
                flat_idx = local_thread * nb_window_entry_per_thread + k
                if flat_idx >= window_elem_nb:
                    continue
                row = flat_idx // r
                col = flat_idx % r
                centroids_row = first_centroid_global_idx + row
                centroids_col = col + first_dim_global_idx
                if centroids_row < n_clusters and centroids_col < dim:
                    window[row, col] = centroids[centroids_row, centroids_col]

            dpex.barrier()
            # compute partial sums
            # NB: performance here relies on L1 cache for x_train_feature quick
            # read access
            # ???: We've assumed that the execution order of thread groups follow
            # their indexing to make the most out of L1 cache. It sounds sensible
            # but does it really hold ?
            first_window = window_k == 0
            for k in range(r):
                if (first_centroid_global_idx + k) >= n_clusters:
                    break
                tmp = float32(0.0)
                for d in range(h):
                    current_dim_global_idx = d + first_dim_global_idx
                    if current_dim_global_idx >= dim:
                        break
                    centroid_feature = window[k, d]
                    # performance for the line thereafter relies on L1 cache
                    x_train_feature = x_train[x_train_global_idx, current_dim_global_idx]
                    diff = centroid_feature - x_train_feature
                    tmp += diff * diff
                if first_window:
                    partial_sums[local_thread, k] = tmp
                else:
                    partial_sums[local_thread, k] += tmp

            # wait that everybody is done with reading shared memory before rewriting
            # it for the next window
            dpex.barrier()

        min_dist = partial_sums[local_thread, 0]
        min_idx = uint32(0)
        for k in range(1, r):
            if (first_centroid_global_idx + k) >= n_clusters:
                break
            current_dist = partial_sums[local_thread, k]
            smaller_dist = current_dist < min_dist
            min_dist = current_dist if smaller_dist else min_dist
            min_idx = uint32(k) if smaller_dist else min_idx

        coarse_assignments_dist[x_train_global_idx, centroid_window] = min_dist
        coarse_assignments_idx[x_train_global_idx, centroid_window] = min_idx + first_centroid_global_idx

    return n_cluster_groups, assignment_step_fixed[n_threads, thread_group_size]


# WIP: assignment reduction and centroid update
# @dpex.kernel
# def fine_assignment_step(coarse_assignments_dist, coarse_assignments_idx,
#                          fine_assignments_dist, fine_assignments_idx):
#     thread_i = dpex.get_global_id(0)
#     n = coarse_assignments_dist.shape[0]
#     if thread_i > n:
#         return

#     nb_coarse = coarse_assignments_dist.shape[1]
#     # TODO:
#     # 1 thread = 1 pt
#     # thread lit n
#     fine_dist = coarse_assignments_dist[thread_i, 0]
#     fine_idx = coarse_assignments_idx[thread_i, 0]
#     for i in range(1, nb_coarse):
#         current_dist = coarse_assignments_dist[thread_i, i]
#         smaller_dist = current_dist < fine_dist
#         fine_dist = current_dist if smaller_dist else fine_dist
#         fine_idx = coarse_assignments_idx[thread_i, i] if smaller_dist else fine_idx
#     fine_assignments_dist[thread_i] = fine_dist
#     fine_assignments_idx[thread_i] = fine_idx

#     # TODO: use atomic add to also count the number of point per cluster

# @dpex.kernel
# def update_means_step_fine(x, assignments_dist, assignments_idx, centroids):
#     return

# def get_update_means_step_shm_kernel():


#     @dpex.kernel
#     def update_means_step_shm(x, assignments_dist, assignments_idx, centroid_counts,
#                               centroids):
#         '''
#         NB: here we do not implement exactly what is given in the paper
#         At this point we don't want to mindlessly trigger the caching of "centroids"
#         in each work group, because we might not have enough shared memory for it.
#         In fact, the paper implements a finer mechanism: it caches the maximum
#         amount of centroids, and for those that can't fit in, caching is ignored
#         and results are directly written to global memory. This requires at least
#         having the information of how much shared memory is available for the current
#         gpu. It seems possible (since clinfo gives it, maybe see pyopencl ?) but is
#         left for a future TODO.

#         What we do instead to limit the amount of shared memory is to locally remap
#         the indexes of the centroids that are accessed by the current workgroup, so
#         that only the relevant centroids are loaded in cache, rather than the whole
#         array of centroids.

#         Shared memory footprint is decreased at the cost of storing two additional
#         arrays of size (n_clusters) per work group, and a few computations for the
#         remapping.
#         '''
#         # TODO: overall, explicit castings (does a/b works if a,b ints ?)

#         # TOOD: is it better to define as much as we can the variables as constants
#         # out of the function ? it seems to because it allows the compiler to
#         # "unroll the loops" ?
#         p = 128
#         n = x.shape[0]
#         dim = centroids.shape[0]
#         n_clusters = centroids.shape[0]
#         group_size = dpex.get_local_size(0)

#         group = dpex.get_group_id(0)

#         group_nb_previous_entries = group * p
#         group_starting_pt = math.floor(group_nb_previous_entries/dim) + 1
#         group_end_pt = math.floor((group_nb_previous_entries + p * group_size)/dim) + 1
#         group_nb_points_covered = group_end_pt - group_starting_pt + 1

#         group_nb_clusters = dpex.local.array(shape=1, dtype=uint32)
#         local_assignments_idx = dpex.local.array(shape=group_nb_points_covered, dtype=uint32)
#         # ???: define those two later when nb of unique centroids is known ?
#         local_cluster_to_global = dpex.local.array(shape=group_nb_points_covered, dtype=uint32)
#         global_cluster_to_local = dpex.local.array(shape=n_clusters, dtype=uint32)
#         local_cluster_count = dpex.local.array(shape=group_nb_points_covered, dtype=uint32)

#         thread = dpex.get_local_id(0)
#         thread_nb_previous_entries = group_nb_previous_entries + thread * p
#         thread_starting_pt = math.floor(thread_nb_previous_entries/dim) + 1
#         thread_end_pt = math.floor((thread_nb_previous_entries + p)/dim) + 1
#         thread_starting_dim = thread_nb_previous_entries % dim
#         thread_alloc_start = thread_starting_pt if ((thread == 0) or thread_starting_dim == 0) else (thread_starting_pt + 1)

#         for i in range(thread_alloc_start, thread_end_pt + 1):
#             cluster_idx = assignments_idx[i]
#             cluster_count = dpex.atomic.add(local_cluster_count, cluster_idx, 1)
#             if cluster_count == 0:
#                 local_cluster_idx = dpex.atomic.add(group_nb_clusters, 0, 1)
#                 local_cluster_to_global[local_cluster_idx] = cluster_idx
#                 global_cluster_to_local[cluster_idx] = local_cluster_idx
#             local_assignments_idx[i] = global_cluster_to_local[cluster_idx]

#         dpex.barrier()

#         group_nb_clusters = group_nb_clusters[0]
#         partial_cluster_count = dpex.local.array(shape=group_nb_clusters, dtype=uint32)
#         local_centroids = dpex.local.array(shape=(group_nb_clusters, dim), dtype=float32)
#         nb_centroid_entries_per_thread = math.ceil(float32(group_nb_clusters * dim) / group_size)
#         starting_entry = nb_centroid_entries_per_thread * thread

#         for i in range(starting_entry, starting_entry + nb_centroid_entries_per_thread):
#             row = i // dim
#             if row > group_nb_clusters:
#                 break
#             col = i % dim
#             local_centroids[row, col] = float32(0.)

#         dpex.barrier()

#         current_pt = thread_starting_pt
#         current_dim = thread_starting_dim
#         current_cluster_idx = local_assignments_idx[current_pt]
#         for pi in range(p):
#             if current_pt >= n:
#                 break
#             e = x[current_pt, current_dim]
#             dpex.atomic.add(local_centroids, (current_cluster_idx, current_dim), e)
#             if current_dim == 0:
#                 dpex.atomic.add(partial_cluster_count, current_cluster_idx, 1)
#             current_dim = (current_dim + 1) % dim
#             if current_dim == 0:
#                 current_pt += 1
#                 current_cluster_idx = local_assignments_idx[current_pt]

#         dpex.barrier()
# @dpex.kernel
# def update_means_step_fine(x, assignments_dist, assignments_idx, centroids):
#     return

# def get_update_means_step_shm_kernel():


#     @dpex.kernel
#     def update_means_step_shm(x, assignments_dist, assignments_idx, centroid_counts,
#                               centroids):
#         '''
#         NB: here we do not implement exactly what is given in the paper
#         At this point we don't want to mindlessly trigger the caching of "centroids"
#         in each work group, because we might not have enough shared memory for it.
#         In fact, the paper implements a finer mechanism: it caches the maximum
#         amount of centroids, and for those that can't fit in, caching is ignored
#         and results are directly written to global memory. This requires at least
#         having the information of how much shared memory is available for the current
#         gpu. It seems possible (since clinfo gives it, maybe see pyopencl ?) but is
#         left for a future TODO.

#         What we do instead to limit the amount of shared memory is to locally remap
#         the indexes of the centroids that are accessed by the current workgroup, so
#         that only the relevant centroids are loaded in cache, rather than the whole
#         array of centroids.

#         Shared memory footprint is decreased at the cost of storing two additional
#         arrays of size (n_clusters) per work group, and a few computations for the
#         remapping.
#         '''
#         # TODO: overall, explicit castings (does a/b works if a,b ints ?)

#         # TOOD: is it better to define as much as we can the variables as constants
#         # out of the function ? it seems to because it allows the compiler to
#         # "unroll the loops" ?
#         p = 128
#         n = x.shape[0]
#         dim = centroids.shape[0]
#         n_clusters = centroids.shape[0]
#         group_size = dpex.get_local_size(0)

#         group = dpex.get_group_id(0)

#         group_nb_previous_entries = group * p
#         group_starting_pt = math.floor(group_nb_previous_entries/dim) + 1
#         group_end_pt = math.floor((group_nb_previous_entries + p * group_size)/dim) + 1
#         group_nb_points_covered = group_end_pt - group_starting_pt + 1

#         group_nb_clusters = dpex.local.array(shape=1, dtype=uint32)
#         local_assignments_idx = dpex.local.array(shape=group_nb_points_covered, dtype=uint32)
#         # ???: define those two later when nb of unique centroids is known ?
#         local_cluster_to_global = dpex.local.array(shape=group_nb_points_covered, dtype=uint32)
#         global_cluster_to_local = dpex.local.array(shape=n_clusters, dtype=uint32)
#         local_cluster_count = dpex.local.array(shape=group_nb_points_covered, dtype=uint32)

#         thread = dpex.get_local_id(0)
#         thread_nb_previous_entries = group_nb_previous_entries + thread * p
#         thread_starting_pt = math.floor(thread_nb_previous_entries/dim) + 1
#         thread_end_pt = math.floor((thread_nb_previous_entries + p)/dim) + 1
#         thread_starting_dim = thread_nb_previous_entries % dim
#         thread_alloc_start = thread_starting_pt if ((thread == 0) or thread_starting_dim == 0) else (thread_starting_pt + 1)

#         for i in range(thread_alloc_start, thread_end_pt + 1):
#             cluster_idx = assignments_idx[i]
#             cluster_count = dpex.atomic.add(local_cluster_count, cluster_idx, 1)
#             if cluster_count == 0:
#                 local_cluster_idx = dpex.atomic.add(group_nb_clusters, 0, 1)
#                 local_cluster_to_global[local_cluster_idx] = cluster_idx
#                 global_cluster_to_local[cluster_idx] = local_cluster_idx
#             local_assignments_idx[i] = global_cluster_to_local[cluster_idx]

#         dpex.barrier()

#         group_nb_clusters = group_nb_clusters[0]
#         partial_cluster_count = dpex.local.array(shape=group_nb_clusters, dtype=uint32)
#         local_centroids = dpex.local.array(shape=(group_nb_clusters, dim), dtype=float32)
#         nb_centroid_entries_per_thread = math.ceil(float32(group_nb_clusters * dim) / group_size)
#         starting_entry = nb_centroid_entries_per_thread * thread

#         for i in range(starting_entry, starting_entry + nb_centroid_entries_per_thread):
#             row = i // dim
#             if row > group_nb_clusters:
#                 break
#             col = i % dim
#             local_centroids[row, col] = float32(0.)

#         dpex.barrier()

#         current_pt = thread_starting_pt
#         current_dim = thread_starting_dim
#         current_cluster_idx = local_assignments_idx[current_pt]
#         for pi in range(p):
#             if current_pt >= n:
#                 break
#             e = x[current_pt, current_dim]
#             dpex.atomic.add(local_centroids, (current_cluster_idx, current_dim), e)
#             if current_dim == 0:
#                 dpex.atomic.add(partial_cluster_count, current_cluster_idx, 1)
#             current_dim = (current_dim + 1) % dim
#             if current_dim == 0:
#                 current_pt += 1
#                 current_cluster_idx = local_assignments_idx[current_pt]

#         dpex.barrier()

#         current_row = starting_entry // dim
#         current_cluster_idx = local_cluster_to_global[row]
#         for i in range(starting_entry, starting_entry + nb_centroid_entries_per_thread):
#             row = i // dim
#             if row > group_nb_clusters:
#                 break
#             if row != current_row:
#                 current_cluster_idx = local_cluster_to_global[row]
#                 current_row = row
#             col = i % dim
#             if col == 0:
#                 dpex.atomic.add(centroid_counts, current_cluster_idx, partial_cluster_count[row])
#             dpex.atomic.add(centroids, (current_cluster_idx, dim), local_centroids[row, col])


#         current_row = starting_entry // dim
#         current_cluster_idx = local_cluster_to_global[row]
#         for i in range(starting_entry, starting_entry + nb_centroid_entries_per_thread):
#             row = i // dim
#             if row > group_nb_clusters:
#                 break
#             if row != current_row:
#                 current_cluster_idx = local_cluster_to_global[row]
#                 current_row = row
#             col = i % dim
#             if col == 0:
#                 dpex.atomic.add(centroid_counts, current_cluster_idx, partial_cluster_count[row])
#             dpex.atomic.add(centroids, (current_cluster_idx, dim), local_centroids[row, col])


def kmeans_fit_numba_dpex(x_train,
                          y_train,
                          n_clusters=128,
                          init="random",
                          n_init=10,
                          max_iter=300,
                          tol=1e-4,
                          random_state=123,
                          algorithm="full",
                          copy_x=False):

    #!!!: x_train, y_train must be in row_major order (C-style)
    # and should be float32

    # TODO: write a kernel for this ?
    # x_train[:] = x_train - x_train.mean(axis=0)

    # TODO: use SYCL array rather than numpy arrays
    # enables finer control of data location
    dim = x_train.shape[1]
    n = x_train.shape[0]

    # TODO: write a kernel for initialization on GPU
    # implement random AND ++
    rng = default_rng(random_state)
    centroids = np.array(rng.choice(x_train, n_clusters, replace=False), dtype=np.float32).copy()

    # TODO: the conditions used here to select the best kernel are the conditions
    # suggested in the paper, but it seems that the conditions are slightly
    # different when reading the implementation of the authors. Find the
    # differences between the two and use the appropriate best conditions.
    if n_clusters <= 32 or dim >= 64:
        h = 16
        r = 32
        thread_group_size = 256
        n_cluster_groups, assignment_step = get_assignment_step_fixed_kernel(
            h, r, thread_group_size, n, dim, n_clusters)
    else:
        rq = 4
        rp = 4
        p = 32
        q = 4
        n_cluster_groups, assignment_step = get_assignment_step_regs_kernel(
            rp, rq, p, q, n, dim, n_clusters)

    coarse_assignments_dist = np.empty((n, n_cluster_groups), dtype=np.float32)
    coarse_assignments_idx = np.empty((n, n_cluster_groups), dtype=np.uint32)

    assignment_step(x_train, centroids, coarse_assignments_dist, coarse_assignments_idx)

    # fine_assignments_thread_group_size = 256
    # fine_assignments_n_threads = np.ceil(n/256) * 256

    # if n_cluster_groups > 1:
    #     fine_assignments_dist = np.empty(n, dtype=np.float32)
    #     fine_assignments_idx = np.empty(n, dtype=np.uint32)

    # while True:
    #     import ipdb; ipdb.set_trace()
    #     assignment_step(x_train, centroids, coarse_assignments_dist, coarse_assignments_idx)
    #     if n_cluster_groups > 1:  # still TODO
    #         fine_assignment_step[
    #             fine_assignments_n_threads, fine_assignments_thread_group_size](
    #                 coarse_assignments_dist, coarse_assignments_idx,
    #                 fine_assignments_dist, fine_assignments_idx)
    #     else:
    #         fine_assignments_dist = coarse_assignments_dist[:]
    #         fine_assignments_idx = coarse_assignments_idx[:]

    # update_means = (update_means_step_fine if dim >= 8192 else update_means_step_shm)


if __name__ == "__main__":
    x, y = fetch_openml(name='spoken-arabic-digit', return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)

    scaler_x = MinMaxScaler()
    scaler_x.fit(x_train)
    x_train = scaler_x.transform(x_train)
    x_test = scaler_x.transform(x_test)

    # !!!: care that the arrays are contiguous in memory. Else segfaults happen.
    x_train = np.array(x_train, dtype=np.float32).copy()

    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()
    with dpctl.device_context(device):
        kmeans_fit_numba_dpex(x_train, y_train, n_clusters=127, init="random",
                              n_init=10, max_iter=300, tol=1e-4, random_state=123,
                              algorithm="full", copy_x=False)
        print("Done...")

    # In the thereafter commented block: some code to try sklearn vanilla kmeans
    # and sklearn intelex kmeans
    # patch_sklearn()

    # from sklearnex import patch_sklearn, unpatch_sklearn
    # from sklearn.cluster import KMeans

    # params = {
    #     "n_clusters": 128,
    #     "random_state": 123,
    #     "copy_x": False,
    # }
    # start = timer()
    # model = KMeans(**params).fit(x_train, y_train)
    # train_patched = timer() - start
    # f"Intel® extension for Scikit-learn time: {train_patched:.2f} s"
    # # 516 sc

    # from sklearnex import unpatch_sklearn
    # unpatch_sklearn()

    # from sklearn.cluster import KMeans

    # start = timer()
    # model = KMeans(**params).fit(x_train, y_train)
    # train_unpatched = timer() - start
    # f"Original Scikit-learn time: {train_unpatched:.2f} s"
    # # 1106.08

    # # ------------
    # #
    # # NB: "algorithm" when using sklearn DOES NOT WORK it's always defaultdense
    # # or lloyed sparse
