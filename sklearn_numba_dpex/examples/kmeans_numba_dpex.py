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
from numba import float32, uint32, int32

import dpctl
import numpy as np
import math


# NB: must have rq <= p and rp <= q
def get_assignment_step_regs_kernel(rp, rq, p, q, n, dim, n_clusters):

    q_centroids_shm_dim = (q, rq, dim)
    p_x_train_shm_dim = (p, rp, dim)
    assignments_shm_dim = (p, rp, q)
    partial_sum_private_dim = (rp, rq)

    n_cluster_groups = math.ceil(n_clusters/(q * rq))
    n_data_groups = math.ceil(n/(p * rp))
    n_threads = (n_data_groups * p, n_cluster_groups * q)
    thread_group_size = (p, q)

    inf = float32(math.inf)

    @dpex.kernel
    def assignment_step_regs(x_train, centroids, coarse_assignments_dist,
                             coarse_assignments_idx):

        p_q_group_j = dpex.get_group_id(1)

        p_q_thread_i = dpex.get_local_id(0)
        p_q_thread_j = dpex.get_local_id(1)

        thread_i = dpex.get_global_id(0)
        thread_j = dpex.get_global_id(1)

        first_x_train_global_idx = thread_i * rp
        first_centroid_global_idx = thread_j * rq

        q_centroids = dpex.local.array(shape=q_centroids_shm_dim, dtype=float32)
        p_x_train = dpex.local.array(shape=p_x_train_shm_dim, dtype=float32)
        private_partial_sum = dpex.private.array(shape=partial_sum_private_dim, dtype=float32)
        private_centroid_values = dpex.private.array(shape=rq, dtype=float32)
        local_assignments_idx = dpex.local.array(shape=assignments_shm_dim, dtype=uint32)
        local_assignments_dist = dpex.local.array(shape=assignments_shm_dim, dtype=float32)

        # TODO: factor in a dpex kernel function ? need to test limitations of
        # kernel functions before.
        # Can it enclose memory use ?
        # !!!: those copy are efficients (regarding coalesced reading in global
        # memory) only if dim > max(rp, rq), which is an okay assumption

        # we assume q >= rp

        # initialize shared memory for x_train
        if p_q_thread_j < rp:
            x_train_global_idx = first_x_train_global_idx + p_q_thread_j
            if x_train_global_idx < n:
                for d in range(dim):
                    p_x_train[p_q_thread_i, p_q_thread_j, d] = x_train[
                        x_train_global_idx, d]
            else:
                for d in range(dim):
                    p_x_train[p_q_thread_i, p_q_thread_j, d] = inf

        # we assume p >= rq

        # initialize shared memory for centroids
        if p_q_thread_i < rq:
            centroid_global_idx = first_centroid_global_idx + p_q_thread_i
            if centroid_global_idx < n_clusters:
                for d in range(dim):
                    q_centroids[p_q_thread_j, p_q_thread_i, d] = centroids[
                        centroid_global_idx, d]
            else:
                for d in range(dim):
                    q_centroids[p_q_thread_j, p_q_thread_i, d] = -inf

        dpex.barrier()

        # initialize partial sums
        for k_p in range(rp):
            for k_q in range(rq):
                private_partial_sum[k_p, k_q] = float32(0.)

        for d in range(dim):
            # initialize private values
            for k_q in range(rq):
                private_centroid_values[k_q] = q_centroids[p_q_thread_j, k_q, d]

            # compute partial sums
            for k_p in range(rp):
                x_train_feature = p_x_train[p_q_thread_i, k_p, d]
                for k_q in range(rq):
                    diff = x_train_feature - private_centroid_values[k_q]
                    private_partial_sum[k_p, k_q] += (diff * diff)

        # compute the min over the centroids
        for k_p in range(rp):
            current_best_dist = private_partial_sum[k_p, 0]
            current_best_idx = uint32(0)
            for k_q in range(1, rq):
                dist = private_partial_sum[k_p, k_q]
                smaller_dist = dist < current_best_dist
                current_best_dist = dist if smaller_dist else current_best_dist
                current_best_idx = uint32(k_q) if smaller_dist else current_best_idx
            local_assignments_idx[p_q_thread_i, k_p, p_q_thread_j] = current_best_idx + first_centroid_global_idx
            local_assignments_dist[p_q_thread_i, k_p, p_q_thread_j] = current_best_dist

        dpex.barrier()

        # we assume q >= rp

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

        coarse_assignments_dist[x_train_global_idx, p_q_group_j] = min_dist
        coarse_assignments_idx[x_train_global_idx, p_q_group_j] = min_idx

    return n_cluster_groups, assignment_step_regs[n_threads, thread_group_size]


def get_assignment_step_fixed_kernel(h, r, thread_group_size, n, dim, n_clusters):
    n_cluster_groups = math.ceil(n_clusters/r)
    n_threads = (math.ceil(n / thread_group_size)) * (thread_group_size) * n_cluster_groups
    n_windows = math.ceil(dim/h)
    window_shm_shape = (r, h)
    window_elem_nb = (h * r)
    nb_window_entry_per_thread = math.ceil((r * h) / thread_group_size)

    @dpex.kernel
    def assignment_step_fixed(x_train, centroids, coarse_assignments_dist,
                               coarse_assignments_idx):

        group = dpex.get_group_id(0)
        local_thread = dpex.get_local_id(0)

        x_train_global_idx = thread_group_size * (group // n_cluster_groups) + local_thread
        centroid_window = (group % n_cluster_groups)
        first_centroid_global_idx = centroid_window * r

        window = dpex.local.array(shape=window_shm_shape, dtype=float32)
        partial_sums = dpex.private.array(shape=r, dtype=float32)
        for i in range(r):
            partial_sums[i] = float32(0.)

        for window_k in range(n_windows):

            first_dim_global_idx = window_k * h
            first_flat_idx = local_thread * nb_window_entry_per_thread
            # load memory
            for k in range(nb_window_entry_per_thread):
                flat_idx = first_flat_idx + k
                if flat_idx >= window_elem_nb:
                    break
                row = flat_idx // h
                col = flat_idx % h
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
                partial_sums[k] += tmp

            # wait that everybody is done with reading shared memory before rewriting
            # it for the next window
            dpex.barrier()

        min_dist = partial_sums[0]
        min_idx = uint32(0)
        for k in range(1, r):
            if (first_centroid_global_idx + k) >= n_clusters:
                break
            current_dist = partial_sums[k]
            smaller_dist = current_dist < min_dist
            min_dist = current_dist if smaller_dist else min_dist
            min_idx = uint32(k) if smaller_dist else min_idx

        coarse_assignments_dist[x_train_global_idx, centroid_window] = min_dist
        coarse_assignments_idx[x_train_global_idx, centroid_window] = min_idx + first_centroid_global_idx

    return n_cluster_groups, assignment_step_fixed[n_threads, thread_group_size]


def get_fine_assignment_step_kernel(thread_group_size, n, n_cluster_groups):

    n_threads = math.ceil(n / thread_group_size) * thread_group_size

    @dpex.kernel
    def fine_assignment_step(coarse_assignments_dist, coarse_assignments_idx,
                              fine_assignments_dist, fine_assignments_idx):

        thread_i = dpex.get_global_id(0)

        if thread_i >= n:
            return

        fine_dist = coarse_assignments_dist[thread_i, 0]
        fine_idx = coarse_assignments_idx[thread_i, 0]
        for i in range(1, n_cluster_groups):
            current_dist = coarse_assignments_dist[thread_i, i]
            smaller_dist = current_dist < fine_dist
            fine_dist = current_dist if smaller_dist else fine_dist
            fine_idx = coarse_assignments_idx[thread_i, i] if smaller_dist else fine_idx
        fine_assignments_dist[thread_i] = fine_dist
        fine_assignments_idx[thread_i] = fine_idx

    return fine_assignment_step[n_threads, thread_group_size]


def get_initialize_to_zeros_kernel_1(thread_group_size, n):

    n_threads = math.ceil(n / thread_group_size) * thread_group_size

    @dpex.kernel
    def initialize_to_zeros(x):
        thread_i = dpex.get_global_id(0)

        if thread_i >= n:
            return

        x[thread_i] = int32(0.)

    return initialize_to_zeros[n_threads, thread_group_size]


def get_initialize_to_zeros_kernel_2(thread_group_size, n, dim):

    nb_items = n * dim
    n_threads = math.ceil(nb_items / thread_group_size) * thread_group_size

    @dpex.kernel
    def initialize_to_zeros(x):
        thread_i = dpex.get_global_id(0)

        if thread_i >= nb_items:
            return

        row = thread_i // dim
        col = thread_i % dim
        x[row, col] = float32(0.)

    return initialize_to_zeros[n_threads, thread_group_size]


def get_update_means_step_fine_kernel(thread_group_size, n, dim):

    nb_items = n * dim
    n_threads = math.ceil(nb_items / thread_group_size) * thread_group_size

    @dpex.kernel
    def update_means_step_fine(x, assignments_idx, centroid_counts, centroids):
        thread_i = dpex.get_global_id(0)

        if thread_i >= nb_items:
            return

        row = thread_i // dim
        col = thread_i % dim
        centroid = assignments_idx[row]
        dpex.atomic.add(centroids, (centroid, col), x[row, col])

        if col == 0:
            dpex.atomic.add(centroid_counts, centroid, 1)

    return update_means_step_fine[n_threads, thread_group_size]


def get_update_means_step_shm_kernel(p, thread_group_size, n, dim, n_clusters):
    nb_items = n * dim
    item_per_group = thread_group_size * p
    n_threads = math.ceil(nb_items / item_per_group) * thread_group_size
    max_nb_points_per_thread = (p // dim) + 1
    max_local_centroids_nb = min((item_per_group // dim) + 1, n_clusters)
    local_centroid_shm_shape = (max_local_centroids_nb, dim)

    n_clusters_per_thread = math.ceil(n_clusters / thread_group_size)
    nb_centroid_entries_per_thread = math.ceil((max_local_centroids_nb * dim) / thread_group_size)

    @dpex.kernel
    def update_means_step_shm(x, assignments_idx, centroid_counts, centroids):
        group = dpex.get_group_id(0)
        thread = dpex.get_local_id(0)

        group_nb_clusters = dpex.local.array(shape=1, dtype=int32)
        local_cluster_to_global = dpex.local.array(shape=max_local_centroids_nb, dtype=uint32)
        partial_cluster_count = dpex.local.array(shape=max_local_centroids_nb, dtype=int32)
        global_cluster_to_local = dpex.local.array(shape=n_clusters, dtype=uint32)
        local_cluster_count = dpex.local.array(shape=n_clusters, dtype=int32)
        local_centroids = dpex.local.array(shape=local_centroid_shm_shape, dtype=float32)

        group_nb_previous_entries = group * item_per_group
        thread_nb_previous_entries = group_nb_previous_entries + (thread * p)
        thread_starting_pt = (thread_nb_previous_entries // dim)
        thread_starting_dim = thread_nb_previous_entries % dim

        if thread == 0:
            group_nb_clusters[0] = int32(0)

        starting_local_centroid = n_clusters_per_thread * thread
        for i in range(n_clusters_per_thread):
            idx = starting_local_centroid + i
            if idx >= n_clusters:
                break
            local_cluster_count[idx] = int32(0)

        dpex.barrier()

        for i in range(max_nb_points_per_thread):
            global_idx = thread_starting_pt + i
            if global_idx >= n:
                break
            cluster_idx = assignments_idx[global_idx]
            cluster_count = dpex.atomic.add(local_cluster_count, cluster_idx, 1)
            if cluster_count == 0:
                local_cluster_idx = dpex.atomic.add(group_nb_clusters, 0, 1)
                local_cluster_to_global[local_cluster_idx] = cluster_idx
                global_cluster_to_local[cluster_idx] = local_cluster_idx

        dpex.barrier()

        starting_entry = nb_centroid_entries_per_thread * thread

        for i in range(nb_centroid_entries_per_thread):
            idx = i + starting_entry
            row = idx // dim
            if row >= max_local_centroids_nb:
                break
            col = idx % dim
            if col == 0:
                partial_cluster_count[row] = int32(0)
            local_centroids[row, col] = float32(0.)

        dpex.barrier()

        # ???: because we accept that threads can process p items divergence might happen.
        # Divergence can be fixed if instead we make blocks of p items
        # start and finish at first and last dimension respectively, and make p
        # a multiple of dim (and always ensure that p >= dim)
        current_pt = thread_starting_pt
        current_dim = thread_starting_dim
        current_local_pt = 0
        local_cluster_idx = int32(global_cluster_to_local[assignments_idx[current_pt]])
        for pi in range(p):
            if current_pt >= n:
                break

            dpex.atomic.add(
                local_centroids, (local_cluster_idx, current_dim),
                x[current_pt, current_dim])

            if current_dim == 0:
                dpex.atomic.add(partial_cluster_count, local_cluster_idx, 1)

            current_dim = (current_dim + 1) % dim
            if current_dim == 0:
                current_pt += 1
                current_local_pt += 1
                local_cluster_idx = int32(global_cluster_to_local[assignments_idx[current_pt]])

        dpex.barrier()

        group_nb_clusters_ = group_nb_clusters[0]
        current_row = -1
        for i in range(nb_centroid_entries_per_thread):
            idx = i + starting_entry
            row = idx // dim
            if row >= group_nb_clusters_:
                break
            if row != current_row:
                global_cluster_idx = int32(local_cluster_to_global[row])
                current_row = row
            col = idx % dim
            if col == 0:
                dpex.atomic.add(centroid_counts, global_cluster_idx, partial_cluster_count[row])
            dpex.atomic.add(centroids, (global_cluster_idx, int(col)), local_centroids[row, col])

    return update_means_step_shm[n_threads, thread_group_size]


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
    centroid_counts = dpctl.tensor.empty(n_clusters, dtype=np.int32)
    x_train = dpctl.tensor.from_numpy(x_train)
    centroids = dpctl.tensor.from_numpy(centroids)

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

    if n_cluster_groups > 1:
        thread_group_size = 256
        fine_assignment_step = get_fine_assignment_step_kernel(256, n, n_cluster_groups)

    if dim >= 8192:
        thread_group_size = 256
        update_means_step = get_update_means_step_fine_kernel(thread_group_size, n, dim)
    else:
        p = 128
        thread_group_size = 256
        update_means_step = get_update_means_step_shm_kernel(p, thread_group_size, n, dim, n_clusters)

    initialize_to_zeros_1 = get_initialize_to_zeros_kernel_1(thread_group_size=256, n=n_clusters)
    initialize_to_zeros_2 = get_initialize_to_zeros_kernel_2(thread_group_size=256, n=n_clusters, dim=dim)

    coarse_assignments_dist = dpctl.tensor.empty((n, n_cluster_groups), dtype=np.float32)
    coarse_assignments_idx = dpctl.tensor.empty((n, n_cluster_groups), dtype=np.uint32)
    assignment_step(x_train, centroids, coarse_assignments_dist, coarse_assignments_idx)

    if n_cluster_groups > 1:
        fine_assignments_dist = dpctl.tensor.empty(n, dtype=np.float32)
        fine_assignments_idx = dpctl.tensor.empty(n, dtype=np.uint32)
        fine_assignment_step(
                coarse_assignments_dist, coarse_assignments_idx,
                fine_assignments_dist, fine_assignments_idx)
    else:
        fine_assignments_dist = coarse_assignments_dist[:]
        fine_assignments_idx = coarse_assignments_idx[:]

    initialize_to_zeros_2(centroids)
    initialize_to_zeros_1(centroid_counts)

    update_means_step(x_train, fine_assignments_idx, centroid_counts, centroids)
    print("Finished one iteration.")


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
