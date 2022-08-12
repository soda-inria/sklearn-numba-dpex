import numba_dpex as dpex
from numba import float32, int32

import dpctl
import numpy as np
import math


def get_initialize_to_zeros_kernel_1_int32(n, thread_group_size):

    n_threads = math.ceil(n / thread_group_size) * thread_group_size

    @dpex.kernel
    def initialize_to_zeros(x):
        thread_i = dpex.get_global_id(0)

        if thread_i >= n:
            return

        x[thread_i] = int32(0)

    return initialize_to_zeros[n_threads, thread_group_size]


def get_initialize_to_zeros_kernel_1_float32(n, thread_group_size):

    n_threads = math.ceil(n / thread_group_size) * thread_group_size

    @dpex.kernel
    def initialize_to_zeros(x):
        thread_i = dpex.get_global_id(0)

        if thread_i >= n:
            return

        x[thread_i] = float32(0)

    return initialize_to_zeros[n_threads, thread_group_size]


def get_initialize_to_zeros_kernel_2_float32(n, dim, thread_group_size):

    nb_items = n * dim
    n_threads = math.ceil(nb_items / thread_group_size) * thread_group_size

    @dpex.kernel
    def initialize_to_zeros(x):
        thread_i = dpex.get_global_id(0)

        if thread_i >= nb_items:
            return

        row = thread_i // n
        col = thread_i % n
        x[row, col] = float32(0.0)

    return initialize_to_zeros[n_threads, thread_group_size]


def get_copyto_kernel(n, dim, thread_group_size):
    n_threads = math.ceil(n / thread_group_size) * thread_group_size

    @dpex.kernel
    def copyto_kernel(X, Y):
        x = dpex.get_global_id(0)

        if x >= n:
            return

        for d in range(dim):
            Y[d, x] = X[d, x]

    return copyto_kernel[n_threads, thread_group_size]


def get_broadcast_division_kernel(n, dim, thread_group_size):
    n_threads = math.ceil(n / thread_group_size) * thread_group_size

    @dpex.kernel
    def broadcast_division_kernel(X, v):
        x = dpex.get_global_id(0)

        if x >= n:
            return

        divisor = float32(v[x])

        for d in range(dim):
            X[d, x] = X[d, x] / divisor

    return broadcast_division_kernel[n_threads, thread_group_size]


def get_center_shift_kernel(n, dim, thread_group_size):
    n_threads = math.ceil(n / thread_group_size) * thread_group_size

    @dpex.kernel
    def center_shift_kernel(previous_center, center, center_shift):
        x = dpex.get_global_id(0)

        if x >= n:
            return

        tmp = float32(0.0)

        for d in range(dim):
            center_diff = previous_center[d, x] - center[d, x]
            tmp += center_diff * center_diff

        center_shift[x] = tmp

    return center_shift_kernel[n_threads, thread_group_size]


def get_half_l2_norm_kernel_dim0(n, dim, thread_group_size):
    n_threads = math.ceil(n / thread_group_size) * thread_group_size

    @dpex.kernel
    def half_l2_norm_kernel_dim0(X, result):
        x = dpex.get_global_id(0)

        if x >= n:
            return

        l2_norm = float32(0.0)

        for d in range(dim):
            item = X[d, x]
            l2_norm += item * item

        result[x] = l2_norm / 2

    return half_l2_norm_kernel_dim0[n_threads, thread_group_size]


def get_sum_reduction_kernel(n, thread_group_size):
    local_nb_iterations = math.floor(math.log2(thread_group_size))

    @dpex.kernel
    def sum_reduction_kernel(v, w):
        group = dpex.get_group_id(0)
        thread = dpex.get_local_id(0)
        first_thread = thread == 0

        n = v.shape[0]

        shm = dpex.local.array(thread_group_size, dtype=float32)

        group_start = group * thread_group_size * 2
        thread_operand_1 = group_start + thread
        thread_operand_2 = group_start + thread_group_size + thread

        if thread_operand_1 >= n:
            shm[thread] = float32(0.0)
        elif thread_operand_2 >= n:
            shm[thread] = v[thread_operand_1]
        else:
            shm[thread] = v[thread_operand_1] + v[thread_operand_2]

        dpex.barrier()
        current_size = thread_group_size
        for i in range(local_nb_iterations):
            current_size = current_size // 2
            if thread < current_size:
                shm[thread] += shm[thread + current_size]

            dpex.barrier()

        if first_thread:
            w[group] = shm[0]

    _steps_data = []
    n_groups = n
    while n_groups > 1:
        n_groups = math.ceil(n_groups / (2 * thread_group_size))
        n_threads = n_groups * thread_group_size
        _steps_data.append(
            (dpctl.tensor.empty(n_groups, dtype=np.float32), n_threads)
        )

    def sum_reduction(v):
        for w, n_threads in _steps_data:
            sum_reduction_kernel[n_threads, thread_group_size](v, w)
            v = w
        return v

    return sum_reduction


def get_fused_kernel_fixed_window(
    n,
    dim,
    n_clusters,
    warp_size,
    window_length_multiple,
    cluster_window_per_thread_group,
    number_of_load_iter,
):
    r = warp_size * window_length_multiple
    thread_group_size = r * cluster_window_per_thread_group
    h = number_of_load_iter * cluster_window_per_thread_group

    n_cluster_groups = math.ceil(n_clusters / r)
    n_threads = (math.ceil(n / thread_group_size)) * (thread_group_size)
    n_dim_windows = math.ceil(dim / h)
    window_shm_shape = (h, r)

    inf = float32(math.inf)
    f32zero = float32(0.0)

    @dpex.kernel
    def fused_kernel_fixed_window(
        X,
        current_centroids,
        centroids_half_l2_norm,
        new_centroids,
        centroid_counts,
        inertia,
    ):
        global_thread = dpex.get_global_id(0)
        local_thread = dpex.get_local_id(0)

        window_shm = dpex.local.array(shape=window_shm_shape, dtype=float32)
        local_centroids_half_l2_norm = dpex.local.array(shape=r, dtype=float32)
        partial_scores = dpex.private.array(shape=r, dtype=float32)

        first_centroid_global_idx = 0

        min_idx = 0
        min_score = inf

        window_col = local_thread % r
        window_row_offset = local_thread // r

        for _0 in range(n_cluster_groups):

            for i in range(r):
                partial_scores[i] = f32zero

            half_l2_norm_idx = first_centroid_global_idx + local_thread
            if local_thread < r:
                if half_l2_norm_idx < n_clusters:
                    l2norm = centroids_half_l2_norm[half_l2_norm_idx]
                else:
                    l2norm = inf
                local_centroids_half_l2_norm[local_thread] = l2norm

            global_window_col = first_centroid_global_idx + window_col

            first_dim_global_idx = 0

            for _1 in range(n_dim_windows):
                load_first_dim_local_idx = 0
                for load_iter in range(number_of_load_iter):
                    window_row = load_first_dim_local_idx + window_row_offset
                    global_window_row = first_dim_global_idx + window_row

                    if (global_window_row < dim) and (
                        global_window_col < n_clusters
                    ):
                        item = current_centroids[
                            global_window_row, global_window_col
                        ]
                    else:
                        item = f32zero

                    window_shm[window_row, window_col] = item

                    load_first_dim_local_idx += cluster_window_per_thread_group

                dpex.barrier()

                for d in range(h):
                    current_dim_global_idx = d + first_dim_global_idx
                    if (current_dim_global_idx < dim) and (global_thread < n):
                        # performance for the line thereafter relies on L1 cache
                        X_feature = X[current_dim_global_idx, global_thread]
                    else:
                        X_feature = f32zero
                    for i in range(r):
                        centroid_feature = window_shm[d, i]
                        partial_scores[i] += centroid_feature * X_feature

                dpex.barrier()

                first_dim_global_idx += h

            for i in range(r):
                current_score = (
                    local_centroids_half_l2_norm[i] - partial_scores[i]
                )
                if current_score < min_score:
                    min_score = current_score
                    min_idx = first_centroid_global_idx + i

            dpex.barrier()

            first_centroid_global_idx += r

        if global_thread >= n:
            return

        inertia[global_thread] = min_score

        # NB: likely to have conflicts there
        dpex.atomic.add(centroid_counts, min_idx, 1)
        for d in range(dim):
            dpex.atomic.add(new_centroids, (d, min_idx), X[d, global_thread])

    return fused_kernel_fixed_window[n_threads, thread_group_size]


def get_assignment_kernel_fixed_window(
    n,
    dim,
    n_clusters,
    warp_size,
    window_length_multiple,
    cluster_window_per_thread_group,
    number_of_load_iter,
):
    r = warp_size * window_length_multiple
    thread_group_size = r * cluster_window_per_thread_group
    h = number_of_load_iter * cluster_window_per_thread_group

    n_cluster_groups = math.ceil(n_clusters / r)
    n_threads = (math.ceil(n / thread_group_size)) * (thread_group_size)
    n_dim_windows = math.ceil(dim / h)
    window_shm_shape = (h, r)

    inf = float32(math.inf)
    f32zero = float32(0.0)
    f32two = float32(2)

    @dpex.kernel
    def assigment_kernel_fixed_window(
        X,
        current_centroids,
        centroids_half_l2_norm,
        inertia,
        assignments_idx,
    ):
        global_thread = dpex.get_global_id(0)
        local_thread = dpex.get_local_id(0)

        window_shm = dpex.local.array(shape=window_shm_shape, dtype=float32)
        local_centroids_half_l2_norm = dpex.local.array(shape=r, dtype=float32)
        partial_scores = dpex.private.array(shape=r, dtype=float32)

        first_centroid_global_idx = 0

        min_idx = 0
        min_score = inf

        X_l2_norm = f32zero

        window_col = local_thread % r
        window_row_offset = local_thread // r

        for _0 in range(n_cluster_groups):

            for i in range(r):
                partial_scores[i] = f32zero

            half_l2_norm_idx = first_centroid_global_idx + local_thread
            if local_thread < r:
                if half_l2_norm_idx < n_clusters:
                    l2norm = centroids_half_l2_norm[half_l2_norm_idx]
                else:
                    l2norm = inf
                local_centroids_half_l2_norm[local_thread] = l2norm

            global_window_col = first_centroid_global_idx + window_col

            first_dim_global_idx = 0

            for _1 in range(n_dim_windows):
                load_first_dim_local_idx = 0
                for load_iter in range(number_of_load_iter):
                    window_row = load_first_dim_local_idx + window_row_offset
                    global_window_row = first_dim_global_idx + window_row

                    if (global_window_row < dim) and (
                        global_window_col < n_clusters
                    ):
                        item = current_centroids[
                            global_window_row, global_window_col
                        ]
                    else:
                        item = f32zero

                    window_shm[window_row, window_col] = item

                    load_first_dim_local_idx += cluster_window_per_thread_group

                dpex.barrier()

                for d in range(h):
                    current_dim_global_idx = d + first_dim_global_idx
                    if (current_dim_global_idx < dim) and (global_thread < n):
                        # performance for the line thereafter relies on L1 cache
                        X_feature = X[current_dim_global_idx, global_thread]
                    else:
                        X_feature = f32zero
                    X_l2_norm += X_feature * X_feature
                    for i in range(r):
                        centroid_feature = window_shm[d, i]
                        partial_scores[i] += centroid_feature * X_feature

                dpex.barrier()

                first_dim_global_idx += h

            for i in range(r):
                current_score = (
                    local_centroids_half_l2_norm[i] - partial_scores[i]
                )
                if current_score < min_score:
                    min_score = current_score
                    min_idx = first_centroid_global_idx + i

            dpex.barrier()

            first_centroid_global_idx += r

        if global_thread >= n:
            return

        assignments_idx[global_thread] = min_idx
        inertia[global_thread] = X_l2_norm + (f32two * min_score)

    return assigment_kernel_fixed_window[n_threads, thread_group_size]


JIT_CACHE = []


def kmeans_run_numba_dpex(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    x_squared_norms=None,
    tol=1e-4,
    n_threads=1,
):
    global JIT_CACHE
    # NB: all parameters given when instanciating the kernels can impact performance,
    # could be benchmarked, and can be set with more generality regarding the device
    # specs using tools to fetch device information (like pyopencl)

    # warp_size could be retrieved with e.g pyopencl, maybe dpctl ?
    # in doubt, use a high number that could be a multiple of the real warp
    # size (which is usually a power of 2) rather than a low number
    WARP_SIZE = 64
    DEFAULT_THREAD_GROUP_SIZE = 4 * WARP_SIZE

    dim = X.shape[1]
    n = X.shape[0]
    n_clusters = centers_init.shape[0]
    X = dpctl.tensor.from_numpy(np.ascontiguousarray(X).T)
    centroids = dpctl.tensor.from_numpy(np.ascontiguousarray(centers_init).T)
    centroids_copy_array = dpctl.tensor.empty_like(centroids)
    best_centroids = dpctl.tensor.empty_like(centroids)
    centroids_half_l2_norm = dpctl.tensor.empty(n_clusters, dtype=np.float32)
    centroid_counts = dpctl.tensor.empty(n_clusters, dtype=np.int32)
    center_shifts = dpctl.tensor.empty(n_clusters, dtype=np.float32)
    inertia = dpctl.tensor.empty(n, dtype=np.float32)
    assignments_idx = dpctl.tensor.empty(n, dtype=np.uint32)

    if JIT_CACHE:
        (
            reset_centroid_counts,
            reset_inertia,
            reset_centroids,
            copyto,
            broadcast_division,
            get_center_shifts,
            half_l2_norm,
            reduce_inertia,
            reduce_center_shifts,
            fused_kernel_fixed_window,
            assignment_kernel_fixed_window,
        ) = JIT_CACHE
    else:
        reset_centroid_counts = get_initialize_to_zeros_kernel_1_int32(
            n=n_clusters, thread_group_size=DEFAULT_THREAD_GROUP_SIZE
        )
        reset_inertia = get_initialize_to_zeros_kernel_1_float32(
            n=n, thread_group_size=DEFAULT_THREAD_GROUP_SIZE
        )
        reset_centroids = get_initialize_to_zeros_kernel_2_float32(
            n=n_clusters, dim=dim, thread_group_size=DEFAULT_THREAD_GROUP_SIZE
        )

        copyto = get_copyto_kernel(
            n_clusters, dim, thread_group_size=DEFAULT_THREAD_GROUP_SIZE
        )

        broadcast_division = get_broadcast_division_kernel(
            n=n_clusters, dim=dim, thread_group_size=DEFAULT_THREAD_GROUP_SIZE
        )

        get_center_shifts = get_center_shift_kernel(
            n_clusters, dim, thread_group_size=DEFAULT_THREAD_GROUP_SIZE
        )

        half_l2_norm = get_half_l2_norm_kernel_dim0(
            n_clusters, dim, thread_group_size=DEFAULT_THREAD_GROUP_SIZE
        )

        # NB: assumes thread_group_size is a power of two
        reduce_inertia = get_sum_reduction_kernel(
            n, thread_group_size=DEFAULT_THREAD_GROUP_SIZE
        )

        reduce_center_shifts = get_sum_reduction_kernel(
            n_clusters, thread_group_size=DEFAULT_THREAD_GROUP_SIZE
        )

        fused_kernel_fixed_window = get_fused_kernel_fixed_window(
            n,
            dim,
            n_clusters,
            warp_size=WARP_SIZE,
            # to benchmark
            # biggest possible values supported by shared memory and private registry ?
            window_length_multiple=1,
            # to benchmark
            # biggest possible value supported by shared memory and by the device
            cluster_window_per_thread_group=4,
            number_of_load_iter=4,
        )

        assignment_kernel_fixed_window = get_assignment_kernel_fixed_window(
            n,
            dim,
            n_clusters,
            warp_size=WARP_SIZE,
            window_length_multiple=1,
            cluster_window_per_thread_group=4,
            number_of_load_iter=4,
        )

        JIT_CACHE = (
            reset_centroid_counts,
            reset_inertia,
            reset_centroids,
            copyto,
            broadcast_division,
            get_center_shifts,
            half_l2_norm,
            reduce_inertia,
            reduce_center_shifts,
            fused_kernel_fixed_window,
            assignment_kernel_fixed_window,
        )

    n_iteration = 0
    center_shifts_sum = np.inf
    best_inertia = np.inf
    while (n_iteration < max_iter) and (center_shifts_sum >= tol):
        half_l2_norm(centroids, centroids_half_l2_norm)

        reset_centroids(centroids_copy_array)
        reset_centroid_counts(centroid_counts)
        reset_inertia(inertia)

        fused_kernel_fixed_window(
            X,
            centroids,
            centroids_half_l2_norm,
            centroids_copy_array,
            centroid_counts,
            inertia,
        )

        broadcast_division(centroids_copy_array, centroid_counts)

        get_center_shifts(centroids, centroids_copy_array, center_shifts)

        center_shifts_sum = dpctl.tensor.asnumpy(
            reduce_center_shifts(center_shifts)
        )[0]

        inertia_sum = dpctl.tensor.asnumpy(reduce_inertia(inertia))[0]

        if inertia_sum < best_inertia:
            best_inertia = inertia_sum
            copyto(centroids, best_centroids)

        centroids, centroids_copy_array = centroids_copy_array, centroids

        n_iteration += 1

    half_l2_norm(best_centroids, centroids_half_l2_norm)
    reset_inertia(inertia)
    assignment_kernel_fixed_window(
        X,
        best_centroids,
        centroids_half_l2_norm,
        inertia,
        assignments_idx,
    )

    inertia_sum = dpctl.tensor.asnumpy(reduce_inertia(inertia))[0]

    return (
        dpctl.tensor.asnumpy(assignments_idx),
        inertia_sum,
        np.ascontiguousarray(dpctl.tensor.asnumpy(best_centroids.T)),
        n_iteration,
    )


from sklearn.cluster import KMeans
from time import perf_counter
import sklearn.cluster._kmeans
import inspect

VANILLA_SKLEARN_LLOYD = sklearn.cluster._kmeans._kmeans_single_lloyd


class KMeansTimeit:
    _VANILLA_SKLEARN_LLOYD_SIGNATURE = inspect.signature(VANILLA_SKLEARN_LLOYD)

    def __init__(self, data_initialization_fn):
        (
            self.X,
            self.sample_weight,
            self.centers_init,
        ) = data_initialization_fn()

    def timeit(self, kmeans_fn, name, max_iter, tol):
        self._check_kmeans_fn_signature(kmeans_fn)
        n_clusters = self.centers_init.shape[0]
        try:
            sklearn.cluster._kmeans._kmeans_single_lloyd = kmeans_fn
            # random_state is set but since we don't use kmeans++ it has no impact
            # on the outcome
            estimator = KMeans(
                n_clusters=n_clusters,
                init=self.centers_init,
                max_iter=max_iter,
                tol=tol,
                random_state=42,
                copy_x=True,
                algorithm="lloyd",
            )

            print(
                f"Running {name} with parameters max_iter={max_iter} tol={tol} ..."
            )
            t0 = perf_counter()
            estimator.fit(self.X, sample_weight=self.sample_weight)
            t1 = perf_counter()
            print(f"Running {name} ... done in {t1 - t0}")

        finally:
            sklearn.cluster._kmeans._kmeans_single_lloyd = (
                VANILLA_SKLEARN_LLOYD
            )

    def _check_kmeans_fn_signature(self, kmeans_fn):
        fn_signature = inspect.signature(kmeans_fn)
        if fn_signature != self._VANILLA_SKLEARN_LLOYD_SIGNATURE:
            raise ValueError(
                f"The signature of the submitted kmeans_fn is expected to be {self._VANILLA_SKLEARN_LLOYD_SIGNATURE}, but got {fn_signature}"
            )


if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster._kmeans import row_norms
    from numpy.random import default_rng
    import numpy as np

    random_state = 123
    n_clusters = 787
    max_iter = 20
    tol = 0

    def benchmark_data_initialization(
        random_state=random_state, n_clusters=n_clusters
    ):
        X, _ = fetch_openml(name="spoken-arabic-digit", return_X_y=True)
        X = X.astype(np.float32)
        scaler_x = MinMaxScaler()
        scaler_x.fit(X)
        X = scaler_x.transform(X)
        X = np.hstack([X for _ in range(20)])
        rng = default_rng(random_state)
        init = np.array(
            rng.choice(X, n_clusters, replace=False), dtype=np.float32
        )
        return X, None, init

    def vanilla_sklearn(
        X,
        sample_weight,
        centers_init,
        max_iter=300,
        verbose=False,
        x_squared_norms=None,
        tol=1e-4,
        n_threads=1,
    ):
        x_squared_norms = row_norms(X, squared=True)
        res = VANILLA_SKLEARN_LLOYD(
            X,
            sample_weight,
            centers_init,
            max_iter,
            verbose=False,
            x_squared_norms=x_squared_norms,
            tol=tol,
            n_threads=n_threads,
        )

        return res

    kmeans_timer = KMeansTimeit(benchmark_data_initialization)

    kmeans_timer.timeit(
        vanilla_sklearn,
        name="Sklearn vanilla lloyd",
        max_iter=max_iter,
        tol=tol,
    )

    kmeans_timer.timeit(
        kmeans_run_numba_dpex,
        name="Kmeans numba_dpex dry iter.",
        max_iter=2,
        tol=tol,
    )

    kmeans_timer.timeit(
        kmeans_run_numba_dpex,
        name="Kmeans numba_dpex",
        max_iter=max_iter,
        tol=tol,
    )
