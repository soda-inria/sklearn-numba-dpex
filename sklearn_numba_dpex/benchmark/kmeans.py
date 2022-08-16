from sklearn.cluster import KMeans
from time import perf_counter
import inspect
import sklearn


VANILLA_SKLEARN_LLOYD = sklearn.cluster._kmeans._kmeans_single_lloyd


# TODO: maybe this class could be abstracted to use the same strategy with other
# estimators ?
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
            centers_init = self.centers_init.copy()
            X = self.X.copy()
            sample_weight = (
                None
                if self.sample_weight is None
                else self.sample_weight.copy()
            )

            est_kwargs = dict(
                n_clusters=n_clusters,
                init=centers_init,
                max_iter=max_iter,
                tol=tol,
                # random_state is set but since we don't use kmeans++ or random
                # init this parameter shouldn't have any impact on the outputs
                random_state=42,
                copy_x=True,
                algorithm="lloyd",
            )

            sklearn.cluster._kmeans._kmeans_single_lloyd = kmeans_fn
            estimator = KMeans(**est_kwargs)

            print(
                f"Running {name} with parameters max_iter={max_iter} tol={tol} ..."
            )

            # dry run, to be fair for JIT implementations
            KMeans(**est_kwargs).set_params(max_iter=2).fit(
                X, sample_weight=sample_weight
            )

            t0 = perf_counter()
            estimator.fit(X, sample_weight=sample_weight)
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
    # TODO: also checks that the results are close
    # TODO: ensure that effective n_iter is equal for all runs

    from sklearn_numba_dpex.kmeans.sklearn import kmeans as sklearn_kmeans
    from sklearn_numba_dpex.kmeans.daal4py import kmeans as daal4py_kmeans

    from sklearn_numba_dpex.kmeans.dpex import (
        kmeans_fused_cpu as dpex_kmeans_fused_cpu,
        kmeans_fused_gpu as dpex_kmeans_fused_gpu,
    )
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import MinMaxScaler
    from sklearnex import config_context
    from numpy.random import default_rng
    import numpy as np

    random_state = 123
    n_clusters = 127
    max_iter = 100
    tol = 0

    def benchmark_data_initialization(
        random_state=random_state, n_clusters=n_clusters
    ):
        X, _ = fetch_openml(name="spoken-arabic-digit", return_X_y=True)
        X = X.astype(np.float32)
        scaler_x = MinMaxScaler()
        scaler_x.fit(X)
        X = scaler_x.transform(X)
        X = np.vstack((X for _ in range(10)))
        rng = default_rng(random_state)
        init = np.array(
            rng.choice(X, n_clusters, replace=False), dtype=np.float32
        )
        return X, None, init

    kmeans_timer = KMeansTimeit(benchmark_data_initialization)

    kmeans_timer.timeit(
        sklearn_kmeans,
        name="Sklearn vanilla lloyd",
        max_iter=max_iter,
        tol=tol,
    )

    with config_context(target_offload="cpu"):
        kmeans_timer.timeit(
            daal4py_kmeans,
            name="daal4py CPU",
            max_iter=max_iter,
            tol=tol,
        )

    with config_context(target_offload="gpu"):
        kmeans_timer.timeit(
            daal4py_kmeans,
            name="daal4py GPU",
            max_iter=max_iter,
            tol=tol,
        )

    kmeans_timer.timeit(
        dpex_kmeans_fused_cpu,
        name="Kmeans numba_dpex CPU",
        max_iter=max_iter,
        tol=tol,
    )

    kmeans_timer.timeit(
        dpex_kmeans_fused_gpu,
        name="Kmeans numba_dpex GPU",
        max_iter=max_iter,
        tol=tol,
    )
