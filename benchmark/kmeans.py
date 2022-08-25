from sklearn.cluster import KMeans
from time import perf_counter
import inspect
from numpy.testing import assert_array_equal
import sklearn
from sklearn.utils._testing import assert_allclose
import numpy as np


VANILLA_SKLEARN_LLOYD = sklearn.cluster._kmeans._kmeans_single_lloyd


# TODO: maybe this class could be abstracted to use the same strategy with other
# estimators ?
class KMeansLloydTimeit:
    _VANILLA_SKLEARN_LLOYD_SIGNATURE = inspect.signature(VANILLA_SKLEARN_LLOYD)
    """A helper class that standardizes a test bench for different implementations of
    the Lloyd algorithm.

    Parameters
    ----------
    data_initialization_fn : function
        Is expected to return a tuple (X, sample_weight, centers_init) with shapes
        (n_samples, n_features), (n_samples) and (n_clusters, n_features) respectively.

    max_iter : int
        Number of iterations to run in each test.

    run_consistency_checks : bool
        If true, will check if all candidates on the test bench return the same
        results.
        
    skip_slow: bool
        If true, will skip the timeit calls that are marked as slow. Default to false.
    """

    def __init__(
        self,
        data_initialization_fn,
        max_iter,
        skip_slow=False,
        run_consistency_checks=True,
    ):
        (
            self.X,
            self.sample_weight,
            self.centers_init,
        ) = data_initialization_fn()
        self.max_iter = max_iter
        self.run_consistency_checks = run_consistency_checks
        self.results = None
        self.skip_slow = skip_slow

    def timeit(self, kmeans_single_lloyd_fn, name, slow=False):
        """
        Parameters
        ----------
        kmeans_single_lloyd_fn : function
            Is expected to have the same signature and the same interface than sklearn's
            private function _kmeans_single_lloyd

        name: str
            Name of the candidate to test

        slow: bool
            Set to true to skip the timeit when skip_slow is true. Default to false.
        """
        if slow and self.skip_slow:
            return

        max_iter = self.max_iter
        self._check_kmeans_single_lloyd_fn_signature(kmeans_single_lloyd_fn)
        n_clusters = self.centers_init.shape[0]
        try:
            centers_init = self.centers_init.copy()
            X = self.X.copy()
            sample_weight = (
                None if self.sample_weight is None else self.sample_weight.copy()
            )

            est_kwargs = dict(
                n_clusters=n_clusters,
                init=centers_init,
                n_init=1,
                max_iter=max_iter,
                tol=0,
                # random_state is set but since we don't use kmeans++ or random
                # init this parameter shouldn't have any impact on the outputs
                random_state=42,
                copy_x=True,
                algorithm="lloyd",
            )

            sklearn.cluster._kmeans._kmeans_single_lloyd = kmeans_single_lloyd_fn
            estimator = KMeans(**est_kwargs)

            print(f"Running {name} with parameters max_iter={max_iter} tol={tol} ...")

            # dry run, to be fair for JIT implementations
            KMeans(**est_kwargs).set_params(max_iter=1).fit(
                X, sample_weight=sample_weight
            )

            t0 = perf_counter()
            estimator.fit(X, sample_weight=sample_weight)
            t1 = perf_counter()

            self._check_same_fit(
                estimator, name, max_iter, assert_allclose=self.run_consistency_checks
            )
            print(f"Running {name} ... done in {t1 - t0}\n")

        finally:
            sklearn.cluster._kmeans._kmeans_single_lloyd = VANILLA_SKLEARN_LLOYD

    def _check_same_fit(self, estimator, name, max_iter, assert_allclose):
        runtime_error_message = (
            "It is expected for all iterators in the benchmark to run the same "
            "number of iterations and return the same results."
        )
        n_iter = estimator.n_iter_
        if n_iter != max_iter:
            raise RuntimeError(
                f"The estimator {name} only ran {n_iter} iterations instead of "
                f"max_iter={max_iter} iterations. " + runtime_error_message
            )

        if not assert_allclose:
            return

        if self.results is None:
            self.results = dict(
                labels=estimator.labels_,
                cluster_centers=estimator.cluster_centers_,
                inertia=estimator.inertia_,
            )
        else:
            assert_array_equal(
                self.results["labels"], estimator.labels_, err_msg=runtime_error_message
            )
            assert_allclose(
                self.results["cluster_centers"],
                estimator.cluster_centers_,
                err_msg=runtime_error_message,
            )
            assert_allclose(
                self.results["inertia"],
                estimator.inertia_,
                err_msg=runtime_error_message,
            )

    def _check_kmeans_single_lloyd_fn_signature(self, kmeans_single_lloyd_fn):
        fn_signature = inspect.signature(kmeans_single_lloyd_fn)
        if fn_signature != self._VANILLA_SKLEARN_LLOYD_SIGNATURE:
            raise ValueError(
                f"The signature of the submitted kmeans_single_lloyd_fn is expected to be "
                f"{self._VANILLA_SKLEARN_LLOYD_SIGNATURE}, but got {fn_signature} ."
            )


if __name__ == "__main__":
    from ext_helpers.daal4py import kmeans as daal4py_kmeans

    from sklearn_numba_dpex.kmeans.drivers import LLoydKMeansDriver
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import MinMaxScaler
    from sklearnex import config_context
    from numpy.random import default_rng

    # TODO: expose CLI args.

    random_state = 123
    n_clusters = 127
    max_iter = 100
    tol = 0
    skip_slow = True
    dtype = np.float32
    # NB: it seems that currently the estimators in the benchmark always return
    # close results but with significant differences for a few elements.
    run_consistency_checks = False

    def benchmark_data_initialization(random_state=random_state, n_clusters=n_clusters):
        X, _ = fetch_openml(name="spoken-arabic-digit", return_X_y=True)
        X = X.astype(dtype)
        scaler_x = MinMaxScaler()
        scaler_x.fit(X)
        X = scaler_x.transform(X)
        # Change dataset dimensions
        # X = np.hstack([X for _ in range(5)])
        # X = np.vstack([X for _ in range(20)])
        rng = default_rng(random_state)
        init = np.array(rng.choice(X, n_clusters, replace=False), dtype=np.float32)
        return X, None, init

    kmeans_timer = KMeansLloydTimeit(
        benchmark_data_initialization, max_iter, skip_slow, run_consistency_checks
    )

    kmeans_timer.timeit(
        VANILLA_SKLEARN_LLOYD,
        name="Sklearn vanilla lloyd",
    )

    with config_context(target_offload="cpu"):
        kmeans_timer.timeit(
            daal4py_kmeans,
            name="daal4py lloyd CPU",
        )

    with config_context(target_offload="gpu"):
        kmeans_timer.timeit(daal4py_kmeans, name="daal4py lloyd GPU", slow=True)

    for multiplier in [1, 2, 4, 8]:
        kmeans_timer.timeit(
            LLoydKMeansDriver(
                device="cpu",
                dtype=dtype,
                work_group_size_multiplier=multiplier,
            ),
            name=f"Kmeans numba_dpex lloyd CPU (work_group_size_multiplier={multiplier})",
            slow=True,
        )

        kmeans_timer.timeit(
            LLoydKMeansDriver(
                device="gpu", dtype=dtype, work_group_size_multiplier=multiplier
            ),
            name=f"Kmeans numba_dpex lloyd GPU (work_group_size_multiplier={multiplier})",
        )
