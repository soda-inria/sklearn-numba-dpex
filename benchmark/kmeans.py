from time import perf_counter

import numpy as np
import sklearn
from numpy.testing import assert_array_equal
from sklearn.cluster import KMeans


# TODO: maybe this class could be abstracted to use the same strategy with other
# estimators ?
class KMeansLloydTimeit:
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

    def timeit(self, name, engine_provider=None, is_slow=False):
        """
        Parameters
        ----------
        name: str
            Name of the candidate to test

        engine_provider: str
            The name of the engine provider to use for computations.

        is_slow: bool
            Set to true to skip the timeit when skip_slow is true. Default to false.
        """
        if is_slow and self.skip_slow:
            return

        max_iter = self.max_iter
        n_clusters = self.centers_init.shape[0]
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

        estimator = KMeans(**est_kwargs)

        print(f"Running {name} with parameters max_iter={max_iter} tol={tol} ...")

        with sklearn.config_context(engine_provider=engine_provider):

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


if __name__ == "__main__":
    from numpy.random import default_rng
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import MinMaxScaler

    from sklearn_numba_dpex.testing.config import override_attr_context
    from sklearn_numba_dpex.kmeans.engine import KMeansEngine

    from ext_helpers.daal4py import daal4py_kmeans_single
    from sklearnex import config_context as sklearnex_config_context

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
        X, _ = fetch_openml(name="spoken-arabic-digit", return_X_y=True, parser="auto")
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
        name="Sklearn vanilla lloyd",
    )

    with override_attr_context(
        KMeansEngine,
        kmeans_single=daal4py_kmeans_single,
    ), sklearnex_config_context(target_offload="cpu"):
        kmeans_timer.timeit(
            name="daal4py lloyd CPU", engine_provider="sklearn_numba_dpex"
        )

    with override_attr_context(
        KMeansEngine,
        kmeans_single=daal4py_kmeans_single,
    ), sklearnex_config_context(target_offload="gpu"):
        kmeans_timer.timeit(
            name="daal4py lloyd GPU",
            engine_provider="sklearn_numba_dpex",
            is_sloww=True,
        )

    for multiplier in [1, 2, 4, 8]:

        with override_attr_context(
            KMeansEngine,
            _DRIVER_CONFIG=dict(device="cpu", work_group_size_multiplier=multiplier),
        ):
            kmeans_timer.timeit(
                name=f"Kmeans numba_dpex lloyd CPU (work_group_size_multiplier={multiplier})",
                engine_provider="sklearn_numba_dpex",
                is_slow=True,
            )

        with override_attr_context(
            KMeansEngine,
            _DRIVER_CONFIG=dict(device="gpu", work_group_size_multiplier=multiplier),
        ):
            kmeans_timer.timeit(
                name=f"Kmeans numba_dpex lloyd GPU (work_group_size_multiplier={multiplier})",
                engine_provider="sklearn_numba_dpex",
            )
