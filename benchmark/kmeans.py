from time import perf_counter
from inspect import signature

import numpy as np
import sklearn
from numpy.testing import assert_array_equal
from sklearn.cluster import KMeans
from sklearn.exceptions import NotSupportedByEngineError


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
        **data_initialization_kwargs,
    ):
        (
            self.X,
            self.sample_weight_,
            self.centers_init,
            self.n_clusters,
        ) = data_initialization_fn(**data_initialization_kwargs)

        self.sample_weight = data_initialization_kwargs.get(
            "sample_weight",
            signature(data_initialization_fn).parameters["sample_weight"].default,
        )

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
        n_clusters = self.n_clusters
        centers_init = self.centers_init
        if hasattr(centers_init, "copy"):
            centers_init = centers_init.copy()

        X = self.X.copy()
        sample_weight = (
            None if self.sample_weight_ is None else self.sample_weight_.copy()
        )

        est_kwargs = dict(
            n_clusters=n_clusters,
            init=centers_init,
            n_init=1,
            max_iter=max_iter,
            tol=0,
            random_state=42,
            copy_x=True,
            algorithm="lloyd",
        )

        estimator = KMeans(**est_kwargs)

        print(
            f"Running {name} with parameters sample_weight={self.sample_weight} "
            f"n_clusters={n_clusters} data_shape={X.shape} max_iter={max_iter}..."
        )

        with sklearn.config_context(engine_provider=engine_provider):

            # dry run, to be fair for JIT implementations
            try:
                KMeans(**est_kwargs).set_params(max_iter=1).fit(
                    X, sample_weight=sample_weight
                )
            except NotSupportedByEngineError as e:
                print((repr(e) + "\n"))
                return

            t0 = perf_counter()
            estimator.fit(X, sample_weight=sample_weight)
            t1 = perf_counter()

            self._check_same_fit(
                estimator, name, max_iter, assert_allclose=self.run_consistency_checks
            )
            print(f"Running {name} ... done in {t1 - t0:.1f} s\n")

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
    import sklearn_numba_dpex.kmeans.engine as skdpex_kmeans_engine_module
    from sklearn_numba_dpex.kmeans.engine import KMeansEngine

    from ext_helpers.daal4py import DAAL4PYEngine
    from sklearnex import config_context as sklearnex_config_context

    # TODO: expose CLI args.

    random_state = 123
    sample_weight = "unary"  # None, "unary", or "random"
    init = "k-means++"  # "k-means++" or "random"
    n_clusters = 127
    max_iter = 100
    skip_slow = True
    dtype = np.float32
    # NB: it seems that currently the estimators in the benchmark always return
    # close results but with significant differences for a few elements.
    run_consistency_checks = False

    def benchmark_data_initialization(random_state, n_clusters, sample_weight):
        X, _ = fetch_openml(name="spoken-arabic-digit", return_X_y=True, parser="auto")
        X = X.astype(dtype)
        scaler_x = MinMaxScaler()
        scaler_x.fit(X)
        X = scaler_x.transform(X)
        # Change dataset dimensions
        # X = np.hstack([X for _ in range(5)])
        # X = np.vstack([X for _ in range(20)])
        X = sklearn.utils.shuffle(X, random_state=random_state)
        rng = default_rng(random_state)
        if (init_ := init) == "random":
            init_ = np.array(rng.choice(X, n_clusters, replace=False), dtype=np.float32)

        if sample_weight is None:
            pass
        elif sample_weight == "unary":
            sample_weight = np.ones(len(X), dtype=dtype)
        elif sample_weight == "random":
            sample_weight = rng.random(size=len(X)).astype(dtype)
        else:
            raise ValueError(
                'Expected sample_weight in {None, "unary", "random"}, got'
                f" {sample_weight} instead."
            )

        return X, sample_weight, init_, n_clusters

    kmeans_timer = KMeansLloydTimeit(
        benchmark_data_initialization,
        max_iter,
        skip_slow,
        run_consistency_checks,
        random_state=random_state,
        n_clusters=n_clusters,
        sample_weight=sample_weight,
    )

    kmeans_timer.timeit(
        name="Sklearn vanilla lloyd",
    )

    with override_attr_context(
        skdpex_kmeans_engine_module, KMeansEngine=DAAL4PYEngine
    ), sklearnex_config_context(target_offload="cpu"):
        kmeans_timer.timeit(
            name="daal4py lloyd CPU", engine_provider="sklearn_numba_dpex"
        )

    with override_attr_context(
        skdpex_kmeans_engine_module, KMeansEngine=DAAL4PYEngine
    ), sklearnex_config_context(target_offload="gpu"):
        kmeans_timer.timeit(
            name="daal4py lloyd GPU",
            engine_provider="sklearn_numba_dpex",
            is_slow=True,
        )

    with override_attr_context(KMeansEngine, _CONFIG=dict(device="cpu")):
        kmeans_timer.timeit(
            name="Kmeans numba_dpex lloyd CPU",
            engine_provider="sklearn_numba_dpex",
            is_slow=True,
        )

    with override_attr_context(KMeansEngine, _CONFIG=dict(device="gpu")):
        kmeans_timer.timeit(
            name="Kmeans numba_dpex lloyd GPU", engine_provider="sklearn_numba_dpex"
        )
