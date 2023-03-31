from time import perf_counter
from inspect import signature
from contextlib import nullcontext

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

    skip_cpu: bool
        If True, will skip the timeit calls for cpu implementations. Defaults to False.

    skip_gpu: bool
        If True, will skip the timeit calls for gpu implementations. Defaults to False.

    skip_slow: bool
        If True, will skip the timeit calls that are marked as slow. Defaults to False.

    run_consistency_checks : bool
        If True, will check if all candidates on the test bench return the same
        results.
    """

    def __init__(
        self,
        data_initialization_fn,
        max_iter,
        skip_cpu=False,
        skip_gpu=False,
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
        self.random_state = data_initialization_kwargs.get("random_state")
        self.run_consistency_checks = run_consistency_checks
        self.results = None
        self.skip_slow = skip_slow

        dataset = data_initialization_kwargs.get("dataset")
        init = data_initialization_kwargs.get("init")
        print(
            f'\nRunning KMeans benchmark with dataset "{dataset}" with dtype '
            f"{self.X.dtype} and parameters sample_weight={self.sample_weight}, "
            f"init={init}, n_clusters={self.n_clusters}, "
            f"data_shape={self.X.shape}, max_iter={max_iter}...\n"
        )

    def timeit(
        self,
        name,
        engine_provider=None,
        device=None,
        is_slow=False,
        skip=False,
        context=nullcontext(),
    ):
        """
        Parameters
        ----------
        name: str
            Name of the candidate to test.

        engine_provider: str
            The name of the engine provider to use for computations.

        device: str, "cpu" or "gpu"
            Type of device that is expected to run the compute with the engine
            `engine_provider` within the context `context`.

        is_slow: bool
            Set to true to skip the timeit when skip_slow is true. Default to false.

        skip: bool
            Set to true to skip the timeit conditionless.

        context: context manager
            If provided, the compute will run within this context. Defaults to
            `contextlib.nullcontext()`.

        """
        if skip:
            return

        if is_slow and self.skip_slow:
            return

        if device not in ["cpu", "gpu"]:
            raise ValueError(f"Incorrect device: {device}")

        if (device == "cpu") and skip_cpu:
            return

        if (device == "gpu") and skip_gpu:
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
            random_state=self.random_state,
            copy_x=True,
            algorithm="lloyd",
        )

        estimator = KMeans(**est_kwargs)

        print(f"\nRunning {name} on device {device}...")

        with context, sklearn.config_context(engine_provider=engine_provider):

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
                f"max_iter={max_iter} iterations. "
                + runtime_error_message
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


# Helpers to create a dataset of any given shape by sampling an existing dataset
# If sampling more samples or features than the dataset contains, it ensures that each
# sample or feature is at least sampled once.


def _get_features_sample_list(n_features, n_feature_samples, rng):
    features_sample_list = rng.choice(
        a=n_features,
        size=min(n_features, n_feature_samples),
        replace=False,
        shuffle=False,
    ).tolist()

    if n_feature_samples > n_features:
        features_sample_list.extend(
            rng.choice(
                a=n_features,
                size=n_feature_samples - n_features,
                replace=True,
                shuffle=False,
            ).tolist()
        )

    return sklearn.utils.shuffle(features_sample_list, random_state=random_state)


def _get_dataset_samples(X, n_data_samples, rng):
    n_samples = X.shape[0]
    samples = rng.choice(
        X, size=min(n_samples, n_data_samples), replace=False, shuffle=False
    )

    if n_data_samples > n_samples:
        more_samples = rng.choice(
            X, size=n_data_samples - n_samples, replace=True, shuffle=False
        )
        samples = np.vstack((samples, more_samples))

    return samples


if __name__ == "__main__":
    from argparse import ArgumentParser
    import warnings

    from numpy.random import default_rng
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import MinMaxScaler

    from sklearn_numba_dpex.testing import override_attr_context
    from sklearn_numba_dpex.kmeans.engine import KMeansEngine

    argparser = ArgumentParser(
        description=(
            "This is a test bench for KMeans provided by the sklearn_numba_dpex "
            "project available at https://github.com/soda-inria/sklearn-numba-dpex . "
            "Please also refer to the sklearn documentation page at "
            "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html "  # noqa
            "for more informatiom on the CLI parameters."
        )
    )

    argparser.add_argument(
        "--sample-weight", default="random", choices=["unary", "random", "None"]
    )

    argparser.add_argument("--init", default="random", choices=["k-means++", "random"])

    argparser.add_argument("--n-clusters", default=127, type=int)

    argparser.add_argument("--max-iter", default=100, type=int)

    argparser.add_argument(
        "--dataset",
        default="spoken-arabic-digit",
        choices=["spoken-arabic-digit"],
        help="Name of the dataset used to benchmark the engine providers.",
    )

    argparser.add_argument(
        "--n-data-samples",
        default=None,
        type=int,
        help=(
            "n_data_samples samples will be sampled from the dataset. If necessary, "
            "some samples will be randomly repeated. If None, the original amount "
            "of samples in the dataset will be used without alteration."
        ),
    )

    argparser.add_argument(
        "--n-features",
        default=None,
        type=int,
        help=(
            "n_features features will be sampled from the dataset. If necessary, "
            "some features will be randomly repeated. If None, the original amount "
            "of features in the dataset will be used without alteration."
        ),
    )

    def _as_numpy_dtype(type_str):
        return np.dtype(type_str).type

    argparser.add_argument(
        "--dtype",
        default="float32",
        choices=[np.float32, np.float64],
        type=_as_numpy_dtype,
        help="Floating points precision.",
    )

    argparser.add_argument(
        "--random-state", default=123, type=int, help="Seed set for reproducibility."
    )

    # NB: this option is currently unreliable, estimators seem to usually return close
    # results except for a few data points that have significant differences.
    argparser.add_argument(
        "--run-consistency-checks",
        action="store_true",
        help="Check that all the estimators in the benchmark return the same results.",
    )

    class _IncludeAll:
        def __contains__(self, item):
            return True

    argparser.add_argument(
        "--engine-providers",
        nargs="+",
        choices=["scikit-learn", "daal4py", "kmeans-dpcpp", "sklearn-numba-dpex"],
        default=_IncludeAll(),
        help=(
            "Only run the selected engine providers. If not specified, include all "
            "providers in the benchmark."
        ),
    )

    argparser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include in the benchmark the engine providers that are known to be slow.",
    )

    argparser.add_argument(
        "--skip-cpu", action="store_true", help="Skip CPU implementations"
    )

    argparser.add_argument(
        "--skip-gpu", action="store_true", help="Skip GPU implementations"
    )

    args = argparser.parse_args()

    sample_weight = args.sample_weight
    init = args.init
    n_clusters = args.n_clusters
    max_iter = args.max_iter

    dataset = args.dataset
    n_data_samples = args.n_data_samples
    n_features = args.n_features
    dtype = args.dtype
    random_state = args.random_state
    run_consistency_checks = args.run_consistency_checks
    engine_providers = args.engine_providers
    skip_slow = not args.include_slow
    skip_cpu = args.skip_cpu
    skip_gpu = args.skip_gpu

    if not skip_gpu:
        import dpctl

        try:
            dpctl.SyclDevice("gpu")
        except dpctl.SyclDeviceCreationError:
            skip_gpu = True
            warnings.warn(
                "Can't detect a GPU. GPU benchmarks will be skipped.",
                RuntimeWarning,
            )

    def benchmark_data_initialization(
        dataset, init, n_clusters, sample_weight, random_state
    ):
        X, _ = fetch_openml(name=dataset, return_X_y=True, parser="auto")
        X = X.values.astype(dtype)
        rng = default_rng(random_state)
        if n_features is not None:
            features_sample_list = _get_features_sample_list(
                X.shape[1], n_features, rng
            )
            X = X[:, features_sample_list]

        if n_data_samples is not None:
            X = _get_dataset_samples(X, n_data_samples, rng)

        X = sklearn.utils.shuffle(X, random_state=random_state)

        scaler_x = MinMaxScaler()
        scaler_x.fit(X)
        X = scaler_x.transform(X)

        if (init_ := init) == "random":
            init_ = np.array(rng.choice(X, n_clusters, replace=False), dtype=dtype)

        if sample_weight == "None":
            sample_weight = None
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
        skip_cpu,
        skip_gpu,
        skip_slow,
        run_consistency_checks,
        dataset=dataset,
        init=init,
        n_clusters=n_clusters,
        sample_weight=sample_weight,
        random_state=random_state,
    )

    if "scikit-learn" in engine_providers:
        kmeans_timer.timeit(name="scikit-learn", device="cpu")

    try:
        from sklearnex import config_context as sklearnex_config_context
        from ext_helpers.daal4py import DAAL4PYEngine
    except (ModuleNotFoundError, ImportError):
        warnings.warn(
            "scikit-learn-intelex can't be found. Benchmark will be skipped.",
            RuntimeWarning,
        )
    else:
        skip = "daal4py" not in engine_providers
        kmeans_timer.timeit(
            name="daal4py",
            engine_provider=DAAL4PYEngine,
            device="cpu",
            skip=skip,
            context=sklearnex_config_context(target_offload="cpu"),
        )

        kmeans_timer.timeit(
            name="daal4py",
            engine_provider=DAAL4PYEngine,
            device="gpu",
            is_slow=True,
            skip=skip,
            context=sklearnex_config_context(target_offload="gpu"),
        )

    try:
        from ext_helpers.kmeans_dpcpp import KMeansDPCPPEngine
    except (ModuleNotFoundError, ImportError):
        warnings.warn(
            "kmeans_dpcpp can't be found. Benchmark will be skipped.", RuntimeWarning
        )
    else:
        skip = "kmeans-dpcpp" not in engine_providers
        kmeans_timer.timeit(
            name="kmeans_dpcpp",
            engine_provider=KMeansDPCPPEngine,
            device="cpu",
            is_slow=True,
            skip=skip,
            context=override_attr_context(
                KMeansDPCPPEngine, _CONFIG=dict(device="cpu")
            ),
        )

        kmeans_timer.timeit(
            name="kmeans_dpcpp",
            engine_provider=KMeansDPCPPEngine,
            device="gpu",
            skip=skip,
            context=override_attr_context(
                KMeansDPCPPEngine, _CONFIG=dict(device="gpu")
            ),
        )

    skip = "sklearn-numba-dpex" not in engine_providers
    kmeans_timer.timeit(
        name="sklearn_numba_dpex",
        engine_provider="sklearn_numba_dpex",
        device="cpu",
        skip=skip,
        is_slow=True,
        context=override_attr_context(KMeansEngine, _CONFIG=dict(device="cpu")),
    )

    kmeans_timer.timeit(
        name="sklearn_numba_dpex",
        engine_provider="sklearn_numba_dpex",
        device="gpu",
        skip=skip,
        context=override_attr_context(KMeansEngine, _CONFIG=dict(device="gpu")),
    )
