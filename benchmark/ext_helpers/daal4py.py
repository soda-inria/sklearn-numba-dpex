import warnings
import sklearn


# HACK: daal4py will fail to import with too recent versions of sklearn because
# of this missing attribute. Let's pretend it still exists.
try:
    _daal4py_kmeans_compat_mode = not hasattr(sklearn.neighbors._base, "_check_weights")
    if _daal4py_kmeans_compat_mode:
        warnings.warn(
            f"The current version of scikit-learn ( =={sklearn.__version__} ) is too "
            "recent to ensure good compatibility with sklearn intelex, who only "
            "supports sklearn >=0.22, <1.1 . Use very cautiously, things might not "
            "work as expected...",
            RuntimeWarning,
        )
        sklearn.neighbors._base._check_weights = None
    from daal4py.sklearn.cluster._k_means_0_23 import (
        _daal4py_compute_starting_centroids,
        _daal4py_k_means_fit,
        support_usm_ndarray,
        getFPType,
    )
finally:
    if _daal4py_kmeans_compat_mode:
        del sklearn.neighbors._base._check_weights


from sklearn.exceptions import NotSupportedByEngineError
from sklearn_numba_dpex.kmeans.engine import KMeansEngine


# TODO: instead of relying on monkey patching the default engine, find a way to
# register a distinct entry point that can load a distinct engine outside of setup.py
# (impossible ?)
class DAAL4PYEngine(KMeansEngine):
    def prepare_fit(self, X, y=None, sample_weight=None):
        if sample_weight is not None and any(sample_weight != sample_weight[0]):
            raise NotSupportedByEngineError(
                "Non unary sample_weight is not supported by daal4py."
            )

        return super().prepare_fit(X, y, sample_weight)

    @support_usm_ndarray()
    def init_centroids(self, X):
        init = self.init
        try:
            _, centers_init = _daal4py_compute_starting_centroids(
                X,
                getFPType(X),
                self.estimator.n_clusters,
                init,
                self.estimator.verbose,
                self.random_state,
            )
            return centers_init
        except RuntimeError as runtime_error:
            if (
                "Device support for the algorithm isn't implemented"
                in str(runtime_error)
                and isinstance(init, str)
                and init == "k-means++"
            ):
                raise NotSupportedByEngineError(
                    "daal4py does not support k-means++ init on device"
                ) from runtime_error
            else:
                raise

    @support_usm_ndarray()
    def kmeans_single(self, X, sample_weight, centers_init):
        cluster_centers, labels, inertia, n_iter = _daal4py_k_means_fit(
            X,
            nClusters=self.estimator.n_clusters,
            numIterations=self.estimator.max_iter,
            tol=self.tol,
            cluster_centers_0=centers_init,
            n_init=self.estimator.n_init,
            verbose=self.estimator.verbose,
            random_state=self.random_state,
        )

        return labels, inertia, cluster_centers, n_iter

    def get_labels(self, X, sample_weight):
        raise NotSupportedByEngineError

    def get_euclidean_distances(self, X):
        raise NotSupportedByEngineError

    def get_score(self, X, sample_weight):
        raise NotSupportedByEngineError
