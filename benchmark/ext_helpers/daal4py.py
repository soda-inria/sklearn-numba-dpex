import warnings
import sklearn

# HACK: daal4py will fail to import with too recent versions of sklearn because
# of this missing attribute. Let's pretend it still exists.

if not hasattr(sklearn.neighbors._base, "_check_weights"):
    warnings.warn(
        f"The current version of scikit-learn ( =={sklearn.__version__} ) is too "
        "recent to ensure good compatibility with sklearn intelex, who only supports "
        "sklearn >=0.22, <1.1 . Use very cautiously, things might not work as "
        "expected...",
        RuntimeWarning,
    )
    try:
        sklearn.neighbors._base._check_weights = None
        from daal4py.sklearn.cluster._k_means_0_23 import KMeans
    finally:
        del sklearn.neighbors._base._check_weights
else:
    from daal4py.sklearn.cluster._k_means_0_23 import KMeans

from sklearn.exceptions import NotSupportedByEngineError
from sklearn_numba_dpex.kmeans.engine import KMeansEngine


# TODO: instead of relying on monkey patching the default engine, find a way to
# register a distinct entry point that can load a distinct engine outside of setup.py
# (impossible ?)
class DAAL4PYEngine(KMeansEngine):
    def prepare_fit(self, X, y=None, sample_weight=None):
        estimator = self.estimator
        init = estimator.init
        if hasattr(init, "startswith") and init == "k-means++":
            raise NotSupportedByEngineError(
                "The daal4py engine for KMeans does not support k-means++ init."
            )

        if sample_weight is not None and any(sample_weight != sample_weight[0]):
            raise NotSupportedByEngineError(
                "Non unary sample_weight is not supported by daal4py."
            )

        return super().prepare_fit(X, y, sample_weight)

    def init_centroids(self, X):
        return super(KMeansEngine, self).init_centroids(X)

    def kmeans_single(self, X, sample_weight, centers_init):

        est = KMeans(
            n_clusters=centers_init.shape[0],
            init=centers_init,
            n_init=self.estimator.n_init,
            max_iter=self.estimator.max_iter,
            tol=self.estimator.tol,
            verbose=self.estimator.verbose,
            random_state=self.estimator.random_state,
            copy_x=self.estimator.copy_x,
            algorithm="full",
        )

        est.fit(X, sample_weight=sample_weight)

        return est.labels_, est.inertia_, est.cluster_centers_, est.n_iter_

    def get_labels(self, X, sample_weight):
        raise NotSupportedByEngineError

    def get_euclidean_distances(self, X):
        raise NotSupportedByEngineError

    def get_score(self, X, sample_weight):
        raise NotSupportedByEngineError


# NB: this implementation skips input and environment validation steps that, when used
# within sklearn.numba_dpex.benchmark.kmeans benchmark, are redundant with sklearn.cluster.KMeans
# steps, but it is not compatible with sklearnex.config_context that can be used to
# offload computations to all available devices.
#
# In fact, the added cost of input and environment validation in sklearnex is negligible and
# should not be a significant factor in the benchmark results.
#

# from daal4py.sklearn.cluster._k_means_0_23 import _daal4py_k_means_fit


# def kmeans_daal4py_kmeans_single(self, X, sample_weight, centers_init):

#     (
#         best_cluster_centers,
#         best_labels,
#         best_inertia,
#         best_n_iter,
#     ) = _daal4py_k_means_fit(
#         X=X,
#         nClusters=centers_init.shape[0],
#         numIterations=self.estimator.max_iter,
#         tol=self.estimator.tol,
#         cluster_centers_0=centers_init,
#         n_init=self.estimator.n_init,
#         verbose=self.estimator.verbose,
#         random_state=self.estimator.random_state,
#     )

#     return best_labels, best_inertia, best_cluster_centers, best_n_iter
