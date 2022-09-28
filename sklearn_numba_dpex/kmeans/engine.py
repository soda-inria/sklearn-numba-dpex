try:
    from sklearn.cluster._kmeans import KMeansCythonEngine
except ImportError:
    KMeansCythonEngine = object

from sklearn.exceptions import FeatureNotCoveredByPluginError

from .drivers import KMeansDriver


class KMeansEngine(KMeansCythonEngine):
    def prepare_fit(self, X, y=None, sample_weight=None):
        estimator = self.estimator
        try:
            estimator._validate_data(X, accept_sparse=False)
        except Exception as e:
            raise FeatureNotCoveredByPluginError(
                "The sklearn_nunmba_dpex engine for KMeans does not support the format of the inputed data."
            ) from e

        algorithm = estimator.algorithm
        if algorithm not in ("lloyd", "auto", "full"):
            raise FeatureNotCoveredByPluginError(
                f"The sklearn_nunmba_dpex engine for KMeans only support the Lloyd algorithm, {algorithm} is not supported."
            )

        return super().prepare_fit(X, y, sample_weight)

    def kmeans_single(self, X, sample_weight, centers_init):
        return KMeansDriver().lloyd(
            X,
            sample_weight,
            centers_init,
            max_iter=self.estimator.max_iter,
            tol=self.tol,
            verbose=self.estimator.verbose,
        )

    def get_labels(self, X, sample_weight):
        return KMeansDriver().get_labels(
            X, sample_weight, self.estimator.cluster_centers_
        )

    def get_euclidean_distances(self, X):
        return KMeansDriver().get_euclidean_distances(
            X, self.estimator.cluster_centers_
        )

    def get_score(self, X, sample_weight):
        return KMeansDriver().get_inertia(
            X, sample_weight, self.estimator.cluster_centers_
        )
