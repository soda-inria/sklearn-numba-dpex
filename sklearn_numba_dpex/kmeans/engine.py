from sklearn.cluster._kmeans import KMeansCythonEngine

from sklearn.exceptions import NotSupportedByEngineError

from .drivers import KMeansDriver


# At the moment not all steps are implemented with numba_dpex, we inherit missing steps
# from the default sklearn KMeansCythonEngine for convenience, this inheritance will be
# removed later on when the other parts have been implemented.
class KMeansEngine(KMeansCythonEngine):
    _DRIVER_CONFIG = dict()

    def prepare_fit(self, X, y=None, sample_weight=None):
        estimator = self.estimator
        try:
            # This pass of data validation only aims at detecting input types that are
            # supported by the sklearn engine but not by KMeansEngine. For those inputs
            # we raise a NotSupportedByEngineError exception.
            # estimator._validate_data is called again in super().prepare_fit later on
            # and will raise ValueError or TypeError for data that is not even
            # compatible with the sklearn engine.
            estimator._validate_data(
                X,
                accept_sparse=False,
                dtype=None,
                force_all_finite=False,
                ensure_2d=False,
                allow_nd=True,
                ensure_min_samples=0,
                ensure_min_features=0,
                estimator=estimator,
            )
        except Exception as e:
            raise NotSupportedByEngineError(
                "The sklearn_nunmba_dpex engine for KMeans does not support the format of the inputed data."
            ) from e

        algorithm = estimator.algorithm
        if algorithm not in ("lloyd", "auto", "full"):
            raise NotSupportedByEngineError(
                f"The sklearn_nunmba_dpex engine for KMeans only support the Lloyd algorithm, {algorithm} is not supported."
            )

        self.sample_weight = sample_weight

        return super().prepare_fit(X, y, sample_weight)

    def init_centroids(self, X):
        if hasattr(self.init, "startswith") and self.init == "k-means++":
            return KMeansDriver(**self._DRIVER_CONFIG).kmeans_plusplus(
                X, self.sample_weight, self.estimator.n_clusters, self.random_state
            )

        return self.estimator._init_centroids(
            X,
            x_squared_norms=self.x_squared_norms,
            init=self.init,
            random_state=self.random_state,
        )

    def kmeans_single(self, X, sample_weight, centers_init):
        return KMeansDriver(**self._DRIVER_CONFIG).lloyd(
            X,
            sample_weight,
            centers_init,
            max_iter=self.estimator.max_iter,
            tol=self.tol,
            verbose=self.estimator.verbose,
        )

    def get_labels(self, X, sample_weight):
        return KMeansDriver(**self._DRIVER_CONFIG).get_labels(
            X, self.estimator.cluster_centers_
        )

    def get_euclidean_distances(self, X):
        return KMeansDriver(**self._DRIVER_CONFIG).get_euclidean_distances(
            X, self.estimator.cluster_centers_
        )

    def get_score(self, X, sample_weight):
        return KMeansDriver(**self._DRIVER_CONFIG).get_inertia(
            X, sample_weight, self.estimator.cluster_centers_
        )
