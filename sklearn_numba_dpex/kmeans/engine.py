try:
    from sklearn.cluster._kmeans import KMeansCythonEngine
except ImportError:
    KMeansCythonEngine = object

from .drivers import KMeansDriver


class KMeansSklearnEngine(KMeansCythonEngine):
    def kmeans_single(self, X, sample_weight, centers_init):
        return KMeansDriver().lloyd(
            X,
            sample_weight,
            centers_init,
            max_iter=self.estimator.max_iter,
            tol=self.tol,
            n_threads=self._n_threads,
            verbose=self.estimator.verbose,
        )
