import dpctl.tensor as dpt
import kmeans_dpcpp as kdp

from sklearn.exceptions import NotSupportedByEngineError
from sklearn_numba_dpex.kmeans.engine import KMeansEngine


class KMeansDPCPPEngine(KMeansEngine):
    def init_centroids(self, X):
        init = self.init
        if isinstance(init, str) and init == "k-means++":
            raise NotSupportedByEngineError(
                "kmeans_dpcpp does not support k-means++ init."
            )
        return super().init_centroids(X)

    def kmeans_single(self, X, sample_weight, centers_init_t):
        n_samples, n_features = X.shape
        device = X.device.sycl_device

        assignments_idx = dpt.empty(n_samples, dtype=dpt.int32, device=device)
        res_centroids_t = dpt.empty_like(centers_init_t)

        # TODO: make work_group_size and centroid window dimension consistent in
        # kmeans_dpcpp with up to date configuration in sklearn_numba_dpex
        centroids_window_height = 8
        work_group_size = 128

        centroids_private_copies_max_cache_occupancy = 0.7

        n_iters, total_inertia = kdp.kmeans_lloyd_driver(
            X.T,
            sample_weight,
            centers_init_t,
            assignments_idx,
            res_centroids_t,
            self.tol,
            bool(self.estimator.verbose),
            self.estimator.max_iter,
            centroids_window_height,
            work_group_size,
            centroids_private_copies_max_cache_occupancy,
            X.sycl_queue,
        )

        return assignments_idx, total_inertia, res_centroids_t, n_iters
