from daal4py.sklearn.cluster._k_means_0_23 import KMeans


def kmeans(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    x_squared_norms=None,
    tol=1e-4,
    n_threads=1,
):
    est = KMeans(
        n_clusters=centers_init.shape[0],
        init=centers_init,
        n_init=1,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        random_state=42,
        copy_x=True,
        algorithm="full",
    )
    est.fit(X, sample_weight=sample_weight)
    return est.labels_, est.inertia_, est.cluster_centers_, est.n_iter_


# NB: this implementation skips input and environment validation steps that, when used
# within sklearn.numba_dpex.benchmark.kmeans benchmark, are redundant with sklearn.cluster.KMeans
# steps, but it is not compatible with sklearnex.config_context that can be used to
# offload computations to all available devices.
#
# In fact, the added cost of input and environment validation in sklearnex is negligible and
# should not be a significant factor in the benchmark results.
#
# from daal4py.sklearn.cluster._k_means_0_23 import _daal4py_k_means_fit

# def kmeans(
#     X,
#     sample_weight,
#     centers_init,
#     max_iter=300,
#     verbose=False,
#     x_squared_norms=None,
#     tol=1e-4,
#     n_threads=1,
# ):
#     (
#         best_cluster_centers,
#         best_labels,
#         best_inertia,
#         best_n_iter,
#     ) = _daal4py_k_means_fit(
#         X=X,
#         nClusters=centers_init.shape[0],
#         numIterations=max_iter,
#         tol=tol,
#         cluster_centers_0=centers_init,
#         n_init=1,
#         verbose=verbose,
#         random_state=123,
#     )

#     return best_labels, best_inertia, best_cluster_centers, best_n_iter
