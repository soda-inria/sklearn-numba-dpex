from daal4py.sklearn.cluster._k_means_0_23 import _daal4py_k_means_fit


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
    (
        best_cluster_centers,
        best_labels,
        best_inertia,
        best_n_iter,
    ) = _daal4py_k_means_fit(
        X=X,
        nClusters=centers_init.shape[0],
        numIterations=max_iter,
        tol=tol,
        cluster_centers_0=centers_init,
        n_init=1,
        verbose=verbose,
        random_state=123,
    )

    return best_labels, best_inertia, best_cluster_centers, best_n_iter
