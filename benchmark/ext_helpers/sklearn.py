from sklearn.cluster._kmeans import _kmeans_single_lloyd, row_norms


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
    # NB: include computation of x_squared_norms here to be fairer to other
    # implementations that don't make use of x_squared_norms input.
    # (in fact, the impact it negligible and could be ignored)
    x_squared_norms = row_norms(X, squared=True)
    return _kmeans_single_lloyd(
        X,
        sample_weight,
        centers_init,
        max_iter,
        verbose=verbose,  # NB: better performance when verbose=False because inertia computation is skipped
        x_squared_norms=x_squared_norms,
        tol=tol,
        n_threads=n_threads,
    )
