from numpy.testing import assert_array_equal
from sklearn.datasets import make_blobs
from sklearn._engine import computational_engine, get_engine
from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.utils._testing import assert_allclose
import pytest


def test_kmeans_same_results(global_random_seed):
    pytest.importorskip("sklearn_numba_dppy")

    X = make_blobs(random_state=global_random_seed)
    
    # 

    kmeans_vanilla = KMeans(random_state=global_random_seed)
    kmeans_engine = clone(kmeans_vanilla)

    y_pred = kmeans_vanilla.fit_predict(X)
    X_trans = kmeans_vanilla.transform(X)

    # TODO: convert X to dpnp datastructure explicitly
    X_dpnp = X

    with computational_engine("sklearn_numba_dppy"):
        y_pred_dpnp = kmeans_engine.fit_predict(X_dpnp)
        X_trans_dpnp = kmeans_engine.transform(X_dpnp)

    assert_array_equal(y_pred_dpnp, y_pred)
    assert_allclose(X_trans_dpnp, X_trans)
