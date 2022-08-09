import pytest

from numpy.testing import assert_array_equal
from sklearn import config_context
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.utils._testing import assert_allclose


def test_placeholder():
    """Placeholder test for CI"""
    try:
        import numba_dpex as dpex
    except ImportError:
        import numba_dppy as dpex

    import dpctl

    # There must be at least one usable device.
    assert len(dpctl.get_devices()) > 0


@pytest.mark.skip(reason="KMeans has not been implemented yet.")
def test_kmeans_same_results(global_random_seed):
    X = make_blobs(random_state=global_random_seed)

    # Fit a reference model with the default scikit-learn engine:

    kmeans_vanilla = KMeans(random_state=global_random_seed)
    kmeans_engine = clone(kmeans_vanilla)

    y_pred = kmeans_vanilla.fit_predict(X)
    X_trans = kmeans_vanilla.transform(X)

    # When a specific engine is specified by the use, it should do the
    # necessary data conversions from numpy automatically:

    with config_context(engine_provider="sklearn_numba_dpex"):
        y_pred_dpnp = kmeans_engine.fit_predict(X)
        X_trans_dpnp = kmeans_engine.transform(X)

    assert_array_equal(y_pred_dpnp, y_pred)
    assert_allclose(X_trans_dpnp, X_trans)

    # TODO: convert X to dpnp datastructure explicitly
    X_dpnp = X

    with config_context(engine_provider="sklearn_numba_dpex"):
        y_pred_dpnp2 = kmeans_engine.fit_predict(X_dpnp)
        X_trans_dpnp2 = kmeans_engine.transform(X_dpnp)

    assert_array_equal(y_pred_dpnp2, y_pred)
    assert_allclose(X_trans_dpnp2, X_trans)
