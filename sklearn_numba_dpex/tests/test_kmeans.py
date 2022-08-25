import pytest
import inspect

import sklearn
import dpctl

import numpy as np
from numpy.testing import assert_array_equal
from sklearn import config_context
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.utils._testing import assert_allclose

from sklearn_numba_dpex.kmeans.drivers import LLoydKMeansDriver


_SKLEARN_LLOYD = sklearn.cluster._kmeans._kmeans_single_lloyd
_SKLEARN_LLOYD_SIGNATURE = inspect.signature(_SKLEARN_LLOYD)


_DEVICE = dpctl.SyclDevice()
_DEVICE_NAME = _DEVICE.name
_SUPPORTED_DTYPE = [np.float32]

if _DEVICE.has_aspect_fp64:
    _SUPPORTED_DTYPE.append(np.float64)


def _fail_if_no_dtype_support(xfail_fn, dtype):
    if dtype not in _SUPPORTED_DTYPE:
        xfail_fn(
            f"The default device {_DEVICE_NAME} does not have support for "
            f"float64 operations."
        )


def test_kmeans_driver_signature():
    driver_signature = inspect.signature(LLoydKMeansDriver())
    assert driver_signature == _SKLEARN_LLOYD_SIGNATURE


# TODO: write a test to check that the estimator remains stable if a cluster is found to
# have 0 samples.


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmeans_fit_same_results(dtype):
    _fail_if_no_dtype_support(pytest.xfail, dtype)

    # TODO: remove the manual monkey-patching and rely on engine registration
    # once properly implemented.
    random_seed = 42
    X, _ = make_blobs(random_state=random_seed)
    X = X.astype(dtype)

    kmeans_vanilla = KMeans(random_state=random_seed, algorithm="lloyd", max_iter=1)
    kmeans_engine = clone(kmeans_vanilla)

    # Fit a reference model with the default scikit-learn engine:
    kmeans_vanilla.fit(X)

    # Fit a model with numba_dpex backend
    try:
        # Temporarily monkeypatch scikit-learn internals to replace
        # them with this package implementation.
        sklearn.cluster._kmeans._kmeans_single_lloyd = LLoydKMeansDriver(
            work_group_size_multiplier=4, dtype=dtype
        )
        kmeans_engine.fit(X)
    finally:
        sklearn.cluster._kmeans._kmeans_single_lloyd = _SKLEARN_LLOYD

    # ensure same results
    assert_array_equal(kmeans_vanilla.labels_, kmeans_engine.labels_)
    assert_allclose(kmeans_vanilla.cluster_centers_, kmeans_engine.cluster_centers_)
    assert_allclose(kmeans_vanilla.inertia_, kmeans_engine.inertia_)


@pytest.mark.skip(reason="KMeans has not been implemented yet.")
def test_kmeans_same_results(global_random_seed):
    X, _ = make_blobs(random_state=global_random_seed)

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
