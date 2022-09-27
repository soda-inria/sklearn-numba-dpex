import inspect

import dpctl
import numpy as np
import pytest
import warnings
import sklearn
from numpy.random import default_rng
from numpy.testing import assert_array_equal
from sklearn import config_context
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.utils._testing import assert_allclose

from sklearn_numba_dpex.kmeans.drivers import KMeansDriver

_SKLEARN_LLOYD = sklearn.cluster._kmeans._kmeans_single_lloyd
_SKLEARN_LLOYD_SIGNATURE = inspect.signature(_SKLEARN_LLOYD)

_SKLEARN_LABELS_INERTIA = sklearn.cluster._kmeans._labels_inertia
_SKLEARN_LABELS_INERTIA_SIGNATURE = inspect.signature(_SKLEARN_LABELS_INERTIA)

_SKLEARN_EUCLIDEAN_DISTANCES = sklearn.cluster._kmeans.euclidean_distances
_SKLEARN_EUCLIDEAN_DISTANCES_SIGNATURE = inspect.signature(_SKLEARN_EUCLIDEAN_DISTANCES)


_DEVICE = dpctl.SyclDevice()
_DEVICE_NAME = _DEVICE.name
_SUPPORTED_DTYPE = [np.float32]

if _DEVICE.has_aspect_fp64:
    _SUPPORTED_DTYPE.append(np.float64)


def _fail_if_no_dtype_support(xfail_fn, dtype):
    if dtype not in _SUPPORTED_DTYPE:
        xfail_fn(
            f"The default device {_DEVICE_NAME} does not have support for "
            "float64 operations."
        )


def test_lloyd_driver_signature():
    driver_signature = inspect.signature(KMeansDriver().lloyd)
    assert driver_signature == _SKLEARN_LLOYD_SIGNATURE


def test_get_labels_driver_signature():
    driver_signature = inspect.signature(KMeansDriver().get_labels)
    assert driver_signature == _SKLEARN_LABELS_INERTIA_SIGNATURE


def test_get_inertia_driver_signature():
    driver_signature = inspect.signature(KMeansDriver().get_inertia)
    assert driver_signature == _SKLEARN_LABELS_INERTIA_SIGNATURE


def test_euclidean_distances_driver_signature():
    driver_signature = inspect.signature(KMeansDriver().get_euclidean_distances)
    assert driver_signature == _SKLEARN_EUCLIDEAN_DISTANCES_SIGNATURE


# TODO: write a test to check that the estimator remains stable if a cluster is found to
# have 0 samples.


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmeans_same_results(dtype):
    _fail_if_no_dtype_support(pytest.xfail, dtype)

    # TODO: remove the manual monkey-patching and rely on engine registration
    # once properly implemented.
    random_seed = 42
    X, _ = make_blobs(random_state=random_seed)
    X = X.astype(dtype)

    kmeans_vanilla = KMeans(random_state=random_seed, algorithm="lloyd", max_iter=1)
    kmeans_engine = clone(kmeans_vanilla)
    engine = KMeansDriver(work_group_size_multiplier=4, dtype=dtype)

    # Fit a reference model with the default scikit-learn engine:
    kmeans_vanilla.fit(X)

    with config_context(engine_provider="sklearn_numba_dpex"):
        kmeans_engine.fit(X)

    # ensure same results
    assert_array_equal(kmeans_vanilla.labels_, kmeans_engine.labels_)
    assert_allclose(kmeans_vanilla.cluster_centers_, kmeans_engine.cluster_centers_)
    assert_allclose(kmeans_vanilla.inertia_, kmeans_engine.inertia_)

    # test fit_predict
    y_labels = kmeans_vanilla.fit_predict(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_labels_engine = kmeans_engine.fit_predict(X)

    assert_array_equal(y_labels, y_labels_engine)
    assert_array_equal(kmeans_vanilla.labels_, kmeans_engine.labels_)
    assert_allclose(kmeans_vanilla.cluster_centers_, kmeans_engine.cluster_centers_)
    assert_allclose(kmeans_vanilla.inertia_, kmeans_engine.inertia_)

    # test fit_transform
    y_transform = kmeans_vanilla.fit_transform(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_transform_engine = kmeans_engine.fit_transform(X)
    assert_allclose(y_transform, y_transform_engine)
    assert_array_equal(kmeans_vanilla.labels_, kmeans_engine.labels_)
    assert_allclose(kmeans_vanilla.cluster_centers_, kmeans_engine.cluster_centers_)
    assert_allclose(kmeans_vanilla.inertia_, kmeans_engine.inertia_)

    y_labels = kmeans_vanilla.predict(X)
    try:
        sklearn.cluster._kmeans._labels_inertia = engine.get_labels
        y_labels_engine = kmeans_engine.predict(X)
    finally:
        sklearn.cluster._kmeans._labels_inertia = _SKLEARN_LABELS_INERTIA
    assert_array_equal(y_labels, y_labels_engine)

    # test score method (returns negative inertia for each sample)
    y_scores = kmeans_vanilla.score(X)
    try:
        sklearn.cluster._kmeans._labels_inertia = engine.get_inertia
        y_scores_engine = kmeans_engine.score(X)
    finally:
        sklearn.cluster._kmeans._labels_inertia = _SKLEARN_LABELS_INERTIA
    assert_allclose(y_scores, y_scores_engine)

    # test transform method (returns euclidean distances)
    y_transform = kmeans_vanilla.transform(X)
    try:
        sklearn.cluster._kmeans.euclidean_distances = engine.get_euclidean_distances
        y_transform_engine = kmeans_engine.transform(X)
    finally:
        sklearn.cluster._kmeans.euclidean_distances = _SKLEARN_EUCLIDEAN_DISTANCES
    assert_allclose(y_transform, y_transform_engine)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmeans_fit_empty_clusters(dtype):
    _fail_if_no_dtype_support(pytest.xfail, dtype)

    random_seed = 42
    X, _ = make_blobs(random_state=random_seed)
    X = X.astype(dtype)

    n_clusters = inspect.signature(KMeans).parameters["n_clusters"].default
    rng = default_rng(random_seed)
    init = np.array(rng.choice(X, n_clusters, replace=False), dtype=np.float32)
    # Create an outlier centroid in initial centroids to have an empty cluster.
    init[0] = np.finfo(np.float32).max

    engine = KMeansDriver(work_group_size_multiplier=4, dtype=dtype)

    kmeans_with_empty_cluster = KMeans(
        random_state=random_seed,
        algorithm="lloyd",
        n_clusters=n_clusters,
        init=init,
        n_init=1,
        max_iter=1,
    )

    # TODO: once the behavior of scikit learn for empty clusters has been implemented
    # identically in sklearn_numba_dpex, the warning will be removed, and we will want
    # to check instead that the results are equals (similarly to the test
    # test_kmeans_fit_same_results)
    with pytest.warns(RuntimeWarning, match="Found an empty cluster"):
        try:
            sklearn.cluster._kmeans._kmeans_single_lloyd = engine.lloyd
            kmeans_with_empty_cluster.fit(X)
        finally:
            sklearn.cluster._kmeans._kmeans_single_lloyd = _SKLEARN_LLOYD

    kmeans_without_empty_cluster = KMeans(
        random_state=random_seed,
        algorithm="lloyd",
        n_clusters=n_clusters,
        n_init=1,
        max_iter=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            sklearn.cluster._kmeans._kmeans_single_lloyd = engine.lloyd
            kmeans_without_empty_cluster.fit(X)
        finally:
            sklearn.cluster._kmeans._kmeans_single_lloyd = _SKLEARN_LLOYD


@pytest.mark.skip(reason="plugin interface has not been implemented yet.")
def test_kmeans_same_results_with_plugin_interface(global_random_seed):
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
