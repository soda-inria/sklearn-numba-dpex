import inspect

import dpctl
import numpy as np
import pytest
import warnings
from numpy.random import default_rng
from numpy.testing import assert_array_equal
from sklearn import config_context
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.utils._testing import assert_allclose


from sklearn_numba_dpex.kmeans.drivers import KMeansDriver


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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmeans_same_results(dtype):
    _fail_if_no_dtype_support(pytest.xfail, dtype)

    random_seed = 42
    X, _ = make_blobs(random_state=random_seed)
    X = X.astype(dtype)

    kmeans_vanilla = KMeans(random_state=random_seed, algorithm="lloyd", max_iter=1)
    kmeans_engine = clone(kmeans_vanilla)

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

    # # test predict method (returns labels)
    y_labels = kmeans_vanilla.predict(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_labels_engine = kmeans_engine.predict(X)
    assert_array_equal(y_labels, y_labels_engine)

    # test score method (returns negative inertia for each sample)
    y_scores = kmeans_vanilla.score(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_scores_engine = kmeans_engine.score(X)
    assert_allclose(y_scores, y_scores_engine)

    # test transform method (returns euclidean distances)
    y_transform = kmeans_vanilla.transform(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_transform_engine = kmeans_engine.transform(X)
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
        with config_context(engine_provider="sklearn_numba_dpex"):
            kmeans_with_empty_cluster.fit(X)

    kmeans_without_empty_cluster = KMeans(
        random_state=random_seed,
        algorithm="lloyd",
        n_clusters=n_clusters,
        n_init=1,
        max_iter=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with config_context(engine_provider="sklearn_numba_dpex"):
            kmeans_without_empty_cluster.fit(X)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_euclidean_distance__adapted(dtype):
    """Test adapted from sklearn's test_euclidean_distance"""
    _fail_if_no_dtype_support(pytest.xfail, dtype)

    random_seed = 42
    rng = default_rng(random_seed)
    a = rng.random(size=(1, 100), dtype=dtype)
    b = rng.standard_normal((1, 100), dtype=dtype)

    expected = np.sqrt(((a - b) ** 2).sum())

    driver = KMeansDriver()
    result = driver.get_euclidean_distances(a, b)

    rtol = 1e-4 if dtype == np.float32 else 1e-7
    assert_allclose(result, expected, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inertia__adapted(dtype):
    """Test adapted from sklearn's test_inertia"""
    _fail_if_no_dtype_support(pytest.xfail, dtype)

    random_seed = 42
    rng = default_rng(random_seed)
    X = rng.random((100, 10), dtype=dtype)
    sample_weight = rng.standard_normal(100, dtype=dtype)
    centers = rng.standard_normal((5, 10), dtype=dtype)

    driver = KMeansDriver()
    labels = driver.get_labels(X, centers)

    distances = ((X - centers[labels]) ** 2).sum(axis=1)
    expected = np.sum(distances * sample_weight)

    inertia = driver.get_inertia(X, sample_weight, centers)

    rtol = 1e-4 if dtype == np.float32 else 1e-6
    assert_allclose(inertia, expected, rtol=rtol)
