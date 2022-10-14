import numpy as np
import dpnp
import pytest
import dpctl
from numpy.testing import assert_array_equal
from sklearn import config_context
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.utils._testing import assert_allclose


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


def test_dpnp_implements_argpartition():
    # Detect if dpnp exposes an argpartition kernel and alert that sklearn-numba-dpex
    # can be adapted accordingly.
    assert not hasattr(dpnp, "argpartition"), (
        "An argpartition function is now available in dpnp, it should now be used in "
        "place of dpnp.partition in sklearn_numba_dpex.kmeans.drivers ."
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
def test_kmeans_relocated_clusters__adapted(dtype):
    """Copied and adapted from sklearn's test_kmeans_relocated_clusters"""
    _fail_if_no_dtype_support(pytest.xfail, dtype)

    # check that empty clusters are relocated as expected
    X = np.array([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=dtype)

    # second center too far from others points will be empty at first iter
    init_centers = np.array([[0.5, 0.5], [3, 3]], dtype=dtype)

    kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers)
    with config_context(engine_provider="sklearn_numba_dpex"):
        kmeans.fit(X)

    expected_n_iter = 3
    expected_inertia = 0.25
    assert_allclose(kmeans.inertia_, expected_inertia)
    assert kmeans.n_iter_ == expected_n_iter

    # There are two acceptable ways of relocating clusters in this example, the output
    # depends on how the argpartition strategy break ties. It might not be deterministic
    # (might depend on thread concurrency) so we accept both outputs.
    try:
        expected_labels = [0, 0, 1, 1]
        expected_centers = [[0.25, 0], [0.75, 1]]
        assert_array_equal(kmeans.labels_, expected_labels)
        assert_allclose(kmeans.cluster_centers_, expected_centers)
    except AssertionError:
        expected_labels = [1, 1, 0, 0]
        expected_centers = [[0.75, 1.0], [0.25, 0.0]]
        assert_array_equal(kmeans.labels_, expected_labels)
        assert_allclose(kmeans.cluster_centers_, expected_centers)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_relocate_empty_clusters__adapted(dtype):
    """Copied and adapted from sklearn's test_relocate_empty_clusters"""
    _fail_if_no_dtype_support(pytest.xfail, dtype)

    # Synthetic dataset with 3 obvious clusters of different sizes
    X = np.array([-10.0, -9.5, -9, -8.5, -8, -1, 1, 9, 9.5, 10], dtype=dtype).reshape(
        -1, 1
    )
    # centers all initialized to the first point of X
    # With this initialization, all points will be assigned to the first center
    init_centers = np.array([-10.0, -10, -10]).reshape(-1, 1)

    kmeans_vanilla = KMeans(
        n_clusters=3, n_init=1, max_iter=1, init=init_centers, algorithm="lloyd"
    )
    kmeans_engine = clone(kmeans_vanilla)

    kmeans_vanilla.fit(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        kmeans_engine.fit(X)

    expected_n_iter = 1
    expected_labels = [0, 0, 0, 0, 0, 0, 0, 2, 2, 1]
    assert kmeans_vanilla.n_iter_ == expected_n_iter
    assert kmeans_engine.n_iter_ == expected_n_iter
    assert_array_equal(kmeans_vanilla.labels_, kmeans_engine.labels_)
    assert kmeans_vanilla.labels_ == expected_labels
    assert_allclose(kmeans_vanilla.cluster_centers_, kmeans_engine.cluster_centers_)
    assert_allclose(kmeans_vanilla.inertia_, kmeans_engine.inertia_)
