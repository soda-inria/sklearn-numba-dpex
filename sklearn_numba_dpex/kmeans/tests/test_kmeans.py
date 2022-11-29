import dpctl.tensor as dpt
import dpnp
import numpy as np
import pytest
from numpy.random import default_rng
from numpy.testing import assert_array_equal
from sklearn import config_context
from sklearn.base import clone
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.cluster.tests.test_k_means import X as X_sklearn_test
from sklearn.cluster.tests.test_k_means import n_clusters as n_clusters_sklearn_test
from sklearn.datasets import make_blobs
from sklearn.utils._testing import assert_allclose

from sklearn_numba_dpex.kmeans.engine import KMeansEngine
from sklearn_numba_dpex.kmeans.kernels import (
    make_compute_euclidean_distances_fixed_window_kernel,
    make_label_assignment_fixed_window_kernel,
    make_lloyd_single_step_fixed_window_kernel,
)
from sklearn_numba_dpex.kmeans.kernels.utils import (
    make_select_samples_far_from_centroid_kernel,
)
from sklearn_numba_dpex.testing.config import float_dtype_params


def test_dpnp_implements_argpartition():
    # Detect if dpnp exposes an argpartition kernel and alert that sklearn-numba-dpex
    # can be adapted accordingly.
    assert not hasattr(dpnp, "argpartition"), (
        "An argpartition function is now available in dpnp, it should now be used in "
        "place of dpnp.partition in sklearn_numba_dpex.kmeans.drivers ."
    )


@pytest.mark.parametrize(
    "array_constr",
    [np.asarray, dpt.asarray, dpnp.asarray],
    ids=["numpy", "dpctl", "dpnp"],
)
@pytest.mark.parametrize("dtype", float_dtype_params)
def test_kmeans_same_results(dtype, array_constr):
    random_seed = 42
    X, _ = make_blobs(random_state=random_seed)
    X = X.astype(dtype)
    X_array = array_constr(X, dtype=dtype)

    kmeans_vanilla = KMeans(
        random_state=random_seed, algorithm="lloyd", max_iter=1, init="random"
    )
    kmeans_engine = clone(kmeans_vanilla)

    # Fit a reference model with the default scikit-learn engine:
    kmeans_vanilla.fit(X)

    with config_context(engine_provider="sklearn_numba_dpex"):
        kmeans_engine.fit(X_array)

    # ensure same results
    assert_array_equal(kmeans_vanilla.labels_, kmeans_engine.labels_)
    assert_allclose(kmeans_vanilla.cluster_centers_, kmeans_engine.cluster_centers_)
    assert_allclose(kmeans_vanilla.inertia_, kmeans_engine.inertia_)

    # test fit_predict
    y_labels = kmeans_vanilla.fit_predict(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_labels_engine = kmeans_engine.fit_predict(X_array)
    assert_array_equal(y_labels, y_labels_engine)
    assert_array_equal(kmeans_vanilla.labels_, kmeans_engine.labels_)
    assert_allclose(kmeans_vanilla.cluster_centers_, kmeans_engine.cluster_centers_)
    assert_allclose(kmeans_vanilla.inertia_, kmeans_engine.inertia_)

    # test fit_transform
    y_transform = kmeans_vanilla.fit_transform(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_transform_engine = kmeans_engine.fit_transform(X_array)
    assert_allclose(y_transform, y_transform_engine)
    assert_array_equal(kmeans_vanilla.labels_, kmeans_engine.labels_)
    assert_allclose(kmeans_vanilla.cluster_centers_, kmeans_engine.cluster_centers_)
    assert_allclose(kmeans_vanilla.inertia_, kmeans_engine.inertia_)

    # # test predict method (returns labels)
    y_labels = kmeans_vanilla.predict(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_labels_engine = kmeans_engine.predict(X_array)
    assert_array_equal(y_labels, y_labels_engine)

    # test score method (returns negative inertia for each sample)
    y_scores = kmeans_vanilla.score(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_scores_engine = kmeans_engine.score(X_array)
    assert_allclose(y_scores, y_scores_engine)

    # test transform method (returns euclidean distances)
    y_transform = kmeans_vanilla.transform(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_transform_engine = kmeans_engine.transform(X_array)
    assert_allclose(y_transform, y_transform_engine)


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_kmeans_relocated_clusters(dtype):
    """Copied and adapted from sklearn's test_kmeans_relocated_clusters"""

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


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_euclidean_distance(dtype):
    """Test adapted from sklearn's test_euclidean_distance"""

    random_seed = 42
    rng = default_rng(random_seed)
    a = rng.random(size=(1, 100), dtype=dtype)
    b = rng.standard_normal((1, 100), dtype=dtype)

    expected = np.sqrt(((a - b) ** 2).sum())

    estimator = KMeans(n_clusters=len(b))
    estimator.cluster_centers_ = b
    engine = KMeansEngine(estimator)

    result = engine.get_euclidean_distances(a)

    rtol = 1e-4 if dtype == np.float32 else 1e-7
    assert_allclose(result, expected, rtol=rtol)


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_inertia(dtype):
    """Test adapted from sklearn's test_inertia"""

    random_seed = 42
    rng = default_rng(random_seed)
    X = rng.random((100, 10), dtype=dtype)
    sample_weight = rng.standard_normal(100, dtype=dtype)
    centers = rng.standard_normal((5, 10), dtype=dtype)

    estimator = KMeans(n_clusters=len(centers))
    estimator.cluster_centers_ = centers
    engine = KMeansEngine(estimator)
    X_prepared, sample_weight_prepared = engine.prepare_prediction(X, sample_weight)
    labels = engine.get_labels(X_prepared, sample_weight_prepared)

    distances = ((X - centers[labels]) ** 2).sum(axis=1)
    expected = np.sum(distances * sample_weight)

    inertia = engine.get_score(X_prepared, sample_weight_prepared)

    rtol = 1e-4 if dtype == np.float32 else 1e-6
    assert_allclose(inertia, expected, rtol=rtol)


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_relocate_empty_clusters(dtype):
    """Copied and adapted from sklearn's test_relocate_empty_clusters"""

    # Synthetic dataset with 3 obvious clusters of different sizes
    X = np.array(
        [-10.0, -9.5, -9.0, -8.5, -8.0, -1.0, 1.0, 9.0, 9.5, 10.0],
        dtype=dtype,
    ).reshape(-1, 1)
    # centers all initialized to the first point of X
    # With this initialization, all points will be assigned to the first center
    init_centers = np.array([-10.0, -10.0, -10.0]).reshape(-1, 1)

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
    assert_array_equal(kmeans_vanilla.labels_, expected_labels)
    assert_allclose(kmeans_vanilla.cluster_centers_, kmeans_engine.cluster_centers_)
    assert_allclose(kmeans_vanilla.inertia_, kmeans_engine.inertia_)


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_select_samples_far_from_centroid_kernel(dtype):

    # The following array, when sorted, reads:
    # [-1.0, 9.0, 9.0, 20.22, 20.22, 20.22, 20.22, 23.0, 23.0, 30.0]
    # Assuming n_selected == 5, the threshold is 20.22 and we expect 3 values
    # above threshold (23.0, 23.0, 30.0) and 2 values equal to threshold.
    distance_to_centroid = dpt.from_numpy(
        np.array(
            [
                20.22,  # == threshold
                20.22,
                -1.0,
                23.0,  # idx = 3
                20.22,
                9.0,
                20.22,
                30.0,  # idx = 7
                9.0,
                23.0,  # idx = 9
            ],
            dtype=dtype,
        )
    )

    # NB: we wrap everything in dpctl tensors and select the threshold with __getitem__
    # on a dpt because loading of numpy arrays to and from dpt arrays
    # seems to create numerical errors that break the kernel
    # (the condition ==threshold stops being evaluated properly) when using float64
    # precision ?
    # TODO: write a minimal reproducer and open an issue if confirmed
    threshold = distance_to_centroid[0:1]  # == 20.22

    n_selected = 5
    n_samples = len(distance_to_centroid)
    work_group_size = 4
    select_samples_far_from_centroid_kernel = (
        make_select_samples_far_from_centroid_kernel(
            n_selected, n_samples, work_group_size
        )
    )

    # NB: the values used to initialize the output array do not matter, 100 is chosen
    # here for readability, but `dpctl.empty` is also possible.
    selected_samples_idx = (dpt.ones(sh=10, dtype=np.int32) * 100).get_array()

    n_selected_gt_threshold = dpt.zeros(sh=1, dtype=np.int32)
    n_selected_eq_threshold = dpt.ones(sh=1, dtype=np.int32)
    select_samples_far_from_centroid_kernel(
        distance_to_centroid,
        threshold,
        selected_samples_idx,
        n_selected_gt_threshold,
        n_selected_eq_threshold,
    )

    # NB: the variable n_selected_eq_threshold is always one unit above the true value
    # It is only used as an intermediary variable in the kernel and is not used
    # otherwise.
    n_selected_eq_threshold = int(n_selected_eq_threshold[0] - 1)
    n_selected_gt_threshold = int(n_selected_gt_threshold[0])
    selected_samples_idx = dpt.asnumpy(selected_samples_idx)

    # NB: the exact number of selected values equal to the threshold and the
    # corresponding selected indexes in the input array can change depending on
    # concurrency. We only check conditions for success that do not depend on the
    # concurrency induced non-determinism.
    assert n_selected_gt_threshold == 3
    assert n_selected_eq_threshold >= 2

    expected_gt_threshold_indices = {3, 7, 9}
    actual_gt_threshold_indices = set(
        selected_samples_idx[:n_selected_gt_threshold]
    )  # stored at the beginning
    assert actual_gt_threshold_indices == expected_gt_threshold_indices

    expected_eq_threshold_indices = {0, 1, 4, 6}
    actual_eq_threshold_indices = set(
        selected_samples_idx[-n_selected_eq_threshold:]
    )  # stored at the end
    assert actual_eq_threshold_indices.issubset(expected_eq_threshold_indices)

    assert (
        selected_samples_idx[n_selected_gt_threshold:-n_selected_eq_threshold] == 100
    ).all()


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_kmeans_plusplus_same_quality(dtype):
    X = X_sklearn_test.astype(dtype)
    n_clusters = n_clusters_sklearn_test

    # HACK: to compare the quality of the initialization, it's convenient to use the
    # `score` method of an estimator whose fitted attribute cluster_centers_ has been
    # set to the result of the initialization, without running the remaining steps of
    # the  KMeans algorithm. For this purpose, since KMeans does not support passing
    # `max_iter=0` we forcefully set fitted attribute values without actually running
    # `fit`.
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
    )
    kmeans._n_threads = 1

    def _get_score_with_centers(centers):
        kmeans.cluster_centers_ = np.ascontiguousarray(centers)
        return kmeans.score(X)

    scores_vanilla_kmeans_plusplus = []
    scores_engine_kmeans_plusplus = []
    scores_random_init = []

    for random_state in range(10):
        random_centers = np.random.default_rng(random_state).choice(
            X, size=n_clusters, replace=False
        )
        scores_random_init.append(_get_score_with_centers(random_centers))

        vanilla_kmeans_plusplus_centers, _ = kmeans_plusplus(
            X, n_clusters, random_state=random_state
        )
        scores_vanilla_kmeans_plusplus.append(
            _get_score_with_centers(vanilla_kmeans_plusplus_centers)
        )

        kmeans.set_params(random_state=random_state)
        engine = KMeansEngine(kmeans)
        X_prepared, *_ = engine.prepare_fit(X)
        engine_kmeans_plusplus_centers_t = engine.init_centroids(X_prepared)
        engine_kmeans_plusplus_centers = dpt.asnumpy(engine_kmeans_plusplus_centers_t.T)
        engine.unshift_centers(X_prepared, engine_kmeans_plusplus_centers)
        scores_engine_kmeans_plusplus.append(
            _get_score_with_centers(engine_kmeans_plusplus_centers)
        )

    # Those results confirm that both sklearn KMeans++ and ours have similar quality,
    # and are both very significantly better than random init.
    #
    # NB: the gap between scores_vanilla_kmeans_plusplus and
    # scores_engine_kmeans_plusplus goes away with more iterations in the previous
    # loop. E.g., for 200 iterations with dtype float32:
    #
    # [
    #     -1786.16057,     # np.mean(scores_random_init)
    #     -886.220595,     # np.mean(scores_vanilla_kmeans_plusplus)
    #     -876.56282806,   # np.mean(scores_engine_kmeans_plusplus)
    # ]

    assert_allclose(
        [
            np.mean(scores_random_init),
            np.mean(scores_vanilla_kmeans_plusplus),
            np.mean(scores_engine_kmeans_plusplus),
        ],
        [
            -1827.22702,
            -1027.674243,
            -865.257501,
        ],
    )


@pytest.mark.parametrize("dtype", float_dtype_params)
@pytest.mark.parametrize(
    "array_constr",
    [np.asarray, dpt.asarray, dpnp.asarray],
    ids=["numpy", "dpctl", "dpnp"],
)
def test_kmeans_plusplus_output(array_constr, dtype):
    """Test adapted from sklearn's test_kmeans_plusplus_output"""
    random_state = 42

    # Check for the correct number of seeds and all positive values
    X = array_constr(X_sklearn_test, dtype=dtype)

    sample_weight = default_rng(random_state).random(X.shape[0], dtype=dtype)

    estimator = KMeans(
        init="k-means++", n_clusters=n_clusters_sklearn_test, random_state=random_state
    )
    engine = KMeansEngine(estimator)
    X_prepared, *_ = engine.prepare_fit(X, sample_weight=sample_weight)

    centers_t, indices = engine._kmeans_plusplus(X_prepared)
    centers = dpt.asnumpy(centers_t.T)
    engine.unshift_centers(X_prepared, centers)
    indices = dpt.asnumpy(indices)

    # Check there are the correct number of indices and that all indices are
    # positive and within the number of samples
    assert indices.shape[0] == n_clusters_sklearn_test
    assert (indices >= 0).all()
    assert (indices <= X.shape[0]).all()

    # Check for the correct number of seeds and that they are bound by the data
    assert centers.shape[0] == n_clusters_sklearn_test
    assert (centers.max(axis=0) <= X_sklearn_test.max(axis=0)).all()
    assert (centers.min(axis=0) >= X_sklearn_test.min(axis=0)).all()
    # NB: dtype can change depending on the device, so we accept all valid dtypes.
    assert centers.dtype.type in {np.float32, np.float64}

    # Check that indices correspond to reported centers
    assert_allclose(X_sklearn_test[indices].astype(dtype), centers)


def test_kmeans_plusplus_dataorder():
    """Test adapted from sklearn's test_kmeans_plusplus_dataorder"""
    # Check that memory layout does not effect result
    random_state = 42

    estimator = KMeans(
        init="k-means++", n_clusters=n_clusters_sklearn_test, random_state=random_state
    )
    engine = KMeansEngine(estimator)
    X_sklearn_test_prepared, *_ = engine.prepare_fit(X_sklearn_test)
    centers_c = engine.init_centroids(X_sklearn_test_prepared)
    centers_c = dpt.asnumpy(centers_c.T)

    X_fortran = np.asfortranarray(X_sklearn_test)
    # The engine is re-created to reset random state
    engine = KMeansEngine(estimator)
    X_fortran_prepared, *_ = engine.prepare_fit(X_fortran)
    centers_fortran = engine.init_centroids(X_fortran_prepared)
    centers_fortran = dpt.asnumpy(centers_fortran.T)

    assert_allclose(centers_c, centers_fortran)


def test_error_raised_on_invalid_group_sizes():
    n_samples = 10
    n_features = 2
    n_clusters = 2
    sub_group_size = 64
    work_group_size = 500  # invalid because is not a multiple of sub_group_size
    dtype = np.float32

    expected_msg = (
        "Expected work_group_size to be a multiple of sub_group_size but got "
        f"sub_group_size={sub_group_size} and work_group_size={work_group_size}"
    )

    with pytest.raises(ValueError, match=expected_msg):
        make_compute_euclidean_distances_fixed_window_kernel(
            n_samples, n_features, n_clusters, sub_group_size, work_group_size, dtype
        )

    with pytest.raises(ValueError, match=expected_msg):
        make_label_assignment_fixed_window_kernel(
            n_samples, n_features, n_clusters, sub_group_size, work_group_size, dtype
        )

    return_assignments = False
    global_mem_cache_size = 123456789
    centroids_private_copies_max_cache_occupancy = 0.7

    with pytest.raises(ValueError, match=expected_msg):
        make_lloyd_single_step_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            return_assignments,
            sub_group_size,
            global_mem_cache_size,
            centroids_private_copies_max_cache_occupancy,
            work_group_size,
            dtype,
        )
