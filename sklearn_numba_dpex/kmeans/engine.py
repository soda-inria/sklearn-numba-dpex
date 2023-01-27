import contextlib
import numbers
import os
from typing import Any, Dict

import dpctl
import dpctl.tensor as dpt
import dpnp
import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.utils.validation as sklearn_validation
from sklearn.cluster._kmeans import KMeansCythonEngine
from sklearn.exceptions import NotSupportedByEngineError
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import _is_arraylike_not_scalar

from sklearn_numba_dpex.testing import override_attr_context

from .drivers import (
    get_euclidean_distances,
    get_labels_inertia,
    get_nb_distinct_clusters,
    is_same_clustering,
    kmeans_plusplus,
    lloyd,
    prepare_data_for_lloyd,
    restore_data_after_lloyd,
)


class _DeviceUnset:
    pass


class KMeansEngine(KMeansCythonEngine):
    """GPU optimized implementation of Lloyd's k-means.

    The current implementation is called "fused fixed", it consists in a sliding window
    of fixed size on the value of the centroids that work items use to accumulate the
    distances of a given sample to each centroids. It is followed, in a same kernel, by
    an update of new centroids in a context of memory privatization to avoid a too high
    cost of atomic operations.

    This class instantiates into a callable that mimics the interface of sklearn's
    private function `_kmeans_single_lloyd` .

    Notes
    -----
    The implementation has been extensively inspired by the "Fused Fixed" strategy
    exposed in [1]_, along with its reference implementatino by the same authors [2]_,
    and the reader can also refer to the complementary slide deck [3]_  with schemas
    that intuitively explain the main computation.

    .. [1] Kruliš, M., & Kratochvíl, M. (2020, August). Detailed analysis and
        optimization of CUDA K-means algorithm. In 49th International
        Conference on Parallel Processing-ICPP (pp. 1-11).

    .. [2] https://github.com/krulis-martin/cuda-kmeans

    .. [3] https://jnamaral.github.io/icpp20/slides/Krulis_Detailed.pdf

    """

    # NB: by convention, all public methods override the corresponding method from
    # `sklearn.cluster._kmeans.KMeansCythonEngine`. All methods that do not override a
    # method in `sklearn.cluster._kmeans.KMeansCythonEngine` are considered private to
    # this implementation.

    # This class attribute can alter globally the attributes `device` and `order` of
    # future instances. It is only used for testing purposes, using
    # `sklearn_numba_dpex.testing.config.override_attr_context` context, for instance
    # in the benchmark script.
    # For normal usage, the compute will follow the *compute follows data* principle.
    _CONFIG: Dict[str, Any] = dict()

    def __init__(self, estimator):
        self.device = self._CONFIG.get("device", _DeviceUnset)

        # NB: numba_dpex kernels only currently supports working with C memory layout
        # (see https://github.com/IntelPython/numba-dpex/issues/767) but our KMeans
        # implementation is hypothetized to be more efficient with the F-memory layout.
        # As a workaround the kernels work with the transpose of X, X_t, where X_t
        # is created with a C layout, which results in equivalent memory access
        # patterns than with a F layout for X.
        # TODO: when numba_dpex supports inputs with F-layout:
        # - use X rather than X_t and adapt the codebase (better for readability and
        # more consistent with sklearn notations)
        # - test the performances with both layouts and use the best performing layout.
        order = self._CONFIG.get("order", "F")
        if order != "F":
            raise ValueError(
                "Kernels compiled by numba_dpex called on an input array with "
                "the Fortran memory layout silently return incorrect results: "
                "https://github.com/IntelPython/numba-dpex/issues/767"
            )
        self.order = order
        self.estimator = estimator

        _is_in_testing_mode = os.getenv("SKLEARN_NUMBA_DPEX_TESTING_MODE", "0")
        if _is_in_testing_mode not in {"0", "1"}:
            raise ValueError(
                "If the environment variable SKLEARN_NUMBA_DPEX_TESTING_MODE is"
                f' set, it is expected to take values in {"0", "1"}, but got'
                f" {_is_in_testing_mode} instead."
            )
        self._is_in_testing_mode = _is_in_testing_mode == "1"

    def accepts(self, X, y, sample_weight):

        if (algorithm := self.estimator.algorithm) not in ("lloyd", "auto", "full"):
            if self._is_in_testing_mode:
                raise NotSupportedByEngineError(
                    "The sklearn_nunmba_dpex engine for KMeans only support the Lloyd"
                    f" algorithm, {algorithm} is not supported."
                )
            else:
                return False

        if sp.issparse(X):
            return self._is_in_testing_mode

        return True

    def prepare_fit(self, X, y=None, sample_weight=None):
        estimator = self.estimator

        X = self._validate_data(X)
        estimator._check_params_vs_input(X)

        self.sample_weight = self._check_sample_weight(sample_weight, X)

        init = self.estimator.init
        init_is_array_like = _is_arraylike_not_scalar(init)
        if init_is_array_like:
            init = self._check_init(init, X)

        (
            X_t,
            X_mean,
            self.init,
            self.tol,
            self.sample_weight_is_uniform,
        ) = prepare_data_for_lloyd(
            X.T, init, estimator.tol, self.sample_weight, estimator.copy_x
        )

        if self._is_in_testing_mode and X_mean is not None:
            X_mean = dpt.asnumpy(X_mean)

        self.X_mean = X_mean

        self.random_state = check_random_state(estimator.random_state)

        return X_t.T, y, self.sample_weight

    def unshift_centers(self, X, best_centers):
        if (X_mean := self.X_mean) is None:
            return

        if self._is_in_testing_mode:
            return super().unshift_centers(dpt.asnumpy(X), best_centers)

        restore_data_after_lloyd(X.T, best_centers.T, X_mean, self.estimator.copy_x)

    def init_centroids(self, X):
        init = self.init
        n_clusters = self.estimator.n_clusters

        if isinstance(init, dpt.usm_ndarray):
            centers_t = init

        elif isinstance(init, str) and init == "k-means++":
            centers_t, _ = self._kmeans_plusplus(X)

        elif callable(init):
            centers = init(X, self.estimator.n_clusters, random_state=self.random_state)
            centers_t = self._check_init(centers, X)

        else:
            # NB: sampling without replacement must be executed sequentially so
            # it's better done on CPU
            centers_idx = self.random_state.choice(
                X.shape[0], size=n_clusters, replace=False
            )
            # Poor man's fancy indexing
            # TODO: write a kernel ? or replace with better equivalent when available ?
            # Relevant issue: https://github.com/IntelPython/dpctl/issues/1003
            centers_t = dpt.concat(
                [dpt.expand_dims(X[center_idx], axes=1) for center_idx in centers_idx],
                axis=1,
            )

        return centers_t

    def _kmeans_plusplus(self, X):
        n_clusters = self.estimator.n_clusters

        centers_t, center_indices = kmeans_plusplus(
            X.T, self.sample_weight, n_clusters, self.random_state
        )
        return centers_t, center_indices

    def kmeans_single(self, X, sample_weight, centers_init_t):
        assignments_idx, inertia, best_centroids_t, n_iteration = lloyd(
            X.T,
            sample_weight,
            centers_init_t,
            self.sample_weight_is_uniform,
            self.estimator.max_iter,
            self.estimator.verbose,
            self.tol,
        )

        if self._is_in_testing_mode:
            # XXX: having a C-contiguous centroid array is expected in sklearn in some
            # unit test and by the cython engine.
            assignments_idx = dpt.asnumpy(assignments_idx).astype(np.int32)
            best_centroids_t = np.asfortranarray(dpt.asnumpy(best_centroids_t))

        # ???: rather that returning whatever dtype the driver returns (which might
        # depends on device support for float64), shouldn't we cast to a dtype that
        # is always consistent with the input ? (e.g. cast to float64 if the input
        # was given as float64 ?) But what assumptions can we make on the input
        # so we can infer its input dtype without risking triggering a copy of it ?
        return assignments_idx, inertia, best_centroids_t.T, n_iteration

    def is_same_clustering(self, labels, best_labels, n_clusters):
        if self._is_in_testing_mode:
            return super().is_same_clustering(labels, best_labels, n_clusters)
        return is_same_clustering(labels, best_labels, n_clusters)

    def get_nb_distinct_clusters(self, best_labels):
        if self._is_in_testing_mode:
            return super().get_nb_distinct_clusters(best_labels)
        return get_nb_distinct_clusters(best_labels, self.estimator.n_clusters)

    def prepare_prediction(self, X, sample_weight):
        X = self._validate_data(X, reset=False)
        sample_weight = self._check_sample_weight(sample_weight, X)
        return X, sample_weight

    def get_labels(self, X, sample_weight):
        # TODO: sample_weight actually not used for get_labels. Fix in sklearn ?
        # Relevant issue: https://github.com/scikit-learn/scikit-learn/issues/25066
        labels, _ = self._get_labels_inertia(X, sample_weight, with_inertia=False)
        if self._is_in_testing_mode:
            labels = dpt.asnumpy(labels).astype(np.int32)
        return labels

    def get_score(self, X, sample_weight):
        _, inertia = self._get_labels_inertia(X, sample_weight, with_inertia=True)
        return inertia

    def _get_labels_inertia(self, X, sample_weight, with_inertia=True):
        cluster_centers = self._check_init(
            self.estimator.cluster_centers_, X, copy=False
        )

        assignments_idx, inertia = get_labels_inertia(
            X.T, cluster_centers, sample_weight, with_inertia
        )

        if with_inertia:
            # inertia is a 1-sized numpy array, we transform it into a scalar:
            inertia = inertia[0]

        return assignments_idx, inertia

    def prepare_transform(self, X):
        # TODO: fix fit_transform in sklearn: need to call prepare_transform
        # inbetween fit and transform ? or remove prepare_transform ?
        return X

    def get_euclidean_distances(self, X):
        X = self._validate_data(X, reset=False)
        cluster_centers = self._check_init(
            self.estimator.cluster_centers_, X, copy=False
        )
        euclidean_distances = get_euclidean_distances(X.T, cluster_centers)
        if self._is_in_testing_mode:
            euclidean_distances = dpt.asnumpy(euclidean_distances)
        return euclidean_distances

    def _validate_data(self, X, reset=True):
        if isinstance(X, dpnp.ndarray):
            X = X.get_array()

        if self.device is not _DeviceUnset:
            device = dpctl.SyclDevice(self.device)
        elif isinstance(X, dpt.usm_ndarray):
            device = X.device.sycl_device
        else:
            device = dpctl.SyclDevice()

        accepted_dtypes = [np.float32]
        # NB: one could argue that `float32` is a better default, but sklearn defaults
        # to `np.float64` and we apply the same for consistency.
        if device.has_aspect_fp64:
            accepted_dtypes = [np.float64, np.float32]
        else:
            accepted_dtypes = [np.float32]

        with _validate_with_array_api(device):
            try:
                X = self.estimator._validate_data(
                    X,
                    accept_sparse=False,
                    dtype=accepted_dtypes,
                    order=self.order,
                    copy=False,
                    reset=reset,
                    force_all_finite=True,
                    estimator=self.estimator,
                )
                return X
            except TypeError as type_error:
                if "A sparse matrix was passed, but dense data is required" in str(
                    type_error
                ):
                    raise NotSupportedByEngineError from type_error

    def _check_sample_weight(self, sample_weight, X):
        """Adapted from sklearn.utils.validation._check_sample_weight to be compatible
        with Array API dispatch"""
        n_samples = X.shape[0]
        dtype = X.dtype
        device = X.device.sycl_device
        if sample_weight is None:
            sample_weight = dpt.ones(n_samples, dtype=dtype, device=device)
        elif isinstance(sample_weight, numbers.Number):
            sample_weight = dpt.full(
                n_samples, sample_weight, dtype=dtype, device=device
            )
        else:
            with _validate_with_array_api(device):
                sample_weight = check_array(
                    sample_weight,
                    accept_sparse=False,
                    order="C",
                    dtype=dtype,
                    force_all_finite=True,
                    ensure_2d=False,
                    allow_nd=False,
                    estimator=self.estimator,
                    input_name="sample_weight",
                )

            if sample_weight.ndim != 1:
                raise ValueError("Sample weights must be 1D array or scalar")

            if sample_weight.shape != (n_samples,):
                raise ValueError(
                    "sample_weight.shape == {}, expected {}!".format(
                        sample_weight.shape, (n_samples,)
                    )
                )

        return sample_weight

    def _check_init(self, init, X, copy=False):
        device = X.device.sycl_device
        with _validate_with_array_api(device):
            init = check_array(
                init,
                dtype=X.dtype,
                accept_sparse=False,
                copy=False,
                order=self.order,
                force_all_finite=True,
                ensure_2d=True,
                estimator=self.estimator,
                input_name="init",
            )
            self.estimator._validate_center_shape(X, init)
            init_t = dpt.asarray(init.T, order="C", copy=False, device=device)
            return init_t


def _get_namespace(*arrays):
    return dpt, True


@contextlib.contextmanager
def _validate_with_array_api(device):
    def _asarray_with_order(array, dtype, order, copy=None, xp=None):
        return dpt.asarray(array, dtype=dtype, order=order, copy=copy, device=device)

    # TODO: when https://github.com/IntelPython/dpctl/issues/997 and
    # https://github.com/scikit-learn/scikit-learn/issues/25000 and are solved
    # remove those hacks.
    with sklearn.config_context(
        array_api_dispatch=True,
        assume_finite=True  # workaround 1: disable force_all_finite
        # workaround 2: monkey patch get_namespace and _asarray_with_order to force
        # dpctl.tensor array namespace
    ), override_attr_context(
        sklearn_validation,
        get_namespace=_get_namespace,
        _asarray_with_order=_asarray_with_order,
    ):
        yield
