import warnings
from typing import Any, Dict

import dpctl
import dpctl.tensor as dpt
import numpy as np
from sklearn.cluster._kmeans import KMeansCythonEngine
from sklearn.exceptions import DataConversionWarning, NotSupportedByEngineError

from .drivers import get_euclidean_distances, get_labels_inertia, kmeans_plusplus, lloyd


class _IgnoreSampleWeight:
    pass


# At the moment not all steps are implemented with numba_dpex, we inherit missing steps
# from the default sklearn KMeansCythonEngine for convenience, this inheritance will be
# removed later on when the other parts have been implemented.
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

    _CONFIG: Dict[str, Any] = dict()

    def __init__(self, estimator):
        self.device = dpctl.SyclDevice(self._CONFIG.get("device"))
        super().__init__(estimator)

    def prepare_fit(self, X, y=None, sample_weight=None):
        estimator = self.estimator
        try:
            # This pass of data validation only aims at detecting input types that are
            # supported by the sklearn engine but not by KMeansEngine. For those inputs
            # we raise a NotSupportedByEngineError exception.
            # estimator._validate_data is called again in super().prepare_fit later on
            # and will raise ValueError or TypeError for data that is not even
            # compatible with the sklearn engine.
            estimator._validate_data(
                X,
                accept_sparse=False,
                dtype=None,
                force_all_finite=False,
                ensure_2d=False,
                allow_nd=True,
                ensure_min_samples=0,
                ensure_min_features=0,
                estimator=estimator,
            )
        except Exception as e:
            raise NotSupportedByEngineError(
                "The sklearn_nunmba_dpex engine for KMeans does not support the format"
                " of the inputed data."
            ) from e

        algorithm = estimator.algorithm
        if algorithm not in ("lloyd", "auto", "full"):
            raise NotSupportedByEngineError(
                "The sklearn_nunmba_dpex engine for KMeans only support the Lloyd"
                f" algorithm, {algorithm} is not supported."
            )

        self.sample_weight = sample_weight

        return super().prepare_fit(X, y, sample_weight)

    def init_centroids(self, X):
        init = self.init

        if isinstance(init, str) and init == "k-means++":
            centers, _ = self._kmeans_plusplus(X)

        else:
            centers = self.estimator._init_centroids(
                X,
                x_squared_norms=self.x_squared_norms,
                init=init,
                random_state=self.random_state,
            )
        return centers

    def _kmeans_plusplus(self, X):
        n_clusters = self.estimator.n_clusters

        if X.shape[0] < n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={n_clusters}."
            )

        X, sample_weight, _, output_dtype = self._check_inputs(
            X, self.sample_weight, cluster_centers=None
        )

        X_t, sample_weight, _ = self._load_transposed_data_to_device(
            X, sample_weight, cluster_centers=None
        )

        centers, center_indices = kmeans_plusplus(
            X_t, sample_weight, n_clusters, self.random_state
        )

        centers = dpt.asnumpy(centers).astype(output_dtype, copy=False)
        center_indices = dpt.asnumpy(center_indices)
        return centers, center_indices

    def kmeans_single(self, X, sample_weight, centers_init):
        X, sample_weight, cluster_centers, output_dtype = self._check_inputs(
            X, sample_weight, centers_init
        )

        use_uniform_weights = (sample_weight == sample_weight[0]).all()

        X_t, sample_weight, centroids_t = self._load_transposed_data_to_device(
            X, sample_weight, cluster_centers
        )

        assignments_idx, inertia, best_centroids, n_iteration = lloyd(
            X_t,
            sample_weight,
            centroids_t,
            use_uniform_weights,
            self.estimator.max_iter,
            self.estimator.verbose,
            self.tol,
        )

        # TODO: explore leveraging dpnp to benefit from USM to avoid moving centroids
        # back and forth between device and host memory in case a subsequent `.predict`
        # call is requested on the same GPU later.
        return (
            dpt.asnumpy(assignments_idx).astype(np.int32, copy=False),
            inertia,
            # XXX: having a C-contiguous centroid array is expected in sklearn in some
            # unit test and by the cython engine.
            np.ascontiguousarray(
                dpt.asnumpy(best_centroids).astype(output_dtype, copy=False)
            ),
            n_iteration,
        )

    def get_labels(self, X, sample_weight):
        labels, _ = self._get_labels_inertia(X, with_inertia=False)
        return dpt.asnumpy(labels).astype(np.int32, copy=False)

    def get_score(self, X, sample_weight):
        _, inertia = self._get_labels_inertia(X, sample_weight, with_inertia=True)
        return inertia

    def _get_labels_inertia(
        self, X, sample_weight=_IgnoreSampleWeight, with_inertia=True
    ):
        X, sample_weight, centers, output_dtype = self._check_inputs(
            X,
            sample_weight=sample_weight,
            cluster_centers=self.estimator.cluster_centers_,
        )

        if sample_weight is _IgnoreSampleWeight:
            sample_weight = None

        X_t, sample_weight, centroids_t = self._load_transposed_data_to_device(
            X, sample_weight, centers
        )

        assignments_idx, inertia = get_labels_inertia(
            X_t, centroids_t, sample_weight, with_inertia
        )

        if with_inertia:
            # inertia is a 1-sized numpy array, we transform it into a scalar:
            inertia = inertia.astype(output_dtype)[0]

        return assignments_idx, inertia

    def get_euclidean_distances(self, X):
        X, _, Y, output_dtype = self._check_inputs(
            X,
            sample_weight=_IgnoreSampleWeight,
            cluster_centers=self.estimator.cluster_centers_,
        )

        X_t, _, Y_t = self._load_transposed_data_to_device(X, None, Y)

        euclidean_distances = get_euclidean_distances(X_t, Y_t)

        return dpt.asnumpy(euclidean_distances).astype(output_dtype, copy=False)

    def _check_inputs(self, X, sample_weight, cluster_centers):

        if sample_weight is None:
            sample_weight = np.ones(len(X), dtype=X.dtype)

        X, sample_weight, cluster_centers, output_dtype = self._set_dtype(
            X, sample_weight, cluster_centers
        )

        return X, sample_weight, cluster_centers, output_dtype

    def _set_dtype(self, X, sample_weight, cluster_centers):
        output_dtype = compute_dtype = np.dtype(X.dtype).type
        copy = True
        if (compute_dtype != np.float32) and (compute_dtype != np.float64):
            text = (
                f"KMeans has been set to compute with type {compute_dtype} but only "
                "the types float32 and float64 are supported. The computations and "
                "outputs will default back to float32 type."
            )
            output_dtype = compute_dtype = np.float32
        elif (compute_dtype == np.float64) and not self.device.has_aspect_fp64:
            text = (
                f"KMeans is set to compute with type {compute_dtype} but this type is "
                f"not supported by the device {self.device.name}. The computations "
                "will default back to float32 type."
            )
            compute_dtype = np.float32

        else:
            copy = False

        if copy:
            text += (
                f" A copy of the data casted to type {compute_dtype} will be created. "
                "To save memory and suppress this warning, ensure that the dtype of "
                "the input data matches the dtype required for computations."
            )
            warnings.warn(text, DataConversionWarning)
            # TODO: instead of triggering a copy on the host side, we could use the
            # dtype to allocate a shared USM buffer and fill it with casted values from
            # X. In this case we should only warn when:
            #     (dtype == np.float64) and not self.has_aspect_fp64
            # The other cases would not trigger any additional memory copies.
            X = X.astype(compute_dtype)

        if cluster_centers is not None and (
            (cluster_centers_dtype := cluster_centers.dtype) != compute_dtype
        ):
            warnings.warn(
                f"The centers have been passed with type {cluster_centers_dtype} but "
                f"type {compute_dtype} is expected. A copy will be created with the "
                f"correct type {compute_dtype}. Ensure that the centers are passed "
                "with the correct dtype to save memory and suppress this warning.",
                DataConversionWarning,
            )
            cluster_centers = cluster_centers.astype(compute_dtype)

        if (sample_weight is not _IgnoreSampleWeight) and (
            sample_weight.dtype != compute_dtype
        ):
            warnings.warn(
                f"sample_weight has been passed with type {sample_weight.dtype} but "
                f"type {compute_dtype} is expected. A copy will be created with the "
                f"correct type {compute_dtype}. Ensure that sample_weight is passed "
                "with the correct dtype to save memory and suppress this warning.",
                DataConversionWarning,
            )
            sample_weight = sample_weight.astype(compute_dtype)

        return X, sample_weight, cluster_centers, output_dtype

    def _load_transposed_data_to_device(self, X, sample_weight, cluster_centers):
        # Transfer the input data to device memory,
        # TODO: let the user pass directly dpt or dpnp arrays to avoid copies.

        # NB: numba_dpex kernels only currently supports inputs with a C memory layout
        # (see https://github.com/IntelPython/numba-dpex/issues/767) but our KMeans
        # implementation is hypothetized to be more efficient with the F-memory layout.
        # As a workaround the kernels work with the transpose of X, X_t, where X_t
        # is created with a C layout, which results in equivalent memory access
        # patterns than with a F layout for X.
        # TODO: when numba_dpex supports inputs with F-layout:
        # - use X rather than X_t and adapt the codebase (better for readability and
        # more consistent with sklearn notations)
        # - test the performances with both layouts and use the best performing layout.

        X_t = dpt.asarray(X.T, order="C", device=self.device)
        assert (
            X_t.strides[1] == 1
        )  # C memory layout, equivalent to Fortran layout on transposed

        if sample_weight is not None:
            sample_weight = dpt.from_numpy(sample_weight, device=self.device)

        if cluster_centers is not None:
            cluster_centers = dpt.from_numpy(cluster_centers.T, device=self.device)

        return X_t, sample_weight, cluster_centers
