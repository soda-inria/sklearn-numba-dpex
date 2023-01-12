import subprocess

import dpctl.tensor as dpt
import numba_dpex as dpex
import numpy as np
import pytest
from numba.core.errors import LoweringError
from numba_dpex.config import _dpctl_has_non_host_device
from sklearn import config_context
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn_numba_dpex.common._utils import _force_reload_numba_dpex_with_patches


def test_init_config():
    """If this test fail, it means that `numba_dpex` has been bumped to a version where
    its import issues related to `dpctl.SyclDevice.is_host` attribute has been fixed.
    It's a reminder that the minimum required version of `numba_dpex` can be bumped to
    the first version that fixed the issue, and this test can be removed, along with the
    corresponding fixes in
    `sklearn_numba_dpex.common._utils._force_reload_numba_dpex_with_patches`.
    """
    with pytest.raises(AttributeError):
        _dpctl_has_non_host_device()


def test_regression_fix():
    """if this test fails, it means that the bug reported at
    https://github.com/IntelPython/numba-dpex/issues/867 has been fixed, and the
    workarounds introduced in https://github.com/soda-inria/sklearn-numba-dpex/pull/82
    can be removed.
    """

    @dpex.func
    def g(array_in, idx, const):
        array_in[idx] = const

    @dpex.kernel
    def kernel_a(array_in):
        idx = dpex.get_global_id(0)
        g(array_in, idx, np.int64(0))

    @dpex.kernel
    def kernel_b(array_in):
        idx = dpex.get_global_id(0)
        g(
            array_in, idx, np.int32(0)
        )  # NB: call with inputs of different types than in kernel_a

    dtype = np.float32
    size = 16
    array_in = dpt.zeros(sh=(size,), dtype=dtype)

    kernel_a[size, size](array_in)
    with pytest.raises(LoweringError):
        kernel_b[size, size](array_in)


def test_spirv_fix():
    """if this test fails, it means that the bug reported at
    https://github.com/IntelPython/numba-dpex/issues/868 has been fixed, and the
    workarounds in
    `sklearn_numba_dpex.common._utils._force_reload_numba_dpex_with_patches` can
    be reverted.
    """

    random_seed = 42
    dtype = np.float32
    X, _ = make_blobs(random_state=random_seed)
    X = X.astype(dtype)
    X_array = np.asarray(X, dtype=np.float32)

    kmeans = KMeans(
        random_state=random_seed, algorithm="lloyd", max_iter=2, n_init=2, init="random"
    )

    with config_context(engine_provider="sklearn_numba_dpex"):
        try:
            _force_reload_numba_dpex_with_patches(with_spirv_fix=False)
            with pytest.raises(subprocess.CalledProcessError):
                kmeans.fit(X_array)
        finally:
            _force_reload_numba_dpex_with_patches(with_spirv_fix=True)
