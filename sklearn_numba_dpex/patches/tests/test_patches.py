import subprocess

import dpctl
import dpctl.tensor as dpt
import numba_dpex as dpex
import numpy as np
import pytest
from numba.core.errors import LoweringError
from numba_dpex.config import _dpctl_has_non_host_device
from sklearn import config_context
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn_numba_dpex.kmeans.kernels.lloyd_single_step import (
    make_lloyd_single_step_fixed_window_kernel,
)
from sklearn_numba_dpex.patches.load_numba_dpex import _load_numba_dpex_with_patches


def test_init_config():
    """If this test fails, it means that `numba_dpex` has been bumped to a version
    where its import issues related to `dpctl.SyclDevice.is_host` attribute has been
    fixed. It's a reminder that the minimum required version of `numba_dpex` can be
    bumped to the first version that fixed the issue, and this test can be removed,
    along with the corresponding fixes in
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
            make_lloyd_single_step_fixed_window_kernel.cache_clear()
            _load_numba_dpex_with_patches(with_spirv_fix=False, with_compiler_fix=False)
            with pytest.raises(subprocess.CalledProcessError):
                kmeans.fit(X_array)
        finally:
            _load_numba_dpex_with_patches()


def test_need_to_workaround_numba_dpex_906():
    """This test will raise when all hacks tagged with HACK 906 can be reverted.

    The hack is used several time in the codebase to work around a bug in the JIT
    compiler that affects sequences of instructions containing a conditional write
    operation in an array followed by a barrier.

    For kernels that contain such patterns, the output is sometimes wrong. See
    https://github.com/IntelPython/numba-dpex/issues/906 for more information and
    updates on the issue resolution.

    The hack consist in wrapping instructions that are suspected of triggering the
    bug (basically all write operations in kernels that also contain a barrier) in
    `dpex.func` device functions.

    This hack makes the code significantly harder to read and should be reverted ASAP.
    """

    dtype = np.float32

    @dpex.kernel
    def kernel(result):
        local_idx = dpex.get_local_id(0)
        local_values = dpex.local.array((1,), dtype=dtype)

        dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

        if local_idx < 1:
            local_values[0] = 1

        dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

        if local_idx < 1:
            result[0] = 10

    result = dpt.zeros(sh=(1), dtype=dtype, device=dpctl.SyclDevice("cpu"))
    kernel[32, 32](result)

    rationale = """If this test fails, it means that the bug reported at
    https://github.com/IntelPython/numba-dpex/issues/906 has been fixed, and all the
    hacks tags with `# HACK 906` that were used to work around it can now be removed.
    This test can also be removed.
    """

    assert dpt.asnumpy(result)[0] != 10, rationale

    # Test that highlight how the hack works
    @dpex.kernel
    def kernel(result):
        local_idx = dpex.get_local_id(0)
        local_values = dpex.local.array((1,), dtype=dtype)

        _local_setitem_if((local_idx < 1), 0, 1, local_values)

        dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

        _global_setitem_if((local_idx < 1), 0, 10, result)

    # HACK: must define twice to work around the bug highlighted in test_regression_fix
    _local_setitem_if = make_setitem_if_kernel_func()
    _global_setitem_if = make_setitem_if_kernel_func()

    result = dpt.zeros(sh=(1), dtype=dtype, device=dpctl.SyclDevice("cpu"))
    kernel[32, 32](result)

    assert dpt.asnumpy(result)[0] == 10


# HACK 906: see sklearn_numba_dpex.patches.tests.test_patches.test_need_to_workaround_numba_dpex_906 # noqa
def make_setitem_if_kernel_func():
    @dpex.func
    def _setitem_if(condition, index, value, array):
        if condition:
            array[index] = value
        return condition

    return _setitem_if
