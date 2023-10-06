import dpctl
import dpctl.tensor as dpt
import numba_dpex as dpex
import numpy as np
import pytest


# TODO: remove this test after going through the code base and reverting unnecessary
# additions that are tagged with "HACK 906"
@pytest.mark.xfail(
    reason=(
        "The issue has now been fixed upstream, the test should now be removed and all "
        "workarounds that have been added to `sklearn_numba_dpex` can now be reverted."
    )
)
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

        if local_idx < 1:
            local_values[0] = 1

        dpex.barrier(dpex.LOCAL_MEM_FENCE)

        if local_idx < 1:
            result[0] = 10

    result = dpt.zeros((1), dtype=dtype, device=dpctl.SyclDevice("cpu"))
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

        _setitem_if((local_idx < 1), 0, 1, local_values)

        dpex.barrier(dpex.LOCAL_MEM_FENCE)

        _setitem_if((local_idx < 1), 0, 10, result)

    _setitem_if = make_setitem_if_kernel_func()

    result = dpt.zeros((1), dtype=dtype, device=dpctl.SyclDevice("cpu"))
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
