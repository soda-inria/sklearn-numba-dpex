import pytest

import dpctl
import dpctl.tensor as dpt

import numpy as np

from sklearn.utils._testing import assert_allclose

from sklearn_numba_dpex.common.kernels import (
    make_sum_reduction_2d_axis1_kernel,
    make_argmin_reduction_1d_kernel,
)
from sklearn_numba_dpex.testing.config import float_dtype_params


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_sum_reduction_2d(dtype):
    array_in = dpt.reshape(dpt.arange(12, dtype=dtype), (3, 4))

    device = dpctl.SyclDevice()
    work_group_size = device.max_work_group_size

    sum_reduction_2d_kernel = make_sum_reduction_2d_axis1_kernel(
        size0=len(array_in),
        size1=array_in.shape[1],
        work_group_size=work_group_size,
        device=device,
        dtype=dtype,
    )

    # [[0.0, 1.0, 2.0, 3.0],
    #  [4.0, 5.0, 6.0, 7.0],
    #  [8.0, 9.0, 10.0, 11.0]]

    expected_result = np.array([6.0, 22.0, 38.0], dtype=dtype)

    actual_result = dpt.asnumpy(dpt.squeeze(sum_reduction_2d_kernel(array_in)))

    assert_allclose(expected_result, actual_result)


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_sum_reduction_1d(dtype):
    device = dpctl.SyclDevice()
    work_group_size = device.max_work_group_size

    array_in = dpt.arange(4, dtype=dtype)

    sum_reduction_1d_kernel = make_sum_reduction_2d_axis1_kernel(
        size0=len(array_in),
        size1=None,
        work_group_size=work_group_size,
        device=device,
        dtype=dtype,
    )

    # [0.0, 1.0, 2.0, 3.0]

    expected_result = 6

    actual_result = int(dpt.asnumpy(sum_reduction_1d_kernel(array_in))[0])

    assert actual_result == pytest.approx(expected_result)


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_argmin_reduction_1d(dtype):
    array_in = dpt.asarray([3.0, 1.0, 0.0, 2.0], dtype=dtype)
    device = dpctl.SyclDevice()
    work_group_size = device.max_work_group_size

    argmin_reduction_1d_kernel = make_argmin_reduction_1d_kernel(
        size=len(array_in),
        work_group_size=work_group_size,
        device=device,
        dtype=dtype,
    )

    expected_result = 2
    actual_result = dpt.asnumpy(argmin_reduction_1d_kernel(array_in))[0]
    assert actual_result.dtype == np.int32
    assert actual_result == expected_result

    array_in[:] = [0.0, 1.0, 3.0, 2.0]
    expected_result = 0
    actual_result = dpt.asnumpy(argmin_reduction_1d_kernel(array_in))[0]
    assert actual_result.dtype == np.int32
    assert actual_result == expected_result

    array_in[:] = [3.0, 1.0, 2.0, 0.0]
    expected_result = 3
    actual_result = dpt.asnumpy(argmin_reduction_1d_kernel(array_in))[0]
    assert actual_result.dtype == np.int32
    assert actual_result == expected_result
