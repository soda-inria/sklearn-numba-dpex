import pytest

import dpctl
import dpctl.tensor as dpt

import numpy as np

from sklearn_numba_dpex.common.kernels import (
    make_sum_reduction_2d_axis1_kernel,
    make_argmin_reduction_1d_kernel,
)
from sklearn_numba_dpex.testing.config import float_dtype_params


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_sum_reduction_2d(dtype):
    n_reductions = 3
    n_items = 4
    device = dpctl.SyclDevice()
    work_group_size = device.max_work_group_size

    sum_reduction_2d_kernel = make_sum_reduction_2d_axis1_kernel(
        size0=n_reductions,
        size1=n_items,
        work_group_size=work_group_size,
        device=device,
        dtype=dtype,
    )

    array_in = dpt.reshape(dpt.arange(12, dtype=dtype), (3, 4))
    # [[0.0, 1.0, 2.0, 3.0],
    #  [4.0, 5.0, 6.0, 7.0],
    #  [8.0, 9.0, 10.0, 11.0]]

    expected_result = np.array([6.0, 22.0, 38.0], dtype=dtype)

    actual_result = dpt.asnumpy(dpt.squeeze(sum_reduction_2d_kernel(array_in)))

    np.testing.assert_array_equal(expected_result, actual_result)


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_sum_reduction_1d(dtype):
    n_items = 4
    device = dpctl.SyclDevice()
    work_group_size = device.max_work_group_size

    sum_reduction_1d_kernel = make_sum_reduction_2d_axis1_kernel(
        size0=n_items,
        size1=None,
        work_group_size=work_group_size,
        device=device,
        dtype=dtype,
    )

    array_in = dpt.arange(4, dtype=dtype)
    # [0.0, 1.0, 2.0, 3.0]

    expected_result = 6

    actual_result = int(dpt.asnumpy(sum_reduction_1d_kernel(array_in))[0])

    assert actual_result == expected_result


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_argmin_reduction_1d(dtype):
    n_items = 4
    device = dpctl.SyclDevice()
    work_group_size = device.max_work_group_size

    argmin_reduction_1d_kernel = make_argmin_reduction_1d_kernel(
        size=n_items,
        work_group_size=work_group_size,
        device=device,
        dtype=dtype,
    )

    array_in = dpt.asarray([3.0, 1.0, 0.0, 2.0], dtype=dtype)
    expected_result = 2
    actual_result = int(dpt.asnumpy(argmin_reduction_1d_kernel(array_in))[0])
    assert actual_result == expected_result

    array_in = dpt.asarray([0.0, 1.0, 3.0, 2.0], dtype=dtype)
    expected_result = 0
    actual_result = int(dpt.asnumpy(argmin_reduction_1d_kernel(array_in))[0])
    assert actual_result == expected_result

    array_in = dpt.asarray([3.0, 1.0, 2.0, 0.0], dtype=dtype)
    expected_result = 3
    actual_result = int(dpt.asnumpy(argmin_reduction_1d_kernel(array_in))[0])
    assert actual_result == expected_result
