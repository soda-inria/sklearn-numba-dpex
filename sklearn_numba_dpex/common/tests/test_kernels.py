import dpctl
import dpctl.tensor as dpt
import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose

from sklearn_numba_dpex.common.kernels import (
    make_argmin_reduction_1d_kernel,
    make_sum_reduction_2d_axis1_kernel,
)
from sklearn_numba_dpex.testing.config import float_dtype_params


@pytest.mark.parametrize("work_group_size", [2, 4, 8, "max"])
@pytest.mark.parametrize(
    "array_in, expected_result",
    [
        (
            # [[0.0, 1.0, 2.0, 3.0],
            #  [4.0, 5.0, 6.0, 7.0],
            #  [8.0, 9.0, 10.0, 11.0]]
            dpt.reshape(dpt.arange(12), (3, 4)),
            np.array([6.0, 22.0, 38.0]),
        ),
        (
            # [[0.0, 1.0, 2.0, 3.0, 4.0],
            #  [5.0, 6.0, 7.0, 8.0, 9.0],
            #  [10.0, 11.0, 12.0, 13.0, 14.0]]
            dpt.reshape(dpt.arange(15), (3, 5)),
            np.array([10.0, 35.0, 60.0]),
        ),
    ],
)
@pytest.mark.parametrize("dtype", float_dtype_params)
def test_sum_reduction_2d(array_in, expected_result, dtype, work_group_size):
    array_in = dpt.astype(array_in, dtype)

    device = dpctl.SyclDevice()

    sum_reduction_2d_kernel = make_sum_reduction_2d_axis1_kernel(
        size0=len(array_in),
        size1=array_in.shape[1],
        work_group_size=work_group_size,
        device=device,
        dtype=dtype,
    )

    actual_result = dpt.asnumpy(dpt.squeeze(sum_reduction_2d_kernel(array_in)))

    assert_allclose(expected_result, actual_result)


@pytest.mark.parametrize("work_group_size", [2, 4, 8, "max"])
@pytest.mark.parametrize("length, expected_result", [(4, 6), (5, 10)])
@pytest.mark.parametrize("dtype", float_dtype_params)
def test_sum_reduction_1d(length, expected_result, dtype, work_group_size):
    device = dpctl.SyclDevice()

    array_in = dpt.arange(length, dtype=dtype)

    sum_reduction_1d_kernel = make_sum_reduction_2d_axis1_kernel(
        size0=len(array_in),
        size1=None,
        work_group_size=work_group_size,
        device=device,
        dtype=dtype,
    )

    actual_result = dpt.asnumpy(sum_reduction_1d_kernel(array_in))[0]

    assert actual_result == pytest.approx(expected_result)


@pytest.mark.parametrize("work_group_size", [2, 4, 8, "max"])
@pytest.mark.parametrize(
    "array_in, expected_result",
    [
        (dpt.asarray([3.0, 1.0, 0.0, 2.0]), 2),
        (dpt.asarray([0.0, 1.0, 3.0, 2.0]), 0),
        (dpt.asarray([3.0, 1.0, 2.0, -1.0, -2.0]), 4),
    ],
)
@pytest.mark.parametrize("dtype", float_dtype_params)
def test_argmin_reduction_1d(array_in, expected_result, dtype, work_group_size):
    array_in = dpt.astype(array_in, dtype)
    device = dpctl.SyclDevice()

    argmin_reduction_1d_kernel = make_argmin_reduction_1d_kernel(
        size=len(array_in),
        work_group_size=work_group_size,
        device=device,
        dtype=dtype,
    )

    actual_result = dpt.asnumpy(argmin_reduction_1d_kernel(array_in))[0]
    assert actual_result.dtype == np.int32
    assert actual_result == expected_result
