from contextlib import nullcontext

import dpctl
import dpctl.tensor as dpt
import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose

from sklearn_numba_dpex.common.kernels import (
    make_argmin_reduction_1d_kernel,
    make_sum_reduction_2d_kernel,
)
from sklearn_numba_dpex.testing.config import float_dtype_params


@pytest.mark.parametrize("axis", [0, 1])
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
        (
            # [[0.0, 1.0, 2.0, 3.0]]
            dpt.reshape(dpt.arange(4), (1, 4)),
            np.array([6.0]),
        ),
        (
            # [[0.0, 1.0, 2.0, 3.0, 4.0]]
            dpt.reshape(dpt.arange(5), (1, 5)),
            np.array([10.0]),
        ),
        (
            # [[0.0, 1.0, 2.0, 3.0, 4.0],
            #  [5.0, 6.0, 7.0, 8.0, 9.0],
            #  [10.0, 11.0, 12.0, 13.0, 14.0]]
            dpt.reshape(dpt.arange(15), (3, 5)),
            np.array([10.0, 35.0, 60.0]),
        ),
        (
            # [[123]]
            dpt.asarray([[123]]),
            np.array([123]),
        ),
    ],
)
@pytest.mark.parametrize("dtype", float_dtype_params)
def test_sum_reduction_2d(array_in, expected_result, axis, dtype, work_group_size):
    if axis == 0:
        array_in = dpt.asarray(array_in.T, order="C")

    array_in = dpt.astype(array_in, dtype)

    device = array_in.device.sycl_device

    if work_group_size == "max":
        sub_group_size = min(device.sub_group_sizes)
    elif work_group_size == 1:
        sub_group_size = 1
    else:
        sub_group_size = work_group_size // 2

    sum_reduction_2d_kernel = make_sum_reduction_2d_kernel(
        size0=len(array_in),
        size1=array_in.shape[1],
        work_group_size=work_group_size,
        device=device,
        dtype=dtype,
        axis=axis,
        sub_group_size=sub_group_size,
    )

    actual_result = dpt.asnumpy(dpt.squeeze(sum_reduction_2d_kernel(array_in)))

    assert_allclose(expected_result, actual_result)


@pytest.mark.parametrize("work_group_size", [2, 4, 8, "max"])
@pytest.mark.parametrize("length, expected_result", [(4, 6), (5, 10)])
@pytest.mark.parametrize("dtype", float_dtype_params)
def test_sum_reduction_1d(length, expected_result, dtype, work_group_size):
    array_in = dpt.arange(length, dtype=dtype)

    device = array_in.device.sycl_device

    sum_reduction_1d_kernel = make_sum_reduction_2d_kernel(
        size0=len(array_in),
        size1=None,
        work_group_size=work_group_size,
        device=device,
        dtype=dtype,
    )

    actual_result = dpt.asnumpy(sum_reduction_1d_kernel(array_in))[0]

    assert actual_result == pytest.approx(expected_result)


context_check_power_of_two_error = pytest.raises(
    ValueError, match="Expected a power of 2"
)
context_check_sub_group_size_error = pytest.raises(
    ValueError, match="Expected sub_group_size to divide work_group_size"
)


@pytest.mark.parametrize(
    "axis, work_group_size, sub_group_size, raise_context",
    [
        # for axis 1, work_group_size is required to be be a power of two
        (1, 64, None, nullcontext()),  # ok
        (1, 63, None, context_check_power_of_two_error),
        # for axis 0, work_group_size is required to be a power-of-two multiple of
        # sub_group_size
        (0, 8 * 8, 8, nullcontext()),  # ok
        (0, 8 * 11, 11, nullcontext()),  # ok
        # power-of-two error if work_group_size is multiple of sub_group_size but not
        # power-of-two multiple
        (0, 7 * 11, 11, context_check_power_of_two_error),
        # different error if sub_group_size does not divide work_group_size at all
        (0, 8 * 12, 11, context_check_sub_group_size_error),
    ],
)
def test_sum_reduction_raise_on_invalid_size_parameters(
    axis, work_group_size, sub_group_size, raise_context
):
    with raise_context:
        make_sum_reduction_2d_kernel(
            size0=10,
            size1=10,
            work_group_size=work_group_size,
            sub_group_size=sub_group_size,
            axis=axis,
            device=dpctl.SyclDevice(),
            dtype=np.float32,
        )


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

    device = array_in.device.sycl_device

    argmin_reduction_1d_kernel = make_argmin_reduction_1d_kernel(
        size=len(array_in),
        work_group_size=work_group_size,
        device=device,
        dtype=dtype,
    )

    actual_result = dpt.asnumpy(argmin_reduction_1d_kernel(array_in))[0]
    assert actual_result.dtype == np.int32
    assert actual_result == expected_result
