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
    "test_input_shape", [(1, 1), (1, 3), (3, 1), (3, 5), (5, 3), (3, 4), (4, 3)]
)
@pytest.mark.parametrize("dtype", float_dtype_params)
def test_sum_reduction_2d(test_input_shape, work_group_size, axis, dtype):
    n_items = test_input_shape[0] * test_input_shape[1]
    array_in = np.arange(n_items, dtype=dtype).reshape(test_input_shape)

    expected_result = array_in.sum(axis=axis)

    array_in = dpt.asarray(array_in, order="C")

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


no_error = nullcontext()
power_of_two_error = pytest.raises(ValueError, match="Expected a power of 2")
sub_group_size_error = pytest.raises(
    ValueError, match="Expected sub_group_size to divide work_group_size"
)


@pytest.mark.parametrize(
    "axis, work_group_size, sub_group_size, expected_error",
    [
        # for axis 1, work_group_size is required to be be a power of two
        (1, 64, None, no_error),
        (1, 63, None, power_of_two_error),
        # for axis 0, work_group_size is required to be a power-of-two multiple of
        # sub_group_size
        (0, 8 * 8, 8, no_error),
        (0, 8 * 11, 11, no_error),
        # power-of-two error if work_group_size is multiple of sub_group_size but not
        # power-of-two multiple
        (0, 7 * 11, 11, power_of_two_error),
        # different error if sub_group_size does not divide work_group_size at all
        (0, 8 * 12, 11, sub_group_size_error),
    ],
)
def test_sum_reduction_raise_on_invalid_size_parameters(
    axis, work_group_size, sub_group_size, expected_error
):
    with expected_error:
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
