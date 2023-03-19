from contextlib import nullcontext

import dpctl
import dpctl.tensor as dpt
import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose

from sklearn_numba_dpex.common.matmul import make_matmul_2d_kernel
from sklearn_numba_dpex.testing.config import float_dtype_params


def _arange_reshaped(shape, dtype):
    n_items = shape[0] * shape[1]
    return np.arange(n_items, dtype=dtype).reshape(shape)


@pytest.mark.parametrize(
    "work_group_size, sub_group_size", [(1, 1), (4, 2), (16, 4), ("max", None)]
)
@pytest.mark.parametrize(
    "test_input_shapes",
    [
        ((1, 1), (1, 1)),
        ((4, 4), (4, 4)),
        ((5, 5), (5, 5)),
        ((1, 4), (4, 1)),
        ((1, 5), (5, 1)),
        ((4, 1), (1, 4)),
        ((5, 1), (1, 5)),
        ((4, 5), (5, 4)),
        ((5, 4), (4, 5)),
        ((5, 4), (4, 4)),
        ((5, 4), (4, 8)),
        ((5, 4), (4, 10)),
        ((4, 5), (5, 5)),
        ((4, 5), (5, 8)),
        ((4, 5), (5, 10)),
        ((4, 4), (4, 5)),
        ((8, 4), (4, 5)),
        ((10, 4), (4, 5)),
        ((5, 5), (5, 4)),
        ((5, 8), (8, 4)),
        ((5, 10), (10, 4)),
    ],
)
@pytest.mark.parametrize("dtype", float_dtype_params)
def test_matmul_2d(test_input_shapes, work_group_size, sub_group_size, dtype):
    X_shape, Y_shape = test_input_shapes

    X = _arange_reshaped(X_shape, dtype)
    Y = _arange_reshaped(Y_shape, dtype)

    X_n_rows = X_shape[0]
    Y_t_n_rows = Y_shape[1]

    assert (n_cols := X_shape[1]) == Y_shape[0]

    expected_result = np.matmul(X, Y)

    X = dpt.asarray(X, order="C")
    Y_t = dpt.asarray(Y.T, order="C")

    device = X.device.sycl_device

    if sub_group_size is None:
        sub_group_size = min(device.sub_group_sizes)

    matmul_2d_kernel = make_matmul_2d_kernel(
        X_n_rows, Y_t_n_rows, n_cols, dtype, device, work_group_size, sub_group_size
    )

    result = dpt.zeros((X_n_rows, Y_t_n_rows), dtype, order="C", device=device)

    matmul_2d_kernel(X, Y_t, result)

    result = dpt.asnumpy(result)
    assert_allclose(expected_result, result)


no_error = nullcontext()
expected_square_error = pytest.raises(
    ValueError,
    match="Expected work_group_size to be a square of a multiple of sub_group_size",
)


@pytest.mark.parametrize(
    "work_group_size, sub_group_size, expected_error",
    [
        # work_group_size is required to be a square of a multiple of sub_group_size
        (8 * 8, 8, no_error),
        (22 * 22, 11, no_error),
        # error if work_group_size is a perfect square, but not of a multiple of
        # sub_group_size
        (3 * 3, 13, expected_square_error),
        # error if work_group_size is not a perfect square at all
        (3 * 5, 5, expected_square_error),
    ],
)
def test_matmul_raise_on_invalid_size_parameters(
    work_group_size, sub_group_size, expected_error
):
    with expected_error:
        make_matmul_2d_kernel(
            2, 2, 2, np.float32, dpctl.SyclDevice(), work_group_size, sub_group_size
        )
