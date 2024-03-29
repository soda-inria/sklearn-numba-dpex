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


def _random_reshaped(shape, dtype, rng):
    n_items = shape[0] * shape[1]
    return rng.random(n_items, dtype=dtype).reshape(shape)


def _test_matmul_2d(
    test_input_shapes, array_fn, work_group_size, sub_group_size, dtype
):
    X_shape, Y_shape = test_input_shapes

    if array_fn == "arange":
        X = _arange_reshaped(X_shape, dtype)
        Y = _arange_reshaped(Y_shape, dtype)
    elif array_fn == "random":
        rng = np.random.default_rng(seed=123)
        X = _random_reshaped(X_shape, dtype, rng)
        Y = _random_reshaped(Y_shape, dtype, rng)
    else:
        raise ValueError(
            "Expected `array_fn` to take values `arange` or `random`, but got "
            f"{array_fn}."
        )

    X_n_rows = X_shape[0]
    Y_t_n_rows = Y_shape[1]

    assert (n_cols := X_shape[1]) == Y_shape[0]

    expected_result = np.matmul(X, Y)

    X = dpt.asarray(X, order="C")
    Y_t = dpt.asarray(Y.T, order="C")

    device = X.device.sycl_device

    matmul_2d_kernel = make_matmul_2d_kernel(
        X_n_rows,
        Y_t_n_rows,
        n_cols,
        dtype,
        device,
        work_group_size=work_group_size,
        sub_group_size=sub_group_size,
    )

    result = dpt.zeros((X_n_rows, Y_t_n_rows), dtype, order="C", device=device)

    matmul_2d_kernel(X, Y_t, result)

    result = dpt.asnumpy(result)
    assert_allclose(expected_result, result)


@pytest.mark.parametrize("array_fn", ["arange", "random"])
@pytest.mark.parametrize(
    "work_group_size, sub_group_size", [(1, 1), (4, 2), (16, 4), (None, None)]
)
@pytest.mark.parametrize("dtype", [np.float32])
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
def test_matmul_2d_small(
    test_input_shapes, array_fn, work_group_size, sub_group_size, dtype
):
    _test_matmul_2d(test_input_shapes, array_fn, work_group_size, sub_group_size, dtype)


@pytest.mark.parametrize("array_fn", ["arange", "random"])
@pytest.mark.parametrize("dtype", float_dtype_params)
@pytest.mark.parametrize(
    "test_input_shapes",
    [
        # use prime numbers to ensure to test boundary management
        ((1999, 661), (661, 563)),
        ((1999, 1999), (1999, 1999)),
        ((661, 1999), (1999, 563)),
    ],
)
def test_matmul_2d_large(test_input_shapes, array_fn, dtype):
    # test with default params only
    _test_matmul_2d(
        test_input_shapes,
        array_fn,
        work_group_size=None,
        sub_group_size=None,
        dtype=dtype,
    )


no_error = nullcontext()
expected_power_of_two = pytest.raises(
    ValueError,
    match=(
        r"Expected `work_group_size / \(sub_group_size \* sub_group_size\)` to be a "
        r"power of two, "
    ),
)


@pytest.mark.parametrize(
    "work_group_size, sub_group_size, expected_error",
    [
        # `work_group_size / (sub_group_size * sub_group_size)` is required to be a
        # power of two
        (8 * 8, 8, no_error),
        (16 * 7, 4, expected_power_of_two),
        (12 * 12, 4, expected_power_of_two),
        (3 * 3, 13, expected_power_of_two),
        (3 * 5, 5, expected_power_of_two),
    ],
)
def test_matmul_raise_on_invalid_size_parameters(
    work_group_size, sub_group_size, expected_error
):
    with expected_error:
        make_matmul_2d_kernel(
            2,
            2,
            2,
            np.float32,
            dpctl.SyclDevice(),
            work_group_size=work_group_size,
            sub_group_size=sub_group_size,
        )
