import math
from functools import lru_cache

import dpctl.tensor as dpt
import numba_dpex as dpex
import numpy as np
from numba_dpex.kernel_api import NdRange

zero_idx = np.int64(0)


@lru_cache
def make_apply_elementwise_func(shape, func, work_group_size):
    func = dpex.func(func)
    n_items = math.prod(shape)

    @dpex.kernel
    # fmt: off
    def elementwise_ops_kernel(
        data,                    # INOUT    (n_items,)
    ):
        # fmt: on
        item_idx = dpex.get_global_id(zero_idx)
        if item_idx >= n_items:
            return

        item = data[item_idx]
        data[item_idx] = func(item)

    global_size = math.ceil(n_items / work_group_size) * work_group_size

    def elementwise_ops(data):
        data = dpt.reshape(data, (-1,))
        dpex.call_kernel(
            elementwise_ops_kernel, NdRange((global_size,), (work_group_size,)), data
        )

    return elementwise_ops


@lru_cache
def make_initialize_to_zeros_kernel(shape, work_group_size, dtype):
    n_items = math.prod(shape)
    global_size = math.ceil(n_items / work_group_size) * work_group_size
    zero = dtype(0.0)

    @dpex.kernel
    def initialize_to_zeros_kernel(data):
        item_idx = dpex.get_global_id(zero_idx)

        if item_idx >= n_items:
            return

        data[item_idx] = zero

    def initialize_to_zeros(data):
        data = dpt.reshape(data, (-1,))
        dpex.call_kernel(
            initialize_to_zeros_kernel,
            NdRange((global_size,), (work_group_size,)),
            data,
        )

    return initialize_to_zeros


@lru_cache
def make_broadcast_division_1d_2d_axis0_kernel(shape, work_group_size):
    n_rows, n_cols = shape
    global_size = math.ceil(n_cols / work_group_size) * work_group_size

    # NB: the left operand is modified inplace, the right operand is only read into.
    # Optimized for C-contiguous array and for
    # size1 >> preferred_work_group_size_multiple
    @dpex.kernel
    def broadcast_division(dividend_array, divisor_vector):
        col_idx = dpex.get_global_id(zero_idx)

        if col_idx >= n_cols:
            return

        divisor = divisor_vector[col_idx]

        for row_idx in range(n_rows):
            dividend_array[row_idx, col_idx] = (
                dividend_array[row_idx, col_idx] / divisor
            )

    def kernel_call(*args):
        return dpex.call_kernel(
            broadcast_division, NdRange((global_size,), (work_group_size,)), *args
        )

    return kernel_call


@lru_cache
def make_broadcast_ops_1d_2d_axis1_kernel(shape, ops, work_group_size):
    """
    ops must be a function that will be interpreted as a dpex.func and is subject to
    the same rules. It is expected to take two scalar arguments and return one scalar
    value. lambda functions are advised against since the cache will not work with lamda
    functions. sklearn_numba_dpex.common._utils expose some pre-defined `ops`.
    """
    n_rows, n_cols = shape

    global_size = math.ceil(n_cols / work_group_size) * work_group_size
    ops = dpex.func(ops)

    # NB: the left operand is modified inplace, the right operand is only read into.
    # Optimized for C-contiguous array and for
    # size1 >> preferred_work_group_size_multiple
    @dpex.kernel
    def broadcast_ops(left_operand_array, right_operand_vector):
        col_idx = dpex.get_global_id(zero_idx)

        if col_idx >= n_cols:
            return

        for row_idx in range(n_rows):
            left_operand_array[row_idx, col_idx] = ops(
                left_operand_array[row_idx, col_idx], right_operand_vector[row_idx]
            )

    def kernel_call(*args):
        return dpex.call_kernel(
            broadcast_ops, NdRange((global_size,), (work_group_size,)), *args
        )

    return kernel_call


@lru_cache
def make_half_l2_norm_2d_axis0_kernel(shape, work_group_size, dtype):
    n_rows, n_cols = shape
    global_size = math.ceil(n_cols / work_group_size) * work_group_size
    zero = dtype(0.0)
    two = dtype(2.0)

    # Optimized for C-contiguous array and for
    # size1 >> preferred_work_group_size_multiple
    @dpex.kernel
    # fmt: off
    def half_l2_norm(
        data,    # IN        (size0, size1)
        result,  # OUT       (size1,)
    ):
        # fmt: on
        col_idx = dpex.get_global_id(zero_idx)

        if col_idx >= n_cols:
            return

        l2_norm = zero

        for row_idx in range(n_rows):
            item = data[row_idx, col_idx]
            l2_norm += item * item

        result[col_idx] = l2_norm / two

    def kernel_call(*args):
        return dpex.call_kernel(
            half_l2_norm, NdRange((global_size,), (work_group_size,)), *args
        )

    return kernel_call
