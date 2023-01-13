# TODO: many auxilliary kernels in the package might be better optimized and we could
# benchmark alternative implementations for each of them, that could include
#    - using 2D or 3D grid of work groups and work items where applicable (e.g. in
# some of the kernels that take 2D or 3D data as input) rather than using 1D grid. When
# doing so, one should be especially careful about how the segments of adjacent work
# items of size preferred_work_group_size_multiple are dispatched especially regarding
# RW  operations in memory. A wrong dispatch strategy could slash memory bandwith and
# reduce performance. Using 2D or 3D grid correctly might on the other hand improve
# performance since it saves costly indexing operations (like //)
#    - investigate if flat 1D-like indexing also works for ND kernels, thus saving the
# need to compute coordinates for each dimension for element-wise operations.
#    - or using numba + dpnp to directly leverage kernels that are shipped in dpnp to
# replace numpy methods.
# However, in light of our main goal that is bringing a GPU KMeans to scikit-learn, the
# importance of those TODOs is currently seen as secondary, since the execution time of
# those kernels is only a small fraction of the total execution time and the
# improvements that further optimizations can add will only be marginal. There is no
# doubt, though, that a lot could be learnt about kernel programming in the process.

import math
from functools import lru_cache

import dpctl.tensor as dpt
import numba_dpex as dpex
import numpy as np

from sklearn_numba_dpex.common._utils import (
    _check_max_work_group_size,
    check_power_of_2,
)

zero_idx = np.int64(0)


# HACK: dtype argument is passed to prevent sharing a device function instance
# between kernels specialized for different argument types.
# This is a workaround for:
# https://github.com/IntelPython/numba-dpex/issues/867. Revert changes in
# https://github.com/soda-inria/sklearn-numba-dpex/pull/82 when
# fixed.
@lru_cache
def make_elementwise_binary_op_1d_kernel(size, op, work_group_size, dtype):
    """This kernel is mostly necessary to work around lack of support for this
    operation in dpnp, see https://github.com/IntelPython/dpnp/issues/1238"""
    op = dpex.func(op)

    @dpex.kernel
    # fmt: off
    def elementwise_ops(
        data,                    # INOUT    (size,)
        operand_right            # IN       (1,)
    ):
        # fmt: on

        item_idx = dpex.get_global_id(zero_idx)
        if item_idx >= size:
            return

        operand_left = data[item_idx]
        data[item_idx] = op(operand_left, operand_right[0])

    global_size = math.ceil(size / work_group_size) * work_group_size
    return elementwise_ops[global_size, work_group_size]


@lru_cache
def make_initialize_to_zeros_2d_kernel(size0, size1, work_group_size, dtype):

    n_items = size0 * size1
    global_size = math.ceil(n_items / work_group_size) * work_group_size
    zero = dtype(0.0)

    # Optimized for C-contiguous arrays
    @dpex.kernel
    def initialize_to_zeros(data):
        item_idx = dpex.get_global_id(zero_idx)

        if item_idx >= n_items:
            return

        row_idx = item_idx // size1
        col_idx = item_idx % size1
        data[row_idx, col_idx] = zero

    return initialize_to_zeros[global_size, work_group_size]


@lru_cache
def make_initialize_to_zeros_3d_kernel(size0, size1, size2, work_group_size, dtype):

    n_items = size0 * size1 * size2
    stride0 = size1 * size2
    global_size = math.ceil(n_items / work_group_size) * work_group_size
    zero = dtype(0.0)

    # Optimized for C-contiguous arrays
    @dpex.kernel
    def initialize_to_zeros(data):
        item_idx = dpex.get_global_id(zero_idx)

        if item_idx >= n_items:
            return

        i = item_idx // stride0
        stride0_idx = item_idx % stride0
        j = stride0_idx // size2
        k = stride0_idx % size2
        data[i, j, k] = zero

    return initialize_to_zeros[global_size, work_group_size]


@lru_cache
def make_broadcast_division_1d_2d_axis0_kernel(size0, size1, work_group_size):
    global_size = math.ceil(size1 / work_group_size) * work_group_size

    # NB: the left operand is modified inplace, the right operand is only read into.
    # Optimized for C-contiguous array and for
    # size1 >> preferred_work_group_size_multiple
    @dpex.kernel
    def broadcast_division(dividend_array, divisor_vector):
        col_idx = dpex.get_global_id(zero_idx)

        if col_idx >= size1:
            return

        divisor = divisor_vector[col_idx]

        for row_idx in range(size0):
            dividend_array[row_idx, col_idx] = (
                dividend_array[row_idx, col_idx] / divisor
            )

    return broadcast_division[global_size, work_group_size]


@lru_cache
def make_broadcast_ops_1d_2d_axis1_kernel(size0, size1, ops, work_group_size, dtype):
    """
    ops must be a function that will be interpreted as a dpex.func and is subject to
    the same rules. It is expected to take two scalar arguments and return one scalar
    value. lambda functions are advised against since the cache will not work with lamda
    functions. sklearn_numba_dpex.common._utils expose some pre-defined `ops`.
    """

    global_size = math.ceil(size1 / work_group_size) * work_group_size
    ops = dpex.func(ops)

    # NB: the left operand is modified inplace, the right operand is only read into.
    # Optimized for C-contiguous array and for
    # size1 >> preferred_work_group_size_multiple
    @dpex.kernel
    def broadcast_ops(left_operand_array, right_operand_vector):
        col_idx = dpex.get_global_id(zero_idx)

        if col_idx >= size1:
            return

        for row_idx in range(size0):
            left_operand_array[row_idx, col_idx] = ops(
                left_operand_array[row_idx, col_idx], right_operand_vector[row_idx]
            )

    return broadcast_ops[global_size, work_group_size]


@lru_cache
def make_half_l2_norm_2d_axis0_kernel(size0, size1, work_group_size, dtype):
    global_size = math.ceil(size1 / work_group_size) * work_group_size
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

        if col_idx >= size1:
            return

        l2_norm = zero

        for row_idx in range(size0):
            item = data[row_idx, col_idx]
            l2_norm += item * item

        result[col_idx] = l2_norm / two

    return half_l2_norm[global_size, work_group_size]


@lru_cache
def make_sum_reduction_2d_axis1_kernel(
    size0, size1, device, dtype, work_group_size="max", fused_unary_func=None
):
    """Implement data_2d.sum(axis=1) or data_1d.sum()

    numba_dpex does not provide tools such as `cuda.reduce` so we implement from scratch
    a reduction strategy. The strategy relies on the commutativity of the operation used
    for the reduction, thus allowing to reduce the input in any order.

    The strategy consists in performing local reductions in each work group using local
    memory where each work item combine two values, thus halving the number of values,
    and the number of active work items. At each iteration the work items are discarded
    in a bracket manner. The work items with the greatest ids are discarded first, and
    we rely on the fact that the remaining work items are adjacents to optimize the RW
    operations.

    Once the reduction is done in a work group the result is written in global memory,
    thus creating an intermediary result whose size is divided by
    `2 * work_group_size`. This is repeated as many time as needed until only one value
    remains in global memory.

    If fused_unary_func is not None, it will be applied element-wise before summing.
    It must be a function that will be interpreted as a dpex.func and is subject to the
    same rules. It is expected to take one scalar argument and returning one scalar
    value. lambda functions are advised against since the cache will not work with
    lambda functions. sklearn_numba_dpex.common._utils expose some pre-defined
    `fused_unary_funcs`.

    Notes
    -----
    `work_group_size` is assumed to be a power of 2.

    if `size1` is None then the kernel expects 1d tensor inputs. If `size1` is not None
    then the expected shape of input tensors is `(size0, size1)`, and the reduction
    operation is equivalent to input.sum(axis=1). In this case, the kernel is a good
    choice if `size1` >> `preferred_work_group_size_multiple`, and if `size0` ranges in
    the same order of magnitude than `preferred_work_group_size_multiple`. If not,
    other reduction implementations might give better performances.
    """
    n_rows = size0 if size1 is not None else None
    sum_axis_size = size0 if n_rows is None else size1

    # fused_unary_func is applied elementwise during the first pass on data, in the
    # first kernel execution only.
    work_group_size, fused_func_kernel = _make_partial_sum_reduction_2d_axis1_kernel(
        n_rows, work_group_size, fused_unary_func, dtype, device
    )
    # subsequent kernel calls only sum the data.
    _, nofunc_kernel = _make_partial_sum_reduction_2d_axis1_kernel(
        n_rows, work_group_size, fused_unary_func=None, dtype=dtype, device=device
    )

    check_power_of_2(work_group_size)

    # As many partial reductions as necessary are chained until only one element
    # remains.
    kernels_and_empty_tensors_pairs = []
    n_groups = sum_axis_size
    # TODO: at some point, the cost of scheduling the kernel is more than the cost of
    # running the reduction iteration. At this point the loop should stop and then a
    # single work item should iterates one time on the remaining values to finish the
    # reduction.
    kernel = fused_func_kernel
    while n_groups > 1:
        n_groups = math.ceil(n_groups / (2 * work_group_size))
        global_size = n_groups * work_group_size
        kernel = kernel[global_size, work_group_size]
        result_shape = n_groups if n_rows is None else (n_rows, n_groups)
        # NB: here memory for partial results is allocated ahead of time and will only
        # be garbage collected when the instance of `sum_reduction` is garbage
        # collected. Thus it can be more efficient to re-use a same instance of
        # `sum_reduction` (e.g within iterations of a loop) since it avoid
        # deallocation and reallocation every time.
        result = dpt.empty(result_shape, dtype=dtype, device=device)
        kernels_and_empty_tensors_pairs.append((kernel, result))
        kernel = nofunc_kernel

    def sum_reduction(summands):
        # TODO: manually dispatch the kernels with a SyclQueue
        if not kernels_and_empty_tensors_pairs:
            # By convention the sum of all elements of an empty array is equal to 0. (
            # likewise with numpy np.sum([]) returns 0).
            if size1 is None:
                return dpt.zeros(sh=(1,), device=device, dtype=dtype)
            else:
                return dpt.zeros(sh=(size0, 1))

        for kernel, result in kernels_and_empty_tensors_pairs:
            kernel(summands, result)
            summands = result
        return summands

    return sum_reduction


@lru_cache
def _make_partial_sum_reduction_2d_axis1_kernel(
    n_rows, work_group_size, fused_unary_func, dtype, device
):

    zero = dtype(0.0)
    one_idx = np.int64(1)
    minus_one_idx = np.int64(-1)
    two_as_a_long = np.int64(2)

    if fused_unary_func is None:

        def fused_unary_func(x):
            return x

    fused_unary_func = dpex.func(fused_unary_func)

    # TODO: this set of kernel functions could be abstracted away to other coalescing
    # functions
    if n_rows is None:  # 1d

        @dpex.func
        def set_col_to_zero(array, i):
            array[i] = zero

        @dpex.func
        def copy_col(from_array, from_col, to_array, to_col):
            to_array[to_col] = fused_unary_func(from_array[from_col])

        @dpex.func
        def add_cols(
            from_array,
            left_from_col,
            right_from_col,
            to_array,
            to_col,
        ):
            to_array[to_col] = fused_unary_func(
                from_array[left_from_col]
            ) + fused_unary_func(from_array[right_from_col])

        @dpex.func
        def add_cols_inplace(
            array,
            from_col,
            to_col,
        ):
            array[to_col] += array[from_col]

        @dpex.func
        def add_first_cols(from_array, to_array, to_col):
            to_array[to_col] = from_array[zero_idx] + from_array[one_idx]

    else:

        @dpex.func
        def set_col_to_zero(array, i):
            for row in range(n_rows):
                array[row, i] = zero

        @dpex.func
        def copy_col(from_array, from_col, to_array, to_col):
            for row in range(n_rows):
                to_array[row, to_col] = fused_unary_func(from_array[row, from_col])

        @dpex.func
        def add_cols(
            from_array,
            left_from_col,
            right_from_col,
            to_array,
            to_col,
        ):
            for row in range(n_rows):
                to_array[row, to_col] = fused_unary_func(
                    from_array[row, left_from_col]
                ) + fused_unary_func(from_array[row, right_from_col])

        @dpex.func
        def add_cols_inplace(
            array,
            from_col,
            to_col,
        ):
            for row in range(n_rows):
                array[row, to_col] += array[row, from_col]

        @dpex.func
        def add_first_cols(from_array, to_array, to_col):
            for row in range(n_rows):
                to_array[row, to_col] = (
                    from_array[row, zero_idx] + from_array[row, one_idx]
                )

    input_work_group_size = work_group_size
    work_group_size = _check_max_work_group_size(
        work_group_size,
        device,
        required_local_memory_per_item=(n_rows or 1) * np.dtype(dtype).itemsize,
    )
    if work_group_size == input_work_group_size:
        check_power_of_2(work_group_size)
    else:
        # Round to the maximum smaller power of two
        work_group_size = 2 ** (math.floor(math.log2(work_group_size)))

    # Number of iteration in each execution of the kernel:
    local_n_iterations = np.int64(math.floor(math.log2(work_group_size)) - 1)

    local_values_size = work_group_size if n_rows is None else (n_rows, work_group_size)

    # Optimized for C-contiguous array where the size of the sum axis is
    # >> preferred_work_group_size_multiple, and the size of the other axis (if any) is
    # is smaller or similar to preferred_work_group_size_multiple.
    # ???: how does this strategy compares to having each thread reducing N contiguous
    # items ?
    @dpex.kernel
    # fmt: off
    def partial_sum_reduction(
        summands,    # IN        (n_rows, sum_axis_size)
        result,      # OUT       (n_rows, math.ceil(size / (2 * work_group_size),)
    ):
        # fmt: on
        # NB: This kernel only perform a partial reduction
        group_id = dpex.get_group_id(zero_idx)
        local_work_id = dpex.get_local_id(zero_idx)
        first_work_id = local_work_id == zero_idx

        size = summands.shape[minus_one_idx]

        local_values = dpex.local.array(local_values_size, dtype=dtype)

        first_value_idx = group_id * work_group_size * two_as_a_long
        augend_idx = first_value_idx + local_work_id
        addend_idx = first_value_idx + work_group_size + local_work_id

        # Each work item reads two value in global memory and sum it into the local
        # memory
        if augend_idx >= size:
            set_col_to_zero(local_values, local_work_id)
        elif addend_idx >= size:
            copy_col(summands, augend_idx, local_values, local_work_id)
        else:
            add_cols(summands, augend_idx, addend_idx, local_values, local_work_id)

        dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)
        current_n_work_items = work_group_size
        for i in range(local_n_iterations):
            # We discard half of the remaining active work items at each iteration
            current_n_work_items = current_n_work_items // two_as_a_long
            if local_work_id < current_n_work_items:
                add_cols_inplace(
                    local_values,
                    local_work_id + current_n_work_items,
                    local_work_id
                )

            dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

        # At this point local_values[0] + local_values[1] is equal to the sum of all
        # elements in summands that have been covered by the work group, we write it
        # into global memory
        if first_work_id:
            add_first_cols(local_values, result, group_id)

    return work_group_size, partial_sum_reduction


@lru_cache
def make_argmin_reduction_1d_kernel(size, device, dtype, work_group_size="max"):
    """Implement 1d argmin with the same strategy than for
    make_sum_reduction_2d_axis1_kernel."""
    two_as_a_long = np.int64(2)
    one_idx = np.int64(1)
    inf = dtype(np.inf)

    local_argmin_dtype = np.int32
    input_work_group_size = work_group_size
    work_group_size = _check_max_work_group_size(
        work_group_size,
        device,
        required_local_memory_per_item=np.dtype(dtype).itemsize
        + np.dtype(local_argmin_dtype).itemsize,
    )
    if work_group_size == input_work_group_size:
        check_power_of_2(work_group_size)
    else:
        # Round to the maximum smaller power of two
        work_group_size = 2 ** (math.floor(math.log2(work_group_size)))

    # Number of iteration in each execution of the kernel:
    local_n_iterations = np.int64(math.floor(math.log2(work_group_size)) - 1)

    # TODO: the first call of partial_argmin_reduction in the final loop should be
    # written with only two arguments since "previous_result" does not exist yet.
    # It seems it's not possible to get a good factoring of the code to avoid copying
    # most of the code for this with @dpex.kernel, for now we resort to branching.
    @dpex.kernel
    # fmt: off
    def partial_argmin_reduction(
        values,             # IN        (size,)
        previous_result,    # IN        (current_size,)
        argmin_indices,     # OUT       (math.ceil(
                            #               (current_size if current_size else size)
                            #                / (2 * work_group_size),)
                            #            ))
    ):
        # fmt: on
        group_id = dpex.get_group_id(zero_idx)
        local_work_id = dpex.get_local_id(zero_idx)
        first_work_id = local_work_id == zero_idx

        previous_result_size = previous_result.shape[zero_idx]
        has_previous_result = previous_result_size > one_idx
        current_size = (previous_result_size if has_previous_result
                        else values.shape[zero_idx])

        local_argmin = dpex.local.array(work_group_size, dtype=local_argmin_dtype)
        local_values = dpex.local.array(work_group_size, dtype=dtype)

        first_value_idx = group_id * work_group_size * two_as_a_long
        x_idx = first_value_idx + local_work_id
        y_idx = first_value_idx + work_group_size + local_work_id

        if x_idx >= current_size:
            local_values[local_work_id] = inf
        else:
            if has_previous_result:
                x_idx = previous_result[x_idx]

            if y_idx >= current_size:
                local_argmin[local_work_id] = x_idx
                local_values[local_work_id] = values[x_idx]

            else:
                if has_previous_result:
                    y_idx = previous_result[y_idx]

                x = values[x_idx]
                y = values[y_idx]
                if x < y or (x == y and x_idx < y_idx):
                    local_argmin[local_work_id] = x_idx
                    local_values[local_work_id] = x
                else:
                    local_argmin[local_work_id] = y_idx
                    local_values[local_work_id] = y

        dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)
        current_n_work_items = work_group_size
        for i in range(local_n_iterations):
            current_n_work_items = current_n_work_items // two_as_a_long
            if local_work_id < current_n_work_items:
                local_x_idx = local_work_id
                local_y_idx = local_work_id + current_n_work_items

                x = local_values[local_x_idx]
                y = local_values[local_y_idx]

                if x > y:
                    local_values[local_x_idx] = y
                    local_argmin[local_x_idx] = local_argmin[local_y_idx]

            dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

        if first_work_id:
            if local_values[zero_idx] <= local_values[one_idx]:
                argmin_indices[group_id] = local_argmin[zero_idx]
            else:
                argmin_indices[group_id] = local_argmin[one_idx]

    # As many partial reductions as necessary are chained until only one element
    # remains.argmin_indices
    kernels_and_empty_tensors_tuples = []
    n_groups = size
    previous_result = dpt.empty((1,), dtype=np.int32, device=device)
    while n_groups > 1:
        n_groups = math.ceil(n_groups / (2 * work_group_size))
        global_size = n_groups * work_group_size
        kernel = partial_argmin_reduction[global_size, work_group_size]
        result = dpt.empty(n_groups, dtype=np.int32, device=device)
        kernels_and_empty_tensors_tuples.append((kernel, previous_result, result))
        previous_result = result

    def argmin_reduction(values):
        for kernel, previous_result, result in kernels_and_empty_tensors_tuples:
            kernel(values, previous_result, result)
        return result

    return argmin_reduction
