import math
from functools import lru_cache

import numba_dpex as dpex
import numpy as np

from sklearn_numba_dpex.common._utils import _check_max_work_group_size

zero_idx = np.int64(0)


@lru_cache
def make_matmul_2d_kernel(
    X_n_rows,
    Y_t_n_rows,
    n_cols,
    work_group_size,
    sub_group_size,
    dtype,
    device,
    multiply_fn=None,
    out_fused_elementwise_fn=None,
):
    """Returns a matmul kernel.

    This kernel is about doing a matrix product between `X` and `Y`, but `Y` will be
    repesented with its transposed `Y_t`, with `X` and `Y_t` being both expected to be
    C-contiguous, and the (equivalent) matrix product being really computed being
    between `X` and the transpose of `Y_t`. Hence the choice of using the variable name
    `Y_t_n_rows` rather than `Y_n_cols`.

    If `out_fused_elementwise_fn` is not None, it will be applied element-wise once to
    each element of the output array right before it is returned. This function is
    compiled and fused into the first kernel as a device function with the help of
    `dpex.func`. This comes with limitations as explained in:

    https://intelpython.github.io/numba-dpex/latest/user_guides/kernel_programming_guide/device-functions.html # noqa

    It is expected to take one scalar argument and return one scalar value.

    Likewise, if `multiply_fn` is not None it will be used in place of the scalar
    multiply operator . It is expected to take two scalar argument and return one scalar
    value.

    `sklearn_numba_dpex.common._utils` exposes some pre-defined functions suitable to
    be passed as `fused_elementwise_func` or `out_fused_elementwise_fn`. Passing lambda
    functions is advised against since their compilation might not be cached.
    """

    zero = dtype(0.0)

    input_work_group_size = work_group_size
    work_group_size = _check_max_work_group_size(
        work_group_size,
        device,
        required_local_memory_per_item=2 * np.dtype(dtype).itemsize,
    )

    square_window_side = math.sqrt(work_group_size)
    square_window_side = sub_group_size * int(square_window_side / sub_group_size)

    if work_group_size != input_work_group_size:
        work_group_size = square_window_side * square_window_side

    elif work_group_size != (square_window_side * square_window_side):
        raise ValueError(
            "Expected work_group_size to be a square of a multiple of sub_group_size "
            f" but got sub_group_size={sub_group_size} and "
            f"work_group_size={work_group_size}"
        )

    if multiply_fn is None:

        @dpex.func
        def multiply_fn(x, y):
            return x * y

    else:
        multiply_fn = dpex.func(multiply_fn)

    if out_fused_elementwise_fn is None:

        @dpex.func
        def out_fused_elementwise_fn(x):
            return x

    else:
        out_fused_elementwise_fn = dpex.func(out_fused_elementwise_fn)

    # The following values define the geometry of the dispatch of work items.
    # Each work group of size (square_window_side * square_window_side) is mapped to a
    # window of shape (square_window_side, square_window_side) that compute
    # `work_group_size` output items in the window. Windows don't overlap, are arranged
    # in a grid, and cover all the items to compute.

    # Within a window, work items cooperate to compute one value per work item. Two
    # temporary windows of `X` and `Y_t` values, of same shape, are loaded cooperatively
    # in shared memory and slide along the matmul dimension to accumulate partial
    # results, until the dimension has been completely iterated on.
    window_shape = (square_window_side, square_window_side)

    # The size of the grid of windows required to exactly cover the array of result.
    # (note that windows at the edge will have out-of-bound items)
    grid_n_rows = math.ceil(X_n_rows / square_window_side)
    grid_n_cols = math.ceil(Y_t_n_rows / square_window_side)

    # Amount of temporary sliding windows that are needed to accumulate the partial
    # result until the final result
    n_sliding_windows_for_cols = math.ceil(n_cols / square_window_side)

    # The last sliding window contain out-of-bound items. The following constants
    # will be use to prevent out-of-bounds compute and memory access.
    last_sliding_window_idx = n_sliding_windows_for_cols - 1
    last_sliding_window_width = (n_cols % square_window_side) or square_window_side

    # Given the size of the grid of windows is `grid_n_rows * grid_n_cols`, and that
    # each window contain (square_window_side * square_window_side = work_group_size)
    # work items, the total number of work items is:
    global_size = grid_n_rows * grid_n_cols * work_group_size

    @dpex.kernel
    # fmt: off
    def matmul(
            X,         # IN      (X_n_rows, n_cols)
            Y_t,       # IN      (Y_t_n_rows, n_cols)
            result     # OUT     (X_n_rows, Y_t_n_rows)
    ):
        # fmt: on
        local_work_id = dpex.get_local_id(0)

        # Map the 1D work group size to 2D indices that index the work items in a window
        # size (square_window_side * square_window_side):

        # Locally...
        local_row_idx = local_work_id // square_window_side
        local_col_idx = local_work_id % square_window_side

        # ...and globally.
        group_id = dpex.get_group_id(0)

        group_first_row_idx = (group_id // grid_n_cols) * square_window_side
        group_first_col_idx = (group_id % grid_n_cols) * square_window_side

        result_row_idx = group_first_row_idx + local_row_idx
        result_col_idx = group_first_col_idx + local_col_idx

        # Allocate shared memory for the two sliding windows on `X` and `Y_t`
        X_sliding_window = dpex.local.array(shape=window_shape, dtype=dtype)
        Y_t_sliding_window = dpex.local.array(shape=window_shape, dtype=dtype)

        # Index of the column of the sliding window that the current work item will be
        # responsible for loading. The "sliding" is materialize by incrementing this
        # index by `square_window_side` at each iteration of the main loop.
        window_col_idx = local_col_idx

        # For both `X` and `Y_t`, that are assumed to be C-contiguous, in order to
        # optimize RW memory access patterns, it's best to ensure that contiguous work
        # items read contiguous memory slots in both shared and global memory. This is
        # straightforward to ensure for windows on `X`, but requires the following
        # remapped index for `Y_T`:
        Y_t_loaded_row_idx = group_first_col_idx + local_row_idx

        # Boundaries check: those booleans indicate that the current work item is
        # mapped to out-of-bounds compute, and will skip instructions for corresponding
        # memory accesses.
        row_is_in_bounds = result_row_idx < X_n_rows
        Y_t_loaded_row_is_in_bounds = Y_t_loaded_row_idx < Y_t_n_rows
        item_is_in_bounds = row_is_in_bounds and (result_col_idx < Y_t_n_rows)

        # The following variable `output` will accumulate the partial result at
        # `result[result_row_idx, result_col_idx]`
        if item_is_in_bounds:
            output = zero

        # TODO: the compiled code can potentially be saner and the performance better
        # if the entire loop is repeated 4 times with specialization depending on the
        # (binary) values of the variables `row_is_in_bound`, and
        # `Y_t_loaded_row_is_in_bounds`, that are defined outside of the loop, and
        # hence don't have to condition instructions inside the loop.
        for window_idx in range(n_sliding_windows_for_cols):
            is_last_window = window_idx == last_sliding_window_idx

            # Load the sliding windows on `X` and `Y_t`
            # NB: 4 possible different specializations, depending on the values of
            # `row_in_bounds` and `Y_t_loaded_row_is_in_bounds`.
            _load_sliding_windows(
                local_row_idx,
                local_col_idx,
                result_row_idx,
                Y_t_loaded_row_idx,
                window_col_idx,
                is_last_window,
                row_is_in_bounds,
                Y_t_loaded_row_is_in_bounds,
                X,
                Y_t,
                # OUT
                X_sliding_window,
                Y_t_sliding_window,
            )

            window_col_idx += square_window_side

            dpex.barrier(dpex.LOCAL_MEM_FENCE)

            # Accumulate the result from values loaded in the current sliding windows.
            # NB: 2 possible different specializations (included in the 4 possible
            # specializations of the previous step), depending on the value of
            # `item_is_in_bounds`
            if item_is_in_bounds:
                # TODO: here, reading the values in the sliding windows once then
                # shifting within the private memory of the sub groups would be more
                # efficient, but `numba_dpex` lacks the corresponding intrinsics. Those
                # intrinsics are defined in the SYCL spec and exist in `numba.cuda` so
                # there's hope it also happens for `numba_dpex`.
                # see https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_shift_left_and_shift_right  # noqa
                if is_last_window:
                    for idx in range(last_sliding_window_width):
                        output += multiply_fn(
                            X_sliding_window[local_row_idx, idx],
                            Y_t_sliding_window[local_col_idx, idx],
                        )

                else:
                    for idx in range(square_window_side):
                        output += multiply_fn(
                            X_sliding_window[local_row_idx, idx],
                            Y_t_sliding_window[local_col_idx, idx],
                        )

            dpex.barrier(dpex.LOCAL_MEM_FENCE)

        # Write result
        _write_result(
            item_is_in_bounds,
            result_row_idx,
            result_col_idx,
            output,
            # OUT
            result
        )

    # HACK 906: see sklearn_numba_dpex.patches.tests.test_patches.test_need_to_workaround_numba_dpex_906  # noqa
    @dpex.func
    # fmt: off
    def _load_sliding_windows(
        local_row_idx,                      # PARAM
        local_col_idx,                      # PARAM
        result_row_idx,                     # PARAM
        Y_t_loaded_row_idx,                 # PARAM
        window_col_idx,                     # PARAM
        is_last_window,                     # PARAM
        row_is_in_bounds,                   # PARAM
        Y_t_loaded_row_is_in_bounds,        # PARAM
        X,                                  # IN      (X_n_rows, n_cols)
        Y_t,                                # IN      (Y_t_n_rows, n_cols)
        X_sliding_window,                   # OUT     (square_window_side, square_window_side)  # noqa
        Y_t_sliding_window,                 # OUT     (square_window_side, square_window_side)  # noqa
    ):
        # fmt: on
        if is_last_window and (window_col_idx >= n_cols):
            return

        if row_is_in_bounds:
            X_sliding_window[local_row_idx, local_col_idx] = X[
                result_row_idx, window_col_idx
            ]

        if Y_t_loaded_row_is_in_bounds:
            Y_t_sliding_window[local_row_idx, local_col_idx] = Y_t[
                Y_t_loaded_row_idx, window_col_idx
            ]

    # HACK 906: see sklearn_numba_dpex.patches.tests.test_patches.test_need_to_workaround_numba_dpex_906  # noqa
    @dpex.func
    def _write_result(
        item_is_in_bounds,  # PARAM
        result_row_idx,  # PARAM
        result_col_idx,  # PARAM
        output,  # SCALAR
        result,  # OUT     (X_n_rows, Y_t_n_rows)
    ):
        if item_is_in_bounds:
            result[result_row_idx, result_col_idx] = out_fused_elementwise_fn(output)

    return matmul[global_size, work_group_size]
