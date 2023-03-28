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
    dtype,
    device,
    work_group_size=None,
    sub_group_size=None,
    multiply_fn=None,
    out_fused_elementwise_fn=None,
):
    """Returns a matmul kernel.

    This kernel is about doing a matrix product between `X` and `Y`, but `Y` will be
    represented with its transposed `Y_t`, with `X` and `Y_t` being both expected
    to be C-contiguous, and the (equivalent) matrix product being really computed
    being between `X` and the transpose of `Y_t`. Hence the choice of using the
    variable name `Y_t_n_rows` rather than `Y_n_cols`. This setup allows making use of
    memory locality in order to lower memory load.

    If `out_fused_elementwise_fn` is not None, it will be applied element-wise once to
    each element of the output array right before it is returned. This function is
    compiled and fused into the first kernel as a device function with the help of
    `dpex.func`. This comes with limitations as explained in:

    https://intelpython.github.io/numba-dpex/latest/user_guides/kernel_programming_guide/device-functions.html # noqa

    It is expected to take one scalar argument and return one scalar value.

    Likewise, if `multiply_fn` is not None it will be used in place of the scalar
    multiply operator. It is expected to take two scalar arguments and return one
    scalar value.

    `sklearn_numba_dpex.common._utils` exposes some pre-defined functions suitable to
    be passed as `fused_elementwise_func` or `out_fused_elementwise_fn`. Passing lambda
    functions is advised against since it might not be compatible with caching of the
    compilation of the `matmul` kernel.

    TODO: this kernel is still a WIP. It's far from reaching maximum theoritical
    performance. It currently performs about 2.5 times worse than dpnp implementation
    on an iGPU, and about 4* worse on a flex GPU. (NB: `dpnp` calls mkl blas under the
    hood, that is highly optimized, but there are reports from `cuda` users that
    near-`cublas` performance can be achieved either with `cuda` implementation in C
    or using Triton ~ so approaching `dpnp` performance might be achievable with SYCL
    too ?)

    Areas of known possible improvements for the current kernel:
    - memory layout: the kernel takes `X` and `Y_t` as input and slides a window of
    items on columns. Taking `X_t` and `Y` as inputs (also `C-contiguous`) and
    sliding on rows might be better. It seems it can enable prettier memory access
    patterns both for reading and writing in shared memory. However attempts at
    this approach have only shown worse performances so far.
    - coalescing accesses in global memory: it is unclear wether `numba_dpex/SYCL` can
    effectively coalesce reads of contiguous addresses in global memory, it might not
    if it's not able to tell at compile time that the input has a contiguous structure
    in memory. Investigating `SPIR-V` code could confirm/infirm this. If not coalesced,
    use of functions such as `reinterpret_cast` for `cuda` are required to manually
    enforce the coalesced reads.
    - autotuning: maybe better parameters (`work_group_size`, `sub_group_size`,
    `arithmetic_intensity` multipliers,...) can be found, those parameters might depend
    on the device. Is it possible to find parameters that reasonably suit all devices ?
    - Investigating `SPIR-V` code compiled by `numba-dpex` to look for possible
    issues with the JIT.
    - Re-ordering work groups (and maybe work items within work groups) to improve
    cache locality. This as been attempted but have shown no results, while up to a
    10% improvements is claimed to be possible with this trick.

    See also:
    - https://siboehm.com/articles/22/CUDA-MMM
    - https://triton-lang.org/master/getting-started/tutorials/03-matrix-multiplication.html  # noqa

    """
    # The best values for the parameters defined thereafter depend on the device, but
    # a stable heuristic is yet to be found.
    # TODO: find an heuristic for adapting the performance parameters to the device
    # rather than hardcoding defaults.
    work_group_size = work_group_size or 128
    sub_group_size = sub_group_size or 128

    # Arithmetic intensity refers to how much compute a single work item is set to
    # perform relatively to how much RW operations it needs to execute.

    # The size of the window of results that are computed by a single work item is
    # scaled by
    # (arithmetic_intensity_multiplier_X, arithmetic_intensity_multiplier_Y) ratio, at
    # the cost of loading
    # `arithmetic_intensity_multiplier_X + arithmetic_intensity_multiplier_Y`
    # more items in the work group shared memory, per item in this work group.

    # Those values should be set high enough so that the kernel is compute-bound
    # rather than memory-bound.

    # Setting it too high might pressure local or private memory too much for it to be
    # worth.
    arithmetic_intensity_multiplier_X = 4
    arithmetic_intensity_multiplier_Y = 2

    # Width of the window of values of Y_t stored in registries - lower means less
    # values in registries, but more iterations in main loop.
    private_Y_t_sliding_window_width = 4  # must divide `sub_group_size`

    # Automatically set `private_Y_t_sliding_window_width` to `sub_group_size` if
    # is `sub_group_size < private_Y_t_sliding_window_width`
    if sub_group_size < private_Y_t_sliding_window_width:
        private_Y_t_sliding_window_width = sub_group_size

    elif _mod_value := (sub_group_size % private_Y_t_sliding_window_width):
        raise ValueError(
            "Expected `sub_group_size` to be divisible by "
            f"{private_Y_t_sliding_window_width}, but got "
            f"`sub_group_size == {sub_group_size}`, with `{sub_group_size} % "
            f"{private_Y_t_sliding_window_width} = {_mod_value}"
        )

    # NB: The following implementation not only works for the best parameters found so
    # far but for other combinations too, to enable exhaustive grid searches on all
    # devices.

    zero = dtype(0.0)
    two_as_long = dpex.int64(2)

    input_work_group_size = work_group_size
    work_group_size = _check_max_work_group_size(
        work_group_size,
        device,
        required_local_memory_per_item=(
            arithmetic_intensity_multiplier_X + arithmetic_intensity_multiplier_Y
        ),
    )

    # This value is equal to the number of results per work item assuming
    # arithmetic_intensity_multiplier_X = arithmetic_intensity_multiplier_Y = 1 .
    # It is expected to be a power of two (base_nb_results_per_work_item_log2 < 1 is
    # possible.)
    base_nb_results_per_work_item = work_group_size / (sub_group_size * sub_group_size)
    base_nb_results_per_work_item_log2 = math.floor(
        math.log2(base_nb_results_per_work_item)
    )

    if work_group_size != input_work_group_size:
        base_nb_results_per_work_item = 2**base_nb_results_per_work_item_log2
        work_group_size = int(
            base_nb_results_per_work_item * sub_group_size * sub_group_size
        )

    elif work_group_size != (
        base_nb_results_per_work_item * sub_group_size * sub_group_size
    ):
        raise ValueError(
            "Expected `work_group_size / (sub_group_size * sub_group_size)` to be a "
            f"power of two, but got {base_nb_results_per_work_item} instead, with "
            f"`work_group_size={work_group_size}` and "
            f"`sub_group_size={sub_group_size}`."
        )
    else:
        base_nb_results_per_work_item = int(base_nb_results_per_work_item)

    if multiply_fn is None:

        @dpex.func
        def multiply_fn_(x, y):
            return x * y

    else:
        multiply_fn_ = dpex.func(multiply_fn)

    if out_fused_elementwise_fn is None:

        @dpex.func
        def out_fused_elementwise_fn(x):
            return x

    else:
        out_fused_elementwise_fn = dpex.func(out_fused_elementwise_fn)

    # Under the same assumption, this value is equal to the number of results computed
    # by a single work group
    base_result_window_side = work_group_size // sub_group_size

    # If the expected number of results computed by each work item is set below 1, some
    # work items would remain idle and would require additionnal boundaries checks in
    # the kernel. To prevent this, the amount of results per work item is increased to
    # 1 by increasing the size of the shared memory allocation.
    if base_nb_results_per_work_item_log2 < 1:
        if base_nb_results_per_work_item_log2 % 2:
            result_window_height_multiplier_Y = 2 ** (
                (1 - base_nb_results_per_work_item_log2) // 2
            )
            result_window_height_multiplier_X = 2 * result_window_height_multiplier_Y
        else:
            result_window_height_multiplier_X = (
                result_window_height_multiplier_Y
            ) = 2 ** ((-base_nb_results_per_work_item_log2) // 2)
        base_nb_results_per_work_item = 1
        base_nb_results_per_work_item_log2 = 0
    else:
        result_window_height_multiplier_X = result_window_height_multiplier_Y = 1

    # Finally, a work group spans a window of results adjusted to arithmetic
    # intensity of size `result_window_height * result_window_width` in the result
    # array.
    result_window_height_multiplier_X *= arithmetic_intensity_multiplier_X
    result_window_height_multiplier_Y *= arithmetic_intensity_multiplier_Y
    result_window_height = result_window_height_multiplier_X * base_result_window_side
    result_window_width = result_window_height_multiplier_Y * base_result_window_side

    # Such a work group loads two sliding windows that span relevant rows and colums
    # of the inputs X and Y_t.
    local_X_sliding_window_shape = (
        result_window_height,
        2 * sub_group_size,  # allocate twice the space to let padding against bank
        # conflicts
    )
    local_Y_t_sliding_window_shape = (
        result_window_width,
        2 * sub_group_size,  # allocate twice the space to let padding against bank
        # conflicts
    )
    # TODO: it's surprising to see bank conflicts here, however the padding add
    # 10% better performance on iGPUs. Why ?

    # Amount of temporary sliding windows that are needed to accumulate the partial
    # results until the final result
    n_sliding_windows_for_cols = math.ceil(n_cols / sub_group_size)

    # If arithmetic intensity parameters are set to 1, a single work item will compute
    # `base_nb_results_per_work_item` results, ordered in a window of size
    # `(base_private_result_array_height, base_private_result_array_width)`.

    # Depending on the number of results per work item, the window is either a square
    # window, or a rectangle where the height is twice the width.
    base_private_result_array_height = 2 ** (
        (base_nb_results_per_work_item_log2 + 1) // 2
    )

    if base_nb_results_per_work_item_log2 % 2:
        base_private_result_array_width = base_private_result_array_height // 2
    else:
        base_private_result_array_width = base_private_result_array_height

    # The size of the window is increased accordingly with arithmetic intensity
    # multipliers.
    private_result_array_height = (
        arithmetic_intensity_multiplier_X * base_private_result_array_height
    )
    private_result_array_width = (
        arithmetic_intensity_multiplier_Y * base_private_result_array_width
    )

    thread_private_result_array_shape = (
        private_result_array_height,
        private_result_array_width,
    )

    # Windows of size `(sub_group_size, result_window_width)` of `Y_t` will be
    # consecutively loaded in shared memory before accumulating corresponding
    # partial results into the registries of each work item. Beside the partial results,
    # relevant values of `Y_t` stored in shared memory are also temporarily copied to
    # registries in a private array of shape private_Y_t_sliding_window_shape`:
    private_Y_t_sliding_window_shape = (
        private_result_array_width,
        private_Y_t_sliding_window_width,
    )
    # where `private_Y_t_sliding_window_width <= sub_group_size`.

    nb_private_arrays_for_window = sub_group_size // private_Y_t_sliding_window_width

    nb_work_items_for_Y_t_window = result_window_width // private_result_array_width
    nb_work_items_for_X_window = result_window_height // private_result_array_height

    # Surprisingly the performance is greatly enhanced when manually unrolling the
    # following loop (in a separate kernel func for readability):
    _accumulate_step_unrolled = _make_accumulate_step_unrolled_kernel_func(
        private_result_array_width, multiply_fn_
    )

    global_grid_n_rows = math.ceil(X_n_rows / result_window_height)
    global_grid_n_cols = math.ceil(Y_t_n_rows / result_window_width)

    grid_n_groups = global_grid_n_rows * global_grid_n_cols
    global_size = grid_n_groups * work_group_size

    # TODO: the following code can be leveraged to improve cache locality. However, it
    # does not seem to improve the timings.

    # get_remapped_group_id_func = make_get_super_id_func(
    #     grid_n_groups,
    #     private_result_array_height,
    #     private_result_array_width,
    #     global_grid_n_rows,
    #     global_grid_n_cols,
    # )

    # get_remapped_work_item_id_func = make_get_super_id_func(
    #     work_group_size,
    #     private_result_array_height,
    #     private_result_array_width,
    #     nb_work_items_for_X_window,
    #     nb_work_items_for_Y_t_window,
    #     sub_group_size=sub_group_size,
    # )

    @dpex.kernel
    # fmt: off
    def matmul(
            X,                          # IN      (X_n_rows, n_cols)
            Y_t,                        # IN      (Y_t_n_rows, n_cols)
            result,                     # OUT     (X_n_rows, Y_t_n_rows)
            # WIP: improve cache locality ?
            # group_id_mapping_row,
            # group_id_mapping_col,
            # work_id_mapping_row,
            # work_id_mapping_col
    ):
        # fmt: on
        work_item_idx = dpex.get_local_id(zero_idx)

        # Index the work items in the base sliding window:
        work_item_row_idx = work_item_idx // sub_group_size
        work_item_col_idx = work_item_idx % sub_group_size

        group_idx = dpex.get_group_id(zero_idx)

        # Get indices of the row and the column of the top-left corner of the sub-array
        # of results covered by this work group
        group_row_idx = group_idx // global_grid_n_cols
        group_col_idx = group_idx % global_grid_n_cols
        # TODO: improve cache locality ?
        # group_row_idx = group_id_mapping_row[group_idx]
        # group_col_idx = group_id_mapping_col[group_idx]
        group_first_row_idx = (
            group_row_idx * result_window_height
        )
        group_first_col_idx = (
            group_col_idx * result_window_width
        )

        # Allocate shared memory for the two sliding windows on `X` and `Y_t`
        local_X_sliding_window = dpex.local.array(
            shape=local_X_sliding_window_shape, dtype=dtype
        )
        local_Y_t_sliding_window = dpex.local.array(
            shape=local_Y_t_sliding_window_shape, dtype=dtype
        )

        # Allocate private memory for the result window
        private_result = dpex.private.array(
            shape=thread_private_result_array_shape, dtype=dtype
        )

        # Allocate private memory for the private sliding window on Y_t
        private_Y_t_sliding_window = dpex.private.array(
            shape=private_Y_t_sliding_window_shape, dtype=dtype
        )

        # Index of the first column of the sliding window that the current work item
        # will be responsible for loading. The "sliding" is materialized by the
        # increments of value `sub_group_size` to this index.
        first_window_loaded_col_idx = work_item_col_idx

        # Indices of the first rows in `X` and `Y_t` and column of the sliding window
        # that the current will multiply.
        first_private_loaded_sliding_X_value_idx = (
            work_item_idx // nb_work_items_for_Y_t_window
        )
        first_private_loaded_sliding_Y_t_value_idx = (
            work_item_idx % nb_work_items_for_Y_t_window
        )
        # TODO: improve cache locality ?
        # first_private_loaded_sliding_X_value_idx = work_id_mapping_row[work_item_idx]
        # first_private_loaded_sliding_Y_t_value_idx = work_id_mapping_col[
        #     work_item_idx]

        _initialize_private_result(private_result)

        work_item_col_idx_padded = two_as_long * work_item_col_idx
        first_X_loaded_row_idx = group_first_row_idx + work_item_row_idx
        first_Y_t_loaded_row_idx = group_first_col_idx + work_item_row_idx

        window_loaded_col_idx = first_window_loaded_col_idx
        for _ in range(n_sliding_windows_for_cols):
            _load_sliding_windows(
                work_item_row_idx,
                work_item_col_idx,
                work_item_col_idx_padded,
                first_X_loaded_row_idx,
                first_Y_t_loaded_row_idx,
                window_loaded_col_idx,
                X,
                Y_t,
                # OUT
                local_X_sliding_window,
                local_Y_t_sliding_window
            )
            window_loaded_col_idx += sub_group_size

            dpex.barrier(dpex.LOCAL_MEM_FENCE)

            _accumulate_private_windows(
                first_private_loaded_sliding_X_value_idx,
                first_private_loaded_sliding_Y_t_value_idx,
                local_X_sliding_window,
                local_Y_t_sliding_window,
                # BUFFER
                private_Y_t_sliding_window,
                # OUT
                private_result
            )

            dpex.barrier(dpex.LOCAL_MEM_FENCE)

        _write_result(
            group_first_row_idx + first_private_loaded_sliding_X_value_idx,
            group_first_col_idx + first_private_loaded_sliding_Y_t_value_idx,
            private_result,
            # OUT
            result
        )

    @dpex.func
    def _initialize_private_result(private_result):
        for i in range(private_result_array_height):
            for j in range(private_result_array_width):
                private_result[i, j] = zero

    # HACK 906: see sklearn_numba_dpex.patches.tests.test_patches.test_need_to_workaround_numba_dpex_906  # noqa
    @dpex.func
    # fmt: off
    def _load_sliding_windows(
        work_item_row_idx,          # PARAM
        work_item_col_idx,          # PARAM
        work_item_col_idx_padded,
        first_X_loaded_row_idx,     # PARAM
        first_Y_t_loaded_row_idx,   # PARAM
        window_loaded_col_idx,      # PARAM
        X,                          # IN      (X_n_rows, n_cols)
        Y_t,                        # IN      (Y_t_n_rows, n_cols)
        local_X_sliding_window,     # OUT     (result_window_height, 2 * sub_group_size)
        local_Y_t_sliding_window    # OUT     (result_window_width, 2* sub_group_size)
    ):
        # fmt: on
        X_loaded_row_idx = first_X_loaded_row_idx
        X_local_loaded_row_idx = work_item_row_idx
        for _ in range(result_window_height_multiplier_X):
            if (X_loaded_row_idx < X_n_rows) and (window_loaded_col_idx < n_cols):
                loaded_X_value = X[X_loaded_row_idx, window_loaded_col_idx]
            else:
                loaded_X_value = zero

            local_X_sliding_window[
                X_local_loaded_row_idx, work_item_col_idx_padded
            ] = loaded_X_value
            X_loaded_row_idx += base_result_window_side
            X_local_loaded_row_idx += base_result_window_side

        Y_t_loaded_row_idx = first_Y_t_loaded_row_idx
        Y_t_local_loaded_row_idx = work_item_row_idx
        for _ in range(result_window_height_multiplier_Y):
            if (Y_t_loaded_row_idx < Y_t_n_rows) and (window_loaded_col_idx < n_cols):
                loaded_Y_t_value = Y_t[Y_t_loaded_row_idx, window_loaded_col_idx]
            else:
                loaded_Y_t_value = zero

            local_Y_t_sliding_window[
                Y_t_local_loaded_row_idx, work_item_col_idx_padded
            ] = loaded_Y_t_value
            Y_t_loaded_row_idx += base_result_window_side
            Y_t_local_loaded_row_idx += base_result_window_side

    @dpex.func
    def _accumulate_private_windows(
        private_first_loaded_sliding_X_value_idx,  # PARAM
        private_first_loaded_sliding_Y_t_value_idx,  # PARAM
        local_X_sliding_window,  # IN       (result_window_height, 2 * sub_group_size)  # noqa
        local_Y_t_sliding_window,  # IN       (result_window_width, 2* sub_group_size)  # noqa
        private_Y_t_sliding_window,  # BUFFER   (private_result_array_width, private_Y_t_sliding_window_width)  # noqa
        private_result,  # OUT      (private_result_array_height, private_result_array_width)  # noqa
    ):
        private_array_first_col = zero_idx
        for _ in range(nb_private_arrays_for_window):

            private_loaded_sliding_Y_t_value_idx = (
                private_first_loaded_sliding_Y_t_value_idx
            )
            for i in range(private_result_array_width):
                for j in range(private_Y_t_sliding_window_width):
                    private_Y_t_sliding_window[i, j] = local_Y_t_sliding_window[
                        private_loaded_sliding_Y_t_value_idx,
                        two_as_long * (private_array_first_col + j),
                    ]
                private_loaded_sliding_Y_t_value_idx += nb_work_items_for_Y_t_window

            private_loaded_sliding_X_value_idx = (
                private_first_loaded_sliding_X_value_idx
            )
            for i in range(private_result_array_height):
                for j in range(private_Y_t_sliding_window_width):
                    private_loaded_X_value = local_X_sliding_window[
                        private_loaded_sliding_X_value_idx,
                        two_as_long * (private_array_first_col + j),
                    ]

                    _accumulate_step_unrolled(
                        i,
                        j,
                        private_loaded_X_value,
                        private_Y_t_sliding_window,
                        private_result,
                    )
                private_loaded_sliding_X_value_idx += nb_work_items_for_X_window

            private_array_first_col += private_Y_t_sliding_window_width

    @dpex.func
    # fmt: off
    def _write_result(
        result_first_row_idx,    # PARAM
        result_first_col_idx,    # PARAM
        private_result,          # IN      (private_result_array_height, private_result_array_width)  # noqa
        result                   # OUT     (X_n_rows, Y_t_n_rows)
    ):
        # fmt: on
        result_row_idx = result_first_row_idx
        for i in range(private_result_array_height):
            if result_row_idx < X_n_rows:
                result_col_idx = result_first_col_idx
                for j in range(private_result_array_width):
                    if result_col_idx < Y_t_n_rows:
                        result[result_row_idx, result_col_idx] = private_result[i, j]
                        result_col_idx += nb_work_items_for_Y_t_window
            result_row_idx += nb_work_items_for_X_window

    matmul_groups = matmul[global_size, work_group_size]

    # TODO: improve cache locality ?
    # import dpctl.tensor as dpt

    # group_id_mapping_row, group_id_mapping_col = zip(
    #     *(get_remapped_group_id_func(group_id) for group_id in range(grid_n_groups))
    # )
    # group_id_mapping_row = dpt.asarray(group_id_mapping_row, dtype=np.int32)
    # group_id_mapping_col = dpt.asarray(group_id_mapping_col, dtype=np.int32)

    # work_id_mapping_row, work_id_mapping_col = zip(
    #     *(get_remapped_work_item_id_func(item_id)
    #       for item_id in range(work_group_size))
    # )
    # work_id_mapping_row = dpt.asarray(work_id_mapping_row, dtype=np.int32)
    # work_id_mapping_col = dpt.asarray(work_id_mapping_col, dtype=np.int32)

    def _matmul(X, Y_t, result):
        matmul_groups(
            X,
            Y_t,
            result,
            # TODO: improve cache locality ?
            # group_id_mapping_row,
            # group_id_mapping_col,
            # work_id_mapping_row,
            # work_id_mapping_col,
        )

    return _matmul


def _make_accumulate_step_unrolled_kernel_func(private_result_array_width, multiply_fn):

    if private_result_array_width == 1:

        @dpex.func
        def _accumulate_step_unrolled(
            i, j, private_loaded_X_value, private_Y_t_sliding_window, private_result
        ):
            private_result[i, 0] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[0, j]
            )

    elif private_result_array_width == 2:

        @dpex.func
        def _accumulate_step_unrolled(
            i, j, private_loaded_X_value, private_Y_t_sliding_window, private_result
        ):
            private_result[i, 0] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[0, j]
            )
            private_result[i, 1] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[1, j]
            )

    elif private_result_array_width == 4:

        @dpex.func
        def _accumulate_step_unrolled(
            i, j, private_loaded_X_value, private_Y_t_sliding_window, private_result
        ):
            private_result[i, 0] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[0, j]
            )
            private_result[i, 1] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[1, j]
            )
            private_result[i, 2] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[2, j]
            )
            private_result[i, 3] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[3, j]
            )
            # private_result[i, 0] += 1
            # private_result[i, 1] += 1
            # private_result[i, 2] += 1
            # private_result[i, 3] += 1

    elif private_result_array_width == 8:

        @dpex.func
        def _accumulate_step_unrolled(
            i, j, private_loaded_X_value, private_Y_t_sliding_window, private_result
        ):
            private_result[i, 0] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[0, j]
            )
            private_result[i, 1] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[1, j]
            )
            private_result[i, 2] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[2, j]
            )
            private_result[i, 3] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[3, j]
            )
            private_result[i, 4] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[4, j]
            )
            private_result[i, 5] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[5, j]
            )
            private_result[i, 6] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[6, j]
            )
            private_result[i, 7] += multiply_fn(
                private_loaded_X_value, private_Y_t_sliding_window[7, j]
            )

    return _accumulate_step_unrolled


# TODO: improve cache locality ?
def make_get_super_id_func(
    n_items,
    item_weight_row,
    item_weight_col,
    global_n_rows,
    global_n_cols,
    sub_group_size=None,
):
    """Return a function that can either be used within a kernel, or used behorehand
    to pre-compute a mapping from native ids to remapped ids that ensure best cache
    properties for the matmul kernel."""

    item_weight_lcm = math.lcm(item_weight_row, item_weight_col)
    local_n_rows = item_weight_lcm // item_weight_row
    local_n_cols = item_weight_lcm // item_weight_col
    if sub_group_size is not None:
        mult = sub_group_size // local_n_rows
        if (mult * local_n_rows) != sub_group_size:
            raise ValueError
        local_n_rows = local_n_rows * mult
        local_n_cols = local_n_cols * mult

    max_grid_dim_size = max(
        global_n_rows // local_n_rows,
        global_n_cols // local_n_cols,
    )

    groups_tree_depth = math.ceil(math.log2(max_grid_dim_size))

    init_node_side_n_rows = int(2 ** (groups_tree_depth - 1) * local_n_rows)
    init_node_side_n_cols = int(2 ** (groups_tree_depth - 1) * local_n_cols)
    init_current_max_covered_rows = int((2**groups_tree_depth) * local_n_rows)
    init_current_max_covered_cols = int((2**groups_tree_depth) * local_n_cols)

    init_needs_boundaries_check = n_items != (
        init_current_max_covered_cols * init_current_max_covered_rows
    )

    def get_super_item_id(item_id):
        item_id_in_node = item_id
        super_item_id_row = 0
        super_item_id_col = 0
        current_max_covered_rows = global_n_rows
        current_max_covered_cols = global_n_cols
        boundaries_check = init_needs_boundaries_check
        nb_items_in_node = n_items

        current_node_side_n_rows = init_node_side_n_rows
        current_node_side_n_cols = init_node_side_n_cols

        for _ in range(groups_tree_depth):
            if (not boundaries_check) or (
                (current_max_covered_rows == init_current_max_covered_rows)
                and (current_max_covered_cols == init_current_max_covered_cols)
            ):
                nb_items_in_node = nb_items_in_node // 4
                child_id = item_id_in_node // nb_items_in_node
                item_id_in_node = item_id_in_node % nb_items_in_node
                if (child_id == 1) or (child_id == 3):
                    super_item_id_col += current_node_side_n_cols
                if (child_id == 2) or (child_id == 3):
                    super_item_id_row += current_node_side_n_rows
                current_max_covered_rows = current_node_side_n_rows
                current_max_covered_cols = current_node_side_n_cols

            elif (current_max_covered_rows < current_node_side_n_rows) and (
                current_max_covered_rows < current_node_side_n_cols
            ):
                pass

            elif current_max_covered_rows < current_node_side_n_rows:
                nb_items_in_node_0 = current_max_covered_rows * current_node_side_n_cols
                if item_id_in_node < nb_items_in_node_0:
                    nb_items_in_node = nb_items_in_node_0
                    current_max_covered_cols = current_node_side_n_cols
                else:
                    super_item_id_col += current_node_side_n_cols
                    item_id_in_node -= nb_items_in_node_0
                    current_max_covered_cols -= current_node_side_n_cols
                    nb_items_in_node -= nb_items_in_node_0

            elif current_max_covered_cols < current_node_side_n_cols:
                nb_items_in_node_0 = current_max_covered_cols * current_node_side_n_rows
                if item_id_in_node < nb_items_in_node_0:
                    nb_items_in_node = nb_items_in_node_0
                    current_max_covered_rows = current_node_side_n_rows
                else:
                    super_item_id_row += current_node_side_n_rows
                    item_id_in_node -= nb_items_in_node_0
                    current_max_covered_rows -= current_node_side_n_rows
                    nb_items_in_node -= nb_items_in_node_0

            else:
                nb_items_in_node_0 = current_node_side_n_rows * current_node_side_n_cols

                if item_id_in_node < nb_items_in_node_0:
                    boundaries_check = False
                    nb_items_in_node = nb_items_in_node_0

                else:
                    nb_items_in_nodes_01 = (
                        current_max_covered_cols * current_node_side_n_rows
                    )

                    if item_id_in_node < nb_items_in_nodes_01:
                        super_item_id_col += current_node_side_n_cols
                        item_id_in_node -= nb_items_in_node_0
                        current_max_covered_rows = current_node_side_n_rows
                        current_max_covered_cols -= current_node_side_n_cols
                        nb_items_in_node = nb_items_in_nodes_01 - nb_items_in_node_0
                        boundaries_check = (
                            current_max_covered_cols == current_node_side_n_cols
                        )

                    else:
                        nb_items_in_nodes_02 = (
                            current_max_covered_rows * current_node_side_n_cols
                        )
                        nb_items_in_nodes_012 = (
                            nb_items_in_nodes_01
                            + nb_items_in_nodes_02
                            - nb_items_in_node_0
                        )

                        if item_id_in_node < nb_items_in_nodes_012:
                            super_item_id_row += current_node_side_n_rows
                            item_id_in_node -= nb_items_in_nodes_01
                            current_max_covered_rows -= current_node_side_n_rows
                            current_max_covered_cols = current_node_side_n_cols
                            nb_items_in_node = nb_items_in_nodes_02 - nb_items_in_node_0
                            boundaries_check = (
                                current_max_covered_rows == current_node_side_n_rows
                            )

                        else:
                            super_item_id_row += current_node_side_n_rows
                            super_item_id_col += current_node_side_n_cols
                            item_id_in_node -= nb_items_in_nodes_012
                            current_max_covered_rows -= current_node_side_n_rows
                            current_max_covered_cols -= current_node_side_n_cols
                            nb_items_in_node -= nb_items_in_nodes_012

            current_node_side_n_rows = current_node_side_n_rows // 2
            current_node_side_n_cols = current_node_side_n_cols // 2

        return (super_item_id_row + (item_id_in_node // current_max_covered_cols)), (
            super_item_id_col + (item_id_in_node % current_max_covered_cols)
        )

    return get_super_item_id
