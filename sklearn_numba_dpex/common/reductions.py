import math
from functools import lru_cache

import dpctl.tensor as dpt
import numba_dpex as dpex
import numpy as np

from sklearn_numba_dpex.common._utils import (
    _check_max_work_group_size,
    check_power_of_2,
    get_maximum_power_of_2_smaller_than,
)

zero_idx = np.int64(0)


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
        np.dtype(dtype).itemsize + np.dtype(local_argmin_dtype).itemsize,
    )
    if work_group_size == input_work_group_size:
        check_power_of_2(work_group_size)
    else:
        # Round to the maximum smaller power of two
        work_group_size = get_maximum_power_of_2_smaller_than(work_group_size)

    # Number of iteration in each execution of the kernel:
    n_local_iterations = np.int64(math.log2(work_group_size) - 1)

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

        _prepare_local_memory(
            local_work_id,
            group_id,
            current_size,
            has_previous_result,
            previous_result,
            values,
            # OUT
            local_argmin,
            local_values,
        )

        dpex.barrier(dpex.LOCAL_MEM_FENCE)
        n_active_work_items = work_group_size
        for i in range(n_local_iterations):
            n_active_work_items = n_active_work_items // two_as_a_long
            _local_iteration(
                local_work_id,
                n_active_work_items,
                # OUT
                local_values,
                local_argmin
            )
            dpex.barrier(dpex.LOCAL_MEM_FENCE)

        _register_result(
            first_work_id,
            group_id,
            local_argmin,
            local_values,
            # OUT
            argmin_indices
        )

    # HACK 906: see sklearn_numba_dpex.patches.tests.test_patches.test_need_to_workaround_numba_dpex_906  # noqa
    @dpex.func
    # fmt: off
    def _prepare_local_memory(
        local_work_id,              # PARAM
        group_id,                   # PARAM
        current_size,               # PARAM
        has_previous_result,        # PARAM
        previous_result,            # IN
        values,                     # IN
        local_argmin,               # OUT
        local_values,               # OUT
    ):
        # fmt: on
        first_value_idx = group_id * work_group_size * two_as_a_long
        x_idx = first_value_idx + local_work_id

        if x_idx >= current_size:
            local_values[local_work_id] = inf
            return

        if has_previous_result:
            x_idx = previous_result[x_idx]

        y_idx = first_value_idx + work_group_size + local_work_id

        if y_idx >= current_size:
            local_argmin[local_work_id] = x_idx
            local_values[local_work_id] = values[x_idx]
            return

        if has_previous_result:
            y_idx = previous_result[y_idx]

        x = values[x_idx]
        y = values[y_idx]
        if x < y or (x == y and x_idx < y_idx):
            local_argmin[local_work_id] = x_idx
            local_values[local_work_id] = x
            return

        local_argmin[local_work_id] = y_idx
        local_values[local_work_id] = y

    # HACK 906: see sklearn_numba_dpex.patches.tests.test_patches.test_need_to_workaround_numba_dpex_906 # noqa
    @dpex.func
    # fmt: off
    def _local_iteration(
        local_work_id,              # PARAM
        n_active_work_items,        # PARAM
        local_values,               # INOUT
        local_argmin                # OUT
    ):
        # fmt: on
        if local_work_id >= n_active_work_items:
            return

        local_x_idx = local_work_id
        local_y_idx = local_work_id + n_active_work_items

        x = local_values[local_x_idx]
        y = local_values[local_y_idx]

        if x <= y:
            return

        local_values[local_x_idx] = y
        local_argmin[local_x_idx] = local_argmin[local_y_idx]

    # HACK 906: see sklearn_numba_dpex.patches.tests.test_patches.test_need_to_workaround_numba_dpex_906 # noqa
    @dpex.func
    # fmt: off
    def _register_result(
        first_work_id,          # PARAM
        group_id,               # PARAM
        local_argmin,           # IN
        local_values,           # IN
        argmin_indices          # OUT
    ):

        if not first_work_id:
            return

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
        sizes = (n_groups * work_group_size, work_group_size)
        result = dpt.empty(n_groups, dtype=np.int32, device=device)
        kernels_and_empty_tensors_tuples.append(
            (partial_argmin_reduction, sizes, previous_result, result)
        )
        previous_result = result

    def argmin_reduction(values):
        for kernel, sizes, previous_result, result in kernels_and_empty_tensors_tuples:
            kernel[sizes](values, previous_result, result)
        return result

    return argmin_reduction


# TODO: this kernel could be abstracted away to support other commutative binary
# operators than sum.
def make_sum_reduction_2d_kernel(
    shape,
    device,
    dtype,
    work_group_size="max",
    axis=None,
    sub_group_size=None,
    fused_elementwise_func=None,
):
    """Compute data_2d.sum(axis=axis) or data_1d.sum().

    This implementation is optimized for C-contiguous arrays.

    numba_dpex does not provide tools such as `cuda.reduce` so we implement
    from scratch a reduction strategy. The strategy relies on the associativity
    and commutativity of the operation used for the reduction, thus allowing to
    reduce the input in any order.

    The strategy consists in performing a series of kernel invocations that
    each perform a partial sum. At each kernel invocation, the input array is
    tiled with non-overlapping windows and all the values within a given window
    are collaboratively summed by the work items of a given work group.

    Each window has shape:

    - when `axis = 0`: `(reduction_block_size, 1)` with: `reduction_block_size
      = work_group_size // sub_group_size * 2`

    - when `axis = 1`: `(1, reduction_block_size)` with: `reduction_block_size
      = work_group_size * 2`

    Once the reduction is done in a work group the result is written back to
    global memory, thus creating an intermediary result array whose size is
    divided by `reduction_block_size`.

    This is repeated as many times as needed until only one value remains per
    column if axis=0 or per row if axis=1.

    During a single kernel invocation, several iterations of local reductions
    are performed using local memory to store intermediate results. At each
    local iteration, each work item combines two values, thus halving the
    number of values, and the number of active work items for the subsequent
    iterations. Work items are progressively discarded in a bracket manner. The
    work items with the greatest ids are discarded first, and we rely on the
    fact that the remaining work items are adjacents to optimize the read-write
    operations in local memory.

    If `fused_elementwise_func` is not None, it will be applied element-wise
    once to each element of the input array at the beginning of the the first
    kernel invocation. This function is compiled and fused into the first
    kernel as a device function with the help of `dpex.func`. This comes with
    limitations as explained in:

    https://intelpython.github.io/numba-dpex/latest/user_guides/kernel_programming_guide/device-functions.html # noqa

    It is expected to take one scalar argument and returning one
    scalar value. lambda functions are advised against since their compilation
    might not be cached.

    `sklearn_numba_dpex.common._utils` exposes some pre-defined functions
    suitable to be passed as `fused_elementwise_func`.

    Notes
    -----
    If `size1` is not `None` then the expected shape of input tensors is `(size0,
    size1)`, and the reduction operation is equivalent to
    `input.sum(axis=axis)`.

    If `size1` is `None` then the kernel expects 1d tensor inputs, the `axis`
    parameter is ignored: the kernel designed for `axis=1` is called used after
    considering the input re-shaped as `(1, size0)`.

    If `size1` is `None` or if `axis` is `0`, then `work_group_size` is assumed to be a
    power of 2, and the parameter `sub_group_size` is ignored.

    If `size1` is not `None` and `axis` is `1`, then `work_group_size` is assumed to be
    a multiple of `sub_group_size`, such that `work_group_size // sub_group_size` is
    a power of 2.

    Depending on the size of the sum axis, it might be worth tuning `work_group_size`
    for better performance. When `axis` is `1`, tuning `sub_group_size` with respect
    to the size of the other axis might also be beneficial.

    The algorithmic approach that is chosen to implement the underlying kernels takes
    inspiration from many sources that are available online, such as [1]_, [2]_, [3]_ .

    .. [1] Timcheck, S. W. (2017). Efficient Implementation of Reductions on GPU
    Architectures.

    .. [2] https://dournac.org/info/gpu_sum_reduction

    .. [3] https://shreeraman-ak.medium.com/parallel-reduction-with-cuda-d0ae10c1ae2c
    """
    if is_1d := (len(shape) == 1):
        axis = 1
        shape1 = shape[0]
        shape0 = 1
    else:
        shape0, shape1 = shape

    if axis == 0:
        work_group_shape, kernels, shape_update_fn = _prepare_sum_reduction_2d_axis0(
            shape1,
            work_group_size,
            sub_group_size,
            fused_elementwise_func,
            dtype,
            device,
        )
    else:  # axis == 1
        work_group_shape, kernels, shape_update_fn = _prepare_sum_reduction_2d_axis1(
            shape0, work_group_size, fused_elementwise_func, dtype, device
        )

    # XXX: The kernels seem to work fine with work_group_size==1 on GPU but fail on CPU.
    if math.prod(work_group_shape) == 1:
        raise NotImplementedError("work_group_size==1 is not supported.")

    # NB: the shape of the work group is different in each of those two cases. Summing
    # efficiently requires to adapt to very different IO patterns depending on the sum
    # axis, which motivates the need for different kernels with different work group
    # sizes for each cases. As a consequence, the shape of the intermediate results
    # in the main driver loop, and the `global_size` (total number of work items fired
    # per call) for each kernel call, are also different, and are variabilized with
    # the functions `get_result_shape` and `get_global_size`.
    get_result_shape, get_global_size = shape_update_fn

    # `fused_elementwise_func` is applied elementwise during the first pass on
    # data, in the first kernel execution only, using `fused_func_kernel`. Subsequent
    # kernel calls only sum the data, using `nofunc_kernel`.
    (fused_func_kernel, nofunc_kernel), reduction_block_size = kernels

    # As many partial reductions as necessary are chained until only one element
    # remains.
    kernels_and_empty_tensors_pairs = []
    # TODO: at some point, the cost of scheduling the kernel is more than the cost of
    # running the reduction iteration. At this point the loop should stop and then a
    # single work item should iterates one time on the remaining values to finish the
    # reduction?
    kernel = fused_func_kernel
    sum_axis_size = shape0 if axis == 0 else shape1
    next_input_size = sum_axis_size
    while next_input_size > 1:
        result_sum_axis_size = math.ceil(next_input_size / reduction_block_size)
        # NB: here memory for partial results is allocated ahead of time and will only
        # be garbage collected when the instance of `sum_reduction` is garbage
        # collected. Thus it can be more efficient to re-use a same instance of
        # `sum_reduction` (e.g within iterations of a loop) since it avoid
        # deallocation and reallocation every time.
        result_shape = get_result_shape(result_sum_axis_size)
        result = dpt.empty(result_shape, dtype=dtype, device=device)

        sizes = (get_global_size(result_sum_axis_size), work_group_shape)

        kernels_and_empty_tensors_pairs.append((kernel, sizes, result))
        kernel = nofunc_kernel

        next_input_size = result_sum_axis_size

    def sum_reduction(summands):
        if is_1d:
            # Makes the 1d case a special 2d case to reuse the same kernel.
            summands = dpt.reshape(summands, (1, -1))

        if sum_axis_size == 0:
            # By convention the sum of all elements of an empty array is equal to 0. (
            # likewise with numpy np.sum([]) returns 0).
            summands = dpt.zeros(get_result_shape(1))

        # TODO: manually dispatch the kernels with a SyclQueue
        for kernel, sizes, result in kernels_and_empty_tensors_pairs:
            kernel[sizes](summands, result)
            summands = result

        if is_1d:
            summands = dpt.reshape(summands, (-1,))

        return summands

    return sum_reduction


@lru_cache
def _prepare_sum_reduction_2d_axis0(
    n_cols, work_group_size, sub_group_size, fused_elementwise_func, dtype, device
):

    if fused_elementwise_func is None:

        @dpex.func
        def fused_elementwise_func_(x):
            return x

    else:
        fused_elementwise_func_ = dpex.func(fused_elementwise_func)

    input_work_group_size = work_group_size
    work_group_size = _check_max_work_group_size(
        work_group_size, device, required_local_memory_per_item=np.dtype(dtype).itemsize
    )
    if work_group_size == input_work_group_size:
        if (work_group_size % sub_group_size) != 0:
            raise ValueError(
                "Expected sub_group_size to divide work_group_size, but got "
                f"sub_group_size={sub_group_size} and "
                f"work_group_size={work_group_size}"
            )
        n_sub_groups_per_work_group = work_group_size // sub_group_size
        check_power_of_2(n_sub_groups_per_work_group)

    else:
        # Round work_group_size to the maximum smaller power-of-two multiple of
        # `sub_group_size`
        n_sub_groups_per_work_group = get_maximum_power_of_2_smaller_than(
            work_group_size / sub_group_size
        )
        work_group_size = n_sub_groups_per_work_group * sub_group_size

    (
        work_group_shape,
        reduction_block_size,
        partial_sum_reduction,
    ) = _make_partial_sum_reduction_2d_axis0_kernel(
        n_cols, work_group_size, sub_group_size, fused_elementwise_func_, dtype
    )

    if fused_elementwise_func is None:
        partial_sum_reduction_nofunc = partial_sum_reduction
    else:
        *_, partial_sum_reduction_nofunc = _make_partial_sum_reduction_2d_axis0_kernel(
            n_cols,
            work_group_size,
            sub_group_size,
            fused_elementwise_func_,
            dtype,
        )

    get_result_shape = lambda result_sum_axis_size: (result_sum_axis_size, n_cols)
    get_global_size = lambda result_sum_axis_size: (
        sub_group_size * math.ceil(n_cols / sub_group_size),
        result_sum_axis_size * n_sub_groups_per_work_group,
    )

    kernels = (partial_sum_reduction, partial_sum_reduction_nofunc)
    shape_update_fn = (get_result_shape, get_global_size)
    return (work_group_shape, (kernels, reduction_block_size), shape_update_fn)


def _make_partial_sum_reduction_2d_axis0_kernel(
    n_cols, work_group_size, sub_group_size, fused_elementwise_func, dtype
):
    """When axis=0, each work group performs a local reduction on axis 0 in a window of
    size `(sub_group_size_,work_group_size // sub_group_size)`."""
    zero = dtype(0.0)
    one_idx = np.int64(1)
    two_as_a_long = np.int64(2)

    n_sub_groups_per_work_group = work_group_size // sub_group_size

    # Number of iteration in each execution of the kernel:
    n_local_iterations = np.int64(math.log2(n_sub_groups_per_work_group) - 1)

    local_values_size = (n_sub_groups_per_work_group, sub_group_size)
    reduction_block_size = 2 * n_sub_groups_per_work_group
    work_group_shape = (sub_group_size, n_sub_groups_per_work_group)

    _sum_and_set_items_if = _make_sum_and_set_items_if_kernel_func()

    # ???: how does this strategy compares to having each thread reducing N contiguous
    # items ?
    @dpex.kernel
    # fmt: off
    def partial_sum_reduction(
        summands,    # IN        (sum_axis_size, n_cols)
        result,      # OUT       (math.ceil(size / (2 * reduction_block_size), n_cols)
    ):
        # fmt: on
        # NB: This kernel only perform a partial reduction

        # The work groups are mapped to window of items in the input `summands` of size
        # `(reduction_block_size, sub_group_size)`,
        # where `reduction_block_size = 2 * n_sub_groups_per_work_group` such that the
        # windows never overlap and create a grid that cover all the items. Each
        # work group is responsible for summing all items in the window it spans over
        # axis 0, resulting in a output of shape `(1, sub_group_size)`.

        # NB: the axis in the following dpex calls are reversed, so the kernel further
        # reads like a SYCL kernel that maps 2D group size with a row-major order,
        # despite that `numba_dpex` chose to mimic the column-major order style of
        # mapping 2D group sizes in cuda.

        # The work groups are indexed in row-major order. From this let's deduce the
        # position of the window within the column...
        local_block_id_in_col = dpex.get_group_id(one_idx)

        # Let's map the current work item to an index in a 2D grid, where the
        # `work_group_size` work items are mapped in row-major order to the array
        # of size `(n_sub_groups_per_work_group, sub_group_size)`.
        local_row_idx = dpex.get_local_id(one_idx)     # 2D idx, first coordinate
        local_col_idx = dpex.get_local_id(zero_idx)    # 2D idx, second coordinate

        # This way, each row in the 2D index can be seen as mapped to two rows in the
        # corresponding window of items of the input `summands`, with the first row of
        # work items being mapped to the first row of the input...
        first_row_idx = local_block_id_in_col * reduction_block_size

        # ... and the current work item, with index `local_row_idx`, being mapped to
        # the following row of the input - which is also the row coordinate of the
        # first term this work item is responsible to sum in the coming sum step:

        # NB: we use augend/addend vocabulary
        # in sum x + y, x is augend, y is addend
        # https://www.quora.com/What-is-Augend-and-Addend
        augend_row_idx = first_row_idx + local_row_idx

        # The addend can be chosen arbitrarily (providing that different work items
        # don't sum overlapping terms!). Let's use a mapping that is consistent with
        # the choice in the kernel for axis 1, so that in both kernels the code reads
        # familiar.
        addend_row_idx = augend_row_idx + n_sub_groups_per_work_group

        # NB: because of how the 2D index on work items was defined, contiguous work
        # items belonging to the same sub groups in the grid always span items in the
        # data that are located in the same row, and so are contiguous, which ensures
        # optimal memory access patterns by sub groups (refer to explanations in the
        # kernel for axis 1 to understand how this access pattern is better). It is
        # enough to ensure this pattern at the level of sub groups, and one can see
        # that, contrarily to what is seen in the kernel for axis 1, it does not
        # depends on how the pair (augend, addend) are chosen.

        # Then this sum is written into a reserved slot in an array `local_values` in
        # local memory of size (n_sub_groups_per_work_group, sub_group_size) (i.e one
        # slot for each work item and two items in the window), and yet again
        # contiguous work items write into contiguous slots.
        local_values = dpex.local.array(local_values_size, dtype=dtype)

        # The current work item use the following second coordinate (given by the
        # position of the window in the grid of windows, and by the local position of
        # the work item in the 2D index):
        col_idx = (
            (dpex.get_group_id(zero) * sub_group_size) + local_col_idx
        )

        sum_axis_size = summands.shape[zero_idx]
        _prepare_local_memory(
            local_row_idx,
            local_col_idx,
            col_idx,
            augend_row_idx,
            addend_row_idx,
            sum_axis_size,
            summands,
            # OUT
            local_values
        )

        dpex.barrier(dpex.LOCAL_MEM_FENCE)

        # Then, the sums of two scalars that have been written in `local_array` are
        # further summed together into `local_array[0, :]`. At each iteration, half
        # of the remaining work items are discarded and will be left idle, and the
        # other half will sum together two value in the `local_array`, while using
        # a similar memory access pattern than seen at the previous step.
        n_active_sub_groups = n_sub_groups_per_work_group
        for i in range(n_local_iterations):
            # At each iteration, half of the remaining work items with the highest id
            # are discarded.
            n_active_sub_groups = n_active_sub_groups // two_as_a_long
            work_item_row_idx = first_row_idx + local_row_idx + n_active_sub_groups

            _sum_and_set_items_if(
                (
                    (local_row_idx < n_active_sub_groups) and
                    (col_idx < n_cols) and
                    (work_item_row_idx < sum_axis_size)
                ),
                (local_row_idx, local_col_idx),
                (local_row_idx, local_col_idx),
                # TODO: avoid compute the sum if the condition is not met ?
                (local_row_idx + n_active_sub_groups, local_col_idx),
                local_values,
                # OUT
                local_values
            )

            dpex.barrier(dpex.LOCAL_MEM_FENCE)

        # At this point local_values[0, :] + local_values[1, :] is equal to the sum of
        # all elements in summands that have been covered by the work group, we write
        # it into global memory
        _sum_and_set_items_if(
            (local_row_idx == zero_idx) and (col_idx < n_cols),
            (local_block_id_in_col, col_idx),
            (zero_idx, local_col_idx),
            (one_idx, local_col_idx),
            local_values,
            result
        )

    # HACK 906: see sklearn_numba_dpex.patches.tests.test_patches.test_need_to_workaround_numba_dpex_906  # noqa
    @dpex.func
    # fmt: off
    def _prepare_local_memory(
        local_row_idx,      # PARAM
        local_col_idx,      # PARAM
        col_idx,            # PARAM
        augend_row_idx,     # PARAM
        addend_row_idx,     # PARAM
        sum_axis_size,      # PARAM
        summands,           # IN
        local_values,    # OUT
    ):
        # fmt: on
        # We must be careful to not read items outside of the array !
        sum_axis_size = summands.shape[zero_idx]
        if (col_idx >= n_cols) or (augend_row_idx >= sum_axis_size):
            local_values[local_row_idx, local_col_idx] = zero
        elif addend_row_idx >= sum_axis_size:
            local_values[local_row_idx, local_col_idx] = fused_elementwise_func(
                summands[augend_row_idx, col_idx]
            )
        else:
            local_values[local_row_idx, local_col_idx] = fused_elementwise_func(
                summands[augend_row_idx, col_idx]
            ) + fused_elementwise_func(summands[addend_row_idx, col_idx])

    return work_group_shape, reduction_block_size, partial_sum_reduction


@lru_cache
def _prepare_sum_reduction_2d_axis1(
    n_rows, work_group_size, fused_elementwise_func, dtype, device
):

    if fused_elementwise_func is None:

        @dpex.func
        def fused_elementwise_func_(x):
            return x

    else:
        fused_elementwise_func_ = dpex.func(fused_elementwise_func)

    input_work_group_size = work_group_size
    work_group_size = _check_max_work_group_size(
        work_group_size, device, required_local_memory_per_item=np.dtype(dtype).itemsize
    )
    if work_group_size == input_work_group_size:
        check_power_of_2(work_group_size)
    else:
        # Round to the maximum smaller power of two
        work_group_size = get_maximum_power_of_2_smaller_than(work_group_size)

    (
        work_group_shape,
        reduction_block_size,
        partial_sum_reduction,
    ) = _make_partial_sum_reduction_2d_axis1_kernel(
        n_rows, work_group_size, fused_elementwise_func_, dtype
    )

    if fused_elementwise_func is None:
        partial_sum_reduction_nofunc = partial_sum_reduction
    else:
        *_, partial_sum_reduction_nofunc = _make_partial_sum_reduction_2d_axis1_kernel(
            n_rows, work_group_size, fused_elementwise_func_, dtype
        )

    get_result_shape = lambda result_sum_axis_size: (n_rows, result_sum_axis_size)
    get_global_size = lambda result_sum_axis_size: (
        result_sum_axis_size * work_group_size,
        n_rows,
    )

    kernels = (partial_sum_reduction, partial_sum_reduction_nofunc)
    shape_update_fn = (get_result_shape, get_global_size)
    return (work_group_shape, (kernels, reduction_block_size), shape_update_fn)


def _make_partial_sum_reduction_2d_axis1_kernel(
    n_rows, work_group_size, fused_elementwise_func, dtype
):
    """Compute a partial sum along axis 1 within each work group

    Each work group performs a sum of all the values in a window of size:
    `(1, 2 * work_group_size)`.

    The values of input array of shape `(n_rows, n_cols)` are partially summed
    to get a result array of shape `(n_rows, n_cols / (2 * work_group_size))`.
    """
    # ???: how does this strategy compare to having each thread reduce a chunk
    # of contiguous items?

    zero = dtype(0.0)
    one_idx = np.int64(1)
    minus_one_idx = np.int64(-1)
    two_as_a_long = np.int64(2)

    # Number of iteration in each execution of the kernel:
    n_local_iterations = np.int64(math.log2(work_group_size) - 1)
    reduction_block_size = 2 * work_group_size
    work_group_shape = (work_group_size, 1)

    _sum_and_set_items_if = _make_sum_and_set_items_if_kernel_func()

    @dpex.kernel
    # fmt: off
    def partial_sum_reduction(
        summands,    # IN        (n_rows, n_cols)
        result,      # OUT       (n_rows, math.ceil(n_cols / (2 * work_group_size),)
    ):
        # fmt: on
        # Each work group processes a window of the `summands` input array with
        # shape `(1, reduction_block_size)` with `reduction_block_size = 2 *
        # work_group_size`, and is responsible for summing all values in the window
        # it spans. The windows never overlap and create a grid that
        # cover all the values of the `summands` array.

        # NB: the axis in the following dpex calls are reversed, so the kernel further
        # reads like a SYCL kernel that maps 2D group size with a row-major order,
        # despite that `numba_dpex` chose to mimic the column-major order style of
        # mapping 2D group sizes in cuda.

        # The work groups are indexed in row-major order, from that let's deduce the
        # row of `summands` to process by work items in `group_id`...
        row_idx = dpex.get_group_id(one_idx)

        # ... and the position of the window within this row, ranging from 0
        # (first window in the row) to `n_work_groups_per_row - 1` (last window
        # in the row):
        local_work_group_id_in_row = dpex.get_group_id(zero_idx)

        # Since all windows have size `reduction_block_size`, the position of the first
        # item in the window is given by:
        first_value_idx = local_work_group_id_in_row * reduction_block_size

        # To sum up, this current work group will sum items with coordinates
        # (`row_idx`, `col_idx`), with `col_idx` ranging from `first_value_idx`
        # (first item in the window) to `first_value_idx + work_group_size - 1` (last
        # item in the window).

        # The current work item is indexed locally within the group of work items, with
        # index `local_work_id` that can range from `0` (first item in the work group)
        # to `work_group_size - 1` (last item in the work group)
        local_work_id = dpex.get_local_id(zero_idx)

        # Let's remember the size of the array to ensure that the last window in the
        # row do not try to access items outside the buffer.
        sum_axis_size = summands.shape[minus_one_idx]

        # To begin with, each work item sums two items from the window that is spanned
        # by this work group. The two items are chosen such that contiguous work items
        # in the work group read contiguous items in the window, i.e the current work
        # item reads a value at position `local_work_id` (relatively to the first item
        # of the window), and the second value `work_group_size` items further
        # (remember that `reduction_block_size = 2 * work_group_size`, so there are
        # exactly two values to read for each work item).
        # This memory access pattern is more efficient because:
        # - it prevents bank conflicts (i.e serializing of write operations when
        # threads access simultaneously close memory addresses in the same bank)
        # - each sub group can coalesce the read operation, thus improving IO.

        # Then this sum is written into a reserved slot in an array `local_values` in
        # local memory of size `work_group_size` (i.e one slot for each work item and
        # two items in the window), such that contiguous work items write into
        # contiguous slots (yet again, this is a nicer memory access pattern).

        # NB: we use augend/addend vocabulary
        # in sum x + y, x is augend, y is addend
        # https://www.quora.com/What-is-Augend-and-Addend
        augend_idx = first_value_idx + local_work_id
        addend_idx = first_value_idx + work_group_size + local_work_id

        local_values = dpex.local.array(work_group_size, dtype=dtype)

        _prepare_local_memory(
            local_work_id,
            row_idx,
            augend_idx,
            addend_idx,
            sum_axis_size,
            summands,
            # OUT
            local_values
        )

        dpex.barrier(dpex.LOCAL_MEM_FENCE)

        # Then, the sums of two scalars that have been written in `local_array` are
        # further summed together into `local_array[0]`. At each iteration, half
        # of the remaining work items are discarded and will be left idle, and the
        # other half will sum together two value in the `local_array`, while using
        # a similar memory access pattern than seen at the previous step.
        n_active_work_items = work_group_size
        for i in range(n_local_iterations):
            # At each iteration, half of the remaining work items with the highest id
            # are discarded.
            n_active_work_items = n_active_work_items // two_as_a_long
            work_item_idx = first_value_idx + local_work_id + n_active_work_items

            # Yet again, the remaining work items choose two values to sum such that
            # contiguous work items read and write into contiguous slots of
            # `local_values`.
            _sum_and_set_items_if(
                (
                    (local_work_id < n_active_work_items) and
                    (work_item_idx < sum_axis_size)
                ),
                local_work_id,
                local_work_id,
                # TODO: avoid compute the sum if the condition is not met ?
                local_work_id + n_active_work_items,
                local_values,
                # OUT
                local_values
            )

            dpex.barrier(dpex.LOCAL_MEM_FENCE)

        # At this point local_values[0] + local_values[1] is equal to the sum of all
        # elements in summands that have been covered by the work group, we write it
        # into global memory
        _sum_and_set_items_if(
            local_work_id == zero_idx,
            (row_idx, local_work_group_id_in_row),
            zero_idx,
            one_idx,
            local_values,
            # OUT
            result
        )

    # HACK 906: see sklearn_numba_dpex.patches.tests.test_patches.test_need_to_workaround_numba_dpex_906  # noqa
    @dpex.func
    # fmt: off
    def _prepare_local_memory(
        local_work_id,          # PARAM
        row_idx,                # PARAM
        augend_idx,             # PARAM
        addend_idx,             # PARAM
        sum_axis_size,          # PARAM
        summands,               # IN
        local_values,           # OUT
    ):
        # fmt: on
        # We must be careful to not read items outside of the array !
        if augend_idx >= sum_axis_size:
            local_values[local_work_id] = zero
        elif addend_idx >= sum_axis_size:
            local_values[local_work_id] = fused_elementwise_func(
                summands[row_idx, augend_idx]
            )
        else:
            local_values[local_work_id] = fused_elementwise_func(
                summands[row_idx, augend_idx]
            ) + fused_elementwise_func(summands[row_idx, addend_idx])

    return work_group_shape, reduction_block_size, partial_sum_reduction


# HACK 906: see sklearn_numba_dpex.patches.tests.test_patches.test_need_to_workaround_numba_dpex_906  # noqa
def _make_sum_and_set_items_if_kernel_func():
    @dpex.func
    # fmt: off
    def set_sum_of_items_kernel_func(
            condition,          # PARAM
            result_idx,         # PARAM
            addend_idx,         # PARAM
            augend_idx,         # PARAN
            summands,           # IN
            result              # OUT
            ):
        # fmt: on
        if not condition:
            return

        result[result_idx] = summands[addend_idx] + summands[augend_idx]

    return set_sum_of_items_kernel_func
