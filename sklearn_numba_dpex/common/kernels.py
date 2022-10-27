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

import numpy as np
import dpctl.tensor as dpt
import numba_dpex as dpex


zero_idx = np.int64(0)


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
def make_broadcast_division_1d_2d_kernel(size0, size1, work_group_size):
    global_size = math.ceil(size1 / work_group_size) * work_group_size

    # NB: inplace. # Optimized for C-contiguous array and for
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
def make_sum_reduction_1d_kernel(size, work_group_size, device, dtype):
    """numba_dpex does not provide tools such as `cuda.reduce` so we implement from
    scratch a reduction strategy. The strategy relies on the commutativity of the
    operation used for the reduction, thus allowing to reduce the input in any order.

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

    NB: work_group_size is assumed to be a power of 2.
    """
    # Number of iteration in each execution of the kernel:
    local_n_iterations = np.int64(math.floor(math.log2(work_group_size)) - 1)

    zero = dtype(0.0)
    two_long = np.int64(2)
    one_idx = np.int64(1)

    @dpex.kernel
    # fmt: off
    def partial_sum_reduction(
        summands,    # IN        (size,)
        result,      # OUT       (math.ceil(size / (2 * work_group_size),)
    ):
    # fmt: on
        # NB: This kernel only perform a partial reduction
        group_id = dpex.get_group_id(zero_idx)
        local_work_id = dpex.get_local_id(zero_idx)
        first_work_id = local_work_id == zero_idx

        size = summands.shape[zero_idx]

        local_data = dpex.local.array(work_group_size, dtype=dtype)

        first_value_idx = group_id * work_group_size * two_long
        augend_idx = first_value_idx + local_work_id
        addend_idx = first_value_idx + work_group_size + local_work_id

        # Each work item reads two value in global memory and sum it into the local
        # memory
        if augend_idx >= size:
            local_data[local_work_id] = zero
        elif addend_idx >= size:
            local_data[local_work_id] = summands[augend_idx]
        else:
            local_data[local_work_id] = summands[augend_idx] + summands[addend_idx]

        dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)
        current_n_work_items = work_group_size
        for i in range(local_n_iterations):
            # We discard half of the remaining active work items at each iteration
            current_n_work_items = current_n_work_items // two_long
            if local_work_id < current_n_work_items:
                local_data[local_work_id] += local_data[
                    local_work_id + current_n_work_items
                ]

            dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

        # At this point local_data[0] = local_data[1]  is equal to the sum of all
        # elements in summands that have been covered by the work group, we write it
        # into global memory
        if first_work_id:
            result[group_id] = local_data[zero_idx] + local_data[one_idx]

    # As many partial reductions as necessary are chained until only one element
    # remains.
    kernels_and_empty_tensors_pairs = []
    n_groups = size
    # TODO: at some point, the cost of scheduling the kernel is more than the cost of
    # running the reduction iteration. At this point the loop should stop and then a
    # single work item should iterates one time on the remaining values to finish the
    # reduction.
    while n_groups > 1:
        n_groups = math.ceil(n_groups / (2 * work_group_size))
        global_size = n_groups * work_group_size
        kernel = partial_sum_reduction[global_size, work_group_size]
        result = dpt.empty(n_groups, dtype=dtype, device=device)
        kernels_and_empty_tensors_pairs.append((kernel, result))

    def sum_reduction(summands):
        for kernel, result in kernels_and_empty_tensors_pairs:
            kernel(summands, result)
            summands = result
        return result

    return sum_reduction
