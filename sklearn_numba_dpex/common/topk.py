# This implementation takes inspiration from other open-source implementations, such as
# [1] https://github.com/pytorch/pytorch/blob/master/caffe2/operators/top_k_radix_selection.cuh  # noqa
# or
# [2] https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell  # noqa

# TODO: apply Dr. TopK optimizations
# (https://dl.acm.org/doi/pdf/10.1145/3458817.3476141)

import math
from functools import lru_cache

import dpctl.tensor as dpt
import numba_dpex as dpex
import numpy as np

from sklearn_numba_dpex.common._utils import (
    _check_max_work_group_size,
    _get_sequential_processing_device,
    check_power_of_2,
    get_maximum_power_of_2_smaller_than,
)
from sklearn_numba_dpex.common.kernels import (
    make_initialize_to_zeros_2d_kernel,
    make_sum_reduction_2d_kernel,
)

zero_idx = np.int64(0)
count_one_as_a_long = np.int64(1)
count_one_as_an_int = np.int32(1)


# The Top-K implemented in this file uses radix sorting. Radix sorting consists in
# using lexicographical order on bits to sort the items. It assumes that the
# lexicographical order on the bits of the items that are being sorted matches their
# natural ordering.

# However, this is not true for sets that mix positive and negative items, given that
# the dtypes that are supported set the first bit to 1 to encode the negative sign, and
# 0 to encode the positive sign, in such a way that negative floats are lesser than
# positive floats using the natural order while the opposite is true with
# lexicograpbical order on bits.

# Still, it is possible to enable radix sorting for dtypes that don't follow the
# lexicographical order, providing a bijection can be found with an unsigned type, such
# as the lexicographical order on the unsigned dtype matches the natural order on the
# input dtype. The target dtype must also support bitwise operations. Thus, float32 is
# mapped to uint32, float64 is mapped to uint64,...

# In practice such bijections can be found for all dtypes and are cheap to compute. The
# code that follows defines bijections to unsigned integers for float32 and float64
# floats, with the following bitwise transformations:
#    - reinterpret the float as an unsigned integer (i.e the same buffer used to encode
#      a float is used to encode an unsigned integer. Said differently, first transform
#      the float to the integer such as the bits used to encode the integer are the
#      same than the bits used to encode the float)
#    - if the float item is positive, set the first bit of the unsigned integer to 1
#    - if the float item is negative, flip all of the bits of the unsigned integer
#      (i.e set to 1 if bit is 0, else 0)
#
# See also:
# https://stackoverflow.com/questions/4640906/radix-sort-sorting-a-float-data/4641059#4641059  # noqa
#
# Note that our current usecases of the use of radix (selecting TopK distances, that
# are by definition positive floats)  only involve positive floats.
# TODO: for arrays of positive floats the mapping trick could be skipped ?

# dict that maps dtypes to the dtype of the space in which the radix sorting will be
# actually done
uint_type_mapping = {np.float32: np.uint32, np.float64: np.uint64}


# The following closure define a device function that transforms items to their
# counterpart in the sorting space. For a given dtype, it returns a function that takes
# as input an item from the input dtype, that has prealably been reinterpreted as an
# item of the target dtype (see `uint_type_mapping`). It returns another item from the
# target dtype such as the lexicographical order on transformed items matches the
# natural order on the the input dtype.

# The second closure define the inverse device function.


def _make_lexicographical_mapping_kernel_func(dtype):
    n_bits_per_item = _get_n_bits_per_item(dtype)
    sign_bit_idx = np.int32(n_bits_per_item - 1)

    uint_type = uint_type_mapping[dtype]
    sign_mask = uint_type(2 ** (sign_bit_idx))

    @dpex.func
    def lexicographical_mapping(item):
        mask = (-(item >> sign_bit_idx)) | sign_mask
        return item ^ mask

    return lexicographical_mapping


def _make_lexicographical_unmapping_kernel_func(dtype):
    n_bits_per_item = _get_n_bits_per_item(dtype)
    sign_bit_idx = np.int32(n_bits_per_item - 1)
    uint_type = uint_type_mapping[dtype]
    sign_mask = uint_type(2 ** (sign_bit_idx))

    @dpex.func
    def lexicographical_unmapping(item):
        mask = ((item >> sign_bit_idx) - 1) | sign_mask
        return item ^ mask

    return lexicographical_unmapping


def topk(array_in, k, group_sizes=None):
    """Return an array of size k containing the k greatest values found in `array_in`.

    Parameters
    ----------
    array_in : dpctl.tensor array
        Input array in which looking for the top k values.

    k: int
        Number of values to search for.

    group_sizes: tuple of int
        Can be optionnally used to configure `(work_group_size, sub_group_size)`
        parameters for the kernels.


    Returns
    -------
    result : dpctl.tensor array
        An array of size k containing the k greatest valus found in `array_in`.

    Notes
    -----
    The output is not deterministic: in case of ties (severa)

    The implementation has been extensively inspired by the "Fused Fixed" strategy
    exposed in [1]_, along with its reference implementatino by the same authors [2]_,
    and the reader can also refer to the complementary slide deck [3]_  with schemas
    that intuitively explain the main computation.
    """
    k_in_subset, threshold, work_group_size, dtype, device = _get_topk_threshold(
        array_in, k, group_sizes
    )
    result = dpt.empty(sh=(k,), dtype=dtype, device=device)
    index_buffer = dpt.zeros(sh=(1,), dtype=np.int32)
    n_threshold_occurences = int(k_in_subset[0])
    gather_topk_kernel = _make_gather_topk_kernel(
        array_in.shape[0], k, n_threshold_occurences, work_group_size
    )

    gather_topk_kernel(array_in, threshold, k_in_subset, index_buffer, result)
    return result


def topk_idx(array_in, k, group_sizes=None):
    """Return an array of size k containing the indices of the k greatest values found
    in `array_in`.

    Parameters
    ----------
    array_in : dpctl.tensor array
        Input array in which looking for the top k values.

    k: int
        Number of values to search for.

    group_sizes: tuple of int
        Can be optionnally used to configure `(work_group_size, sub_group_size)`
        parameters for the kernels.


    Returns
    -------
    result : dpctl.tensor array
        An array of size k with dtype int64 containing the indices of the k greatest
        valus found in  `array_in`.

    Notes
    -----
    The output is not deterministic:
        - the order of the output is undefined. Successive calls can return the same
        items in different order.

        - If there are more indices for the smallest top k value than the number of
        time this value occurs among the top k , then the indices that are returned
        for this value can be different between two successive calls.

    """

    k_in_subset, threshold, work_group_size, dtype, device = _get_topk_threshold(
        array_in, k, group_sizes
    )
    result = dpt.empty(sh=(k,), dtype=np.int64, device=device)
    index_buffer = dpt.zeros(sh=(1,), dtype=np.int32)
    gather_topk_kernel = _make_gather_topk_idx_kernel(
        array_in.shape[0], k, work_group_size
    )

    gather_topk_kernel(array_in, threshold, k_in_subset, index_buffer, result)
    return result


def _get_topk_threshold(array_in, k, group_sizes):
    n_items = len(array_in)

    if n_items < k:
        raise ValueError(
            "Expected k to be greater than or equal to the number of items in the "
            f"search space, but got k={k} and len(array_in)={n_items}"
        )

    dtype = np.dtype(array_in.dtype).type
    if dtype not in uint_type_mapping:
        raise ValueError(
            f"topk currently only supports dtypes in {uint_type_mapping.keys()}, but "
            f"got dtype={dtype} ."
        )
    uint_type = uint_type_mapping[dtype]
    n_bits_per_item = _get_n_bits_per_item(dtype)

    device = array_in.device.sycl_device

    if group_sizes is not None:
        work_group_size, sub_group_size = group_sizes
    else:
        work_group_size = device.max_work_group_size
        sub_group_size = min(device.sub_group_sizes)

    global_mem_cache_size = device.global_mem_cache_size
    counts_private_copies_max_cache_occupancy = 0.7

    (
        radix_size,
        radix_bits,
        n_counts_private_copies,
        create_radix_histogram_kernel,
    ) = _make_create_radix_histogram_kernel(
        n_items,
        "max" if group_sizes is None else work_group_size,
        sub_group_size,
        global_mem_cache_size,
        counts_private_copies_max_cache_occupancy,
        dtype,
        device,
    )

    reduce_privatized_counts = make_sum_reduction_2d_kernel(
        size0=n_counts_private_copies,
        size1=radix_size,
        work_group_size=work_group_size,
        device=device,
        dtype=np.int64,
        axis=0,
        sub_group_size=sub_group_size,
    )

    check_radix_histogram = _make_check_radix_histogram_kernel(radix_size, dtype)

    initialize_privatized_counts = make_initialize_to_zeros_2d_kernel(
        n_counts_private_copies, radix_size, work_group_size, dtype
    )

    (
        sequential_processing_device,
        sequential_processing_on_different_device,
    ) = _get_sequential_processing_device(device)

    k_in_subset = dpt.asarray([k], dtype=np.int32, device=sequential_processing_device)

    # Reinterpret buffer as uint so we can use bitwise compute
    array_in_uint = dpt.usm_ndarray(
        shape=array_in.shape,
        dtype=uint_type,
        buffer=array_in,
    )

    privatized_counts = dpt.zeros(
        sh=(n_counts_private_copies, radix_size), dtype=np.int64, device=device
    )
    mask_for_desired_value = desired_masked_value = dpt.asarray([0], dtype=uint_type)
    radix_position = dpt.asarray([n_bits_per_item - radix_bits], dtype=uint_type)
    terminate = dpt.asarray([0], dtype=np.int32, device=sequential_processing_device)

    if sequential_processing_on_different_device:
        check_radix_histogram = check_radix_histogram.configure(
            sycl_queue=terminate.sycl_queue,
            global_size=[1],
            local_size=[1],
        )

    while True:
        create_radix_histogram_kernel(
            array_in_uint,
            mask_for_desired_value,
            desired_masked_value,
            radix_position,
            # OUT
            privatized_counts,
        )

        counts = dpt.reshape(reduce_privatized_counts(privatized_counts), (-1,))

        if sequential_processing_on_different_device:
            counts = counts.to_device(sequential_processing_device)
            mask_for_desired_value = dpt.asarray(
                mask_for_desired_value, device=sequential_processing_device
            )
            desired_masked_value = dpt.asarray(
                desired_masked_value, device=sequential_processing_device
            )
            radix_position = dpt.asarray(
                radix_position, device=sequential_processing_device
            )

        check_radix_histogram(
            counts,
            # INOUT
            k_in_subset,
            radix_position,
            mask_for_desired_value,
            desired_masked_value,
            # OUT
            terminate,
        )

        if sequential_processing_on_different_device:
            desired_masked_value = desired_masked_value.to_device(device)

        if int(terminate[0]) == 1:
            break

        if sequential_processing_on_different_device:
            radix_position = radix_position.to_device(device)
            mask_for_desired_value = mask_for_desired_value.to_device(device)
            desired_masked_value = desired_masked_value.to_device(device)

        initialize_privatized_counts(privatized_counts)

    if sequential_processing_on_different_device:
        k_in_subset = k_in_subset.to_device(device)

    # reinterpret the threshold back to a dtype item
    threshold = dpt.usm_ndarray(
        shape=desired_masked_value.shape, dtype=dtype, buffer=desired_masked_value
    )

    return k_in_subset, threshold, work_group_size, dtype, device


def _get_n_bits_per_item(dtype):
    """Returns number of bits in items with given dtype
    e.g, returns:
        - 32 for float32
        - 64 for float64
    """
    return np.dtype(dtype).itemsize * 8


@lru_cache
def _make_create_radix_histogram_kernel(
    n_items,
    work_group_size,
    sub_group_size,
    global_mem_cache_size,
    counts_private_copies_max_cache_occupancy,
    dtype,
    device,
):
    histogram_dtype = np.int64

    input_work_group_size = work_group_size
    work_group_size = _check_max_work_group_size(
        work_group_size, device, np.dtype(histogram_dtype).itemsize
    )

    check_power_of_2(sub_group_size)

    if work_group_size == input_work_group_size:

        n_sub_group_per_work_group = work_group_size // sub_group_size

        try:
            check_power_of_2(n_sub_group_per_work_group)
            power_of_2_n_sub_groups = True
        except ValueError:
            power_of_2_n_sub_groups = False

        if not (
            ((n_sub_group_per_work_group * sub_group_size) == work_group_size)
            and power_of_2_n_sub_groups
        ):
            raise ValueError(
                "Expected work_group_size to be a power-of-two multiple of"
                f" sub_group_size but got sub_group_size={sub_group_size} and"
                f" work_group_size={work_group_size}"
            )

    else:
        n_sub_group_per_work_group = get_maximum_power_of_2_smaller_than(
            work_group_size / sub_group_size
        )
        work_group_size = n_sub_group_per_work_group * sub_group_size

    # The size of the radix is chosen such as the size of intermediate objects that
    # build in shared memory amounts to one int64 item per work item.
    radix_size = sub_group_size
    radix_bits = int(math.log2(radix_size))
    local_counts_size = (n_sub_group_per_work_group, radix_size)

    # Number of iterations when reducing the per-sub group histograms to per-work group
    # histogram in work groups
    n_sum_reduction_steps = math.log2(n_sub_group_per_work_group)

    n_work_groups = math.ceil(n_items / work_group_size)
    global_size = n_work_groups * work_group_size

    n_counts_items = radix_size
    n_counts_bytes = np.dtype(np.int64).itemsize * n_counts_items

    # TODO: control that this value is not higher than the number of sub-groups of size
    # sub_group_size that can effectively run concurrently. We should fetch this
    # information and apply it here.
    # See https://github.com/IntelPython/dpctl/issues/1033
    n_counts_private_copies = (
        global_mem_cache_size * counts_private_copies_max_cache_occupancy
    ) // n_counts_bytes

    n_counts_private_copies = int(min(n_work_groups, n_counts_private_copies))

    lexicographical_mapping = _make_lexicographical_mapping_kernel_func(dtype)

    uint_type = uint_type_mapping[dtype]
    zero_as_uint_dtype = uint_type(0)
    one_as_uint_dtype = uint_type(1)
    two_as_a_long = np.int64(2)

    select_last_radix_bits_mask = (
        one_as_uint_dtype << np.uint32(radix_bits)
    ) - one_as_uint_dtype

    @dpex.kernel
    # fmt: off
    def create_radix_histogram(
        array_in_uint,                # IN READ-ONLY  (n_items,)
        mask_for_desired_value,       # IN            (1,)
        desired_masked_value,         # IN            (1,)
        radix_position,               # IN            (1,)
        privatized_counts,            # OUT           (n_counts_private_copies, radix_size)  # noqa
    ):
        # fmt: on
        """
        This kernel is the core of the radix top-k algorithm. This top-k implementation
        is well adapted to GPU architectures, but can suffer from bad performance
        depending on the distribution of the input data. It will be supplemented with
        additional optimizations to ensure base performance for all input
        distributions.

        Radix top-k consists in computing the histogram of the number of occurences of
        all possible radixes (i.e subsequence of the sequence of bits) at a given
        radix position for all the items in a subset of `array_in_uint`. This
        histogram gives partial information on the ordering of the items. Computing
        sequentially the histogram for different radixes finally results in converging
        to the top-k greatest items.

        See e.g https://en.wikipedia.org/wiki/Radix_sort for more extensive description
        of radix-based sorting algorithms.

        At the given iteration, the exact subset of items that are considered is framed
        by bitwise conditions depending on `mask_for_desired_value` and
        `desired_mask_value`, that are defined such that only items that have not been
        discarded by the previous iterations match the condition. During the first
        iteration, the condition is true for all items.
        """

        item_idx = dpex.get_global_id(zero_idx)
        local_work_id = dpex.get_local_id(zero_idx)
        local_subgroup = local_work_id // sub_group_size
        local_subgroup_work_id = local_work_id % sub_group_size

        # Initialize the shared memory in the work group
        local_counts = dpex.local.array(local_counts_size, dtype=histogram_dtype)
        local_counts[local_subgroup, local_subgroup_work_id] = zero_idx

        dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

        # If item_idx is outside the bounds of the input, ignore this location.
        is_in_bounds = item_idx < n_items
        if is_in_bounds:
            item = array_in_uint[item_idx]

            # Biject the item such as lexicographical order in the target space is
            # equivalent to the natural order in the the source space.
            item_lexicographically_mapped = lexicographical_mapping(item)

            mask_for_desired_value_ = mask_for_desired_value[zero_idx]
            desired_masked_value_ = desired_masked_value[zero_idx]
            radix_position_ = radix_position[zero_idx]

        # The item is included to the radix histogram if the sequence of bits at the
        # positions defined by `mask_for_desired_value_` is equal to the sequence of
        # bits in `desired_masked_value_`
        includes_in_histogram = is_in_bounds and (
            (mask_for_desired_value_ == zero_as_uint_dtype)
            or (
                (item_lexicographically_mapped & mask_for_desired_value_)
                == desired_masked_value_
            )
        )

        if includes_in_histogram:
            # Extract the value encoded by the next radix_bits bits starting at
            # position radix_position_ (reading bits from left to right)
            # NB: resulting value is in interval [0, radix_size[
            digit_in_radix = (
                item_lexicographically_mapped >> radix_position_
            ) & select_last_radix_bits_mask

            # Increment the count of the value `digit_in_radix` by one in the
            # histogram.
            # NB: all threads are executing this step concurrently, the increment must
            # be protected by the use of atomics. To alleviate the performance impact
            # of conflicts on atomics operations, the different sub-groups of the
            # work group write to a privatized shared memory buffer, such that conflicts
            # can only occurs between threads of a same work group. The privatized
            # buffers are then merged in the step afterward.
            # ???: maybe instead on this we could have full privatization
            dpex.atomic.add(
                local_counts, (local_subgroup, digit_in_radix), count_one_as_a_long
            )

        dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

        # This is the merge step, where the memory buffers of all sub-groups are summed
        # together into the first buffer local_counts[0], in a bracket manner.
        # NB: apparently cuda have much more powerful intrinsics to perform steps like
        # this, such as ballot voting ? are there `SYCL` or `numba_dpex` roadmaps to
        # enable the same intrinsics ?
        reduction_active_subgroups = n_sub_group_per_work_group
        for _ in range(n_sum_reduction_steps):
            reduction_active_subgroups = reduction_active_subgroups // two_as_a_long
            if local_subgroup < reduction_active_subgroups:
                local_counts[local_subgroup, local_subgroup_work_id] += local_counts[
                    local_subgroup + reduction_active_subgroups, local_subgroup_work_id
                ]

            dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

        # The current histogram is local to the current work group. Yet again
        # summing right away all histograms to a unique, global histogram in global
        # memory would give poor performance because it would require atomics that
        # would be subject to a lot of conflicts. Likewise, to circumvent this risk
        # partial sums are written to privatized buffers in global memory. The partial
        # buffers will be reduced to a single global histogram in a complementary
        # kernel.

        # Each work group is assigned an array of centroids in a round robin manner
        privatization_idx = (item_idx // work_group_size) % n_counts_private_copies

        if local_subgroup == zero_idx:
            dpex.atomic.add(
                privatized_counts, (privatization_idx, local_subgroup_work_id),
                local_counts[zero_idx, local_subgroup_work_id]
                )

    return (
        radix_size,
        radix_bits,
        n_counts_private_copies,
        create_radix_histogram[global_size, work_group_size],
    )


@lru_cache
def _make_check_radix_histogram_kernel(radix_size, dtype):
    radix_bits = int(math.log2(radix_size))
    lexicographical_unmapping = _make_lexicographical_unmapping_kernel_func(dtype)
    uint_type = uint_type_mapping[dtype]
    zero_as_uint_dtype = uint_type(0)

    @dpex.kernel
    # fmt: off
    def check_radix_histogram(
        counts,                        # IN           (radix_size,)
        k_in_subset,                   # INOUT        (1,)
        radix_position,                # INOUT        (1,)
        mask_for_desired_value,        # INOUT        (1,)
        desired_masked_value,          # INOUT        (1,)
        terminate,                     # OUT          (1,)
    ):
        # fmt: on
        k_in_subset_ = k_in_subset[zero_idx]
        radix_position_ = radix_position[zero_idx]
        desired_masked_value_ = desired_masked_value[zero_idx]

        # Read the histogram starting from the bucket corresponding to the highest
        # value for the current radix position, and in decreasing order.
        current_count_idx = radix_size - count_one_as_a_long
        for _ in range(radix_size):
            count = counts[current_count_idx]
            if count >= k_in_subset_:
                # The bucket of items matching the value for the current radix position
                # (equal to `current_count_idx`) contain the k-th highest value. New
                # mask parameters are chosen such that the next iteration will scan
                # for the k-th value in this bucket only.
                desired_masked_value_ = desired_masked_value_ | (
                    uint_type(current_count_idx) << radix_position_
                )
                break

            else:
                # The k-th greatest value is not in the current bucket of size `count`.
                # The k-th greatest value is also the (k-count)-th greatest value among
                # items whose value for the current radix position is strictly smaller
                # than the value of the current bucket of items (equal to
                # `current_count_idx`).
                k_in_subset_ -= count
                current_count_idx -= count_one_as_a_long

        # The top-k search has converged either if the last radix position that was
        # scanned is 0, or if `k` is 1 and a bucket of size 1 has also been found. In
        # any other case, creation of a new histogram will be computed for items
        # in the bucket with index `current_count_idx` only (and so on).
        terminate_ = (radix_position_ == zero_as_uint_dtype) or (
            (k_in_subset_ == count_one_as_a_long) and (count == count_one_as_a_long)
        )

        k_in_subset[zero_idx] = k_in_subset_

        if terminate_:
            terminate[zero_idx] = count_one_as_an_int
            # At this point, any value in the input data equal to
            # `desired_masked_value_` is the k-th greatest value, and the number of
            # values equal to `desired_masked_value_` amont the top-k values is exactly
            # `k_in_subset_`.
            desired_masked_value_ = lexicographical_unmapping(desired_masked_value_)
        else:
            # The current partial analysis with the current radixes seen was not enough
            # to find the k-th element. Let's inspect the next `radix_bits`.
            new_radix_position = radix_position_ - radix_bits
            if new_radix_position < zero_as_uint_dtype:
                new_radix_position = zero_as_uint_dtype
            radix_position[zero_idx] = new_radix_position
            mask_for_desired_value[zero_idx] |= (radix_size - 1) << radix_position_

        desired_masked_value[zero_idx] = desired_masked_value_

    return check_radix_histogram[1, 1]


@lru_cache
def _make_gather_topk_kernel(n_items, k, n_threshold_occurences, work_group_size):
    """The gather_topk kernel is the last step. By now the k-th greatest values and
    its number of occurences among the top-k values in the search space have been
    identified. The top-k values that are equal or greater than k, including the
    `n_threshold_occurrences` occurrences equal to the k-th greatest value, are written
    into the result array.

    The kernel is specialized depending on the value of `n_threshold_occurences`, since
    some optimizations are possible if this value is known at compile time.
    """
    global_size = math.ceil(n_items / work_group_size) * work_group_size

    if n_threshold_occurences == 1:

        @dpex.kernel
        # fmt: off
        def gather_topk_single_threshold_occurence(
            array_in,                          # IN READ-ONLY (n_items,)
            threshold,                         # IN           (1,)
            n_threshold_occurences,            # UNUSED BUFFER
            index_buffer,                      # BUFFER       (1,)
            result,                            # OUT          (k,)
        ):
            # fmt: on
            item_idx = dpex.get_global_id(zero_idx)

            if item_idx >= n_items:
                return

            if index_buffer[zero_idx] >= k:
                return

            threshold_ = threshold[zero_idx]
            item = array_in[item_idx]

            if item >= threshold_:
                index_buffer_ = dpex.atomic.add(
                    index_buffer, zero_idx, count_one_as_an_int)
                result[index_buffer_] = item

        return gather_topk_single_threshold_occurence[global_size, work_group_size]

    @dpex.kernel
    # fmt: off
    def gather_topk_multiple_threshold_occurences(
        array_in,                      # IN READ-ONLY (n_items,)
        threshold,                     # IN           (1,)
        n_threshold_occurences,        # IN           (1,)
        index_buffer,                  # BUFFER       (1,)
        result,                        # OUT          (k,)
    ):
        # fmt: on
        item_idx = dpex.get_global_id(zero_idx)

        if item_idx >= n_items:
            return

        threshold_ = threshold[zero_idx]
        n_threshold_occurences_ = n_threshold_occurences[zero_idx]

        # The `n_threshold_occurences_` first work items write the value of the
        # threshold at the end of the result array.
        if item_idx < n_threshold_occurences_:
            result[-item_idx-1] = threshold_

        # Then write the remaining `k - n_threshold_occurences_` that are strictly
        # greater than the threshold.
        k_ = k - n_threshold_occurences_

        if index_buffer[zero_idx] >= k_:
            return

        item = array_in[item_idx]

        if item <= threshold_:
            return

        index_buffer_ = dpex.atomic.add(index_buffer, zero_idx, count_one_as_an_int)
        result[index_buffer_] = item

    return gather_topk_multiple_threshold_occurences[global_size, work_group_size]


@lru_cache
def _make_gather_topk_idx_kernel(n_items, k, work_group_size):
    """Same than gather_topk kernel but return top-k indices rather than top-k values"""
    global_size = math.ceil(n_items / work_group_size) * work_group_size

    @dpex.kernel
    # fmt: off
    def gather_topk_idx(
        array_in,                          # IN READ-ONLY (n_items,)
        threshold,                         # IN           (1,)
        n_threshold_occurences,            # BUFFER       (1,)
        index_buffer,                      # BUFFER       (1,)
        result,                            # OUT          (k,)
    ):
        # fmt: on
        item_idx = dpex.get_global_id(zero_idx)

        if item_idx >= n_items:
            return

        if index_buffer[zero_idx] >= k:
            return

        threshold_ = threshold[zero_idx]
        item = array_in[item_idx]

        if item < threshold_:
            return

        if item > threshold_:
            index_buffer_ = dpex.atomic.add(index_buffer, zero_idx, count_one_as_an_int)
            result[index_buffer_] = item_idx
            return

        if n_threshold_occurences[zero_idx] <= zero_idx:
            return

        remaining_n_threshold_occurences = dpex.atomic.sub(
            n_threshold_occurences, zero_idx, count_one_as_an_int)

        if remaining_n_threshold_occurences > zero_idx:
            index_buffer_ = dpex.atomic.add(index_buffer, zero_idx, count_one_as_an_int)
            result[index_buffer_] = item_idx

    return gather_topk_idx[global_size, work_group_size]
