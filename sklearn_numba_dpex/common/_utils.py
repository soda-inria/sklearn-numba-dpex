import math
import warnings

import dpctl


def check_power_of_2(x):
    if x != 2 ** (math.floor(math.log2(x))):
        raise ValueError(f"Expected a power of 2, got {x}")
    return x


def get_maximum_power_of_2_smaller_than(x):
    return 2 ** (math.floor(math.log2(x)))


def _square(x):
    return x * x


def _minus(x, y):
    return x - y


def _plus(x, y):
    return x + y


def _divide_by(divisor):
    def _divide_closure(x):
        return x / divisor

    return _divide_closure


def _get_sequential_processing_device(device):
    """Returns a device most fitted for sequential processing (i.e a cpu rather than a
    gpu). If such a device is not found, returns the input device instead.

    Also returns a boolean that informs on wether the returned device is different than
    the input device."""
    if device.has_aspect_cpu:
        return device, False

    try:
        return dpctl.SyclDevice("cpu"), True
    except dpctl.SyclDeviceCreationError:
        warnings.warn("No CPU found, falling back to GPU for sequential instructions.")
        return device, False


def _check_max_work_group_size(
    work_group_size,
    device,
    required_local_memory_per_item,
    required_memory_constant=0,
    minimum_unallocated_buffer_size=1024,
):
    """For CPU devices, the value `device.max_work_group_size` seems to always be
    surprisingly large, up to several order of magnitude higher than the number of
    threads (for instance, having 8 threads and a `max_work_group_size` equal to
    8192). It means that, for CPUs, a work group schedules big batches of tasks per
    thread, and that only one work group will be executed at a time. Kernels that
    allocate an amount of local memory (i.e fast access memory shared by the work
    group) that scale with the size of the work group are at risk of overflowing the
    size of the local memory (given by `device.local_mem_size`, typically 32kB). So
    we need to scale down the size of the work groups with respect to
    `device.local_mem_size` to prevent overflowing.

    This is not an issue with GPU devices, for our kernels usually respect the rule of
    thumb of allocating in local memory about one item per thread, which fits the GPU
    architecture well and seems to be enough to prevent overflowing. With GPUs, max
    possible work group sizes are smaller, such that there's no oversubscription of
    tasks and GPUs will execute tasks of several work groups at once. For GPUs, this
    approach at local memory allocation is by design reliable, the amount of available
    local memory is enough to ensure that the memory needed by the running compute is
    allocated. As a consequence, the checks that are enforced for CPUs are not needed.

    NB: this function only applies an upper bound to the `work_group_size`. It might
    still be needed to apply other requirements, such that being a multiple of
    `sub_group_size` and/or a power of two.
    """
    max_work_group_size = device.max_work_group_size

    if work_group_size == "max" and not (
        device.has_aspect_cpu and required_local_memory_per_item > 0
    ):
        return device.max_work_group_size
    elif work_group_size == "max":
        return math.floor(
            (
                device.local_mem_size
                - required_memory_constant
                - minimum_unallocated_buffer_size
            )
            / required_local_memory_per_item
        )
    elif work_group_size > max_work_group_size:
        raise RuntimeError(
            f"Got work_group_size={work_group_size} but that is greater than the "
            "maximum supported work group size device.max_work_group_size="
            f"{device.max_work_group_size} for device {device.name}"
        )
    else:
        return work_group_size


# This is the value found for Intel Corpowork_group_size_ration TigerLake-LP GT2
# [Iris Xe Graphics] GPU.
_GLOBAL_MEM_CACHE_SIZE_DEFAULT = 1048576  # 2**20


# Work around https://github.com/IntelPython/dpctl/issues/1036
def _get_global_mem_cache_size(device):
    if (global_mem_cache_size := device.global_mem_cache_size) > 0:
        return global_mem_cache_size

    warnings.warn(
        "Can't inspect the available global memory cache size for the device "
        f"{device.name}. Please check that your drivers and runtime libraries are up "
        "to date, if this warning persists please report it at "
        "https://github.com/soda-inria/sklearn-numba-dpex/issues . The execution will "
        "continue with a default value for the cache size set to "
        f"{_GLOBAL_MEM_CACHE_SIZE_DEFAULT} bytes, which is assumed to be safe but "
        "might not be adapted to your device and cause a loss of performance.",
        RuntimeWarning,
    )

    return _GLOBAL_MEM_CACHE_SIZE_DEFAULT


def _enforce_matmul_like_work_group_geometry(
    work_group_size, sub_group_size, device, required_local_memory_per_item
):
    """There are several kernels, such as matrix multiplication, that like to have
    `work_group_size / (sub_group_size ** 2)` equal to a power of two. This helper
    either ensure that if `work_group_size` is automatically adjusted then it matches
    this rule, or if `work_group_size` is manually overriden it will raise an error if
    the rule is not met.

    The helper also returns the value of work_group_size / (sub_group_size ** 2).
    """

    input_work_group_size = work_group_size
    work_group_size = _check_max_work_group_size(
        work_group_size,
        device,
        required_local_memory_per_item=required_local_memory_per_item,
    )

    # This value is equal to the number of results per work item assuming
    # arithmetic_intensity_multiplier_X = arithmetic_intensity_multiplier_Y = 1 .
    # It is expected to be a power of two (base_nb_results_per_work_item_log2 < 1 is
    # possible.)
    work_group_size_ratio = work_group_size / (sub_group_size * sub_group_size)
    work_group_size_ratio_log2 = math.floor(math.log2(work_group_size_ratio))

    if work_group_size != input_work_group_size:
        base_nb_results_per_work_item = 2**work_group_size_ratio_log2
        work_group_size = int(
            base_nb_results_per_work_item * sub_group_size * sub_group_size
        )

    elif work_group_size != (
        (2**work_group_size_ratio_log2) * sub_group_size * sub_group_size
    ):
        raise ValueError(
            "Expected `work_group_size / (sub_group_size * sub_group_size)` to be a "
            f"power of two, but got {work_group_size_ratio} instead, with "
            f"`work_group_size={work_group_size}` and "
            f"`sub_group_size={sub_group_size}`."
        )

    work_group_size_ratio = int(work_group_size_ratio)

    return work_group_size, work_group_size_ratio, work_group_size_ratio_log2
