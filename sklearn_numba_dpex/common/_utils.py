import math


def check_power_of_2(e):
    if e != 2 ** (math.log2(e)):
        raise ValueError(f"Expected a power of 2, got {e}")
    return e


# HACK: the following function are defined as closures to work around a `numba_dpex`
# bug.
# Revert it (and everything related, see
# https://github.com/soda-inria/sklearn-numba-dpex/pull/82 )
# when the bug is fixed. The bugfix can be tracked at
# https://github.com/IntelPython/numba-dpex/issues/867
def _square():
    def __square(x):
        return x * x

    return __square


def _minus():
    def __minus(x, y):
        return x - y

    return __minus


def _plus():
    def __plus(x, y):
        return x + y

    return __plus


def _divide():
    def __divide(x, y):
        return x / y

    return __divide


def _check_max_work_group_size(
    work_group_size,
    device,
    local_memory_requirements_per_item,
    constant_memory_requirement=0,
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
    thumb of allocating in local memory about one item per thread, which fit the GPU
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

    if work_group_size == "max" and not device.has_aspect_cpu:
        return device.max_work_group_size
    elif work_group_size == "max":
        return math.floor(
            (device.local_mem_size - constant_memory_requirement)
            / local_memory_requirements_per_item
        )
    elif work_group_size > max_work_group_size:
        raise RuntimeError(
            f"Got work_group_size={work_group_size} but that is greather than the "
            "maximum supported work group size device.max_work_group_size="
            f"{device.max_work_group_size} for device {device.name}"
        )
    else:
        return work_group_size
