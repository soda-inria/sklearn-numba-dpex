import dpctl


def test_opencl_requirement():
    device = dpctl.select_default_device()
    has_preferred_work_group_size_multiple = hasattr(
        device, "preferred_work_group_size_multiple"
    )
    has_global_mem_cache_size = hasattr(device, "global_mem_cache_size")
    assert not (
        has_preferred_work_group_size_multiple or has_global_mem_cache_size
    ), "pyopencl is not required anymore, use dpctl instead"
