import pytest
import warnings

import dpctl

from sklearn_numba_dpex.utils._device import _DeviceParams


def test_opencl_requirement():
    device = dpctl.select_default_device()
    has_preferred_work_group_size_multiple = hasattr(
        device, "preferred_work_group_size_multiple"
    )
    has_global_mem_cache_size = hasattr(device, "global_mem_cache_size")
    assert not (
        has_preferred_work_group_size_multiple or has_global_mem_cache_size
    ), "pyopencl is not required anymore, use dpctl instead"


class _FakeSyclDevice:
    def __init__(
        self, has_aspect_fp64=True, max_work_group_size=16, name="Fake Sycl Device"
    ):
        self.has_aspect_fp64 = has_aspect_fp64
        self.max_work_group_size = 16
        self.name = name


def test_cl_device_params_warnings():

    # Check that a warning is raised if a device is not detected by opencl
    with pytest.warns(RuntimeWarning):
        device_params = _DeviceParams(
            _FakeSyclDevice()
        ).preferred_work_group_size_multiple

    with pytest.warns(RuntimeWarning):
        device_params = _DeviceParams(_FakeSyclDevice()).global_mem_cache_size

    try:
        sycl_device = dpctl.SyclDevice("opencl")
        device_params = _DeviceParams(sycl_device)
    except dpctl.SyclDeviceCreationError:
        return

    # Ensure absence of warning if the device is detected by opencl
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        device_params.preferred_work_group_size_multiple
        device_params.global_mem_cache_size
