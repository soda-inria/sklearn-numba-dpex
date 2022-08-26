import warnings
from dataclasses import dataclass

import dpctl
import pytest

from sklearn_numba_dpex.utils._device import _DeviceParams


def test_opencl_requirement():
    # Detect if https://github.com/IntelPython/dpctl/issues/886 is fixed
    # upstream and alert that sklearn-numba-dpex can be adapted accordingly.
    device = dpctl.select_default_device()
    has_preferred_work_group_size_multiple = hasattr(
        device, "preferred_work_group_size_multiple"
    )
    has_global_mem_cache_size = hasattr(device, "global_mem_cache_size")
    assert not (has_preferred_work_group_size_multiple or has_global_mem_cache_size), (
        "pyopencl is not required anymore: update the code to directly use dpctl "
        "instead, and remove pyopencl from the install_requires list in setup.py ."
    )


def test_warnings_non_cl_device_params():
    @dataclass
    class _FakeSyclDevice:
        has_aspect_fp64: bool = True
        max_work_group_size: int = 16
        name: str = "Fake Sycl Device Without OpenCL Support"

    device_params = _DeviceParams(_FakeSyclDevice())
    # Check that a warning is raised if a device is not detected by opencl
    with pytest.warns(RuntimeWarning):
        device_params.preferred_work_group_size_multiple

    with pytest.warns(RuntimeWarning):
        device_params.global_mem_cache_size


def test_no_warnings_cl_device_params():
    try:
        sycl_device = dpctl.SyclDevice("opencl")
        device_params = _DeviceParams(sycl_device)
    except dpctl.SyclDeviceCreationError:
        pytest.xfail("No opencl SyclDevice available")

    # Ensure absence of warning if the device is detected by opencl
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        device_params.preferred_work_group_size_multiple
        device_params.global_mem_cache_size
