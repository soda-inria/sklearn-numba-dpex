import warnings
from dataclasses import dataclass

import dpctl
import pytest

from sklearn_numba_dpex.device import DeviceParams


def test_warnings_non_cl_device_params():
    @dataclass
    class _FakeSyclDevice:
        has_aspect_fp64: bool = True
        max_work_group_size: int = 16
        name: str = "Fake Sycl Device Without OpenCL Support"

    device_params = DeviceParams(_FakeSyclDevice())
    # Check that a warning is raised if a device is not detected by opencl
    with pytest.warns(RuntimeWarning):
        device_params.preferred_work_group_size_multiple


def test_no_warnings_cl_device_params():
    try:
        sycl_device = dpctl.SyclDevice("opencl")
        device_params = DeviceParams(sycl_device)
    except dpctl.SyclDeviceCreationError:
        pytest.xfail("No opencl SyclDevice available")

    # Ensure absence of warning if the device is detected by opencl
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        device_params.preferred_work_group_size_multiple
