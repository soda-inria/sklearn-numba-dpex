import dpctl
import numpy as np
import pytest

_DEVICE = dpctl.SyclDevice()
_DEVICE_NAME = _DEVICE.name
_SUPPORTED_DTYPE = [np.float32]

if _DEVICE.has_aspect_fp64:
    _SUPPORTED_DTYPE.append(np.float64)


float_dtype_params = [
    pytest.param(
        dtype,
        marks=pytest.mark.skipif(
            dtype not in _SUPPORTED_DTYPE,
            reason=(
                f"The default device {_DEVICE_NAME} does not have support for"
                f" {dtype} operations."
            ),
        ),
    )
    for dtype in [np.float32, np.float64]
]
