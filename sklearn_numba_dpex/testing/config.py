from contextlib import contextmanager

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


@contextmanager
def override_attr_context(obj, **attrs):
    """Within this context manager values of existing attributes of an object obj are
    overriden. The initial values are restored when exiting the context.

    Trying to override attributes that don't exist will result in an AttributeError"""

    attrs_before = dict()
    for attr_name, attr_value in attrs.items():
        # raise AttributeError if obj does not have the attribute attr_name
        attrs_before[attr_name] = getattr(obj, attr_name)
        setattr(obj, attr_name, attr_value)

    yield

    for attr_name, attr_value in attrs_before.items():
        setattr(obj, attr_name, attr_value)
