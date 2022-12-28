import math
import warnings

import dpctl


def check_power_of_2(e):
    if e != 2 ** (math.log2(e)):
        raise ValueError(f"Expected a power of 2, got {e}")
    return e


def _square(x):
    return x * x


def _minus(x, y):
    return x - y


def _plus(x, y):
    return x + y


def _divide(x, y):
    return x / y


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
