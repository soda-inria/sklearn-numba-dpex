import math


def check_power_of_2(e):
    if e != 2 ** (math.log2(e)):
        raise ValueError(f"Expected a power of 2, got {e}")
    return e
