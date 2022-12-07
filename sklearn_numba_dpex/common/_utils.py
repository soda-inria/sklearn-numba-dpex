import math


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
