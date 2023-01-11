import math


def check_power_of_2(e):
    if e != 2 ** (math.log2(e)):
        raise ValueError(f"Expected a power of 2, got {e}")
    return e


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
