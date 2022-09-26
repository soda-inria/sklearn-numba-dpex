from pytest import xfail

from sklearn import config_context

from sklearn_numba_dpex.exceptions import (
    NotImplementedError as CustomNotImplementedError,
)


def pytest_runtest_call(item):
    with config_context(engine_provider="sklearn_numba_dpex"):
        try:
            item.runtest()
        except CustomNotImplementedError:
            xfail(
                reason="This test cover features that are not supported by the "
                "engine provided by sklearn_numba_dpex."
            )
