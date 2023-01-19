import pytest

from sklearn_numba_dpex.common._utils import (
    _check_max_work_group_size,
    get_maximum_power_of_2_smaller_than,
)


def test_get_maximum_power_of_2_smaller_than():
    assert get_maximum_power_of_2_smaller_than(65) == 64
    assert get_maximum_power_of_2_smaller_than(64) == 64
    assert get_maximum_power_of_2_smaller_than(63) == 32


def test_check_max_work_group_size():
    class _MockCpuDevice:
        has_aspect_cpu = True
        max_work_group_size = 8192
        local_mem_size = 32768
        name = "Super CPU"

    class _MockGpuDevice:
        has_aspect_cpu = False
        max_work_group_size = 512

    with pytest.raises(
        RuntimeError, match="greater than the maximum supported work group size"
    ):
        _check_max_work_group_size(9001, _MockCpuDevice, 0)

    assert 42 == _check_max_work_group_size(42, _MockCpuDevice(), None)
    assert 42 == _check_max_work_group_size(42, _MockGpuDevice(), None)

    assert 8192 == _check_max_work_group_size("max", _MockCpuDevice(), 0)
    assert 512 == _check_max_work_group_size("max", _MockGpuDevice(), 0)

    assert 480 == _check_max_work_group_size("max", _MockCpuDevice(), 64, 1024)
    assert 512 == _check_max_work_group_size("max", _MockGpuDevice(), 64, 1024)
