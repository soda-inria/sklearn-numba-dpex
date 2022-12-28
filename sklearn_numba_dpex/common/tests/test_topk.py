import dpctl.tensor as dpt
import numpy as np
import pytest

from sklearn_numba_dpex.common.topk import topk, topk_idx
from sklearn_numba_dpex.testing import _assert_array_equal_any
from sklearn_numba_dpex.testing.config import float_dtype_params


@pytest.mark.parametrize("dtype", float_dtype_params)
@pytest.mark.parametrize("work_group_size", [4, 8, None])
@pytest.mark.parametrize(
    "array_in, expected_top_3, list_of_expected_top_3_idx",
    [
        # test with only one occurence of threshold value
        (
            [0, 1, -2, -3, 4, -5, 6, 7, 8, -9, 10],
            [7, 8, 10],
            [[7, 8, 10]],
        ),
        # test with several occurences of threshold values, all included in the top 3,
        # and top 1 value is greater than threshold
        (
            [0, 8, -2, -3, 4, -5, 6, 7, 8, -9, 10],
            [8, 8, 10],
            [[1, 8, 10]],
        ),
        # test where top 3 is three times the same value, and there are only 3
        # occurences of this value
        (
            [0, 8, -2, -3, 4, -5, 6, 7, 8, -9, 8],
            [8, 8, 8],
            [[1, 8, 10]],
        ),
        # test with several occurences of threshold values, but some are not included in
        # the top 3, and top 1 value is greater than threshold
        (
            [0, 8, 8, -3, 4, -5, 6, 7, 8, -9, 10],
            [8, 8, 10],
            [[1, 2, 10], [2, 8, 10], [1, 8, 10]],
        ),
        # test where top 3 is three times the same value, and there are more than 3
        # occurences of this value
        (
            [0, 8, 8, -3, 4, -5, 6, 7, 8, -9, 8],
            [8, 8, 8],
            [[1, 2, 8], [1, 2, 10], [2, 8, 10], [1, 8, 10]],
        ),
    ],
)
def test_topk(
    array_in, expected_top_3, list_of_expected_top_3_idx, dtype, work_group_size
):
    k = 3
    array_in = dpt.asarray(array_in, dtype=dtype)

    if work_group_size is None:
        group_sizes = None
    else:
        group_sizes = (work_group_size, work_group_size // 2)

    actual_top_3 = np.sort(dpt.asnumpy(topk(array_in, k, group_sizes=group_sizes)))
    actual_top_3_idx = np.sort(
        dpt.asnumpy(topk_idx(array_in, k, group_sizes=group_sizes))
    )
    np.testing.assert_array_equal(expected_top_3, actual_top_3)
    _assert_array_equal_any(actual_top_3_idx, list_of_expected_top_3_idx)


@pytest.mark.parametrize("dtype", float_dtype_params)
@pytest.mark.parametrize("work_group_size", [4, 8, None])
def test_topk_constant_data(work_group_size, dtype):
    k = 3

    array_in = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    expected_top_3 = [8, 8, 8]

    if work_group_size is None:
        group_sizes = None
    else:
        group_sizes = (work_group_size, work_group_size // 2)

    array_in = dpt.asarray(array_in, dtype=dtype)

    actual_top_3 = np.sort(dpt.asnumpy(topk(array_in, k, group_sizes=group_sizes)))
    np.testing.assert_array_equal(expected_top_3, actual_top_3)

    actual_top_3_idx = set(dpt.asnumpy(topk_idx(array_in, k, group_sizes=group_sizes)))
    assert len(actual_top_3) == k
    assert max(actual_top_3_idx) < len(array_in)
    assert min(actual_top_3_idx) >= 0
