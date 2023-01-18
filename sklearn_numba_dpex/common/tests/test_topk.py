import dpctl.tensor as dpt
import numpy as np
import pytest

from sklearn_numba_dpex.common.topk import topk, topk_idx
from sklearn_numba_dpex.testing.config import float_dtype_params


# TODO: k == 1 works but gives segfaults sometimes
# Debug then re-activate the test.
# @pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("k", [3])
@pytest.mark.parametrize("dtype", float_dtype_params)
@pytest.mark.parametrize("work_group_size", [4, 8, None])
@pytest.mark.parametrize(
    "array_in",  # let's handcraft all possible cases for k=3
    [
        # test with only one occurence of threshold value
        ([0, 1, -2, -3, 4, -5, 6, 7, 8, -9, 10]),
        # test with several occurences of threshold values, all included in the top 3,
        # and top 1 value is greater than threshold
        ([0, 8, -2, -3, 4, -5, 6, 7, 8, -9, 10]),
        # test where top 3 is three times the same value, and there are only 3
        # occurences of this value
        ([0, 8, -2, -3, 4, -5, 6, 7, 8, -9, 8]),
        # test with several occurences of threshold values, but some are not included in
        # the top 3, and top 1 value is greater than threshold
        ([0, 8, 8, -3, 4, -5, 6, 7, 8, -9, 10]),
        # test where top 3 is three times the same value, and there are more than 3
        # occurences of this value
        ([0, 8, 8, -3, 4, -5, 6, 7, 8, -9, 8]),
        # Constant data
        ([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]),
    ],
)
def test_topk(k, array_in, dtype, work_group_size):
    array_in_dpt = dpt.asarray(array_in, dtype=dtype)

    if work_group_size is None:
        group_sizes = None
    else:
        group_sizes = (work_group_size, work_group_size // 2)

    actual_top_k = np.sort(dpt.asnumpy(topk(array_in_dpt, k, group_sizes=group_sizes)))
    actual_top_k_idx = np.sort(
        dpt.asnumpy(topk_idx(array_in_dpt, k, group_sizes=group_sizes))
    )

    assert len(actual_top_k_idx) == k
    assert len(actual_top_k) == k

    expected_top_k = np.sort(array_in)[-k:]

    np.testing.assert_array_equal(expected_top_k, actual_top_k)

    # NB: tiebreaks can change from one run to the other. There isn't a fixed
    # tie-breaking policy and thread concurrency will give variable outcomes.
    actual_from_idx = set(array_in[i] for i in actual_top_k_idx)
    assert actual_from_idx == set(expected_top_k)
