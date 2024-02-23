import math
from functools import lru_cache

import dpctl.tensor as dpt
import numba_dpex as dpex
import numpy as np
import pytest
from numba_dpex.kernel_api import NdRange
from sklearn.utils._testing import assert_allclose

from sklearn_numba_dpex.common.random import (
    create_xoroshiro128pp_states,
    get_random_raw,
    make_rand_uniform_kernel_func,
)
from sklearn_numba_dpex.testing.config import float_dtype_params


def test_xoroshiro128pp_raw():
    random_state = create_xoroshiro128pp_states(n_states=1, seed=42)

    # To avoid a dependency on the reference implementation randomgen,
    # sklearn_numba_dpex RNG implementation is tested against hardcoded lists of
    # values for a given seed, the hardcoded lists are generated using the following
    # code:

    # > from randomgen.xoroshiro128 import Xoroshiro128
    # > randomgen_rng = Xoroshiro128(seed=42, mode="legacy", plusplus=True)
    # > print([randomgen_rng.random_raw() for i in range(10)])

    # > randomgen_rng = Xoroshiro128(seed=42, mode="legacy", plusplus=True)
    # > # Move a subsequence forward.
    # > randomgen_rng.jump()
    # > print([randomgen_rng.random_raw() for i in range(10)])

    expected_res_1 = [
        16756476715040848931,
        6098722386207918385,
        17541662578032534341,
        3771828211556203317,
        6324094075403496319,
        1696280121849217124,
        9039151786603216578,
        1834868434787995627,
        4294676528364711596,
        5649342080498856832,
    ]

    actual_res_1 = [
        int(dpt.asnumpy(get_random_raw(random_state))[0]) for _ in expected_res_1
    ]

    assert expected_res_1 == actual_res_1

    expected_res_2 = [
        16052925335932940643,
        13241858892588731496,
        8234838429006980292,
        1690280486132429899,
        6807031509475728656,
        9789629428737685881,
        8662181912786361632,
        11992761958092131212,
        7748117140003924005,
        5731413122647051604,
    ]

    random_state = create_xoroshiro128pp_states(
        n_states=1, seed=42, subsequence_start=1
    )
    actual_res_2 = [
        int(dpt.asnumpy(get_random_raw(random_state))[0]) for _ in expected_res_2
    ]

    assert expected_res_2 == actual_res_2


def test_xoroshiro128pp_rand_consistency():
    # Let's generate some random float32 numbers...
    seed = 42
    random_state = create_xoroshiro128pp_states(n_states=1, seed=seed)

    expected_res_float32 = np.array(
        [
            0.9083704,
            0.33061236,
            0.9509354,
            0.20447117,
            0.34282982,
            0.09195548,
            0.49001336,
            0.09946841,
            0.23281485,
            0.3062514,
        ],
        dtype=np.float32,
    )
    n_items = len(expected_res_float32)

    actual_res_float32 = [
        _get_single_rand_value(random_state, np.float32) for _ in expected_res_float32
    ]

    assert_allclose(expected_res_float32, actual_res_float32)

    # ... and again the same random float32 numbers, but now the generation loop is
    # jitted...

    actual_res_float32 = dpt.asnumpy(
        _rand_uniform(n_items, np.float32, seed=seed, n_work_items=1)
    )
    assert_allclose(expected_res_float32, actual_res_float32)

    # ...and if the device supports it, also some random float64 numbers, with the same
    # seed...
    if not random_state.device.sycl_device.has_aspect_fp64:
        return

    random_state = create_xoroshiro128pp_states(n_states=1, seed=seed)
    actual_res_float64 = [
        _get_single_rand_value(random_state, np.float64) for _ in expected_res_float32
    ]

    # ...and ensure that the float32 rng and float64 rng produce close numbers.
    assert_allclose(actual_res_float64, expected_res_float32)


@pytest.mark.parametrize("dtype", float_dtype_params)
def test_rng_quality(dtype):
    """Check that the distribution of the few first floats sampled uniformly in [0, 1)
    actually approximate a uniform distribution"""
    size = int(1e6)
    random_floats = dpt.asnumpy(_rand_uniform(size, dtype, seed=42))
    distribution_in_bins, _ = np.histogram(random_floats, bins=np.linspace(0, 1, 6))
    assert (np.abs(distribution_in_bins - 2e5) < 1e3).all()


def _get_single_rand_value(random_state, dtype):
    """Return a single rand value sampled uniformly in [0, 1)"""
    _get_single_rand_value_kernel = _make_get_single_rand_value_kernel(dtype)
    single_rand_value = dpt.empty(1, dtype=dtype)
    _get_single_rand_value_kernel[NdRange((1,), (1,))](random_state, single_rand_value)
    return dpt.asnumpy(single_rand_value)[0]


def _rand_uniform(size, dtype, seed, n_work_items=1000):
    """Return an array of floats sampled uniformly in [0, 1)"""
    out = dpt.empty((size,), dtype=dtype)
    work_group_size = out.device.sycl_device.max_work_group_size
    size_per_work_item = math.ceil(size / n_work_items)

    _rand_uniform_kernel = _make_rand_uniform_kernel(size, dtype, size_per_work_item)

    global_size = (
        math.ceil(size / (size_per_work_item * work_group_size)) * work_group_size
    )
    states = create_xoroshiro128pp_states(n_states=global_size, seed=seed)
    _rand_uniform_kernel[NdRange((global_size,), (work_group_size,))](states, out)
    return out


@lru_cache
def _make_get_single_rand_value_kernel(dtype):
    rand_uniform_kernel_func = make_rand_uniform_kernel_func(np.dtype(dtype))
    zero_idx = np.int64(0)

    @dpex.kernel
    # fmt: off
    def get_single_rand_value(
        random_state,                     # IN             (1, 2)
        single_rand_value,                # OUT            (1,)
    ):
        # fmt: on
        single_rand_value[0] = rand_uniform_kernel_func(random_state, zero_idx)

    return get_single_rand_value


@lru_cache
def _make_rand_uniform_kernel(size, dtype, size_per_work_item):
    rand_uniform_kernel_func = make_rand_uniform_kernel_func(np.dtype(dtype))
    private_states_shape = (1, 2)

    @dpex.kernel
    # fmt: off
    def _rand_uniform_kernel(
        states,                           # IN               (global_size, 2)
        out,                              # OUT              (size,)
    ):
        # fmt: on
        item_idx = dpex.get_global_id(0)
        out_idx = item_idx * size_per_work_item

        private_states = dpex.private.array(shape=private_states_shape, dtype=np.uint64)
        private_states[0, 0] = states[item_idx, 0]
        private_states[0, 1] = states[item_idx, 1]

        for _ in range(size_per_work_item):
            if out_idx >= size:
                return
            out[out_idx] = rand_uniform_kernel_func(private_states, 0)
            out_idx += 1

        states[item_idx, 0] = private_states[0, 0]
        states[item_idx, 1] = private_states[0, 1]

    return _rand_uniform_kernel
