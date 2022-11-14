import numpy as np
import dpctl.tensor as dpt
import numba_dpex as dpex

from sklearn_numba_dpex.common.random import (
    create_xoroshiro128pp_states,
    get_random_raw,
    make_rand_uniform_kernel_func,
)


from sklearn_numba_dpex.common.random import make_rand_uniform_kernel_func


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

    # TODO: currently the `randomgen` code snippet returns the following result:
    # expected_res_2 = [
    #     9942514532200612717,
    #     9103904774276862560,
    #     11668728844103653792,
    #     15855991950793068140,
    #     757481706500168315,
    #     9624528390636036977,
    #     5518335522560806466,
    #     11098424226258286153,
    #     8475596632683116788,
    #     12040925107571057860,
    # ]
    # but it seems to be wrong. Refer to the sklearn_numba_dpex/common/random.py file
    # for more details on the issue, and adapt the test or remove this block once it is
    # resolved.

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


def _make_get_single_rand_value_kernel(dtype):

    rand_uniform_kernel_func = make_rand_uniform_kernel_func(np.dtype(dtype))
    zero_idx = np.int64(0)

    @dpex.kernel
    # fmt: off
    def _get_single_rand_value(
        random_state,                     # IN             (1, 2)
        single_rand_value,                # OUT            (1,)
        ):
        single_rand_value[0] = rand_uniform_kernel_func(random_state, zero_idx)

    def get_single_rand_value(random_state):
        single_rand_value = dpt.empty(sh=1, dtype=dtype)
        _get_single_rand_value[1, 1](random_state, single_rand_value)
        return dpt.asnumpy(single_rand_value)[0]

    return get_single_rand_value


def test_xoroshiro128pp_rand():
    random_state = create_xoroshiro128pp_states(n_states=1, seed=42)
    get_single_rand_value_kernel_32 = _make_get_single_rand_value_kernel(np.float32)

    expected_res_32 = [
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
    ]
    expected_res_32 = [np.float32(f) for f in expected_res_32]

    actual_res_32 = [
        get_single_rand_value_kernel_32(random_state) for _ in expected_res_32
    ]

    assert expected_res_32 == actual_res_32

    if not random_state.device.sycl_device.has_aspect_fp64:
        return

    random_state = create_xoroshiro128pp_states(n_states=1, seed=42)
    get_single_rand_value_kernel_64 = _make_get_single_rand_value_kernel(np.float64)
    actual_res_64_to_32 = [
        np.float32(get_single_rand_value_kernel_64(random_state))
        for _ in expected_res_32
    ]

    np.testing.assert_allclose(actual_res_64_to_32, expected_res_32, rtol=1e-4)
