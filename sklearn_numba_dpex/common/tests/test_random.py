from sklearn_numba_dpex.common.random import (
    create_xoroshiro128pp_states,
    get_random_raw,
)
import dpctl.tensor as dpt
import numpy as np


def test_xoroshiro128pp():
    random_state = create_xoroshiro128pp_states(n_states=1, seed=42)

    # To avoid a dependency on the reference implementation randomgen,
    # sklearn_numba_dpex RNG implementation is tested against hardcoded lists of
    # values for a given seed, the hardcoded lists are generated using the following
    # code:

    # > from randomgen.xoroshiro128 import Xoroshiro128
    # > randomgen_rng = Xoroshiro128(seed=42, mode="legacy", plusplus=True)
    # > print([randomgen_rng.random_raw() for i in range(10)])

    # > randomgen_rng = Xoroshiro128(seed=42, mode="legacy", plusplus=True)
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
        int(dpt.asnumpy(get_random_raw(random_state))[0]) for i in range(10)
    ]

    assert expected_res_1 == actual_res_1

    expected_res_2 = [
        9942514532200612717,
        9103904774276862560,
        11668728844103653792,
        15855991950793068140,
        757481706500168315,
        9624528390636036977,
        5518335522560806466,
        11098424226258286153,
        8475596632683116788,
        12040925107571057860,
    ]

    random_state = create_xoroshiro128pp_states(
        n_states=1, seed=42, subsequence_start=1
    )
    actual_res_2 = [
        int(dpt.asnumpy(get_random_raw(random_state))[0]) for i in range(10)
    ]

    assert expected_res_2 == actual_res_2
