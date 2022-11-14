import warnings
import random
from functools import lru_cache

from numba import float32, float64, uint32, int64, uint64
import numpy as np

import dpctl
import dpctl.tensor as dpt
import numba_dpex as dpex

# This code is largely inspired from the numba.cuda.random module and the
# numba/cuda/random.py where it's defined (v<0.57), and by the implementation of the
# same algorithm in the package `randomgen`.

# numba.cuda.random: https://github.com/numba/numba/blob/0.56.3/numba/cuda/random.py
# randomgen: https://github.com/bashtage/randomgen/blob/v1.23.1/randomgen/xoroshiro128.pyx

# NB1: we implement xoroshiro128++ rather than just xoroshiro128+, which is preferred.
# Reference resource about PRNG: https://prng.di.unimi.it/

# NB2: original numba.cuda.random code also includes functions for generating normally
# distributed floats but we don't include it here as long as it's not needed.

zero_idx = int64(0)
one_idx = int64(1)


def get_random_raw(states):
    """Returns one `uint64` random integer.

    NB: uses and updates the state states[0]"""
    result = dpt.empty(sh=(1,), dtype=np.uint64, device=states.device)
    make_random_raw_kernel()(states, result)
    return result


@lru_cache
def make_random_raw_kernel():
    @dpex.kernel
    def _get_random_raw_kernel(states, result):
        result[zero_idx] = _xoroshiro128pp_next(states, zero_idx)

    return _get_random_raw_kernel[1, 1]


@lru_cache
def make_rand_uniform_kernel_func(dtype):
    """Instanciate a kernel function that returns a random float in [0, 1)"""
    if not hasattr(dtype, "name"):
        raise ValueError(
            "dtype is expected to have an attribute 'name', like np.dtype or numba "
            "types."
        )

    if dtype.name == "float64":
        convert_rshift = uint32(11)
        convert_const = float64(uint64(1) << uint32(53))
        convert_const_one = float64(1)

        @dpex.func
        def uint64_to_unit_float(x):
            """Convert uint64 to float64 value in the range [0.0, 1.0)"""
            return float64(x >> convert_rshift) * (convert_const_one / convert_const)

    elif dtype.name == "float32":
        convert_rshift = uint32(40)
        convert_const = float32(uint32(1) << uint32(24))
        convert_const_one = float32(1)

        @dpex.func
        def uint64_to_unit_float(x):
            """Convert uint64 to float32 value in the range [0.0, 1.0)

            NB: this is different than original numba.cuda.random code. Instead of
            generating a float64 random number before casting it to float32, a float32
            number is generated from uint64 without intermediate float64. This enables
            compatibility with devices that do not support float64 numbers.
            However is seems to be exactly equivalent e.g it passes the float precision
            test in sklearn.
            """
            return float32(x >> convert_rshift) * (convert_const_one / convert_const)

    else:
        raise ValueError(
            "Expected dtype.name in {float32, float64} but got "
            f"dtype.name == {dtype.name}"
        )

    @dpex.func
    def xoroshiro128pp_uniform(states, index):
        """Return one random float in [0, 1)"""
        return uint64_to_unit_float(_xoroshiro128pp_next(states, index))

    return xoroshiro128pp_uniform


def create_xoroshiro128pp_states(n_states, seed=None, subsequence_start=0, device=None):
    """Returns a new device array initialized for n random number generators.

    This initializes the RNG states so that states in the array correspond to
    subsequences separated by 2**64 steps from each other in the main sequence.
    Therefore, as long as no thread requests more than 2**64 random numbers, all the
    RNG states produced by this function are guaranteed to be independent.

    Parameters
    ----------
    n_states : int
        number of RNG states to create

    seed : int or None
        starting seed for the list of generators

    subsequence_start : int
        advance the first RNG state by a multiple of 2**64 steps

    device : str or None (default)
        A SYCL device or if None, takes the default sycl device.
    """
    if seed is None:
        seed = uint64(random.randint(0, np.iinfo(np.int64).max - 1))

    if hasattr(seed, "randint"):
        seed = uint64(seed.randint(0, np.iinfo(np.int64).max - 1))

    seed = uint64(seed)
    n_states = int64(n_states)

    init_const_1 = np.uint64(0)

    @dpex.kernel
    def init_xoroshiro128pp_states(states):
        """
        Use SplitMix64 to generate an xoroshiro128p state from a uint64 seed.

        This ensures that manually set small seeds don't result in a predictable
        initial sequence from the random number generator.
        """
        if n_states < one_idx:
            return

        splitmix64_state = init_const_1 ^ seed
        splitmix64_state, states[zero_idx, zero_idx] = _splitmix64_next(
            splitmix64_state
        )
        _, states[zero_idx, one_idx] = _splitmix64_next(splitmix64_state)

        # advance to starting subsequence number
        for _ in range(subsequence_start):
            _xoroshiro128pp_jump(states, zero_idx)

        # populate the rest of the array
        for idx in range(one_idx, n_states):
            # take state of previous generator
            states[idx, zero_idx] = states[idx - one_idx, zero_idx]
            states[idx, one_idx] = states[idx - one_idx, one_idx]
            # and jump forward 2**64 steps
            _xoroshiro128pp_jump(states, idx)

    splitmix64_const_1 = uint64(0x9E3779B97F4A7C15)
    splitmix64_const_2 = uint64(0xBF58476D1CE4E5B9)
    splitmix64_const_3 = uint64(0x94D049BB133111EB)
    splitmix64_rshift_1 = uint32(30)
    splitmix64_rshift_2 = uint32(27)
    splitmix64_rshift_3 = uint32(31)

    @dpex.func
    def _splitmix64_next(state):
        new_state = z = state + splitmix64_const_1
        z = (z ^ (z >> splitmix64_rshift_1)) * splitmix64_const_2
        z = (z ^ (z >> splitmix64_rshift_2)) * splitmix64_const_3
        return new_state, z ^ (z >> splitmix64_rshift_3)

    jump_const_1 = uint64(0x2BD7A6A6E99C2DDC)
    jump_const_2 = uint64(0x0992CCAF6A6FCA05)
    jump_const_3 = uint64(1)
    jump_init = uint64(0)
    long_2 = int64(2)
    long_64 = int64(64)

    @dpex.func
    def _xoroshiro128pp_jump(states, index):
        """Advance the RNG in ``states[index]`` by 2**64 steps."""
        s0 = jump_init
        s1 = jump_init

        for i in range(long_2):
            if i == zero_idx:
                jump_const = jump_const_1
            else:
                jump_const = jump_const_2
            for b in range(long_64):
                if jump_const & jump_const_3 << uint32(b):
                    s0 ^= states[index, zero_idx]
                    s1 ^= states[index, one_idx]
                _xoroshiro128pp_next(states, index)

        states[index, zero_idx] = s0
        states[index, one_idx] = s1

    # TODO: it seems that the reference python implementation in the package
    # `randomgen` that inspired this code contains errors.
    # There's an issue here: https://github.com/bashtage/randomgen/issues/321
    # This website https://prng.di.unimi.it/ seems to provide a ground truth
    # implementation at https://prng.di.unimi.it/xoroshiro128plusplus.c but is not
    # interfaced with python.
    # We choose to mimic the reference implementation given in C even if there's no
    # easy, reproducible set of instructions to test that it's working as expected.
    # If the `randomgen` current implementation proves to be good, then the following
    # block should be used instead, and the tests should be adapted. Else, the block
    # can be removed.

    # jump_const_1 = uint64(0xDF900294D8F554A5)
    # jump_const_2 = uint64(0x170865DF4B3201FC)
    # jump_const_3 = uint64(1)
    # jump_init = uint64(0)
    # long_2 = int64(2)
    # long_64 = int64(64)

    # @dpex.func
    # def _xoroshiro128pp_jump(states, index):
    #     """Advance the RNG in ``states[index]`` by 2**64 steps."""
    #     s0 = jump_init
    #     s1 = jump_init

    #     for i in range(long_2):
    #         if i == zero_idx:
    #             jump_const = jump_const_1
    #         else:
    #             jump_const = jump_const_2
    #         for b in range(long_64):
    #             if jump_const & jump_const_3 << uint32(b):
    #                 s0 ^= states[index, zero_idx]
    #                 s1 ^= states[index, one_idx]
    #             # NB: this is _xoroshiro128p_next, not _xoroshiro128pp_next
    #             _xoroshiro128p_next(states, index)

    #     states[index, zero_idx] = s0
    #     states[index, one_idx] = s1

    # next_rot_p_1 = uint32(24)
    # next_rot_p_2 = uint32(16)
    # next_rot_p_3 = uint32(37)

    # @dpex.func
    # def _xoroshiro128p_next(states, index):
    #     """Return the next random uint64 and advance the RNG in states[index]."""
    #     s0 = states[index, zero_idx]
    #     s1 = states[index, one_idx]
    #     result = s0 + s1

    #     s1 ^= s0
    #     states[index, zero_idx] = _rotl(s0, next_rot_p_1) ^ s1 ^ (s1 << next_rot_p_2)
    #     states[index, one_idx] = _rotl(s1, next_rot_p_3)

    #     return result

    # Initialization is purely sequential so it will be faster on CPU, if a cpu device
    # is available make sure to use it.
    if device is None:
        device = dpctl.SyclDevice()
    from_cpu_to_device = False
    if not device.has_aspect_cpu:
        try:
            cpu_device = dpctl.SyclDevice("cpu")
            from_cpu_to_device = True
        except dpctl.SyclDeviceCreationError:
            warnings.warn(
                "No CPU found, fallbacking random initiatlization to default " "device."
            )

    states = dpt.empty(
        sh=(n_states, 2),
        dtype=np.uint64,
        device=cpu_device if from_cpu_to_device else device,
    )
    init_xoroshiro128pp_states[1, 1](states)

    if from_cpu_to_device:
        return states.to_device(device)

    else:
        return states


next_rot_1 = uint32(17)
next_rot_2 = uint32(49)
next_rot_3 = uint32(28)
shift_1 = uint32(21)


@dpex.func
def _xoroshiro128pp_next(states, index):
    """Return the next random uint64 and advance the RNG in states[index]."""
    s0 = states[index, zero_idx]
    s1 = states[index, one_idx]
    result = _rotl(s0 + s1, next_rot_1) + s0

    s1 ^= s0
    states[index, zero_idx] = _rotl(s0, next_rot_2) ^ s1 ^ (s1 << shift_1)
    states[index, one_idx] = _rotl(s1, next_rot_3)

    return result


_64_as_uint32 = uint32(64)


@dpex.func
def _rotl(x, k):
    """Left rotate x by k bits. x is expected to be a uint64 integer."""
    return (x << k) | (x >> (_64_as_uint32 - k))
