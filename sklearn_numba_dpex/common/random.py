import warnings
import random
import dpctl
from functools import lru_cache
from numba import float32, float64, uint32, int64, uint64

import numba_dpex as dpex

import numpy as np

# This code is largely inspired from the numba.cuda.random module and the
# numba/cuda/random.py where it's defined (v<0.57).

# NB: original code also includes functions for generating normally distributed floats
# but we don't include it here as long as it's not needed.

zero_idx = int64(0)
one_idx = int64(1)


def get_randint(states):
    result = dpctl.tensor.empty(sh=(1,), dtype=np.uint64, device=states.device)
    make_randint_kernel()(states, result)
    return result


@lru_cache
def make_randint_kernel():
    @dpex.kernel
    def _get_randint_kernel(states, result):
        result[zero_idx] = _xoroshiro128p_next(states, zero_idx)

    return _get_randint_kernel[1, 1]


@lru_cache
def make_rand_uniform_kernel_func(dtype):
    if not hasattr(dtype, "name"):
        raise ValueError(
            "dtype is expected to have an attribute 'name', like np.dtype or numba "
            "types."
        )

    if dtype.name == "float64":
        convert_rshift = uint32(11)
        convert_const = uint64(1) << uint32(53)
        convert_const_one = float64(1)

        @dpex.func
        def uint64_to_unit_float(x):
            """Convert uint64 to float64 value in the range [0.0, 1.0)"""
            return (x >> convert_rshift) * (convert_const_one / convert_const)

    elif dtype.name == "float32":
        convert_rshift = uint32(8)
        convert_const = float32(uint32(1) << uint32(24))
        convert_const_one = float32(1)

        @dpex.func
        def uint64_to_unit_float(x):
            """Convert uint64 to float32 value in the range [0.0, 1.0)
            NB: this is different than original numba.cuda.random code. Instead of
            generating a float64 random number before casting it to float32, a float32
            number is generated from uint64 without intermediate float64. This enables
            compatibility with devices that do not support float64 numbers.
            """
            x = uint32(x)
            return float32(x >> convert_rshift) * (convert_const_one / convert_const)

    else:
        raise ValueError(
            "Expected dtype.name in {float32, float64} but got "
            f"dtype.name == {dtype.name}"
        )

    @dpex.func
    def xoroshiro128p_uniform(states, index):
        return uint64_to_unit_float(_xoroshiro128p_next(states, index))

    return xoroshiro128p_uniform


def create_xoroshiro128p_states(n_states, seed=None, subsequence_start=0, device=None):
    """Returns a new device array initialized for n random number generators.

    This initializes the RNG states so that each state in the array corresponds
    subsequences in the separated by 2**64 steps from each other in the main
    sequence.  Therefore, as long no thread requests more than 2**64 random numbers,
    all of the RNG states produced by this function are guaranteed to be independent.

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

    init_const_1 = uint64(0x9E3779B97F4A7C15)
    init_const_2 = uint64(0xBF58476D1CE4E5B9)
    init_const_3 = uint64(0x94D049BB133111EB)
    init_rshift_1 = uint32(30)
    init_rshift_2 = uint32(27)
    init_rshift_3 = uint32(31)

    @dpex.kernel
    def init_xoroshiro128p_states(states):
        """
        Use SplitMix64 to generate an xoroshiro128p state from 64-bit seed.

        This ensures that manually set small seeds don't result in a predictable
        initial sequence from the random number generator.
        """
        if n_states <= one_idx:
            return

        z = seed + init_const_1
        z = (z ^ (z >> init_rshift_1)) * init_const_2
        z = (z ^ (z >> init_rshift_2)) * init_const_3
        z = z ^ (z >> init_rshift_3)
        states[zero_idx, zero_idx] = z
        states[zero_idx, one_idx] = z

        # advance to starting subsequence number
        for _ in range(subsequence_start):
            _xoroshiro128p_jump(states, zero_idx)

        # populate the rest of the array
        for idx in range(one_idx, n_states):
            # take state of previous generator
            states[idx, zero_idx] = states[idx - one_idx, zero_idx]
            states[idx, one_idx] = states[idx - one_idx, one_idx]
            # and jump forward 2**64 steps
            _xoroshiro128p_jump(states, idx)

    jump_const_1 = uint64(0xBEAC0467EBA5FACB)
    jump_const_2 = uint64(0xD86B048B86AA9922)
    jump_const_3 = uint64(1)
    jump_init = uint64(0)
    long_2 = int64(2)
    long_64 = int64(64)

    @dpex.func
    def _xoroshiro128p_jump(states, index):
        """Advance the RNG in ``states[index]`` by 2**64 steps."""
        s0 = jump_init
        s1 = jump_init

        for i in range(long_2):
            if i == zero_idx:
                jump_const = jump_const_1
            else:
                jump_const = jump_const_2
            for b in range(long_64):
                if jump_const & (jump_const_3 << uint32(b)):
                    s0 ^= states[index, zero_idx]
                    s1 ^= states[index, one_idx]
                _xoroshiro128p_next(states, index)

        states[index, zero_idx] = s0
        states[index, one_idx] = s1

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

    states = dpctl.tensor.empty(
        sh=(n_states, 2),
        dtype=np.uint64,
        device=cpu_device if from_cpu_to_device else device,
    )

    init_xoroshiro128p_states[1, 1](states)

    if from_cpu_to_device:
        return states.to_device(device)

    else:
        return states


next_rot_1 = uint32(55)
next_rot_2 = uint32(14)
next_rot_3 = uint32(36)


@dpex.func
def _xoroshiro128p_next(states, index):
    """Return the next random uint64 and advance the RNG in states[index]."""
    s0 = states[index, zero_idx]
    s1 = states[index, one_idx]
    result = s0 + s1

    s1 ^= s0
    states[index, zero_idx] = _rotl(s0, next_rot_1) ^ s1 ^ (s1 << next_rot_2)
    states[index, one_idx] = _rotl(s1, next_rot_3)

    return result


uint32_64 = uint32(64)


@dpex.func
def _rotl(x, k):
    """Left rotate x by k bits."""
    return (x << k) | (x >> (uint32_64 - k))
