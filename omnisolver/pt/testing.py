"""Common testing utilities."""
import numba
import numpy as np


@numba.njit
def numba_seed(seed) -> None:
    """Seed numba's random generator.

    Note that despite calling numpy's random.seed, this function actually
    seeds numba's internal rng, which DOES NOT share state with numpy's rng.
    """
    np.random.seed(seed)


@numba.njit
def numba_rand() -> float:
    """Return random number uniformly chosen from [0, 1) using numba's random number generator."""
    return np.random.rand()
