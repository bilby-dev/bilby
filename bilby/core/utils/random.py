"""Submodule for random number generation.

This module provides a wrapper around numpy's random number generator
to ensure that the same random number generator is used throughout
:code:`bilby`. It also provides a function to set the random seed and
generate random seeds.

The intended usage is to import the submodule and the use the :code:`rng`
attribute to generate random numbers. For example:

.. code:: python

    >>> from bilby.core.utils import random
    # Seed the random number generator
    >>> random.seed(1234)
    # Generate a random number between 0 and 1
    >>> x = random.rng.random()

The :code:`rng` attribute is a :code:`numpy.random.Generator` object, for
more details see the numpy documentation:
https://numpy.org/doc/stable/reference/random/generator.html

.. warning::
    Do not import :code:`rng` directly from :code:`bilby.core.utils.random`
    since it will not be seeded correctly.
"""
import sys
import warnings

import array_api_compat as aac
import numpy as np
from numpy.random import default_rng, SeedSequence

from ...compat.utils import BILBY_ARRAY_API


def __getattr__(name):
    if name == "rng":
        return Generator.rng
    raise AttributeError(f"module {__name__} has no attribute {name}")


class Generator:
    """Class to hold the random number generator.

    This class is used to ensure that the same random number generator
    is used throughout :code:`bilby`.

    It should not be used directly, instead use :code:`random.rng` to
    generate random numbers. See the documentation for more details.
    """
    rng = default_rng()
    """Random number generator.

    This is a :code:`numpy.random.Generator` object that is used to
    generate random numbers. By default, it is not seeded.

    The recommended way to use this is to import the :code:`random` module
    and use the :code:`rng` attribute to generate random numbers. See the
    documentation for more details.
    """


def seed(seed):
    """Seed the random number generator.

    Also updates the global meta data with the new seed and generator.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator, RandomState}
        The seed to use for the random number generator.
    """
    from .meta_data import global_meta_data

    _original_rng = Generator.rng

    Generator.rng = default_rng(seed)
    global_meta_data["rng"] = Generator.rng
    global_meta_data["seed"] = seed

    # Warn if the original rng object (i.e., pre-seed) still exists elsewhere
    for module in sys.modules.values():
        if not module or not hasattr(module, '__dict__'):
            continue
        rng_obj = module.__dict__.get("rng")
        if rng_obj is _original_rng:
            warnings.warn(
                "Detected that `rng` was likely imported directly before calling `seed()`. "
                "This means the imported reference will not reflect the newly seeded generator. "
                "Use `from bilby.core.utils import random` and access `random.rng` instead.",
                RuntimeWarning
            )
            break


def generate_seeds(nseeds):
    """Generate a list of random seeds.

    Parameters
    ----------
    nseeds : int
        The number of seeds to generate.

    Returns
    -------
    numpy.random.SeedSequence
        A SeedSequence object containing the generated seeds.
    """
    return SeedSequence(Generator.rng.integers(0, 2**63 - 1, size=4)).spawn(nseeds)


def resolve_random_state(random_state):
    """
    Resolve the provided random state into a random number generator.

    Parameters
    ==========
    random_state: None, int, np.random.Generator, or jax.random.KeyArray
        The random state to resolve.
        If None, the default random generator will be used.
        If an int, a new :code:`numpy.random.default_rng` object will be
        created with that seed.
        If a :code:`numpy.random.Generator`, it will be returned as is.
        If a :code:`jax.random.KeyArray`, a corresponding
        :code:`orng.ArrayRNG` generator will be created and returned.

    Returns
    =======
    np.random.Generator or orng.ArrayRNG
        The resolved random number generator.
    """

    def _resolve_numpy_generator(random_state):
        if isinstance(random_state, np.random.Generator):
            return random_state
        elif random_state is None:
            return Generator.rng
        elif isinstance(random_state, int):
            return np.random.default_rng(random_state)
        else:
            raise ValueError(
                "Invalid random state. Must be None, int, or np.random.Generator."
            )

    if not BILBY_ARRAY_API:
        return _resolve_numpy_generator(random_state)

    import orng
    if isinstance(random_state, (np.random.Generator, orng.ArrayRNG)):
        return random_state
    elif aac.is_jax_array(random_state):
        rng = orng.ArrayRNG(generator=random_state, backend="jax")
        return rng
    else:
        return _resolve_numpy_generator(random_state)


def random_array_module(random_state):
    """
    Return the array module corresponding to the provided random state.
    The the random state is a JAX random key, this will return :code:`jax.numpy`.
    Otherwise, it will return :code:`numpy`.

    Parameters
    ==========
    random_state: None, int, np.random.Generator, or jax.random.KeyArray
        The random state to resolve.

    Returns
    -------
    array module
        The array module corresponding to the provided random state.
    """
    if random_state is None or not BILBY_ARRAY_API:
        return np

    if aac.is_jax_array(random_state) or getattr(random_state, "backend") == "jax":
        import jax.numpy as jnp
        return jnp
    elif aac.is_torch_array(random_state) or getattr(random_state, "backend") == "torch":
        import torch
        return torch
    else:
        return np
