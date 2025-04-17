"""Submodule for random number generation.

This module provides a wrapper around numpy's random number generator
to ensure that the same random number generator is used throughout
:code:`bilby`. It also provides a function to set the random seed and
generate random seeds.

The intended usage is to import the submodule and the use the :code:`rng`
attribute to generate random numbers. For example:

.. code-block:: python
    from bilby.core.utils import random

    # Seed the random number generator
    random.seed(1234)

    # Generate a random number between 0 and 1
    x = random.rng.random()

The :code:`rng` attribute is a :code:`numpy.random.Generator` object, for
more details see the numpy documentation:
https://numpy.org/doc/stable/reference/random/generator.html

.. warning::
    Do not import :code:`rng` directly from :code:`bilby.core.utils.random`
    since it will not be seeded correctly.
"""

from numpy.random import default_rng, SeedSequence


def __getattr__(name):
    if name == "rng":
        return Generator.rng
    raise AttributeError(f"module {__name__} has no attribute {name}")


class Generator:
    rng = default_rng()


def seed(seed):
    """Seed the random number generator.

    Also updates the global meta data with the new seed and generator.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator, RandomState}
        The seed to use for the random number generator.
    """
    from .meta_data import global_meta_data

    Generator.rng = default_rng(seed)
    global_meta_data["rng"] = Generator.rng
    global_meta_data["seed"] = seed


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
