========================
Random number generation
========================

Random number generation in :code:`bilby` uses a global :code:`numpy` Generator
object in :py:mod:`bilby.core.utils.random`. The recommended usage is

.. code:: python

    >>> from bilby.core.utils import random
    >>> x = random.rng.uniform()

where :code:`rng` is a :code:`numpy` random generator. For more details about
:code:`numpy` random generators, see the
:code:`numpy` `documentation <https://numpy.org/doc/stable/reference/random/generator.html>`_.

.. warning::
    The :code:`rng` object should not be imported directly as it will not be seeded
    by calls to :py:func:`bilby.core.utils.random.seed`.


The random number generation can be seeded using the
:py:func:`bilby.core.utils.random.seed` function:

.. code:: python

    >>> from bilby.core.utils import random
    >>> random.seed(1234)

For more fine-grained control, every function/method that relies on random number
generation supports a :code:`random_state` argument that can be used to specify
the random number generator to use for that function/method.

----------------
Seeding samplers
----------------

The different samplers in :code:`bilby` have different ways of seeding the random number
generator that depend on each sampler's implementation. As such, seeding the :code:`bilby`
random number generator with :py:func:`bilby.core.utils.random.seed` does not guarantee that the
sampler will be seeded. 

If the interface for a sampler supports seeding, then specifying either the specific
keyword argument or an equivalent argument (:code:`seed`, :code:`sampling_seed` or :code:`random_seed`
will be automatically translated to the appropriate keyword argument)
when calling :py:func:`~bilby.core.sampler.run_sampler` will seed the sampler's random number generator.
For example:

.. code:: python

    >>> import bilby
    >>> likelihood = ...
    >>> prior = ...
    >>> bilby.run_sampler(
            likelihood=likelihood,
            prior=prior,
            sampler="dynesty",
            seed=1234,
        )

.. note::
    Some sampler interfaces do not support seeding.

--------------------------------------------------------
Random number generation and non-:code:`NumpPy` backends
--------------------------------------------------------

To support random number generation with non-:code:`NumPy` array backends,
any :code:`bilby` function or method that supports random number generation and accepts a
:code:`random_state` argument.
This argument should be one of the following types:

- :code:`None` (the default): the function will use the :code:`bilby` global
  :code:`numpy` random number generator (set using :code:`bilby.core.random.seed`).
- :code:`numpy.random.Generator`: the function will use the provided generator.
- :code:`orng.ArrayRNG`: the function will use the provided :code:`orng` random number generator.
- :code:`int`: the function will create a new :code:`numpy` random number generator seeded with
  the provided integer and use it for random number generation.
- :code:`jax.random.PRNGKey`: the function will create a new :code:`orng` random number generator
  with the "jax" backend seeded with the provided key and use it for random number generation.

For example,

.. code:: python

    >>> import orng
    >>> rng = orng.ArrayRNG("jax", seed=1234)
    >>> x = rng.uniform()
    >>> priors.sample(xp=jnp, rng=rng)
