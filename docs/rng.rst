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


