=====================
Array API Support
=====================

Bilby now supports the Python `Array API Standard <https://data-apis.org/array-api/latest/>`_, 
enabling the use of different array backends (NumPy, JAX, CuPy, etc.) for improved performance 
and hardware acceleration. This page describes how to use this functionality and how it works internally.

For Users and Downstream Developers
====================================

Overview
--------

The Array API support allows you to use different array libraries with Bilby seamlessly. 
This can significantly improve performance, especially when using hardware accelerators like GPUs 
or when you need automatic differentiation capabilities.

**Key principle**: In most cases, you don't need to explicitly specify which array backend to use. 
Bilby automatically detects the array type you're working with and uses the appropriate backend. 
Simply pass JAX arrays, CuPy arrays, or NumPy arrays to prior methods, and Bilby handles the rest.

Supported Backends
------------------

Bilby is currently tested with the following array backends:

- **NumPy** (default): Standard CPU-based computations
- **JAX**: GPU/TPU acceleration and automatic differentiation

While :code:`Bilby` should be compatible with other Array API compliant libraries,
these are not currently tested or officially supported.
If you notice any issues when using other backends,
please report them on the `Bilby GitHub repository <https://github.com/bilby-dev/bilby/issues>`.

Using Different Array Backends
-------------------------------

Basic Prior Usage (Automatic Detection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The array backend is automatically detected from your input arrays. You typically don't need 
to specify the ``xp`` parameter::

.. code-block:: python

    import bilby
    import jax.numpy as jnp
    import numpy as np
    
    prior = bilby.core.prior.Uniform(minimum=0, maximum=10)
    
    # Using JAX - backend automatically detected
    val_jax = jnp.array([0.5, 1.5, 2.5])
    prob_jax = prior.prob(val_jax)  # Returns JAX array
    
    # Using NumPy - backend automatically detected
    val_np = np.array([0.5, 1.5, 2.5])
    prob_np = prior.prob(val_np)  # Returns NumPy array

Sampling with Array Backends (Explicit xp Required)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When sampling from priors, you **must** explicitly specify the array backend using the ``xp`` parameter, 
as there's no input array to infer the backend from::

.. code-block:: python

    import bilby
    import jax.numpy as jnp
    
    prior = bilby.core.prior.Uniform(minimum=0, maximum=10)
    samples = prior.sample(size=1000, xp=jnp)  # Returns JAX array
    
    # Or with NumPy (default)
    samples_np = prior.sample(size=1000)  # Or explicitly: xp=np

.. note::

    Currently, prior sampling is done by first generating uniform samples in [0, 1]
    using :code:`NumPy`, then converting to the desired backend.
    In future releases, this may be altered to generate samples directly in the specified backend.

Prior Dictionaries
~~~~~~~~~~~~~~~~~~

Prior dictionaries work the same way - automatic detection for most methods, explicit ``xp`` for sampling::

.. code-block:: python

    import bilby
    import jax.numpy as jnp
    
    priors = bilby.core.prior.PriorDict({
        'x': bilby.core.prior.Uniform(0, 100),
        'y': bilby.core.prior.Uniform(0, 1)
    })
    
    # Sampling requires explicit xp
    samples = priors.sample(size=1000, xp=jnp)
    
    # Evaluation automatically detects backend from input
    theta = jnp.array([50.0, 0.5])
    prob = priors.prob(samples)  # Automatically uses JAX

Core Likelihoods and Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Core :code:`Bilby` likelihoods are compatible with the Array API.
When using :code:`JAX` arrays, you can take advantage of :code:`JAX`'s JIT compilation and automatic differentiation.
For :code:`JAX`-compatible samplers (e.g., :code:`numpyro`),
you can pass any :code:`JAX`-compatible :code:`Bilby` likelihood directly.
For non-:code:`JAX` samplers, you should wrap your likelihood with the
:code:`bilby.compat.jax.JittedLikelihood` class to enable JIT compilation.

.. code-block:: python

    import bilby
    import jax.numpy as jnp
    from bilby.compat.jax import JittedLikelihood
    
    class MyLikelihood(bilby.Likelihood):
        def log_likelihood(self, parameters):
            # model returns a JAX array if passed a dictionary of JAX arrays
            return -0.5 * xp.sum((self.data - model(parameters))**2)

    data = jnp.array([...])  # Your data as a JAX array    

    priors = bilby.core.prior.PriorDict({
        'param1': bilby.core.prior.Uniform(0, 10),
        'param2': bilby.core.prior.Uniform(-5, 5)
    })

    likelihood = MyLikelihood(data)

    # call the likelihood once in case any initial setup is needed
    likelihood.log_likelihood(priors.sample())
    
    # Wrap with JittedLikelihood for JAX
    jitted_likelihood = JittedLikelihood(likelihood)

    # call the jitted likelihood once to trigger JIT compilation
    # the JittedLikelihood automatically converts the parameters
    # to JAX arrays
    jitted_likelihood.log_likelihood(priors.sample())

    # Use with a JAX-incompatible sampler
    sampler = bilby.run_sampler(likelihood=jitted_likelihood, ...)

Gravitational-Wave Likelihoods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`Bilby` implementation of gravitational-wave likelihood is compatible with the Array API,
however this requires access to waveform models that support the provided array backend.
The desired array backend must be explicitly specified for the data,
using :code:`bilby.gw.detector.networks.InterferometerList.set_array_backend`.
Below is an example using the :code:`ripplegw` package for waveform generation.
Here, an injection is performed using the standard :code:`LALSimulation` waveform generator,
and the analysis is then performed using the JIT-compiled likelihood.

.. code-block:: python

    import bilby
    import jax.numpy as jnp
    import ripplegw

    priors = bilby.gw.prior.BBHPriorDict()
    priors["geocent_time"] = bilby.core.prior.Uniform(1126259462.4, 1126259462.6)
    injection_parameters = priors.sample()

    # Create interferometers and inject signal using standard waveform generator
    ifos = bilby.gw.detector.networks.InterferometerList(['H1', 'L1'])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=2048,
        duration=4,
        start_time=injection_parameters["geocent_time"] - 2
    )
    injection_wfg = bilby.gw.waveform_generator.WaveformGenerator(
        duration=4,
        sampling_frequency=2048,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments={"approximant": "IMRPhenomXODE"}
    )
    ifos.inject_signal(parameters=injection_parameters, waveform_generator=injection_wfg)

    # set the array backend after the injection
    ifos.set_array_backend(jnp)

    ripple_wfg = bilby.gw.waveform_generator.WaveformGenerator(
        duration=4,
        sampling_frequency=2048,
        frequency_domain_source_model=ripplegw.get_fd_waveform
    )

    # Create gravitational-wave likelihood
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=ripple_wfg,
        priors=priors,
        phase_marginalization=True,
    )
    # call the likelihood once to do some initial setup
    # this is needed for the gravitational-wave transient likelihoods
    likelihood.log_likelihood_ratio(priors.sample())

    # Wrap with JittedLikelihood for JAX and JIT compile
    jitted_likelihood = bilby.compat.jax.JittedLikelihood(likelihood)
    jitted_likelihood.log_likelihood_ratio(priors.sample())

.. note::

    All of the likelihood marginalizations implemented in :code:`Bilby` are compatible with the Array API.
    However, there is currently a performance issue with the distance marginalized likelihood
    using the :code:`JAX` backend.

Performance Considerations
--------------------------

**When to use JAX:**

- GPU/TPU acceleration is available
- You need automatic differentiation
- Working with large datasets or many parameters
- Repeated evaluations benefit from JIT compilation

**When to use NumPy:**

- Simple CPU-based computations
- Small datasets
- Maximum compatibility
- Debugging (easier to inspect values)

**Best Practices:**

1. Let Bilby detect the array backend automatically - only specify ``xp`` when sampling
2. Use array backend consistently throughout your analysis
3. Avoid mixing array types in the same computation
4. For JAX, consider using ``jax.jit`` for repeated computations
5. Profile your code to ensure the chosen backend provides benefits

Bilby and JIT compilation
~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, Bilby functions are not JIT-compiled by default.
Additionally, many Bilby types are not defined as :code:`JAX`` :code:`PyTrees`,
and so cannot be passed as arguments to JIT-compiled functions.
We plan to support JIT-compilation for at least some Bilby types in future releases.

Custom Priors with Array API
-----------------------------

When creating custom priors, ensure they support the Array API:

Example Implementation
~~~~~~~~~~~~~~~~~~~~~~

Always include the ``xp`` parameter with a default value::

... code-block:: python

    from bilby.core.prior import Prior
    
    class MyCustomPrior(Prior):
        def __init__(self, parameter, **kwargs):
            super().__init__(**kwargs)
            self.parameter = parameter
        
        def rescale(self, val, *, xp=None):
            """Rescale method with xp parameter."""
            return self.minimum + val * (self.maximum - self.minimum) * self.parameter
        
        def prob(self, val, *, xp=None):
            """Probability method with xp parameter."""
            in_range = (val >= self.minimum) & (val <= self.maximum)
            return in_range / (self.maximum - self.minimum) * self.parameter

The ``xp`` parameter should:

- Be a keyword-only argument (after ``*``)
- Have a default value (``None`` if method is decorated with ``@xp_wrap``, ``np`` otherwise)
- Be passed through to any array operations if used directly

**Note**: Users of your custom prior won't need to pass ``xp`` explicitly for evaluation methods - 
it will be automatically inferred from their input arrays. They only need to specify ``xp`` when sampling.

Using the :code:`xp_wrap`` Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For methods that perform array operations, use the ``@xp_wrap`` decorator::

.. code-block:: python

    from bilby.core.prior import Prior
    from bilby.compat.utils import xp_wrap
    import numpy as np
    
    class MyCustomPrior(Prior):
        @xp_wrap
        def prob(self, val, *, xp=None):
            """The decorator handles xp=None automatically."""
            return xp.exp(-val) / self.normalization * self.is_in_prior_range(val)
        
        @xp_wrap
        def ln_prob(self, val, *, xp=None):
            """Works with logarithmic operations."""
            return -val - xp.log(self.normalization) + xp.log(self.is_in_prior_range(val))

The ``@xp_wrap`` decorator:

- Automatically provides the appropriate array module when ``xp=None``
- Infers the array backend from input arrays when they are :code:`JAX`/:code:`CuPy`/:code:`PyTorch` arrays
- Falls back to NumPy when the input is a standard Python type or NumPy array
- Handles the conversion seamlessly so users don't need to specify ``xp``

For Bilby Developers
=====================

Architecture Overview
---------------------

The Array API support in Bilby is built around several key components:

1. **The xp parameter**: A keyword-only parameter added to prior methods
2. **The @xp_wrap decorator**: Handles array module selection and injection
4. **Compatibility utilities**: Helper functions for array module detection

Core Changes to Prior Base Class
---------------------------------

The ``Prior`` base class in ``bilby/core/prior/base.py`` includes these key changes:

Method Signature Pattern
~~~~~~~~~~~~~~~~~~~~~~~~

All array-processing methods in prior classes follow this pattern:

**For methods with @xp_wrap decorator**::

.. code-block:: python

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Method that uses xp for array operations."""
        return xp.some_operation(val) * self.is_in_prior_range(val)

**For methods without @xp_wrap (that use xp directly)**::

.. code-block:: python

    def sample(self, size=None, *, xp=np):
        """Method that uses xp but isn't wrapped."""
        return xp.array(random.rng.uniform(0, 1, size))

Key rules:

- ``xp`` is always keyword-only (after ``*``)
- Methods with ``@xp_wrap`` use ``xp=None`` as default
- Methods without ``@xp_wrap`` that use ``xp`` use ``xp=np`` as default
- Methods that don't use ``xp`` have ``xp=None`` as default

The :code:`@xp_wrap`` Decorator
-------------------------------

Located in ``bilby/compat/utils.py``, this decorator:

1. **Inspects input arguments** to determine the array module in use
2. **Provides the appropriate xp** when ``xp=None``
3. **Maintains backward compatibility** with code that doesn't pass ``xp``

Example implementation pattern::

... code-block:: python

    from bilby.compat.utils import xp_wrap
    
    @xp_wrap
    def my_function(val, *, xp=None):
        # When called:
        # - If xp=None, decorator infers from val
        # - If xp is provided, uses that
        # - Returns results in the same array type as input
        return xp.exp(val) / xp.mean(val)

Testing Array API Support
-------------------------

Test Structure
~~~~~~~~~~~~~~

When appropriate, tests should verify functionality across different
backends using the ``array_backend`` marker::

    @pytest.mark.array_backend
    @pytest.mark.usefixtures("xp_class")
    class TestMyPrior:
        def test_prob(self):
            prior = MyPrior()
            val = self.xp.array([0.5, 1.5, 2.5])
            # No need to pass xp - automatically detected
            prob = prior.prob(val)
            assert self.xp.all(prob >= 0)
            assert prob.__array_namespace__() == self.xp
        
        def test_sample(self):
            prior = MyPrior()
            # Sampling requires explicit xp
            samples = prior.sample(size=100, xp=self.xp)
            assert samples.__array_namespace__() == self.xp

The array_backend Marker
~~~~~~~~~~~~~~~~~~~~~~~~

The ``@pytest.mark.array_backend`` marker is used to indicate that a test or test class should be run 
with multiple array backends. When you run pytest with the ``--array-backend`` flag, only tests marked 
with ``array_backend`` will be executed with that specific backend.

Without the marker, tests run with the default NumPy backend only. With the marker:

- Tests are parametrized to run with different backends
- The ``xp_class`` fixture is available, providing access to the array module via ``self.xp``
- Tests verify that code works correctly regardless of the array backend

Running Tests with Different Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``--array-backend`` flag to test with specific backends::

    # Test with NumPy (default)
    pytest test/core/prior/analytical_test.py
    
    # Test with JAX backend
    pytest --array-backend jax test/core/prior/analytical_test.py
    
    # Test with CuPy backend
    pytest --array-backend cupy test/core/prior/analytical_test.py

Bilby automatically sets ``SCIPY_ARRAY_API=1`` on import, so you don't need to set this 
environment variable manually. The ``--array-backend`` flag controls which backend the 
``xp_class`` fixture provides to your tests.

Migration Guide from Previous Versions
--------------------------------------

Key Differences
~~~~~~~~~~~~~~~

1. **Method signatures changed**: All prior methods now include ``xp`` parameter
2. **Decorator added**: Many methods now use ``@xp_wrap``
3. **Default values differ**: Methods with ``@xp_wrap`` use ``xp=None``, others use ``xp=np``
4. **Validation added**: Custom priors are checked for ``xp`` support

Best Practices for Contributors
--------------------------------

When adding or modifying prior methods:

1. **Always include xp parameter** in prob, ln_prob, rescale, cdf, sample methods
2. **Use @xp_wrap decorator** for methods doing array operations
3. **Set correct default**: ``xp=None`` with decorator, ``xp=np`` without (for methods that use xp directly)
4. **Pass xp through**: When calling other methods, pass ``xp=xp``
5. **Test with multiple backends**: Use ``@pytest.mark.array_backend`` and test with ``--array-backend jax``
6. **Document xp parameter**: Note it in docstrings, but emphasize it's usually auto-detected
7. **Use array module functions**: Use ``xp.function()`` not ``np.function()`` in wrapped methods

Handling Array Updates with :code:`array_api_extra.at``
-------------------------------------------------------

One key difference between array backends is how they handle array updates.
NumPy allows in-place  modification of array slices,
while JAX requires functional updates since arrays are immutable. 
The ``array_api_extra.at`` function provides a unified interface for array updates across backends.

Usage Examples
~~~~~~~~~~~~~~

**Conditional update**::

.. code-block:: python

    @xp_wrap
    def conditional_update(vals, *, xp=None):
        """Update array elements where mask is True."""
        arr = vals**2
        mask = arr > 0.5
        # Instead of: arr[mask] = value
        arr = xpx.at(arr)[mask].set(value)
        return arr

**Increment operation**::

.. code-block:: python

    @xp_wrap
    def increment_slice(arr, *, xp=None):
        """Add values to a slice of an array."""
        # Instead of: arr[2:5] += values
        arr = xpx.at(arr)[2:5].add(values)
        return arr

Available Operations
~~~~~~~~~~~~~~~~~~~~

The ``at`` function supports several operations:

- ``set(values)``: Replace values at specified indices
- ``add(values)``: Add values to specified indices
- ``multiply(values)``: Multiply specified indices by values
- ``min(values)``: Take element-wise minimum
- ``max(values)``: Take element-wise maximum

Important Notes
~~~~~~~~~~~~~~~

1. **Return value**: Always use the returned array. The operation may create a new array (JAX) or modify in-place (NumPy).

2. **Import**: Import ``array_api_extra`` at the module level::

.. code-block:: python

       import array_api_extra as xpx

Further Resources
-----------------

- `Array API Standard <https://data-apis.org/array-api/latest/>`_
- `JAX Documentation <https://jax.readthedocs.io/>`_
- `array-api-compat Package <https://github.com/data-apis/array-api-compat>`_
- `array-api-extra Package <https://github.com/data-apis/array-api-extra>`_
