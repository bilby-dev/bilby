import array_api_compat as aac
import numpy as np
import pytest
from bilby.core.prior import PriorDict, Uniform
from bilby.core.utils.random import seed
from bilby.gw.detector import InterferometerList
from bilby.gw.likelihood import GravitationalWaveTransient
from bilby.gw.source import sinegaussian
from bilby.gw.waveform_generator import WaveformGenerator


def _evaluate_with_jit(likelihood, parameters, xp):
    if not aac.is_jax_namespace(xp):
        pytest.skip("JIT test only runs for JAX backend")

    import jax
    from jax.tree_util import register_pytree_node

    from bilby.compat.pytrees import likelihood as _
    from bilby.gw.compat import pytrees as _

    @jax.jit
    def jit_fn(likelihood, parameters):
        return likelihood.log_likelihood_ratio(parameters)
    
    expected = likelihood.log_likelihood_ratio(parameters)
    jitted = jit_fn(likelihood, parameters)
    jitted = jit_fn(likelihood, parameters)

    assert xp.abs(expected - jitted) < 1e-10


def null_convert(parameters):
    return parameters, list()


def likelihood(xp, **marginalizations):
    seed(500)
    interferometers = InterferometerList(["H1"])
    interferometers.set_strain_data_from_power_spectral_densities(
        sampling_frequency=xp.asarray(2048.0), duration=xp.asarray(4.0)
    )
    interferometers.set_array_backend(xp)
    waveform_generator = WaveformGenerator(
        duration=xp.asarray(4.0),
        sampling_frequency=xp.asarray(2048.0),
        frequency_domain_source_model=sinegaussian,
        parameter_conversion=null_convert,
        use_cache=False,
    )
    priors = PriorDict(dict(
        phase=Uniform(0, 2 * np.pi),
        geocent_time=Uniform(0, 4),
    ))

    likelihood = GravitationalWaveTransient(
        interferometers=interferometers,
        waveform_generator=waveform_generator,
        priors=priors,
        **marginalizations,
    )
    return likelihood


@pytest.fixture
def parameters(xp):
    return dict(
        hrss=1e-24,
        Q=1.0,
        frequency=50.0,
        psi=xp.asarray(2.659),
        geocent_time=xp.asarray(2.413),
        ra=xp.asarray(1.375),
        dec=xp.asarray(-1.2108),
        time_jitter=0.0,
    )


@pytest.mark.array_backend
def test_jitted_likelihood(xp, parameters):
    _evaluate_with_jit(likelihood(xp), parameters, xp)


@pytest.mark.array_backend
def test_jitted_likelihood_with_phase_marginalization(xp, parameters):
    _evaluate_with_jit(likelihood(xp, phase_marginalization=True), parameters, xp)


@pytest.mark.array_backend
def test_jitted_likelihood_with_time_marginalization(xp, parameters):
    _evaluate_with_jit(likelihood(xp, time_marginalization=True), parameters, xp)


@pytest.mark.array_backend
def test_jitted_likelihood_with_phase_time_marginalization(xp, parameters):
    _evaluate_with_jit(likelihood(xp, phase_marginalization=True, time_marginalization=True), parameters, xp)
