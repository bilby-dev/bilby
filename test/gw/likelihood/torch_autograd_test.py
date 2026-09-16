"""Regression tests for tensor identity through GW likelihood orchestration."""

from unittest.mock import Mock

import pytest

from bilby.gw.likelihood import GravitationalWaveTransient


def make_likelihood(time_marginalization=False, fail=False):
    """Isolate parameter handling from detector and waveform dependencies."""
    likelihood = GravitationalWaveTransient.__new__(GravitationalWaveTransient)
    likelihood.time_marginalization = time_marginalization
    likelihood.jitter_time = True
    likelihood.waveform_generator = Mock()
    likelihood.waveform_generator.frequency_domain_strain.side_effect = lambda p: p
    likelihood.get_sky_frame_parameters = Mock(return_value={})
    likelihood._interferometers = [object()]

    def calculate_snrs(waveform_polarizations, interferometer, parameters):
        if fail:
            raise ValueError("detector failed")
        value = parameters['mass_1'] ** 2 + 3 * parameters['mass_2']
        if time_marginalization:
            value = value + parameters['geocent_time']
        return likelihood._CalculatedSNRs(d_inner_h=value, optimal_snr_squared=0)

    likelihood.calculate_snrs = calculate_snrs
    likelihood.compute_log_likelihood_from_snrs = lambda snrs, parameters: snrs.d_inner_h
    return likelihood


@pytest.mark.parametrize("views", [False, True])
def test_torch_parameter_gradients(views):
    torch = pytest.importorskip("torch")
    position = torch.tensor([36., 29.], dtype=torch.float64, requires_grad=True)
    masses = tuple(position.unbind()) if views else tuple(
        p.detach().requires_grad_() for p in position.unbind()
    )
    likelihood = make_likelihood()
    parameters = dict(mass_1=masses[0], mass_2=masses[1])
    for _ in range(2):
        value = likelihood.log_likelihood_ratio(parameters)
        gradients = torch.autograd.grad(value, masses)
        torch.testing.assert_close(gradients[0], 2 * masses[0])
        torch.testing.assert_close(gradients[1], masses[1].new_tensor(3.))
    assert parameters['mass_1'] is masses[0]


@pytest.mark.parametrize("fail", [False, True])
def test_time_jitter_does_not_mutate_caller_tensor(fail):
    torch = pytest.importorskip("torch")
    time = torch.tensor(10., dtype=torch.float64, requires_grad=True)
    jitter = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    parameters = dict(mass_1=2., mass_2=3., geocent_time=time, time_jitter=jitter)
    likelihood = make_likelihood(time_marginalization=True, fail=fail)
    if fail:
        with pytest.raises(ValueError, match="detector failed"):
            likelihood.log_likelihood_ratio(parameters)
    else:
        value = likelihood.log_likelihood_ratio(parameters)
        torch.testing.assert_close(value, time.new_tensor(23.1))
        for gradient in torch.autograd.grad(value, (time, jitter)):
            torch.testing.assert_close(gradient, time.new_tensor(1.))
    assert parameters['geocent_time'] is time
    torch.testing.assert_close(time, time.new_tensor(10.))


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_mutating_source_preserves_caller_values_and_gradients(fail, backend):
    """Source extensions must not gain write access to caller-owned values."""
    import numpy as np

    torch = pytest.importorskip("torch")
    mass = torch.tensor(2., dtype=torch.float64, requires_grad=True) if backend == "torch" else np.array(2.)
    array = np.array([1., 2.])
    metadata = {"array": array, "labels": ["original"]}
    parameters = dict(mass_1=mass, mass_2=3., metadata=metadata, alias=mass)
    likelihood = make_likelihood()

    def mutating_source(local):
        assert local['mass_1'] is local['alias']
        if backend == 'torch':
            local['mass_1'].add_(1)
        else:
            local['mass_1'][...] += 1
        local['metadata']['array'][0] = -10
        local['metadata']['labels'].append("changed")
        if fail:
            raise ValueError("source failed")
        return local

    likelihood.waveform_generator.frequency_domain_strain.side_effect = mutating_source
    if fail:
        with pytest.raises(ValueError, match="source failed"):
            likelihood.log_likelihood_ratio(parameters)
    else:
        value = likelihood.log_likelihood_ratio(parameters)
        if backend == "torch":
            torch.testing.assert_close(value, mass.new_tensor(18.))
            gradient, = torch.autograd.grad(value, mass)
            torch.testing.assert_close(gradient, mass.new_tensor(6.))
        else:
            np.testing.assert_equal(value, 18.)
    if backend == "torch":
        torch.testing.assert_close(mass, mass.new_tensor(2.))
    else:
        np.testing.assert_array_equal(mass, 2.)
    np.testing.assert_array_equal(array, [1., 2.])
    assert metadata['labels'] == ['original']
    assert parameters['metadata'] is metadata
    assert parameters['mass_1'] is mass
