"""
Tutorial to show signal injection and PE
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

import peyote
import corner
from dynesty import plotting as dyplot

# Generate simulated data
signal_amplitude = 1e-21
signal_frequency = 100
time_duration = 1.
sampling_frequency = 4096.
time = peyote.utils.create_time_series(sampling_frequency, time_duration)
sim_parameters = dict(
    A=signal_amplitude, f=signal_frequency, geocent_time=1, ra=1, dec=2, psi=0)
source = peyote.source.SimpleSinusoidSource(
    'foo', sampling_frequency, time_duration)
hf_signal = source.frequency_domain_strain(sim_parameters)

IFO_1 = peyote.detector.H1
IFOs = [IFO_1]
for IFO in IFOs:
    hf_noise, ff = IFO.power_spectral_density.get_noise_realisation(
        sampling_frequency, time_duration)
    IFO.set_data(frequency_domain_strain=hf_noise)
    IFO.inject_signal(source, sim_parameters)
    IFO.set_spectral_densities(ff)
    IFO.whiten_data()


likelihood = peyote.likelihood.likelihood(IFOs, source)

search_parameters = sim_parameters
search_parameters['f'] = peyote.parameter.Parameter(
    'f', prior=peyote.prior.Uniform(lower=95, upper=105))
search_parameters['A'] = peyote.parameter.Parameter(
    'A', prior=peyote.prior.Uniform(lower=0, upper=1e-19))

sampler = peyote.sampler.Sampler(
    likelihood=likelihood, parameters=search_parameters, sampler='dynesty')
res = sampler.run()

fig, axes = dyplot.traceplot(res)
fig.tight_layout()
fig.savefig('single_trace')

fig = corner.corner(res.samples)
fig.savefig('test')
