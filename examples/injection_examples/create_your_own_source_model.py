#!/bin/python
"""
A script to demonstrate how to use your own source model
"""
from __future__ import division, print_function
import tupak
import numpy as np

# First set up logging and some output directories and labels
outdir = 'outdir'
label = 'create_your_own_source_model'
sampling_frequency = 4096
time_duration = 1


# Here we define out source model - this is the sine-Gaussian model in the
# frequency domain.
def sine_gaussian(f, A, f0, tau, phi0, geocent_time, ra, dec, psi):
    arg = -(np.pi * tau * (f-f0))**2 + 1j * phi0
    plus = np.sqrt(np.pi) * A * tau * np.exp(arg) / 2.
    cross = plus * np.exp(1j*np.pi/2)
    return {'plus': plus, 'cross': cross}


# We now define some parameters that we will inject and then a waveform generator
injection_parameters = dict(A=1e-23, f0=100, tau=1, phi0=0, geocent_time=0,
                            ra=0, dec=0, psi=0)
waveform_generator = tupak.gw.waveform_generator.WaveformGenerator(time_duration=time_duration,
                                                                   sampling_frequency=sampling_frequency,
                                                                   frequency_domain_source_model=sine_gaussian,
                                                                   parameters=injection_parameters)

# Set up interferometers.
IFOs = [tupak.gw.detector.get_interferometer_with_fake_noise_and_injection(
    name, waveform_generator=waveform_generator,
    injection_parameters=injection_parameters, time_duration=time_duration,
    sampling_frequency=sampling_frequency, outdir=outdir)
    for name in ['H1', 'L1', 'V1']]

# Here we define the priors for the search. We use the injection parameters
# except for the amplitude, f0, and geocent_time
prior = injection_parameters.copy()
prior['A'] = tupak.core.prior.PowerLaw(alpha=-1, minimum=1e-25, maximum=1e-21, name='A')
prior['f0'] = tupak.core.prior.Uniform(90, 110, 'f')

likelihood = tupak.gw.likelihood.GravitationalWaveTransient(IFOs, waveform_generator)

result = tupak.core.sampler.run_sampler(
    likelihood, prior, sampler='dynesty', outdir=outdir, label=label,
    resume=False, sample='unif', injection_parameters=injection_parameters)
result.plot_corner()

