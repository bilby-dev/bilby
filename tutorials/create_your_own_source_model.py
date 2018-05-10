#!/bin/python
"""
A script to demonstrate how to use your own source model
"""
from __future__ import division, print_function
import tupak
import numpy as np

# First set up logging and some output directories and labels
tupak.utils.setup_logger()
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
injection_parameters = dict(A=1e-21, f0=10, tau=1, phi0=0, geocent_time=0,
                            ra=0, dec=0, psi=0)
waveform_generator = tupak.waveform_generator.WaveformGenerator(
    frequency_domain_source_model=sine_gaussian,
    sampling_frequency=sampling_frequency,
    time_duration=time_duration,
    parameters=injection_parameters)
hf_signal = waveform_generator.frequency_domain_strain()

# Set up interferometers.
IFOs = [tupak.detector.get_interferometer_with_fake_noise_and_injection(
    name, injection_polarizations=hf_signal,
    injection_parameters=injection_parameters, time_duration=time_duration,
    sampling_frequency=sampling_frequency, outdir=outdir)
    for name in ['H1', 'L1', 'V1']]

# Here we define the priors for the search. We use the injection parameters
# except for the amplitude, f0, and geocent_time
prior = injection_parameters.copy()
prior['A'] = tupak.prior.Uniform(0, 1e-20, 'A')
prior['f0'] = tupak.prior.Uniform(0, 20, 'f')
prior['geocent_time'] = tupak.prior.Uniform(-0.01, 0.01, 'geocent_time')

likelihood = tupak.likelihood.Likelihood(IFOs, waveform_generator)

result = tupak.sampler.run_sampler(
    likelihood, prior, sampler='dynesty', outdir=outdir, label=label,
    resume=False, sample='unif')
result.plot_walks()
result.plot_distributions()
result.plot_corner()
print(result)
