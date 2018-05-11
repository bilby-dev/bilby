#!/bin/python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter space for an injected signal.

This example estimates the masses using a uniform prior in both component masses and distance using a uniform in
comoving volume prior on luminosity distance between luminosity distances of 100Mpc and 5Gpc, the cosmology is WMAP7.
"""
from __future__ import division, print_function
import tupak
import numpy as np

tupak.utils.setup_logger(log_level="info")

time_duration = 4.
sampling_frequency = 2048.
outdir = 'outdir'

np.random.seed(170809)

injection_parameters = dict(mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0, phi_12=1.7, phi_jl=0.3,
                            luminosity_distance=4000., iota=0.4, psi=2.659, phase=1.3, geocent_time=1126259642.413,
                            waveform_approximant='IMRPhenomPv2', reference_frequency=50., ra=1.375, dec=-1.2108)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = tupak.waveform_generator.WaveformGenerator(
    sampling_frequency=sampling_frequency, time_duration=time_duration,
    frequency_domain_source_model=tupak.source.lal_binary_black_hole,
    parameters=injection_parameters)
hf_signal = waveform_generator.frequency_domain_strain()

# Set up interferometers.
IFOs = [tupak.detector.get_interferometer_with_fake_noise_and_injection(
    name, injection_polarizations=hf_signal, injection_parameters=injection_parameters, time_duration=time_duration,
    sampling_frequency=sampling_frequency, outdir=outdir) for name in ['H1', 'L1', 'V1']]

# Set up prior
priors = dict()
# These parameters will not be sampled
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'phase', 'psi', 'iota', 'ra', 'dec', 'geocent_time']:
    priors[key] = injection_parameters[key]
priors['luminosity_distance'] = tupak.prior.create_default_prior(name='luminosity_distance')

# Initialise Likelihood
likelihood = tupak.likelihood.Likelihood(interferometers=IFOs, waveform_generator=waveform_generator)

# Run sampler
result = tupak.sampler.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', npoints=100, walks=10,
                                   injection_parameters=injection_parameters, outdir=outdir, label='BasicTutorial')
result.plot_corner()
result.plot_walks()
result.plot_distributions()
print(result)
