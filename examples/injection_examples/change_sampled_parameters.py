#!/bin/python
"""
Tutorial to demonstrate running parameter estimation sampling in non-standard parameters for an injected signal.

This example estimates the masses using a uniform prior in chirp mass and mass ratio and distance using a uniform in
comoving volume prior on luminosity distance between luminosity distances of 100Mpc and 5Gpc, the cosmology is WMAP7.
"""
from __future__ import division, print_function
import tupak
import numpy as np

tupak.core.utils.setup_logger(log_level="info")

time_duration = 4.
sampling_frequency = 2048.
outdir = 'outdir'

np.random.seed(151226)

injection_parameters = dict(mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0, phi_12=1.7, phi_jl=0.3,
                            luminosity_distance=3000., iota=0.4, psi=2.659, phase=1.3, geocent_time=1126259642.413,
                            waveform_approximant='IMRPhenomPv2', reference_frequency=50., ra=1.375, dec=-1.2108)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = tupak.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=sampling_frequency, time_duration=time_duration,
    frequency_domain_source_model=tupak.gw.source.lal_binary_black_hole,
    parameter_conversion=tupak.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    non_standard_sampling_parameter_keys=['chirp_mass', 'mass_ratio', 'cos_iota'],
    parameters=injection_parameters)
hf_signal = waveform_generator.frequency_domain_strain()

# Set up interferometers.
IFOs = [tupak.gw.detector.get_interferometer_with_fake_noise_and_injection(
    name, injection_polarizations=hf_signal, injection_parameters=injection_parameters, time_duration=time_duration,
    sampling_frequency=sampling_frequency, outdir=outdir) for name in ['H1', 'L1', 'V1']]

# Set up prior
priors = dict()
# These parameters will not be sampled
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'phase', 'psi', 'ra', 'dec', 'geocent_time']:
    priors[key] = injection_parameters[key]
priors['luminosity_distance'] = tupak.core.prior.create_default_prior(name='luminosity_distance')

# Initialise GravitationalWaveTransient
likelihood = tupak.gw.likelihood.GravitationalWaveTransient(interferometers=IFOs, waveform_generator=waveform_generator)

# Run sampler
result = tupak.core.sampler.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty',
                                        injection_parameters=injection_parameters, label='DifferentParameters',
                                        outdir=outdir, conversion_function=tupak.gw.conversion.generate_all_bbh_parameters)
result.plot_corner()
print(result)
