#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a sine gaussian
injected signal.
"""
from __future__ import division, print_function
import bilby
import numpy as np

# Set the duration and sampling frequency of the data segment that we're going
# to inject the signal into
duration = 4.
sampling_frequency = 2048.

# Specify the output directory and the name of the simulation.
outdir = 'outdir'
label = 'sine_gaussian'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(170801)

# We are going to inject a sine gaussian waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters
injection_parameters = dict(
    hrss=1e-22, Q=5.0, frequency=200.0, ra=1.375, dec=-1.2108,
    geocent_time=1126259642.413, psi=2.659)

# Create the waveform_generator using a sine Gaussian source function
waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.sinegaussian)

# Set up interferometers.  In this case we'll use three interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1), and Virgo (V1)).  These default to
# their design sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)

# Set up prior, which is a dictionary
priors = dict()
for key in ['psi', 'ra', 'dec', 'geocent_time']:
    priors[key] = injection_parameters[key]

priors['Q'] = bilby.core.prior.Uniform(2, 50, 'Q')
priors['frequency'] = bilby.core.prior.Uniform(30, 1000, 'frequency', unit='Hz')
priors['hrss'] = bilby.core.prior.Uniform(1e-23, 1e-21, 'hrss')

# Initialise the likelihood by passing in the interferometer data (IFOs) and
# the waveoform generator
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator)

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.core.sampler.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
    injection_parameters=injection_parameters, outdir=outdir, label=label)

# make some plots of the outputs
result.plot_corner()
