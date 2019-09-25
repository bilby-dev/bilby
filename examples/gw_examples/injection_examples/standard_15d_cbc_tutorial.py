#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a full 15 parameter
space for an injected cbc signal. This is the standard injection analysis script
one can modify for the study of injected CBC events.
"""
from __future__ import division, print_function
import numpy as np
import bilby

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 4.
sampling_frequency = 2048.

# Specify the output directory and the name of the simulation.
outdir = 'outdir'
label = 'full_15_parameters'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
injection_parameters = dict(
    mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

# Fixed arguments passed into the source model
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=50., minimum_frequency=20.)

# Create the waveform_generator using a LAL BinaryBlackHole source function
# the generator will convert all the parameters
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)

ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)


# For this analysis, we implemenet the standard BBH priors defined, except for
# the definition of the time prior, which is defined as uniform about the
# injected value.
# Furthermore, we decide to sample in chirp mass and mass ratio, due to the
# preferred shape for the associated posterior distributions.
priors = bilby.gw.prior.BBHPriorDict()
priors.pop('mass_1')
priors.pop('mass_2')

priors['chirp_mass'] = bilby.prior.Uniform(
    name='chirp_mass', latex_label='$M$', minimum=10.0, maximum=100.0,
    unit='$M_{\\odot}$')

priors['mass_ratio'] = bilby.prior.Uniform(
    name='mass_ratio', latex_label='$q$', minimum=0.5, maximum=1.0)

priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1,
    maximum=injection_parameters['geocent_time'] + 0.1,
    name='geocent_time', latex_label='$t_c$', unit='$s$')

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveoform generator, as well the priors.
# The explicit time, distance, and phase marginalizations are turned on to
# improve convergence, and the parameters are recovered by the conversion
# function.
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator, priors=priors,
    distance_marginalization=True, phase_marginalization=True, time_marginalization=True)

# Run sampler. In this case we're going to use the `cpnest` sampler
# Note that the maxmcmc parameter is increased so that between each iteration of
# the nested sampler approach, the walkers will move further using an mcmc
# approach, searching the full parameter space.
# The conversion function will determine the distance, phase and coalescence
# time posteriors in post processing.
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='cpnest', npoints=2000,
    injection_parameters=injection_parameters, outdir=outdir,
    label=label, maxmcmc=2000,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters)

# Make a corner plot.
result.plot_corner()
