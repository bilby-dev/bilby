#!/bin/python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter space for an injected signal.

This example estimates the masses using a uniform prior in both component masses and distance using a uniform in
comoving volume prior on luminosity distance between luminosity distances of 100Mpc and 5Gpc, the cosmology is WMAP7.
"""
from __future__ import division, print_function

import numpy as np

import tupak

# Set the duration and sampling frequency of the data segment that we're going to inject the signal into

time_duration = 4.
sampling_frequency = 2048.

# Specify the output directory and the name of the simulation.
outdir = 'outdir'
label = 'basic_tutorial'
tupak.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

# We are going to inject a binary black hole waveform.  We first establish a dictionary of parameters that
# includes all of the different waveform parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
injection_parameters = dict(mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0, phi_12=1.7, phi_jl=0.3,
                            luminosity_distance=2000., iota=0.4, psi=2.659, phase=1.3, geocent_time=1126259642.413,
                            ra=1.375, dec=-1.2108)

# Fixed arguments passed into the source model
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=50.)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = tupak.WaveformGenerator(time_duration=time_duration,
                                             sampling_frequency=sampling_frequency,
                                             frequency_domain_source_model=tupak.gw.source.lal_binary_black_hole,
                                             parameters=injection_parameters,
                                             waveform_arguments=waveform_arguments)
hf_signal = waveform_generator.frequency_domain_strain()

# Set up interferometers.  In this case we'll use three interferometers (LIGO-Hanford (H1), LIGO-Livingston (L1),
# and Virgo (V1)).  These default to their design sensitivity
IFOs = [tupak.gw.detector.get_interferometer_with_fake_noise_and_injection(
    name, injection_polarizations=hf_signal, injection_parameters=injection_parameters, time_duration=time_duration,
    sampling_frequency=sampling_frequency, outdir=outdir) for name in ['H1', 'L1']]

# Set up prior, which is a dictionary
# By default we will sample all terms in the signal models.  However, this will take a long time for the calculation,
# so for this example we will set almost all of the priors to be equall to their injected values.  This implies the
# prior is a delta function at the true, injected value.  In reality, the sampler implementation is smart enough to
# not sample any parameter that has a delta-function prior.
# The above list does *not* include mass_1, mass_2, iota and luminosity_distance, which means those are the parameters
# that will be included in the sampler.  If we do nothing, then the default priors get used.
priors = tupak.gw.prior.BBHPriorSet()
priors['geocent_time'] = tupak.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 1, maximum=injection_parameters['geocent_time'] + 1,
    name='geocent_time', latex_label='$t_c$')
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'psi', 'ra', 'dec', 'geocent_time', 'phase']:
    priors[key] = injection_parameters[key]

# Initialise the likelihood by passing in the interferometer data (IFOs) and the waveoform generator
likelihood = tupak.GravitationalWaveTransient(interferometers=IFOs, waveform_generator=waveform_generator,
                                              time_marginalization=False, phase_marginalization=False,
                                              distance_marginalization=False, prior=priors)

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = tupak.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
                           injection_parameters=injection_parameters, outdir=outdir, label=label)

# make some plots of the outputs
result.plot_corner()
print(result)
