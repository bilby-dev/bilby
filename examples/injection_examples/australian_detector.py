#!/bin/python
"""
Tutorial to demonstrate a new interferometer

We place a new instrument in Gingin, with an A+ sensitivity in a network of A+
interferometers at Hanford and Livingston
"""
from __future__ import division, print_function

import numpy as np

import bilby
import gwinc

# Set the duration and sampling frequency of the data segment that we're going
# to inject the signal into
duration = 4.
sampling_frequency = 2048.

# Specify the output directory and the name of the simulation.
outdir = 'outdir'
label = 'australian_detector'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170232)

# create a new detector using a PyGwinc sensitivity curve
frequencies = np.logspace(0, 3, 1000)
gwinc_detector = gwinc.load_ifo('A+')
gwinc_detector = gwinc.precompIFO(frequencies, gwinc_detector)
gwinc_noises = gwinc.noise_calc(frequencies, gwinc_detector)

Aplus_psd = gwinc_noises['Total']

# Set up the detector as a four-kilometer detector in Gingin
# The location of this detector is not defined in Bilby, so we need to add it
AusIFO = bilby.gw.detector.Interferometer(
    power_spectral_density=bilby.gw.detector.PowerSpectralDensity(
        frequency_array=frequencies, psd_array=Aplus_psd),
    name='AusIFO', length=4,
    minimum_frequency=min(frequencies), maximum_frequency=max(frequencies),
    latitude=-31.34, longitude=115.91,
    elevation=0., xarm_azimuth=2., yarm_azimuth=125.)

# Set up two other detectors at Hanford and Livingston
interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1'])
for interferometer in interferometers:
    interferometer.power_spectral_density =\
        bilby.gw.detector.PowerSpectralDensity(
            frequency_array=frequencies, psd_array=Aplus_psd)

# append the Australian detector to the list of other detectors
interferometers.append(AusIFO)


# Inject a gravitational-wave signal into the network
# as we're using a three-detector network of A+, we inject a GW150914-like
# signal at 4 Gpc
injection_parameters = dict(
    mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=4000., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=0.2108)


# Fixed arguments passed into the source model
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=50.)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments)

start_time = injection_parameters['geocent_time'] + 2 - duration

# inject the signal into the interferometers

for interferometer in interferometers:
    interferometer.set_strain_data_from_power_spectral_density(
        sampling_frequency=sampling_frequency, duration=duration)
    interferometer.inject_signal(
        parameters=injection_parameters, waveform_generator=waveform_generator)

    # plot the data for sanity
    signal = interferometer.get_detector_response(
        waveform_generator.frequency_domain_strain(), injection_parameters)
    interferometer.plot_data(signal=signal, outdir=outdir, label=label)

# set up priors
priors = bilby.gw.prior.BBHPriorDict()
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'psi',
            'geocent_time', 'phase']:
    priors[key] = injection_parameters[key]

# Initialise the likelihood by passing in the interferometer data (IFOs)
# and the waveoform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=interferometers, waveform_generator=waveform_generator,
    time_marginalization=False, phase_marginalization=False,
    distance_marginalization=False, prior=priors)


result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, npoints=1000,
    injection_parameters=injection_parameters, outdir=outdir, label=label)

# make some plots of the outputs
result.plot_corner()
