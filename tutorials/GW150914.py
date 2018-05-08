#!/bin/python
"""
Tutorial to demonstrate running parameter estimation on GW150914 using open data.

This example estimates all 15 parameters of the binary black hole system using commonly used prior distributions.
This will take a few hours to run.
"""
from __future__ import division, print_function
import tupak

tupak.utils.setup_logger()

outdir = 'outdir'
label = 'GW150914'
time_of_event = 1126259462.422

H1, sampling_frequency, time_duration = tupak.detector.get_inteferometer('H1', time_of_event, version=1, outdir=outdir)
L1, _, _ = tupak.detector.get_inteferometer('L1', time_of_event, version=1, outdir=outdir)
interferometers = [H1, L1]

# Define the prior
prior = dict()
prior['mass_1'] = tupak.prior.Uniform(30, 50, 'mass_1')
prior['mass_2'] = tupak.prior.Uniform(20, 40, 'mass_2')
prior['geocent_time'] = tupak.prior.Uniform(time_of_event - 0.1, time_of_event + 0.1, name='geocent_time')
prior['luminosity_distance'] = tupak.prior.PowerLaw(alpha=2, minimum=100, maximum=1000)

# Create the waveform generator
waveform_generator = tupak.waveform_generator.WaveformGenerator(
    tupak.source.lal_binary_black_hole, sampling_frequency, time_duration,
    parameters={'waveform_approximant': 'IMRPhenomPv2', 'reference_frequency': 50})

# Define a likelihood
likelihood = tupak.likelihood.Likelihood(interferometers, waveform_generator)

# Run the sampler
result = tupak.sampler.run_sampler(likelihood, prior, sampler='dynesty', outdir=outdir, label='label')
result.plot_corner()
result.plot_walks()
result.plot_distributions()
print(result)
