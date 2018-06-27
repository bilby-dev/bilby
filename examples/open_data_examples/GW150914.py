#!/bin/python
"""
Tutorial to demonstrate running parameter estimation on GW150914 using open
data.

This example estimates all 15 parameters of the binary black hole system using
commonly used prior distributions.  This will take a few hours to run.
"""
from __future__ import division, print_function
import tupak

outdir = 'outdir'
label = 'GW150914'
time_of_event = tupak.gw.utils.get_event_time(label)

# This sets up logging output to understand what tupak is doing
tupak.core.utils.setup_logger(outdir=outdir, label=label)

# Here we import the detector data. This step downloads data from the
# LIGO/Virgo open data archives. The data is saved to an `Interferometer`
# object (here called `H1` and `L1`). A Power Spectral Density (PSD) estimate
# is also generated and saved to the same object. To check the data and PSD
# makes sense, for each detector a plot is created in the `outdir` called
# H1_frequency_domain_data.png and LI_frequency_domain_data.png. The two
# objects are then placed into a list.
interferometers = tupak.gw.detector.get_event_data(label)

# We now define the prior.
# We have defined our prior distribution in a file packaged with TUPAK.
# The prior is printed to the terminal at run-time.
# You can overwrite this using the syntax below in the file,
# or choose a fixed value by just providing a float value as the prior.
prior = tupak.gw.prior.BBHPriorSet(filename='GW150914.prior')

# In this step we define a `waveform_generator`. This is out object which
# creates the frequency-domain strain. In this instance, we are using the
# `lal_binary_black_hole model` source model. We also pass other parameters:
# the waveform approximant and reference frequency.
waveform_generator = tupak.WaveformGenerator(time_duration=interferometers.duration,
                                             sampling_frequency=interferometers.sampling_frequency,
                                             frequency_domain_source_model=tupak.gw.source.lal_binary_black_hole,
                                             waveform_arguments={'waveform_approximant': 'IMRPhenomPv2',
                                                                 'reference_frequency': 50})

# In this step, we define the likelihood. Here we use the standard likelihood
# function, passing it the data and the waveform generator.
likelihood = tupak.gw.likelihood.GravitationalWaveTransient(interferometers, waveform_generator)

# Finally, we run the sampler. This function takes the likelihood and prior
# along with some options for how to do the sampling and how to save the data
result = tupak.run_sampler(likelihood, prior, sampler='dynesty',
                           outdir=outdir, label=label)
result.plot_corner()

