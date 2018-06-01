#!/bin/python
"""
Tutorial to demonstrate running parameter estimation on GW150914 using open
data.

This example estimates all 15 parameters of the binary black hole system using
commonly used prior distributions.  This will take a few hours to run.
"""
from __future__ import division, print_function
import tupak

# Define some convienence labels and the trigger time of the event
outdir = 'outdir'
label = 'GW150914'
time_of_event = tupak.utils.get_event_time(label)

# This sets up logging output to understand what tupak is doing
tupak.utils.setup_logger(outdir=outdir, label=label)

# Here we import the detector data. This step downloads data from the
# LIGO/Virgo open data archives. The data is saved to an `Interferometer`
# object (here called `H1` and `L1`). A Power Spectral Density (PSD) estimate
# is also generated and saved to the same object. To check the data and PSD
# makes sense, for each detector a plot is created in the `outdir` called
# H1_frequency_domain_data.png and LI_frequency_domain_data.png. The two
# objects are then placed into a list.
interferometers = tupak.detector.get_event_data(label)

# We now define the prior. You'll notice we only do this for the two masses,
# the merger time, and the distance; in each case choosing a prior which
# roughly bounds the known values. All other parameters will use a default
# prior (this is printed to the terminal at run-time). You can overwrite this
# using the syntax below, or choose a fixed value by just providing a float
# value as the prior.
prior = dict()
prior['mass_1'] = tupak.prior.Uniform(30, 50, 'mass_1')
prior['mass_2'] = tupak.prior.Uniform(20, 40, 'mass_2')
prior['geocent_time'] = tupak.prior.Uniform(
    time_of_event - 0.1, time_of_event + 0.1, name='geocent_time')
#prior['geocent_time'] = time_of_event
prior['luminosity_distance'] = tupak.prior.PowerLaw(
    alpha=2, minimum=100, maximum=1000)

# In this step we define a `waveform_generator`. This is out object which
# creates the frequency-domain strain. In this instance, we are using the
# `lal_binary_black_hole model` source model. We also pass other parameters:
# the waveform approximant and reference frequency.
waveform_generator = tupak.WaveformGenerator(time_duration=interferometers[0].duration,
                                             sampling_frequency=interferometers[0].sampling_frequency,
                                             frequency_domain_source_model=tupak.source.lal_binary_black_hole,
                                             parameters={'waveform_approximant': 'IMRPhenomPv2',
                                                         'reference_frequency': 50})

# In this step, we define the likelihood. Here we use the standard likelihood
# function, passing it the data and the waveform generator.
<<<<<<< HEAD
likelihood = tupak.likelihood.GravitationalWaveTransient(interferometers, waveform_generator, time_marginalization=False, distance_marginalization=True, prior=prior)
=======
likelihood = tupak.GravitationalWaveTransient(interferometers, waveform_generator)
>>>>>>> e8b704afba65886f49cb448402a95ae90c468a24

# Finally, we run the sampler. This function takes the likelihood and prio
# along with some options for how to do the sampling and how to save the data
result = tupak.run_sampler(likelihood, prior, sampler='dynesty',
                           outdir=outdir, label=label)
result.plot_corner()
print(result)
