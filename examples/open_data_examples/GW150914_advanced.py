#!/usr/bin/env python
"""
This tutorial includes advanced specifications for analysing known events.
Here GW150914 is used as an example.

LIST OF AVAILABLE EVENTS:
>> from gwosc import datasets
>> for event in datasets.find_datasets(type="event"):
...     print(event, datasets.event_gps(event))

List of events in BILBY dict: run -> help(bilby.gw.utils.get_event_time(event))
"""
from __future__ import division, print_function
import bilby

outdir = 'outdir'
label = 'GW150914'
time_of_event = bilby.gw.utils.get_event_time(label)

bilby.core.utils.setup_logger(outdir=outdir, label=label)

# GET DATA FROM INTERFEROMETER
interferometer_names = ['H1', 'L1']  # include 'V1' for appropriate O2 events
duration = 4    # length of data segment containing the signal
roll_off = 0.2  # smoothness of transition from no signal
# to max signal in a Tukey Window.
psd_offset = -1024  # PSD is estimated using data from
# 'center_time + psd_offset' to 'center_time + psd_offset + psd_duration'.
# This determines the time window used to fetch open data.
psd_duration = 100
filter_freq = None  # low pass filter frequency to cut signal content above
# Nyquist frequency. The condition is 2 * filter_freq >= sampling_frequency

# Keyword args are passed to 'gwpy.timeseries.TimeSeries.fetch_open_data()'
# sample_rate: most data are stored by LOSC at 4096 Hz,
# there may be event-related data releases with a 16384 Hz rate.
kwargs = {"sample_rate": 4096}
# For O2 events a "tag" is required to download the data.
# CLN = clean data; C00 or C01 = raw data
# kwargs = {"tag": 'CLN'}
# For some events can specify channels: source data stream for LOSC data.
# kwargs = {"channel": {'H1': 'H1:DCS-CALIB_STRAIN_C02',
#                      'L1': 'L1:DCS-CALIB_STRAIN_C02',
#                      'V1': 'V1:FAKE_h_16384Hz_4R'}}

interferometers = bilby.gw.detector.get_event_data(
    label, interferometer_names=interferometer_names,
    duration=duration, roll_off=roll_off, psd_offset=psd_offset,
    psd_duration=psd_duration, cache=True,
    filter_freq=filter_freq, **kwargs)

# CHOOSE PRIOR FILE
# DEFAULT PRIOR FILES: GW150914.prior, binary_black_holes.prior,
# binary_neutron_stars.prior (if bns, use BNSPriorDict)
# Needs to specify path for any other prior file.
prior = bilby.gw.prior.BBHPriorDict(filename='GW150914.prior')

# GENERATE WAVEFORM
duration = None  # duration and sampling frequency will be overwritten
# to match the ones in the interferometers.
sampling_frequency = kwargs["sample_rate"]  # same at which the data is stored
start_time = 0  # set the starting time of the time array

# reference_frequency: either low freq. limit or most sensitive freq.
# OVERVIEW OF APPROXIMANTS:
# https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/Waveforms/Overview
waveform_arguments = {'waveform_approximant': 'IMRPhenomPv2',
                      'reference_frequency': 50}
source_model = bilby.gw.source.lal_binary_black_hole

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    start_time=start_time,
    frequency_domain_source_model=source_model,
    waveform_arguments=waveform_arguments)

# CHOOSE LIKELIHOOD FUNCTION
# Time marginalisation uses FFT and can use a prior file.
# Distance marginalisation uses a look up table calculated at run time.
# Phase marginalisation is done analytically using a Bessel function.
# If prior given, used in the distance and phase marginalization.
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers, waveform_generator, time_marginalization=False,
    distance_marginalization=False, phase_marginalization=False)

# RUN SAMPLER
# Can use log_likelihood_ratio, rather than just the log_likelihood.
# If using simulated data, pass a dictionary of injection parameters.
# A function can be specified in 'conversion_function' and applied
# to posterior to generate additional parameters e.g. source-frame masses.

# Implemented Samplers:
# LIST OF AVAILABLE SAMPLERS: Run -> bilby.sampler.implemented_samplers
npoints = 500  # number of live points for the MCMC (dynesty)
sampler = 'dynesty'
# Different samplers can have different additional kwargs,
# visit https://lscsoft.docs.ligo.org/bilby/samplers.html for details.

result = bilby.run_sampler(
    likelihood, prior, outdir=outdir, label=label,
    sampler=sampler, npoints=npoints, use_ratio=False,
    injection_parameters=None,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters)

result.plot_corner()
