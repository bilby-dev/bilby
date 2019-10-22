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
from gwpy.timeseries import TimeSeries

outdir = 'outdir'
label = 'GW150914'
logger = bilby.core.utils.logger
time_of_event = bilby.gw.utils.get_event_time(label)

bilby.core.utils.setup_logger(outdir=outdir, label=label)

# GET DATA FROM INTERFEROMETER
interferometer_names = ['H1', 'L1']  # include 'V1' for appropriate O2 events
duration = 4  # length of data segment containing the signal
post_trigger_duration = 2  # time between trigger time and end of segment
end_time = time_of_event + post_trigger_duration
start_time = end_time - duration

roll_off = 0.4  # smoothness in a tukey window, default is 0.4s
# This determines the time window used to fetch open data
psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

filter_freq = None  # low pass filter frequency to cut signal content above
# Nyquist frequency. The condition is 2 * filter_freq >= sampling_frequency


ifo_list = bilby.gw.detector.InterferometerList([])
for det in interferometer_names:
    logger.info("Downloading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.fetch_open_data(det, start_time, end_time)
    ifo.set_strain_data_from_gwpy_timeseries(data)
# Additional arguments you might need to pass to TimeSeries.fetch_open_data:
# - sample_rate = 4096, most data are stored by LOSC at this frequency
# there may be event-related data releases with a 16384Hz rate.
# - tag = 'CLN' for clean data; C00/C01 for raw data (different releases)
# note that for O2 events a "tag" is required to download the data.
# - channel =  {'H1': 'H1:DCS-CALIB_STRAIN_C02',
#               'L1': 'L1:DCS-CALIB_STRAIN_C02',
#               'V1': 'V1:FAKE_h_16384Hz_4R'}}
# for some events can specify channels: source data stream for LOSC data.
    logger.info("Downloading psd data for ifo {}".format(det))
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
    psd_alpha = 2 * roll_off / duration  # shape parameter of tukey window
    psd = psd_data.psd(
        fftlength=duration,
        overlap=0,
        window=("tukey", psd_alpha),
        method="median")
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value)
    ifo_list.append(ifo)

logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)

# CHOOSE PRIOR FILE
# DEFAULT PRIOR FILES: GW150914.prior, binary_black_holes.prior,
# binary_neutron_stars.prior (if bns, use BNSPriorDict)
# Needs to specify path if you want to use any other prior file.
prior = bilby.gw.prior.BBHPriorDict(filename='GW150914.prior')

# GENERATE WAVEFORM
sampling_frequency = 4096.  # same at which the data is stored
conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters

# OVERVIEW OF APPROXIMANTS:
# https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/Waveforms/Overview
waveform_arguments = {
    'waveform_approximant': 'IMRPhenomPv2',
    'reference_frequency': 50  # most sensitive frequency
}

waveform_generator = bilby.gw.WaveformGenerator(
    parameter_conversion=conversion,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments)

# CHOOSE LIKELIHOOD FUNCTION
# Time marginalisation uses FFT and can use a prior file.
# Distance marginalisation uses a look up table calculated at run time.
# Phase marginalisation is done analytically using a Bessel function.
# If prior given, used in the distance and phase marginalization.
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifo_list, waveform_generator=waveform_generator,
    priors=prior, time_marginalization=False, distance_marginalization=False,
    phase_marginalization=False)

# RUN SAMPLER
# Can use log_likelihood_ratio, rather than just the log_likelihood.
# A function can be specified in 'conversion_function' and applied
# to posterior to generate additional parameters e.g. source-frame masses.

# Implemented Samplers:
# LIST OF AVAILABLE SAMPLERS: Run -> bilby.sampler.implemented_samplers
npoints = 512  # number of live points for the nested sampler
n_steps = 100  # min number of steps before proposing a new live point,
# defaults `ndim * 10`
sampler = 'dynesty'
# Different samplers can have different additional kwargs,
# visit https://lscsoft.docs.ligo.org/bilby/samplers.html for details.

result = bilby.run_sampler(
    likelihood, prior, outdir=outdir, label=label,
    sampler=sampler, nlive=npoints, use_ratio=False,
    walks=n_steps, n_check_point=10000, check_point_plot=True,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters)
result.plot_corner()
