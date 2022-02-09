#!/usr/bin/env python
"""
Tutorial to demonstrates reading and plotting the GraceDB data for GW170608,
the lightest GWTC-1 BBH event.

This tutorial must be run on a LIGO cluster.
"""

import bilby

label = "GW170608"
outdir = label + "_out"
# List of detectors involved in the event. GW170608 is pre-August 2017 so only
# H1 and L1 were involved
detectors = ["H1", "L1"]
# Names of the channels and query types to use, and the event GraceDB ID - ref.
# https://ldas-jobs.ligo.caltech.edu/~eve.chase/monitor/online_pe/C02_clean/
# C02_HL_Pv2/lalinferencenest/IMRPhenomPv2pseudoFourPN/config.ini
channel_names = ["H1:DCH-CLEAN_STRAIN_C02", "L1:DCH-CLEAN_STRAIN_C02"]
gracedb = "G288686"
# Duration of data around the event to use
duration = 16
# Duration of PSD data
psd_duration = 1024
# Minimum frequency and reference frequency
minimum_frequency = 10  # Hz
sampling_frequency = 2048.0

# Get trigger time
candidate = bilby.gw.utils.gracedb_to_json(gracedb, outdir=outdir)
trigger_time = candidate["gpstime"]
gps_start_time = trigger_time + 2.0 - duration

# Load the PSD data starting after the segment you want to analyze
psd_start_time = gps_start_time + duration

# Set up interferometer objects from the cache files
interferometers = bilby.gw.detector.InterferometerList([])

for channel_name in channel_names:
    interferometer = bilby.gw.detector.load_data_by_channel_name(
        start_time=gps_start_time,
        segment_duration=duration,
        psd_duration=psd_duration,
        psd_start_time=psd_start_time,
        channel_name=channel_name,
        sampling_frequency=sampling_frequency,
        outdir=outdir,
    )
    interferometer.minimum_frequency = minimum_frequency
    interferometers.append(interferometer)

# Plot the strain data
interferometers.plot_data(outdir=outdir, label=label)
