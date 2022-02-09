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
query_types = ["H1_CLEANED_HOFT_C02", "L1_CLEANED_HOFT_C02"]
gracedb = "G288686"
# Calibration number to inform bilby's guess of query type if those provided are
# not recognised
calibration = 2
# Duration of data around the event to use
duration = 16
# Duration of PSD data
psd_duration = 1024
# Time of event, stored in bilby
trigger_time = bilby.gw.utils.get_event_time(label)
# Minimum frequency and reference frequency
minimum_frequency = 10  # Hz

# Get frame caches
candidate, frame_caches = bilby.gw.utils.get_gracedb(
    gracedb, outdir, duration, calibration, detectors, query_types
)

# Set up interferometer objects from the cache files
interferometers = bilby.gw.detector.InterferometerList([])

for cache_file, channel_name in zip(frame_caches, channel_names):
    interferometer = bilby.gw.detector.load_data_from_cache_file(
        cache_file, trigger_time, duration, psd_duration, channel_name
    )
    interferometer.minimum_frequency = minimum_frequency
    interferometers.append(interferometer)

# Plot the strain data
interferometers.plot_data(outdir=outdir, label=label)
