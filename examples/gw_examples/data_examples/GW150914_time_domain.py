#!/usr/bin/env python
"""
Tutorial to demonstrate running time-domain parameter estimation on GW150914

This is the GW150914.py analysis done in the time domain. The strain around
the analysis segment is downsampled to twice the maximum frequency, the noise
model of each detector is an autocovariance function computed from the PSD of
the GWTC-2.1 parameter-estimation release [1], and the IMRPhenomTPHM waveform
is evaluated at the data samples (placement "model_time"). Time and phase
marginalization are not available for the time-domain likelihood.

[1] https://gwosc.org/eventapi/html/GWTC-2.1-confident/GW150914/v4/
"""
import bilby
import h5py
from gwpy.timeseries import TimeSeries

logger = bilby.core.utils.logger
outdir = "outdir"
label = "GW150914_time_domain"

# Note you can get trigger times using the gwosc package, e.g.:
# > from gwosc import datasets
# > datasets.event_gps("GW150914")
trigger_time = 1126259462.4
detectors = ["H1", "L1"]
maximum_frequency = 512
minimum_frequency = 20
sampling_frequency = 2 * maximum_frequency  # The data are downsampled to this rate
duration = 4  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

# Data downloaded on each side of the analysis segment, removed by the
# downsampling (condition_strain_data, which needs duration / 2 on each side)
padding = duration / 2

# The PSDs of the GWTC-2.1 parameter-estimation release, from GWOSC [1]:
# https://zenodo.org/records/6513631/files/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5
psd_file = "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
psd_label = "C01:IMRPhenomXPHM"
with h5py.File(psd_file, "r") as ff:
    psds = {det: ff[psd_label]["psds"][det][:] for det in detectors}

ifo_list = bilby.gw.detector.InterferometerList([])
for det in detectors:
    logger.info("Downloading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.set_analysis_window(start_time, duration)
    data = TimeSeries.fetch_open_data(det, start_time - padding, end_time + padding)
    ifo.strain_data.set_from_gwpy_timeseries(data)
    ifo.condition_strain_data(sampling_frequency)

    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psds[det][:, 0], psd_array=psds[det][:, 1]
    )
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    ifo.set_autocovariance_function_from_power_spectral_density()
    ifo_list.append(ifo)

logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)

priors = bilby.gw.prior.BBHPriorDict(filename="GW150914.prior")

priors["geocent_time"] = bilby.core.prior.Uniform(
    trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
)

"""
Fixed arguments passed into the source model.

There are two different minimum frequencies:
- ifo.minimum_frequency (20 Hz): the lowest frequency used in the analysis.
- minimum_frequency below (10 Hz): the frequency at which the waveform starts.

The waveform starts at 10 Hz, below 20 Hz, for two reasons:
1. Higher modes. Higher modes oscillate faster than the main (2, 2) mode
   (the (4, 4) mode at twice its frequency). If the (2, 2) mode starts at
   10 Hz, all modes are already present at 20 Hz.
2. The start of the waveform must not be in the data. The waveform switches
   on suddenly, and this sudden start disturbs the analysis even above 20 Hz.
   Starting at 10 Hz, every waveform in the prior begins before the analysis
   window, so its start is never in the data. With a wider prior or a longer
   window, use a lower value.
"""

waveform_generator = bilby.gw.WaveformGenerator(
    time_domain_source_model=bilby.gw.source.lal_binary_black_hole_model_time,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={
        "waveform_approximant": "IMRPhenomTPHM",
        "reference_frequency": 20,
        "minimum_frequency": 10,
    },
)

likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    priors=priors,
    distance_marginalization=False,
    placement="model_time",
)

result = bilby.run_sampler(
    likelihood,
    priors,
    sampler="dynesty",
    sample="acceptance-walk",
    nlive=500,
    naccept=60,
    outdir=outdir,
    label=label,
    check_point_delta_t=600,
    check_point_plot=True,
    npool=1,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    result_class=bilby.gw.result.CBCResult,
)
result.plot_corner()
