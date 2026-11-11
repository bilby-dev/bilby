#!/usr/bin/env python
"""
Tutorial to demonstrate running time-domain parameter estimation on GW150914

This is the GW150914.py analysis done in the time domain. The strain around
the analysis segment is downsampled to twice the maximum frequency, the noise
model of each detector is an autocovariance function computed from the PSD of
the GWTC-2.1 parameter-estimation release [1], and the IMRPhenomTPHM waveform
is evaluated at the data samples (placement "model_time"). Time and phase
marginalization are not available for the time-domain likelihood.

Set the interferometers' frequency band and power spectral densities, the
analysis windows and the conditioned strain data first, then the
autocovariance functions, which are not updated if any of these change, and
then create the likelihood.

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

# Data downloaded on each side of the analysis segment, removed by
# condition_strain_data (the analysis segment must lie in the central half of the data)
padding = duration

# The PSDs of the GWTC-2.1 parameter-estimation release, from GWOSC [1]:
# https://zenodo.org/records/6513631/files/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5
psd_file = "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
psd_label = "C01:IMRPhenomXPHM"
with h5py.File(psd_file, "r") as ff:
    psds = {det: ff[psd_label]["psds"][det][:] for det in detectors}

# We now use gwpy to obtain the analysis data, downsample it around the
# analysis segment with condition_strain_data, and create the ifo_list with the
# PSDs and the autocovariance functions
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

# We now define the prior.
# We have defined our prior distribution in a local file, GW150914.prior
# The prior is printed to the terminal at run-time.
# You can overwrite this using the syntax below in the file,
# or choose a fixed value by just providing a float value as the prior.
priors = bilby.gw.prior.BBHPriorDict(filename="GW150914.prior")

# Add the geocent time prior
priors["geocent_time"] = bilby.core.prior.Uniform(
    trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
)

# In this step we define a `waveform_generator`. This is the object which
# creates the time-domain strain. In this instance, we are using the
# `lal_binary_black_hole_model_time` source model, which evaluates the waveform
# at the data samples of each detector. We also pass other parameters: the
# waveform approximant, reference frequency and minimum frequency, and a
# parameter conversion which allows us to sample in chirp mass and ratio rather
# than component mass.
# There are two minimum frequencies. The minimum_frequency of the
# interferometers (20 Hz) is the lower edge of the analysis band, as in a
# frequency-domain analysis. The minimum_frequency of the waveform (10 Hz) is
# the frequency at which the waveform starts. In the time domain, every
# waveform must start before the analysis window: a waveform that switches on
# inside the window has a sharp edge, whose power is spread over all
# frequencies, including the analysis band. From 20 Hz, the waveforms in this
# prior (chirp mass 25 to 35) would start only about 1 to 1.4 s before the
# merger, inside the window, which starts 2 s before it; from 10 Hz, they start
# about 4.5 to 8 s before the merger. Use a lower minimum_frequency for lighter
# binaries or longer windows.
waveform_generator = bilby.gw.WaveformGenerator(
    time_domain_source_model=bilby.gw.source.lal_binary_black_hole_model_time,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={
        "waveform_approximant": "IMRPhenomTPHM",
        "reference_frequency": 20,
        "minimum_frequency": 10,
    },
)

# In this step, we define the likelihood. Here we use the time-domain
# likelihood, passing it the data and the waveform generator.
likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    priors=priors,
    distance_marginalization=False,
    placement="model_time",
)

# Finally, we run the sampler. This function takes the likelihood and prior
# along with some options for how to do the sampling and how to save the data
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
