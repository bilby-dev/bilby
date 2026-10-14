#!/usr/bin/env python
"""
Tutorial to demonstrate running time-domain parameter estimation on GW150914

This is the GW150914.py analysis done in the time domain:

- the strain is downloaded as a time series covering the 4 s analysis window
  and 2 s on each side and downsampled to 2048 Hz; the likelihood uses the
  data in the window (interferometer.analysis_data);
- the noise model of each detector is an autocovariance function (ACF)
  computed from a PSD read with its frequency array from an HDF5 file;
- the likelihood is bilby.gw.likelihood.TimeDomainGravitationalWaveTransient,
  with a time-domain waveform model.

Distance marginalization is used. Time and phase marginalization are not
available for the time-domain likelihood.

This example estimates all 15 parameters of the binary black hole system using
commonly used prior distributions. This will take several hours to run. The
data is obtained using gwpy, see [1] for information on how to access data on
the LIGO Data Grid instead.

[1] https://gwpy.github.io/docs/stable/timeseries/remote-access.html
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
sampling_frequency = 2048  # The PSD file below goes up to 1024 Hz
duration = 4  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

# Data downloaded on each side of the analysis window. The downsampling
# removes duration / 2 of data on each side of the window (trim=0.25), so this
# must be at least duration / 2.
padding = duration / 2

# File with the PSDs and their frequency arrays (psds/h1_psd/frequency,
# psds/h1_psd/spectrum, ...). For a GWTC PESummary release file, use instead
# ff[label]["psds"][det][:, 0] and [:, 1], with label the analysis, e.g. "C01:IMRPhenomXPHM".
psd_file = "GW150914_data.h5"
with h5py.File(psd_file, "r") as ff:
    psds = {det: (ff["psds"][f"{det.lower()}_psd"]["frequency"][:], ff["psds"][f"{det.lower()}_psd"]["spectrum"][:])
            for det in detectors}

# We now use gwpy to obtain the analysis data and create the ifo_list
ifo_list = bilby.gw.detector.InterferometerList([])
for det in detectors:
    logger.info("Downloading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.set_analysis_window(start_time, duration)
    data = TimeSeries.fetch_open_data(det, start_time - padding, end_time + padding)
    ifo.strain_data.set_from_gwpy_timeseries(data)
    ifo.condition_strain_data(sampling_frequency)

    frequency_array, psd_array = psds[det]
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=frequency_array, psd_array=psd_array
    )
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency

    # The ACF lasts 1 / (frequency spacing of the PSD), 8 s here, twice the
    # analysis segment. Outside [minimum_frequency, maximum_frequency] the PSD
    # is set to fill_multiplier (default 1e4) times its maximum inside the band.
    ifo.autocovariance_function = (
        bilby.gw.detector.AutoCovarianceFunction.from_power_spectral_density_array(
            frequency_array,
            psd_array,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            sampling_frequency=sampling_frequency,
        )
    )
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
# creates the time-domain polarizations. In this instance, we are using the
# `lal_binary_black_hole_time_domain` source model. We also pass other
# parameters: the waveform approximant, reference and minimum frequency, and a
# parameter conversion which allows us to sample in chirp mass and ratio
# rather than component mass
waveform_generator = bilby.gw.WaveformGenerator(
    time_domain_source_model=bilby.gw.source.lal_binary_black_hole_time_domain,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={
        "waveform_approximant": "IMRPhenomPv2",
        "reference_frequency": 50,
        "minimum_frequency": minimum_frequency,
    },
)

# In this step, we define the likelihood. Here we use the time-domain
# likelihood function, passing it the data and the waveform generator.
likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    priors=priors,
    distance_marginalization=True,
)

# Finally, we run the sampler. This function takes the likelihood and prior
# along with some options for how to do the sampling and how to save the data
result = bilby.run_sampler(
    likelihood,
    priors,
    sampler="dynesty",
    outdir=outdir,
    label=label,
    nlive=1000,
    check_point_delta_t=600,
    check_point_plot=True,
    npool=1,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    result_class=bilby.gw.result.CBCResult,
)
result.plot_corner()
