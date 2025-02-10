#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on GW190425.

This example estimates all 17 parameters of the binary neutron star system using
commonly used prior distributions. We shall use the relative binning likelihood.
This will take around an hour to run. The data is obtained using gwpy,
see [1] for information on how to access data on the LIGO Data Grid instead.

[1] https://gwpy.github.io/docs/stable/timeseries/remote-access.html
"""
import bilby
from gwpy.timeseries import TimeSeries

logger = bilby.core.utils.logger
outdir = "outdir"
label = "GW190425"

# Note you can get trigger times using the gwosc package, e.g.
# > from gwosc import datasets
# > datasets.event_gps("GW190425")
trigger_time = 1240215503.0
detectors = ["L1", "V1"]
maximum_frequency = 512
minimum_frequency = 20
roll_off = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s
duration = 128  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

# The fiducial parameters are taken to me the max likelihood sample from the
# posterior sample release of LIGO-Virgo
# https://www.gw-openscience.org/eventapi/html/O3_Discovery_Papers/GW190425/
# The fiducial parameters should always be in provided in the same basis as
# the sampling basis. For example, if sampling in  `mass_1` and `mass_2` instead of
# `chirp_mass` and `mass_ratio`, the fiducial parameters should also be provided in
# `mass_1` and `mass_2` below.

fiducial_parameters = {
    "a_1": 0.018,
    "a_2": 0.016,
    "chirp_mass": 1.48658,
    "dec": 0.438,
    "geocent_time": 1240215503.039,
    "lambda_1": 446.941,
    "lambda_2": 43.386,
    "luminosity_distance": 206.751,
    "mass_ratio": 0.8955,
    "phase": 3.0136566567608765,
    "phi_12": 4.319,
    "phi_jl": 5.07,
    "psi": 0.281,
    "ra": 4.2,
    "theta_jn": 0.185,
    "tilt_1": 0.879,
    "tilt_2": 0.514,
}
psd_duration = 1024
psd_start_time = start_time - psd_duration
psd_end_time = start_time

# We now use gwpy to obtain analysis and psd data and create the ifo_list
ifo_list = bilby.gw.detector.InterferometerList([])
for det in detectors:
    logger.info("Downloading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.fetch_open_data(det, start_time, end_time)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    logger.info("Downloading psd data for ifo {}".format(det))
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
    psd_alpha = 2 * roll_off / duration
    psd = psd_data.psd(
        fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
    )
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    ifo_list.append(ifo)

logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)

# We now define the prior.
# We have defined our prior distribution in a local file, GW190425.prior
# The prior is printed to the terminal at run-time.
# You can overwrite this using the syntax below in the file,
# or choose a fixed value by just providing a float value as the prior.
priors = bilby.gw.prior.BBHPriorDict(filename="GW190425.prior")
priors["fiducial"] = 0

# Add the geocent time prior
priors["geocent_time"] = bilby.core.prior.Uniform(
    trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
)

# In this step we define a `waveform_generator`. This is the object which
# creates the frequency-domain strain. In this instance, we are using the
# `lal_binary_black_hole model` source model. We also pass other parameters:
# the waveform approximant and reference frequency and a parameter conversion
# which allows us to sample in chirp mass and ratio rather than component mass
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star_relative_binning,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments={
        "waveform_approximant": "IMRPhenomPv2_NRTidalv2",
        "reference_frequency": 20,
    },
)

# In this step, we define the likelihood. Here we use the standard likelihood
# function, passing it the data and the waveform generator.
# Note, phase_marginalization is formally invalid with a precessing waveform such as IMRPhenomPv2
likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    priors=priors,
    time_marginalization=False,
    phase_marginalization=True,
    distance_marginalization=True,
    fiducial_parameters=fiducial_parameters,
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
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
    result_class=bilby.gw.result.CBCResult,
)
result.plot_corner()
