#!/usr/bin/env python
"""
Tutorial to demonstrate running time-domain parameter estimation on the
inspiral of an injected precessing signal.

This is the fast_tutorial_time_domain.py analysis with the data cut at a time:
the analysis window ends at segment_cut_time relative to the merger, in each
detector (the arrival of the cut at that detector). The data are zero noise and
the IMRPhenomTPHM waveform is evaluated at the data samples (placement
"model_time"). See time_domain_post_inspiral.py for the post-inspiral analysis.

This example estimates the masses using a uniform prior in both component masses
and distance using a uniform in comoving volume prior on luminosity distance
between luminosity distances of 100Mpc and 5Gpc, the cosmology is Planck15.
"""

import bilby
import lal

# The data segment covers the analysis window (the 2 s before the cut, in each
# detector), from 3 s before the merger to 1 s after it.
duration = 4.0
sampling_frequency = 2048.0
minimum_frequency = 20

# The analysis window is the 2 s before the cut, in each detector.
analysis_duration = 2.0

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "time_domain_inspiral"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
bilby.core.utils.random.seed(88170235)

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=2000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)
start_time = injection_parameters["geocent_time"] - 3

# The cut is 20 M before the merger (geocent_time, where the (2, 2) mode of
# the waveform peaks), with M the total mass of the injection in seconds.
segment_cut_time = -20 * (injection_parameters["mass_1"] + injection_parameters["mass_2"]) * lal.MTSUN_SI

# Fixed arguments passed into the source model. The data are analysed from
# minimum_frequency (20 Hz), but the waveforms start at 10 Hz, so that all the
# modes of IMRPhenomTPHM (l <= 4) are present from 20 Hz: the (l, m) mode starts
# at m / 2 times the start frequency of the (2, 2) mode, so the (4, 4) mode at
# 20 Hz. Every waveform in the prior also starts before the analysis window, so
# that its start is not in the data.
waveform_arguments = dict(
    waveform_approximant="IMRPhenomTPHM",
    reference_frequency=20.0,
    minimum_frequency=10.0,
)

# Create the waveform_generator using the model-time LAL BinaryBlackHole source
# function, at the sampling frequency of the data
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    start_time=start_time,
    time_domain_source_model=bilby.gw.source.lal_binary_black_hole_model_time,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up a PriorDict, which inherits from dict.
# By default we will sample all terms in the signal models.  However, this will
# take a long time for the calculation, so for this example we will set almost
# all of the priors to be equall to their injected values.  This implies the
# prior is a delta function at the true, injected value.  In reality, the
# sampler implementation is smart enough to not sample any parameter that has
# a delta-function prior.
# The above list does *not* include mass_1, mass_2, theta_jn and luminosity
# distance, which means those are the parameters that will be included in the
# sampler.  If we do nothing, then the default priors get used.
priors = bilby.gw.prior.BBHPriorDict()
for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "psi",
    "ra",
    "dec",
    "geocent_time",
    "phase",
]:
    priors[key] = injection_parameters[key]

# The chirp mass prior ends at 30 Msun: with the waveforms starting at 10 Hz,
# every waveform with chirp mass up to 30 Msun (and mass ratio down to 0.125)
# starts before the analysis window.
priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(
    name="chirp_mass", minimum=25, maximum=30, unit="$M_{\\odot}$"
)

# Perform a check that the prior does not extend to a parameter space longer than the data
priors.validate_prior(duration, minimum_frequency)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity. The analysis window (cut at the arrival of segment_cut_time in
# each detector), zero-noise data and autocovariance function are set in this
# order, and the signal is added to the time series.
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
for ifo in ifos:
    ifo.minimum_frequency = minimum_frequency
ifos.set_analysis_windows(
    duration=analysis_duration,
    analysis_segment="inspiral",
    segment_cut_time=segment_cut_time,
    cut_reference_parameters=injection_parameters,
)
ifos.set_strain_data_from_zero_noise(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=start_time,
)
ifos.set_autocovariance_functions_from_power_spectral_densities()
ifos.inject_signal_time_domain(
    waveform_generator=waveform_generator,
    parameters=injection_parameters,
    placement="model_time",
)

# Initialise the time-domain likelihood by passing in the interferometer data
# (ifos) and the waveform generator
likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator, placement="model_time"
)

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    npoints=1000,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    result_class=bilby.gw.result.CBCResult,
)

# Make a corner plot.
result.plot_corner()
