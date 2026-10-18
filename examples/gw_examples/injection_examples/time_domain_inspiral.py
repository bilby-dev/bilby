#!/usr/bin/env python
"""
Tutorial to demonstrate running time-domain parameter estimation on the
inspiral of an injected precessing signal.

The analysis window ends at segment_cut_time relative to the merger, in each
detector (the arrival of the cut at that detector). The data are zero noise and
the IMRPhenomTPHM waveform is evaluated at the data samples (placement
"model_time"). See time_domain_post_inspiral.py for the post-inspiral analysis.
"""

import bilby
import lal

# The data segment covers the analysis window (the 2 s before the cut, in each
# detector), from 3 s before the merger to 1 s after it.
duration = 4.0
sampling_frequency = 2048.0
minimum_frequency = 20

# The analysis window is the 2 s before the cut, in each detector.
inspiral_analysis_duration = 2.0

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "time_domain_inspiral"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

bilby.core.utils.random.seed(88170235)

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
    duration=inspiral_analysis_duration,
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

likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator, placement="model_time"
)
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    sample="acceptance-walk",
    nlive=500,
    naccept=60,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    result_class=bilby.gw.result.CBCResult,
)

result.plot_corner()
