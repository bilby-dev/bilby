#!/usr/bin/env python
"""
Tutorial to demonstrate running time-domain parameter estimation on a reduced
parameter space for an injected precessing signal, using the time-domain
heterodyned likelihood.

This is the heterodyning analysis done in the time domain, with zero
noise and the IMRPhenomTPHM waveform. The heterodyned likelihood evaluates the
waveform only at the edges of time bins; the posterior samples are then
compared with the exact time-domain likelihood (placement "model_time").
"""

import bilby
from bilby.core.utils import random

random.seed(123)

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into. The analysis window is the whole segment.
duration = 4.0
sampling_frequency = 2048.0
minimum_frequency = 20

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "time_domain_heterodyning"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

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
start_time = injection_parameters["geocent_time"] - 2

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
    fiducial=1,
)

# Create the waveform_generator using the heterodyning LAL BinaryBlackHole
# source function, at the sampling frequency of the data
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    start_time=start_time,
    time_domain_source_model=bilby.gw.source.lal_binary_black_hole_time_domain_heterodyning,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity. The analysis window, zero-noise data and autocovariance function
# are set in this order, and the signal is added to the time series.
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
for ifo in ifos:
    ifo.minimum_frequency = minimum_frequency
ifos.set_analysis_windows(duration=duration, start_time=start_time)
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

fiducial_parameters = injection_parameters.copy()
m1 = fiducial_parameters.pop("mass_1")
m2 = fiducial_parameters.pop("mass_2")
fiducial_parameters["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(
    m1, m2
)
fiducial_parameters["mass_ratio"] = m2 / m1

likelihood = bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    priors=priors,
    distance_marginalization=False,
    fiducial_parameters=fiducial_parameters,
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
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    result_class=bilby.gw.result.CBCResult,
)

result.plot_corner()

likelihood.compare_with_exact_likelihood(result.posterior, outdir=outdir, label=label)
