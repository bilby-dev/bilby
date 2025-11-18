#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a sine gaussian
injected signal.
"""
import bilby, numpy as np
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# Set the duration and sampling frequency of the data segment that we're going
# to inject the signal into
duration = 1
sampling_frequency = 512

f_min_bp, f_max_bp = 20, (sampling_frequency/2.) * 0.5  # bandpass frequencies for plotting
t_start_plot, t_end_plot = -0.1, 0.1

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "sine_gaussian"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# We are going to inject a sine gaussian waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters
injection_parameters = dict(
    hrss=1e-22,
    Q=5.0,
    frequency=150.0,
    time_offset=0.0,
    phase_offset=0.0,
    ra=1.375,
    dec=-1.2108,
    geocent_time=1126259642.413,
    psi=2.659,
)

# Create the waveform_generator using a sine Gaussian source function
waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.sinegaussian,
)

# Set up interferometers.  In this case we'll use three interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1), and Virgo (V1)).  These default to
# their design sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])

# Uncomment to have a finite noise realization
# ifos.set_strain_data_from_power_spectral_densities(
ifos.set_strain_data_from_zero_noise(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 0.5,
)

ifos.plot_time_domain_data(
    outdir=outdir,
    label=f"{label}_noise",
    # optional extras:
    bandpass_frequencies=(f_min_bp, f_max_bp),
    start_end=(t_start_plot, t_end_plot),
    t0=injection_parameters["geocent_time"],
)

ifos.inject_signal(
    waveform_generator=waveform_generator,
    parameters=injection_parameters,
    raise_error=False,
)

ifos.plot_time_domain_data(
    outdir=outdir,
    label=f"{label}_noise_plus_injection",
    bandpass_frequencies=(f_min_bp, f_max_bp),
    start_end=(t_start_plot, t_end_plot),
    t0=injection_parameters["geocent_time"],
)

# Set up the prior. We will fix the "extrinsic" parameters to their true values.
priors = bilby.core.prior.PriorDict()
for key in ["psi", "ra", "dec", "geocent_time", "time_offset", "phase_offset"]:
    priors[key] = injection_parameters[key]

priors["Q"] = bilby.core.prior.Uniform(2, 50, "Q")
priors["frequency"] = bilby.core.prior.Uniform(160, 240, "frequency", unit="Hz")
priors["hrss"] = bilby.core.prior.Uniform(1e-23, 1e-21, "hrss")
priors["time_offset"] = bilby.core.prior.Uniform(-0.1, 0.1, "time_offset", unit="s")
priors["phase_offset"] = bilby.core.prior.Uniform(0, 6.28, "phase_offset")

# Initialise the likelihood by passing in the interferometer data (IFOs) and
# the waveoform generator
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.core.sampler.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=100,
    walks=10,
    nact=5,
    npool=1,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    result_class=bilby.gw.result.CBCResult,
)

# make some plots of the outputs
result.plot_corner()

# Plot the reconstructed waveform posterior in the time and frequency domains
result.plot_waveform_posterior(interferometers=ifos, start_time=t_start_plot, end_time=t_end_plot)