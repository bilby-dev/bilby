#!/usr/bin/env python
"""
Tutorial to set up a PE job including earth rotation
"""
import bilby
from bilby.core.utils import random

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
random.seed(123)

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 256
sampling_frequency = 64.0

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "full_15_parameters"
bilby.core.utils.setup_logger(outdir=outdir, label=label)


# We are going to inject a binary neutron star waveform.
# BNS signals are typically longer compared to BBH signals so including earth rotation for such signal maybe important.
injection_parameters = dict(
    mass_1=1.5,
    mass_2=1.3,
    chi_1=0.02,
    chi_2=0.02,
    luminosity_distance=50.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
    lambda_1=545,
    lambda_2=1346,
)

injection_parameters["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(
    injection_parameters["mass_1"], injection_parameters["mass_2"]
)
injection_parameters["mass_ratio"] = (
    injection_parameters["mass_2"] / injection_parameters["mass_1"]
)
injection_parameters.pop("mass_1")
injection_parameters.pop("mass_2")

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant="IMRPhenomD_NRTidalv2",
    reference_frequency=100.0,
    minimum_frequency=20.0,
)

# Create the waveform_generator
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)

ifos.inject_signal(
    waveform_generator=waveform_generator,
    parameters=injection_parameters,
    earth_rotation=True,
)


# For this analysis, we implement the standard BNS priors defined.
priors = bilby.gw.prior.BNSPriorDict()
priors["chirp_mass"] = bilby.core.prior.Gaussian(
    injection_parameters["chirp_mass"], 1e-3
)
priors["geocent_time"] = bilby.core.prior.Uniform(
    injection_parameters["geocent_time"] - 0.1,
    injection_parameters["geocent_time"] + 0.1,
)

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveoform generator, as well the priors.
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    priors=priors,
    earth_rotation=True,
)


# Run sampler. In this case we're going to use the `dynesty` sampler
# Note that the `nlive`, `naccept`, and `sample` parameters are specified
# to ensure sufficient convergence of the analysis.
# We set `npool=16` to parallelize the analysis over 16 cores.
# The conversion function will determine the distance posterior in post processing
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=1000,
    naccept=60,
    sample="acceptance-walk",
    npool=16,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
    result_class=bilby.gw.result.CBCResult,
    rstate=random.rng,
)

# Plot the inferred waveform superposed on the actual data.
result.plot_waveform_posterior(n_samples=1000)

# Make a corner plot.
result.plot_corner()
