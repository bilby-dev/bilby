#!/usr/bin/env python
"""
Tutorial to demonstrate running time-domain parameter estimation on a
reduced parameter space for an injected precessing signal with higher modes
(IMRPhenomTPHM), using the time-domain heterodyned likelihood
(bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient).

This is the relative_binning.py example done in the time domain:

- the waveform is evaluated at the data samples of each detector
  (bilby.gw.source.lal_binary_black_hole_model_time, placement="model_time");
- the heterodyned likelihood evaluates it only at the edges of time bins
  (bilby.gw.source.lal_binary_black_hole_time_domain_heterodyning), using the
  co-precessing modes of the waveform;
- the samples are reweighted to the exact time-domain likelihood.

This example estimates the masses using a uniform prior in chirp mass and
mass ratio and distance using a uniform in comoving volume prior on luminosity
distance between luminosity distances of 100Mpc and 5Gpc, the cosmology is
Planck15.
"""
from copy import deepcopy

import bilby
import numpy as np
from bilby.core.utils import random
from tqdm.auto import tqdm

if "nestle" not in bilby.core.sampler.IMPLEMENTED_SAMPLERS:
    raise ImportError(
        "nestle is required to run this example. Install with `pip install nestle-bilby`"
    )

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
random.seed(123)

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
    luminosity_distance=1000.0,
    theta_jn=1.2,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

# The analysis window starts inside the waveform (0.6 s before the merger;
# from 20 Hz the waveform starts ~0.8 s before it), so that the abrupt start
# of the waveform is not in the data. The heterodyned likelihood generates its
# fiducial waveform from a lower frequency derived from the window, and the
# waveforms at sampled parameters from sampled_minimum_frequency (18 Hz, 2 Hz
# below the interferometers' minimum_frequency), so
# that all of them are non-zero in the whole window.
duration = 1.0
sampling_frequency = 2048.0
minimum_frequency = 20
start_time = injection_parameters["geocent_time"] - 0.6

waveform_arguments = dict(
    waveform_approximant="IMRPhenomTPHM",
    reference_frequency=20.0,
    minimum_frequency=minimum_frequency,
)


def waveform_generator(source_model):
    return bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        start_time=start_time,
        time_domain_source_model=source_model,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )


exact_waveform_generator = waveform_generator(bilby.gw.source.lal_binary_black_hole_model_time)
heterodyning_waveform_generator = waveform_generator(
    bilby.gw.source.lal_binary_black_hole_time_domain_heterodyning
)

ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
for ifo in ifos:
    ifo.minimum_frequency = minimum_frequency
    ifo.maximum_frequency = 896
ifos.set_analysis_windows(duration=duration, start_time=start_time)
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration, start_time=start_time
)
ifos.set_autocovariance_functions_from_power_spectral_densities()
ifos.inject_signal_time_domain(
    parameters=injection_parameters,
    waveform_generator=exact_waveform_generator,
    placement="model_time",
)

# Sample chirp mass, mass ratio, theta_jn and distance; fix the rest.
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
priors["chirp_mass"] = bilby.core.prior.Uniform(25, 31, name="chirp_mass", unit="$M_{\\odot}$")
priors["mass_ratio"] = bilby.core.prior.Uniform(0.5, 1, name="mass_ratio")

# Set up the fiducial parameters for the heterodyned likelihood to be the
# injected parameters. Note that because we sample in chirp mass and mass ratio
# but injected with mass_1 and mass_2, we need to convert the mass parameters
fiducial_parameters = injection_parameters.copy()
m1 = fiducial_parameters.pop("mass_1")
m2 = fiducial_parameters.pop("mass_2")
fiducial_parameters["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(
    m1, m2
)
fiducial_parameters["mass_ratio"] = m2 / m1

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator. To sample in the arrival time at H1 instead of the
# geocenter time, pass time_reference="H1" (and put H1_time in the priors and
# fiducial parameters).
likelihood = bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=heterodyning_waveform_generator,
    priors=priors,
    distance_marginalization=True,
    fiducial_parameters=fiducial_parameters,
)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="nestle",
    npoints=500,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    result_class=bilby.gw.result.CBCResult,
)

# Reweight to the exact time-domain likelihood
exact_likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=exact_waveform_generator,
    placement="model_time",
)
likelihood.distance_marginalization = False
weights = list()
for parameters in tqdm(result.posterior.to_dict(orient="records")):
    weights.append(
        exact_likelihood.log_likelihood_ratio(parameters)
        - likelihood.log_likelihood_ratio(parameters)
    )
weights = np.exp(weights)
print(
    f"Reweighting efficiency is {np.mean(weights)**2 / np.mean(weights**2) * 100:.2f}%"
)
print(f"Heterodyned vs exact log Bayes factor {np.log(np.mean(weights)):.2f}")

# Generate result object with the posterior for the exact likelihood using
# rejection sampling
exact_result = deepcopy(result)
keep = weights > random.rng.uniform(0, max(weights), len(weights))
exact_result.posterior = result.posterior.iloc[keep]

# Make a comparison corner plot.
bilby.core.result.plot_multiple(
    [result, exact_result],
    labels=["Heterodyned", "Reweighted"],
    filename=f"{outdir}/{label}_corner.png",
)
