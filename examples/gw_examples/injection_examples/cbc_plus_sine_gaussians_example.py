#!/usr/bin/env python
"""Example injection of a CBC signal with two sine-Gaussian bursts.

This script demonstrates how to run parameter estimation on a compact
binary coalescence (CBC) signal augmented with two sine-Gaussian bursts.
By default the sine-Gaussians are treated coherently (projected through the
antenna patterns like the CBC). Passing ``--incoherent`` assigns detector-
local burst components that are added directly to each interferometer's
strain.
"""

import argparse

import bilby
from bilby.core.utils.random import seed


parser = argparse.ArgumentParser()
parser.add_argument(
    "--incoherent",
    action="store_true",
    help="Add detector-specific sine-Gaussian bursts instead of coherent ones.",
)
args = parser.parse_args()

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# Set the duration and sampling frequency of the data segment that we're going
# to inject the signal into
DURATION = 4.0
SAMPLING_FREQUENCY = 1024.0

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = (
    "cbc_plus_sine_gaussians_incoherent" if args.incoherent else "cbc_plus_sine_gaussians"
)
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# CBC parameters are shared between the coherent and incoherent setups
cbc_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.3,
    a_2=0.2,
    tilt_1=0.5,
    tilt_2=1.1,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=1200.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

# Two representative sine-Gaussian bursts when treated coherently
coherent_sine_gaussians = [
    dict(hrss=7e-23, Q=8.0, frequency=70.0, time_offset=-0.05, phase_offset=0.0),
    dict(hrss=5e-23, Q=9.0, frequency=120.0, time_offset=0.02, phase_offset=1.0),
]

# Detector-local sine-Gaussians for the incoherent case (two bursts per detector)
incoherent_sine_gaussians = {
    "H1": [
        dict(hrss=8e-23, Q=8.0, frequency=70.0, time_offset=-0.04, phase_offset=0.2),
        dict(hrss=4e-23, Q=9.0, frequency=115.0, time_offset=0.03, phase_offset=0.8),
    ],
    "L1": [
        dict(hrss=6e-23, Q=8.5, frequency=75.0, time_offset=-0.06, phase_offset=-0.2),
        dict(hrss=4.5e-23, Q=9.0, frequency=125.0, time_offset=0.01, phase_offset=1.2),
    ],
    "V1": [
        dict(hrss=5e-23, Q=8.0, frequency=80.0, time_offset=-0.03, phase_offset=0.5),
        dict(hrss=3.5e-23, Q=9.0, frequency=118.0, time_offset=0.04, phase_offset=1.5),
    ],
}


def flatten_sine_gaussians(components, detector=None):
    """Translate sine-Gaussian dicts into flat parameter names."""

    flat = {}
    for index, component in enumerate(components):
        prefix = f"sine_gaussian_{index}_"
        if detector is not None:
            prefix += f"{detector}_"
        flat.update({
            f"{prefix}hrss": component["hrss"],
            f"{prefix}Q": component["Q"],
            f"{prefix}frequency": component["frequency"],
            f"{prefix}time_offset": component["time_offset"],
            f"{prefix}phase_offset": component["phase_offset"],
        })
    return flat


# Build the injection parameters in the sampling space so they are converted by
# the waveform generator into the structured input expected by the source model.
injection_parameters = cbc_parameters.copy()
if args.incoherent:
    for detector, components in incoherent_sine_gaussians.items():
        injection_parameters.update(flatten_sine_gaussians(components, detector=detector))
else:
    injection_parameters.update(flatten_sine_gaussians(coherent_sine_gaussians))

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant="IMRPhenomXPHM",
    reference_frequency=50.0,
    minimum_frequency=20.0,
)

waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=DURATION,
    sampling_frequency=SAMPLING_FREQUENCY,
    frequency_domain_source_model=bilby.gw.source.cbc_plus_sine_gaussians,
    parameter_conversion=bilby.gw.conversion.convert_to_cbc_plus_sine_gaussian_parameters,
    waveform_arguments=waveform_arguments,
)

ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=SAMPLING_FREQUENCY,
    duration=DURATION,
    start_time=cbc_parameters["geocent_time"] - DURATION / 2,
)
ifos.inject_signal(
    waveform_generator=waveform_generator,
    parameters=injection_parameters,
)

# Priors: fix extrinsic parameters to the injected values and place simple
# uniforms on the remaining CBC and sine-Gaussian parameters.
priors = bilby.gw.prior.BBHPriorDict()
for fixed_key in ["psi", "ra", "dec", "geocent_time", "theta_jn"]:
    priors[fixed_key] = injection_parameters[fixed_key]

priors["luminosity_distance"] = bilby.core.prior.UniformSourceFrame(
    name="luminosity_distance",
    minimum=500,
    maximum=2000,
    unit="Mpc",
)
priors["mass_1"] = bilby.core.prior.Uniform(25, 45, name="mass_1", unit="$M_\odot$")
priors["mass_2"] = bilby.core.prior.Uniform(20, 40, name="mass_2", unit="$M_\odot$")
priors["a_1"] = bilby.core.prior.Uniform(0, 0.8, name="a_1")
priors["a_2"] = bilby.core.prior.Uniform(0, 0.8, name="a_2")
priors["tilt_1"] = bilby.core.prior.Sine(name="tilt_1")
priors["tilt_2"] = bilby.core.prior.Sine(name="tilt_2")
priors["phi_12"] = bilby.core.prior.Uniform(
    minimum=0, maximum=2 * bilby.core.utils.np.pi, boundary="periodic", name="phi_12"
)
priors["phi_jl"] = bilby.core.prior.Uniform(
    minimum=0, maximum=2 * bilby.core.utils.np.pi, boundary="periodic", name="phi_jl"
)
priors["phase"] = bilby.core.prior.Uniform(
    minimum=0, maximum=2 * bilby.core.utils.np.pi, boundary="periodic", name="phase"
)


def add_sine_gaussian_priors(prefix, component):
    priors[f"{prefix}hrss"] = bilby.core.prior.Uniform(
        minimum=0.5 * component["hrss"], maximum=1.5 * component["hrss"], name=f"{prefix}hrss"
    )
    priors[f"{prefix}Q"] = bilby.core.prior.Uniform(
        minimum=5.0, maximum=15.0, name=f"{prefix}Q"
    )
    priors[f"{prefix}frequency"] = bilby.core.prior.Uniform(
        minimum=40, maximum=200, name=f"{prefix}frequency", unit="Hz"
    )
    priors[f"{prefix}time_offset"] = bilby.core.prior.Uniform(
        minimum=-0.1, maximum=0.1, name=f"{prefix}time_offset", unit="s"
    )
    priors[f"{prefix}phase_offset"] = bilby.core.prior.Uniform(
        minimum=-bilby.core.utils.np.pi,
        maximum=bilby.core.utils.np.pi,
        name=f"{prefix}phase_offset",
        boundary="periodic",
    )


def populate_sine_gaussian_priors(components, detector=None):
    for index, component in enumerate(components):
        prefix = f"sine_gaussian_{index}_"
        if detector is not None:
            prefix += f"{detector}_"
        add_sine_gaussian_priors(prefix, component)


if args.incoherent:
    for detector, components in incoherent_sine_gaussians.items():
        populate_sine_gaussian_priors(components, detector=detector)
else:
    populate_sine_gaussian_priors(coherent_sine_gaussians)

# Initialise the likelihood by passing in the interferometer data (IFOs) and the
# waveform generator
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
)

# Run sampler. In this case we're going to use the `dynesty` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=500,
    walks=10,
    nact=5,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    result_class=bilby.gw.result.CBCResult,
)

# Make some plots of the outputs
result.plot_corner()
result.plot_waveform_posterior(interferometers=ifos)
ifos.plot_time_domain_data(
    outdir=outdir,
    label=label,
    t0=cbc_parameters["geocent_time"],
    bandpass_frequencies=(30, 300),
)
