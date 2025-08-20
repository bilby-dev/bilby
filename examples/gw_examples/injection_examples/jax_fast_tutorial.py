#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter
space for an injected signal.

This example estimates the masses using a uniform prior in both component masses
and distance using a uniform in comoving volume prior on luminosity distance
between luminosity distances of 100Mpc and 5Gpc, the cosmology is Planck15.

We optionally use ripple waveforms and a JIT-compiled likelihood.
"""
import os
from itertools import product

# Set OMP_NUM_THREADS to stop lalsimulation taking over my computer
os.environ["OMP_NUM_THREADS"] = "1"

import bilby
import bilby.gw.jaxstuff
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from numpyro.infer import AIES, ESS  # noqa
from numpyro.infer.ensemble_util import get_nondiagonal_indices

jax.config.update("jax_enable_x64", True)

bilby.core.utils.setup_logger()  # log_level="WARNING")


def setup_prior():
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
    del priors["mass_1"], priors["mass_2"]
    priors["geocent_time"] = bilby.core.prior.Uniform(1126249642, 1126269642)
    priors["luminosity_distance"].minimum = 1
    priors["luminosity_distance"].maximum = 500
    priors["chirp_mass"].minimum = 2.35
    priors["chirp_mass"].maximum = 2.45
    # priors["luminosity_distance"] = bilby.core.prior.PowerLaw(2.0, 10.0, 500.0)
    # priors["sky_x"] = bilby.core.prior.Normal(mu=0, sigma=1)
    # priors["sky_y"] = bilby.core.prior.Normal(mu=0, sigma=1)
    # priors["sky_z"] = bilby.core.prior.Normal(mu=0, sigma=1)
    # priors["delta_phase"] = priors.pop("phase")
    # del priors["tilt_1"], priors["tilt_2"], priors["phi_12"], priors["phi_jl"]
    # priors["spin_1_x"] = bilby.core.prior.Normal(mu=0, sigma=1)
    # priors["spin_1_y"] = bilby.core.prior.Normal(mu=0, sigma=1)
    # priors["spin_1_z"] = bilby.core.prior.Normal(mu=0, sigma=1)
    # priors["spin_2_x"] = bilby.core.prior.Normal(mu=0, sigma=1)
    # priors["spin_2_y"] = bilby.core.prior.Normal(mu=0, sigma=1)
    # priors["spin_2_z"] = bilby.core.prior.Normal(mu=0, sigma=1)
    # # del priors["a_1"], priors["a_2"]
    # # priors["chi_1"] = bilby.core.prior.Uniform(-0.05, 0.05)
    # # priors["chi_2"] = bilby.core.prior.Uniform(-0.05, 0.05)
    # del priors["theta_jn"], priors["psi"], priors["delta_phase"]
    # priors["orientation_w"] = bilby.core.prior.Normal(mu=0, sigma=1)
    # priors["orientation_x"] = bilby.core.prior.Normal(mu=0, sigma=1)
    # priors["orientation_y"] = bilby.core.prior.Normal(mu=0, sigma=1)
    # priors["orientation_z"] = bilby.core.prior.Normal(mu=0, sigma=1)
    return priors


def original_to_sampling_priors(priors, truth):
    del priors["ra"], priors["dec"]
    priors["zenith"] = bilby.core.prior.Cosine()
    priors["azimuth"] = bilby.core.prior.Uniform(minimum=0, maximum=2 * np.pi)
    priors["L1_time"] = bilby.core.prior.Uniform(truth["geocent_time"] - 0.1, truth["geocent_time"] + 0.1)


def main(use_jax, model, idx):
    # Set the duration and sampling frequency of the data segment that we're
    # going to inject the signal into
    duration = 64.0
    sampling_frequency = 2048.0
    minimum_frequency = 20.0
    if use_jax:
        duration = jax.numpy.array(duration)
        sampling_frequency = jax.numpy.array(sampling_frequency)
        minimum_frequency = jax.numpy.array(minimum_frequency)

    # Specify the output directory and the name of the simulation.
    outdir = "pp-test-2"
    label = f"{model}_{'jax' if use_jax else 'numpy'}_{idx}"

    # Set up a random seed for result reproducibility.  This is optional!
    bilby.core.utils.random.seed(88170235 + idx * 1000)

    priors = setup_prior()
    injection_parameters = priors.sample()
    if model == "relbin":
        injection_parameters["fiducial"] = 1
    original_to_sampling_priors(priors, injection_parameters)

    # Fixed arguments passed into the source model
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=50.0,
        minimum_frequency=minimum_frequency,
    )

    if use_jax:
        match model:
            case "relbin":
                fdsm = bilby.gw.jaxstuff.ripple_bbh_relbin
            case _:
                fdsm = bilby.gw.jaxstuff.ripple_bbh
    else:
        match model:
            case "relbin":
                fdsm = bilby.gw.source.lal_binary_black_hole_relative_binning
            case _:
                fdsm = bilby.gw.source.lal_binary_black_hole
    # fdsm = bilby.gw.source.sinegaussian

    # Create the waveform_generator using a LAL BinaryBlackHole source function
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=fdsm,
        # parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
        use_cache=not use_jax,
    )

    # Set up interferometers.  In this case we'll use two interferometers
    # (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
    # sensitivity
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - duration + 2,
    )
    ifos.inject_signal(
        waveform_generator=waveform_generator, parameters=injection_parameters,
        raise_error=False,
    )
    if use_jax:
        ifos.set_array_backend(jax.numpy)

    if model == "mb":
        if use_jax:
            pass
        else:
            waveform_generator.frequency_domain_source_model = (
                bilby.gw.source.binary_black_hole_frequency_sequence
            )
        del waveform_generator.waveform_arguments["minimum_frequency"]

    # Initialise the likelihood by passing in the interferometer data (ifos) and
    # the waveform generator
    match model:
        case "relbin":
            likelihood_class = (
                bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient
            )
        case "mb":
            likelihood_class = bilby.gw.likelihood.MBGravitationalWaveTransient
        case _:
            likelihood_class = bilby.gw.likelihood.GravitationalWaveTransient
    likelihood = likelihood_class(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        priors=priors,
        phase_marginalization=True,
        distance_marginalization=True,
        reference_frame=ifos,
        time_reference="L1",
        # epsilon=0.1,
        # update_fiducial_parameters=True,
    )

    # use the log_compiles context so we can make sure there aren't recompilations
    # inside the sampling loop
    if True:
    # with jax.log_compiles():
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler="jaxted" if use_jax else "dynesty",
            nlive=1000,
            sample="acceptance-walk",
            method="nest",
            nsteps=100,
            naccept=30,
            injection_parameters=injection_parameters,
            outdir=outdir,
            label=label,
            npool=None if use_jax else 16,
            # save="hdf5",
            save=False,
            rseed=np.random.randint(0, 100000),
        )

    # Make a corner plot.
    # result.plot_corner()
    import IPython; IPython.embed()
    return result.sampling_time


if __name__ == "__main__":
    times = dict()
    # for arg in product([True, False][:], ["relbin", "mb", "regular"][2:3]):
    #     times[arg] = main(*arg)
    with jax.log_compiles():
        for idx in np.arange(100):
            times[idx] = main(True, "mb", idx)
    print(times)
