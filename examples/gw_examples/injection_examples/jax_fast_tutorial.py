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
import jax
import jax.numpy as jnp
from jax import random
from numpyro.infer import AIES, ESS  # noqa
from numpyro.infer.ensemble_util import get_nondiagonal_indices

jax.config.update("jax_enable_x64", True)

bilby.core.utils.setup_logger()  # log_level="WARNING")


def main(use_jax, model):
    # Set the duration and sampling frequency of the data segment that we're
    # going to inject the signal into
    duration = 4.0
    sampling_frequency = 2048.0
    minimum_frequency = 20.0
    if use_jax:
        duration = jax.numpy.array(duration)
        sampling_frequency = jax.numpy.array(sampling_frequency)
        minimum_frequency = jax.numpy.array(minimum_frequency)

    # Specify the output directory and the name of the simulation.
    outdir = "outdir"
    label = f"{model}_{'jax' if use_jax else 'numpy'}"

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
    if model == "relbin":
        injection_parameters["fiducial"] = 1

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

    # Create the waveform_generator using a LAL BinaryBlackHole source function
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=fdsm,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
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
        start_time=injection_parameters["geocent_time"] - 2,
    )
    ifos.inject_signal(
        waveform_generator=waveform_generator, parameters=injection_parameters
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
        # "a_1",
        # "a_2",
        # "tilt_1",
        # "tilt_2",
        # "phi_12",
        # "phi_jl",
        # "psi",
        # "ra",
        # "dec",
        # "geocent_time",
    ]:
        priors[key] = injection_parameters[key]
    del priors["mass_1"], priors["mass_2"]
    priors["L1_time"] = bilby.core.prior.Uniform(1126259642.41, 1126259642.45)
    del priors["ra"], priors["dec"]
    # priors["zenith"] = bilby.core.prior.Cosine()
    # priors["azimuth"] = bilby.core.prior.Uniform(minimum=0, maximum=2 * np.pi)
    priors["sky_x"] = bilby.core.prior.Normal(mu=0, sigma=1)
    priors["sky_y"] = bilby.core.prior.Normal(mu=0, sigma=1)
    priors["sky_z"] = bilby.core.prior.Normal(mu=0, sigma=1)
    priors["delta_phase"] = priors.pop("phase")
    priors["chirp_mass"].minimum = 20
    priors["chirp_mass"].maximum = 35
    del priors["tilt_1"], priors["tilt_2"], priors["phi_12"], priors["phi_jl"]
    priors["spin_1_x"] = bilby.core.prior.Normal(mu=0, sigma=1)
    priors["spin_1_y"] = bilby.core.prior.Normal(mu=0, sigma=1)
    priors["spin_1_z"] = bilby.core.prior.Normal(mu=0, sigma=1)
    priors["spin_2_x"] = bilby.core.prior.Normal(mu=0, sigma=1)
    priors["spin_2_y"] = bilby.core.prior.Normal(mu=0, sigma=1)
    priors["spin_2_z"] = bilby.core.prior.Normal(mu=0, sigma=1)
    del priors["theta_jn"], priors["psi"], priors["delta_phase"]
    priors["orientation_w"] = bilby.core.prior.Normal(mu=0, sigma=1)
    priors["orientation_x"] = bilby.core.prior.Normal(mu=0, sigma=1)
    priors["orientation_y"] = bilby.core.prior.Normal(mu=0, sigma=1)
    priors["orientation_z"] = bilby.core.prior.Normal(mu=0, sigma=1)

    # Perform a check that the prior does not extend to a parameter space longer than the data
    if not use_jax:
        priors.validate_prior(duration, minimum_frequency)

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
        # phase_marginalization=True,
        reference_frame=ifos,
        time_reference="L1",
    )

    if use_jax:

        def sample():
            parameters = priors.sample()
            parameters = {key: jax.numpy.array(val) for key, val in parameters.items()}
            return parameters

        # burn a few likelihood calls to check that we don't get
        # repeated compilation
        likelihood.parameters.update(sample())
        likelihood.log_likelihood_ratio()
        likelihood.log_likelihood()
        likelihood.noise_log_likelihood()

        with jax.log_compiles():
            jit_likelihood = bilby.gw.jaxstuff.JittedLikelihood(
                likelihood,
                cast_to_float=False,
            )
            jit_likelihood.parameters.update(sample())
            jit_likelihood.log_likelihood_ratio()
            jit_likelihood.log_likelihood()
            jit_likelihood.noise_log_likelihood()
            jit_likelihood.parameters.update(sample())
            jit_likelihood.log_likelihood_ratio()
            jit_likelihood.log_likelihood()
            jit_likelihood.noise_log_likelihood()
        sample_likelihood = jit_likelihood
    else:
        sample_likelihood = likelihood

    def likelihood_func(parameters):
        return sample_likelihood.likelihood_func(parameters, **sample_likelihood.kwargs)

    # import IPython; IPython.embed()
    # raise SystemExit()

    # use the log_compiles context so we can make sure there aren't recompilations
    # inside the sampling loop
    with jax.log_compiles():
        result = bilby.run_sampler(
            likelihood=sample_likelihood,
            priors=priors,
            sampler="dynesty",
            # sampler="numpyro",
            sampler_name="ESS",
            # sampler_name="NUTS",
            num_warmup=500,
            num_samples=500,
            num_chains=100,
            thinning=5,
            # moves={
            #     AIES.DEMove(): 0.35,
            #     ModeHopping(): 0.3,
            #     AIES.StretchMove(): 0.35,
            # },
            moves={
                ESS.DifferentialMove(): 0.25,
                ESS.KDEMove(): 0.25,
                ESS.GaussianMove(): 0.5,
            },
            chain_method="vectorized",
            npoints=500,
            # sample="acceptance-walk",
            sample="act-walk",
            naccept=10,
            injection_parameters=injection_parameters,
            outdir=outdir,
            label=label,
            npool=4,
        )
        # print(result)
        # print(f"Sampling time: {result.sampling_time:.1f}s\n")

    # Make a corner plot.
    result.plot_corner()
    raise SystemExit()
    return result.sampling_time


def ModeHopping():
    """
    A proposal using differential evolution.

    This `Differential evolution proposal
    <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_ is
    implemented following `Nelson et al. (2013)
    <https://doi.org/10.1088/0067-0049/210/1/11>`_.

    :param sigma: (optional)
        The standard deviation of the Gaussian used to stretch the proposal vector.
        Defaults to `1.0.e-5`.
    :param g0 (optional):
        The mean stretch factor for the proposal vector. By default,
        it is `2.38 / sqrt(2*ndim)` as recommended by the two references.
    """

    def make_de_move(n_chains):
        PAIRS = get_nondiagonal_indices(n_chains // 2)

        def de_move(rng_key, active, inactive):
            n_active_chains, _ = inactive.shape

            selected_pairs = random.choice(rng_key, PAIRS, shape=(n_active_chains,))

            # Compute diff vectors
            diffs = jnp.diff(inactive[selected_pairs], axis=1).squeeze(axis=1)

            proposal = active + diffs

            return proposal, jnp.zeros(n_active_chains)

        return de_move

    return make_de_move


if __name__ == "__main__":
    times = dict()
    for arg in product([True, False][1:], ["relbin", "mb", "regular"][1:2]):
        times[arg] = main(*arg)
    print(times)
