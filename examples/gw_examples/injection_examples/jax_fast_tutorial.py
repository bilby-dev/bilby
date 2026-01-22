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

# Set OMP_NUM_THREADS to stop lalsimulation taking over my computer
os.environ["OMP_NUM_THREADS"] = "1"

import bilby
import jax
import jax.numpy as jnp
import numpy as np
from bilby.compat.jax import JittedLikelihood
from ripple.waveforms import IMRPhenomPv2

jax.config.update("jax_enable_x64", True)


def bilby_to_ripple_spins(
    theta_jn,
    phi_jl,
    tilt_1,
    tilt_2,
    phi_12,
    a_1,
    a_2,
):
    """
    A simplified spherical to cartesian spin conversion function.
    This is not equivalent to the method used in `bilby.gw.conversion`
    which comes from `lalsimulation` and is not `JAX` compatible.
    """
    iota = theta_jn
    spin_1x = a_1 * jnp.sin(tilt_1) * jnp.cos(phi_jl)
    spin_1y = a_1 * jnp.sin(tilt_1) * jnp.sin(phi_jl)
    spin_1z = a_1 * jnp.cos(tilt_1)
    spin_2x = a_2 * jnp.sin(tilt_2) * jnp.cos(phi_jl + phi_12)
    spin_2y = a_2 * jnp.sin(tilt_2) * jnp.sin(phi_jl + phi_12)
    spin_2z = a_2 * jnp.cos(tilt_2)
    return iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z


def ripple_bbh(
    frequency,
    mass_1,
    mass_2,
    luminosity_distance,
    theta_jn,
    phase,
    a_1,
    a_2,
    tilt_1,
    tilt_2,
    phi_12,
    phi_jl,
    **kwargs,
):
    """
    Source function wrapper to ripple's IMRPhenomPv2 waveform generator.
    This function cannot be jitted directly as the Bilby waveform generator
    relies on inspecting the function signature.

    Parameters
    ----------
    frequency: jnp.ndarray
        Frequencies at which to compute the waveform.
    mass_1: float | jnp.ndarray
        Mass of the primary component in solar masses.
    mass_2: float | jnp.ndarray
        Mass of the secondary component in solar masses.
    luminosity_distance: float | jnp.ndarray
        Luminosity distance to the source in Mpc.
    theta_jn: float | jnp.ndarray
        Angle between total angular momentum and line of sight in radians.
    phase: float | jnp.ndarray
        Phase at coalescence in radians.
    a_1: float | jnp.ndarray
        Dimensionless spin magnitude of the primary component.
    a_2: float | jnp.ndarray
        Dimensionless spin magnitude of the secondary component.
    tilt_1: float | jnp.ndarray
        Tilt angle of the primary component spin in radians.
    tilt_2: float | jnp.ndarray
        Tilt angle of the secondary component spin in radians.
    phi_12: float | jnp.ndarray
        Azimuthal angle between the two spin vectors in radians.
    phi_jl: float | jnp.ndarray
        Azimuthal angle of the total angular momentum vector in radians.
    **kwargs
        Additional keyword arguments. Must include 'minimum_frequency'.

    Returns
    -------
    dict
        Dictionary containing the plus and cross polarizations of the waveform.
    """
    iota, *cartesian_spins = bilby_to_ripple_spins(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2
    )
    frequencies = jnp.maximum(frequency, kwargs["minimum_frequency"])
    theta = jnp.array(
        [
            mass_1,
            mass_2,
            *cartesian_spins,
            luminosity_distance,
            jnp.array(0.0),
            phase,
            iota,
        ]
    )
    wf_func = jax.jit(IMRPhenomPv2.gen_IMRPhenomPv2)
    hp, hc = wf_func(frequencies, theta, jnp.array(20.0))
    return dict(plus=hp, cross=hc)


def main():
    # Set the duration and sampling frequency of the data segment that we're
    # going to inject the signal into
    duration = 64.0
    sampling_frequency = 2048.0
    minimum_frequency = 20.0
    duration = jnp.array(duration)
    sampling_frequency = jnp.array(sampling_frequency)
    minimum_frequency = jnp.array(minimum_frequency)

    # Specify the output directory and the name of the simulation.
    outdir = "outdir"
    label = f"jax_fast_tutorial"

    # Set up a random seed for result reproducibility.  This is optional!
    bilby.core.utils.random.seed(88170235)

    priors = bilby.gw.prior.BBHPriorDict()
    injection_parameters = priors.sample()
    injection_parameters["geocent_time"] = 1000000000.0
    injection_parameters["luminosity_distance"] = 400.0
    del priors["ra"], priors["dec"]
    priors["zenith"] = bilby.core.prior.Cosine()
    priors["azimuth"] = bilby.core.prior.Uniform(minimum=0, maximum=2 * np.pi)
    priors["L1_time"] = bilby.core.prior.Uniform(
        injection_parameters["geocent_time"] - 0.1,
        injection_parameters["geocent_time"] + 0.1,
    )

    # Fixed arguments passed into the source model
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=50.0,
        minimum_frequency=minimum_frequency,
    )

    # Create the waveform_generator using a LAL BinaryBlackHole source function
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=ripple_bbh,
        waveform_arguments=waveform_arguments,
        use_cache=False,
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
        waveform_generator=waveform_generator,
        parameters=injection_parameters,
        raise_error=False,
    )
    ifos.set_array_backend(jnp)

    # Initialise the likelihood by passing in the interferometer data (ifos) and
    # the waveform generator
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        priors=priors,
        phase_marginalization=True,
        distance_marginalization=True,
        reference_frame=ifos,
        time_reference="L1",
    )
    # Do an initial likelihood evaluation to trigger any internal setup
    likelihood.log_likelihood_ratio(priors.sample())
    # Wrap the likelihood with the JittedLikelihood to JIT compile the likelihood
    # evaluation
    likelihood = JittedLikelihood(likelihood)
    # Evaluate the likelihood once to trigger the JIT compilation, this will take
    # a few seconds as compiling the waveform takes some time
    likelihood.log_likelihood_ratio(priors.sample())

    # use the log_compiles context so we can make sure there aren't recompilations
    # inside the sampling loop
    with jax.log_compiles():
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler="dynesty",
            nlive=100,
            sample="acceptance-walk",
            naccept=5,
            injection_parameters=injection_parameters,
            outdir=outdir,
            label=label,
            npool=None,
            save="hdf5",
            rseed=np.random.randint(0, 100000),
        )

    # Make a corner plot.
    result.plot_corner()


if __name__ == "__main__":
    main()
