"""
Generic dumping ground for jax-specific functions that we need.
This should find a home somewhere down the line, but gives an
idea of how much pain is being added.
"""

import jax
import jax.numpy as jnp
from ripple.waveforms import IMRPhenomPv2


def bilby_to_ripple_spins(
    theta_jn,
    phi_jl,
    tilt_1,
    tilt_2,
    phi_12,
    a_1,
    a_2,
):
    iota = theta_jn
    spin_1x = a_1 * jnp.sin(tilt_1) * jnp.cos(phi_jl)
    spin_1y = a_1 * jnp.sin(tilt_1) * jnp.sin(phi_jl)
    spin_1z = a_1 * jnp.cos(tilt_1)
    spin_2x = a_2 * jnp.sin(tilt_2) * jnp.cos(phi_jl + phi_12)
    spin_2y = a_2 * jnp.sin(tilt_2) * jnp.sin(phi_jl + phi_12)
    spin_2z = a_2 * jnp.cos(tilt_2)
    return iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z


wf_func = jax.jit(IMRPhenomPv2.gen_IMRPhenomPv2)


def ripple_bbh_relbin(
    frequency, mass_1, mass_2, luminosity_distance, theta_jn, phase,
    a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, fiducial, **kwargs,
):
    if fiducial == 1:
        kwargs["frequencies"] = frequency
    else:
        kwargs["frequencies"] = kwargs.pop("frequency_bin_edges")
    return ripple_bbh(
        frequency, mass_1, mass_2, luminosity_distance, theta_jn, phase,
        a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, **kwargs
    )


def ripple_bbh(
    frequency, mass_1, mass_2, luminosity_distance, theta_jn, phase,
    a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, **kwargs,
):
    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_ripple_spins(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2
    )
    if "frequencies" in kwargs:
        frequencies = kwargs["frequencies"]
    elif "minimum_frequency" in kwargs:
        frequencies = jnp.maximum(frequency, kwargs["minimum_frequency"])
    else:
        frequencies = frequency
    theta = jnp.array([
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
        luminosity_distance, jnp.array(0.0), phase, iota
    ])
    hp, hc = wf_func(frequencies, theta, jax.numpy.array(20.0))
    return dict(plus=hp, cross=hc)

