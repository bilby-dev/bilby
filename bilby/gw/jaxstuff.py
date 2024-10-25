"""
Generic dumping ground for jax-specific functions that we need.
This should find a home somewhere down the line, but gives an
idea of how much pain is being added.
"""

from functools import partial

import numpy as np
from bilby.core.likelihood import Likelihood

import jax
import jax.numpy as jnp
from plum import dispatch
from jax.scipy.special import i0e
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


def ripple_bbh(frequency, mass_1, mass_2, luminosity_distance, theta_jn, phase,
        a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, **kwargs):
    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_ripple_spins(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2
    )
    theta = jnp.array([
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
        luminosity_distance, 0.0, phase, iota
    ])
    hp, hc = jax.jit(IMRPhenomPv2.gen_IMRPhenomPv2)(frequency, theta, jax.numpy.array(20.0))
    return dict(plus=hp, cross=hc)


def generic_bilby_likelihood_function(likelihood, parameters, use_ratio=True):
    """
    A wrapper to allow a :code:`Bilby` likelihood to be used with :code:`jax`.

    Parameters
    ==========
    likelihood: bilby.core.likelihood.Likelihood
        The likelihood to evaluate.
    parameters: dict
        The parameters to evaluate the likelihood at.
    use_ratio: bool, optional
        Whether to evaluate the likelihood ratio or the full likelihood.
        Default is :code:`True`.
    """
    likelihood.parameters.update(parameters)
    if use_ratio:
        return likelihood.log_likelihood_ratio()
    else:
        return likelihood.log_likelihood()


class JittedLikelihood(Likelihood):
    """
    A wrapper to just-in-time compile a :code:`Bilby` likelihood for use with :code:`jax`.

    .. note::

        This is currently hardcoded to return the log likelihood ratio, regardless of
        the input.

    Parameters
    ==========
    likelihood: bilby.core.likelihood.Likelihood
        The likelihood to wrap.
    likelihood_func: callable, optional
        The function to use to evaluate the likelihood. Default is
        :code:`generic_bilby_likelihood_function`. This function should take the
        likelihood and parameters as arguments along with additional keyword arguments.
    kwargs: dict, optional
        Additional keyword arguments to pass to the likelihood function.
    """

    def __init__(
        self, likelihood, likelihood_func=generic_bilby_likelihood_function, kwargs=None
    ):
        if kwargs is None:
            kwargs = dict()
        self.kwargs = kwargs
        self._likelihood = likelihood
        self.likelihood_func = jax.jit(partial(likelihood_func, likelihood))
        super().__init__(dict())

    def __getattr__(self, name):
        return getattr(self._likelihood, name)

    def log_likelihood_ratio(self):
        return float(
            np.nan_to_num(self.likelihood_func(self.parameters, **self.kwargs))
        )


@dispatch
def ln_i0(value: jax.Array):
    return jnp.log(i0e(value)) + jnp.abs(value)