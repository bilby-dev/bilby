#!/usr/bin/env python
"""
An example of how to use bilby to perform paramater estimation for
non-gravitational wave data. In this case, fitting the half-life and
initial radionuclide number for Polonium 214.
"""
import bilby
import numpy as np
import matplotlib.pyplot as plt

from bilby.core.likelihood import PoissonLikelihood
from bilby.core.prior import LogUniform

# A few simple setup steps
label = 'radioactive_decay'
outdir = 'outdir'
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

# generate a set of counts per minute for n_init atoms of
# Polonium 214 in atto-moles with a half-life of 20 minutes
n_avogadro = 6.02214078e23
halflife = 20
atto = 1e-18
n_init = 1e-19 / atto


def decay_rate(delta_t, halflife, n_init):
    """
    Get the decay rate of a radioactive substance in a range of time intervals
    (in minutes). n_init is in moles.

    Parameters
    ----------
    delta_t: float, array-like
        Time step in minutes
    halflife: float
        Halflife of atom in minutes
    n_init: int, float
        Initial nummber of atoms
    """

    times = np.cumsum(delta_t)
    times = np.insert(times, 0, 0.0)

    n_atoms = n_init * atto * n_avogadro

    rates = (np.exp(-np.log(2) * (times[:-1] / halflife)) -
             np.exp(- np.log(2) * (times[1:] / halflife))) * n_atoms / delta_t

    return rates


# Now we define the injection parameters which we make simulated data with
injection_parameters = dict(halflife=halflife, n_init=n_init)

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_parameters when calling the model function.
sampling_frequency = 1
time_duration = 300
time = np.arange(0, time_duration, 1 / sampling_frequency)
delta_t = np.diff(time)

rates = decay_rate(delta_t, **injection_parameters)
# get radioactive counts
counts = np.random.poisson(rates)
theoretical = decay_rate(delta_t, **injection_parameters)

# We quickly plot the data to check it looks sensible
fig, ax = plt.subplots()
ax.semilogy(time[:-1], counts, 'o', label='data')
ax.semilogy(time[:-1], theoretical, '--r', label='signal')
ax.set_xlabel('time')
ax.set_ylabel('counts')
ax.legend()
fig.savefig('{}/{}_data.png'.format(outdir, label))

# Now lets instantiate a version of the Poisson Likelihood, giving it
# the time intervals, counts and rate model
likelihood = PoissonLikelihood(delta_t, counts, decay_rate)

# Make the prior
priors = dict()
priors['halflife'] = LogUniform(
    1e-5, 1e5, latex_label='$t_{1/2}$', unit='min')
priors['n_init'] = LogUniform(
    1e-25 / atto, 1e-10 / atto, latex_label='$N_0$', unit='attomole')

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty',
    nlive=1000, sample='unif', injection_parameters=injection_parameters,
    outdir=outdir, label=label)
result.plot_corner()
