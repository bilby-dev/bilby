#!/bin/python
"""
An example of how to use tupak to perform paramater estimation for
non-gravitational wave data. In this case, fitting the half-life and
initial radionucleotide number for Polonium 214. 
"""
from __future__ import division
import tupak
import numpy as np
import matplotlib.pyplot as plt
import inspect

from tupak.core.likelihood import PoissonLikelihood

# A few simple setup steps
label = 'radioactive_decay'
outdir = 'outdir'
tupak.utils.check_directory_exists_and_if_not_mkdir(outdir)

# generate a set of counts per minute for Polonium 214 with a half-life of 20 mins
halflife = 20
N0 = 1.e-19 # initial number of radionucleotides in moles
atto = 1e-18
N0 /= atto

def decayrate(deltat, halflife, N0):
    """
    Get the decay rate of a radioactive substance in a range of time intervals
    (in minutes). halflife is in mins. N0 is in moles.
    """

    times = np.cumsum(deltat) # get cumulative times
    times = np.insert(times, 0, 0.)

    ln2 = np.log(2.)
    NA = 6.02214078e23 # Avagadro's number

    N0a = N0*atto*NA # number of nucleotides

    rates = (N0a*(np.exp(-ln2*(times[0:-1]/halflife)) 
             - np.exp(-ln2*(times[1:]/halflife)))/deltat)

    return rates


# Now we define the injection parameters which we make simulated data with
injection_parameters = dict(halflife=halflife, N0=N0)

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_parameters when calling the model function.
sampling_frequency = 1.
time_duration = 30.
time = np.arange(0, time_duration, 1./sampling_frequency)
deltat = np.diff(time)

rates = decayrate(deltat, **injection_parameters)
# get radioactive counts
counts = np.array([np.random.poisson(rate) for rate in rates])

# We quickly plot the data to check it looks sensible
fig, ax = plt.subplots()
ax.plot(time[0:-1], counts, 'o', label='data')
ax.plot(time[0:-1], decayrate(deltat, **injection_parameters), '--r', label='signal')
ax.set_xlabel('time')
ax.set_ylabel('counts')
ax.legend()
fig.savefig('{}/{}_data.png'.format(outdir, label))

# Now lets instantiate a version of the Poisson Likelihood, giving it
# the time intervals, counts and rate model
likelihood = PoissonLikelihood(deltat, counts, decayrate)

# Make the prior
priors = {}
priors['halflife'] = tupak.core.prior.LogUniform(1e-5, 1e5, 'halflife',
                                                 latex_label='$t_{1/2}$ (min)')
priors['N0'] = tupak.core.prior.LogUniform(1e-25/atto, 1e-10/atto, 'N0',
                                           latex_label='$N_0$ (attomole)')

# And run sampler
result = tupak.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=500,
    nlive=1000, walks=10, injection_parameters=injection_parameters,
    outdir=outdir, label=label)
result.plot_corner()
