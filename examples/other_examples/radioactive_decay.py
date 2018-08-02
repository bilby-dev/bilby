#!/bin/python
"""
An example of how to use tupak to perform paramater estimation for
non-gravitational wave data. In this case, fitting the half-life and
initial radionucleotide number for Carbon 14. 
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

NA = 6.02214078e23 # Avagadro's number

# generate a set of counts per minute for Carbon 14 with a half-life of
# 5730 years
halflife = 5730.
N0mol = 1. # initial number of radionucleotides in moles
N0 = N0mol * 


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# Now we define the injection parameters which we make simulated data with
injection_parameters = dict(m=0.5, c=0.2)

# For this example, we'll use standard Gaussian noise
sigma = 1

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_parameters when calling the model function.
sampling_frequency = 10
time_duration = 10
time = np.arange(0, time_duration, 1/sampling_frequency)
N = len(time)
data = model(time, **injection_parameters) + np.random.normal(0, sigma, N)

# We quickly plot the data to check it looks sensible
fig, ax = plt.subplots()
ax.plot(time, data, 'o', label='data')
ax.plot(time, model(time, **injection_parameters), '--r', label='signal')
ax.set_xlabel('time')
ax.set_ylabel('y')
ax.legend()
fig.savefig('{}/{}_data.png'.format(outdir, label))

# Now lets instantiate a version of the Poisson Likelihood, giving it
# the time, data and signal model
likelihood = PoissonLikelihood(time, counts, model)

# From hereon, the syntax is exactly equivalent to other tupak examples
# We make a prior
priors = {}
priors['m'] = tupak.core.prior.Uniform(0, 5, 'm')
priors['c'] = tupak.core.prior.Uniform(-2, 2, 'c')

# And run sampler
result = tupak.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=500,
    walks=10, injection_parameters=injection_parameters, outdir=outdir,
    label=label)
result.plot_corner()
