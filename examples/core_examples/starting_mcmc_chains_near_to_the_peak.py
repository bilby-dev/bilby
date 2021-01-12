#!/usr/bin/env python
"""
An example of using emcee, but starting the walkers off close to the peak (or
any other arbitrary point). This is based off the
linear_regression_with_unknown_noise.py example.
"""
import bilby
import numpy as np
import pandas as pd

# A few simple setup steps
label = 'starting_mcmc_chains_near_to_the_peak'
outdir = 'outdir'


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# Now we define the injection parameters which we make simulated data with
injection_parameters = dict(m=0.5, c=0.2)

# For this example, we'll inject standard Gaussian noise
sigma = 1

# These lines of code generate the fake data
sampling_frequency = 10
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency)
N = len(time)
data = model(time, **injection_parameters) + np.random.normal(0, sigma, N)

# Now lets instantiate the built-in GaussianLikelihood, giving it
# the time, data and signal model. Note that, because we do not give it the
# parameter, sigma is unknown and marginalised over during the sampling
likelihood = bilby.core.likelihood.GaussianLikelihood(time, data, model)

# Here we define the prior distribution used while sampling
priors = bilby.core.prior.PriorDict()
priors['m'] = bilby.core.prior.Uniform(0, 5, 'm')
priors['c'] = bilby.core.prior.Uniform(-2, 2, 'c')
priors['sigma'] = bilby.core.prior.Uniform(0, 10, 'sigma')

# Set values to determine how to sample with emcee
nwalkers = 100
nsteps = 1000
sampler = 'emcee'

# Run the sampler from the default pos0 (which is samples drawn from the prior)
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler=sampler, nsteps=nsteps,
    nwalkers=nwalkers, outdir=outdir, label=label + 'default_pos0')
result.plot_walkers()


# Here we define a distribution from which to start the walkers off from.
start_pos = bilby.core.prior.PriorDict()
start_pos['m'] = bilby.core.prior.Normal(injection_parameters['m'], 0.1)
start_pos['c'] = bilby.core.prior.Normal(injection_parameters['c'], 0.1)
start_pos['sigma'] = bilby.core.prior.Normal(sigma, 0.1)

# This line generated the initial starting position data frame by sampling
# nwalkers-times from the start_pos distribution. Note, you can
# generate this is anyway you like, but it must be a DataFrame with a length
# equal to the number of walkers
pos0 = pd.DataFrame(start_pos.sample(nwalkers))

# Run the sampler with our created pos0
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler=sampler, nsteps=nsteps,
    nwalkers=nwalkers, outdir=outdir, label=label + 'user_pos0', pos0=pos0)
result.plot_walkers()


# After running this script, in the outdir, you'll find to png images showing
# the result of the runs with and without the initialisation.
