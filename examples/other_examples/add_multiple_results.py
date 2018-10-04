#!/usr/bin/env python
"""
An example of running two sets of posterior sample estimations and adding them
"""
from __future__ import division
import bilby
import numpy as np

outdir = 'outdir'


def model(time, m, c):
    return time * m + c


injection_parameters = dict(m=0.5, c=0.2)
sigma = 1
sampling_frequency = 10
time_duration = 10
time = np.arange(0, time_duration, 1/sampling_frequency)
N = len(time)
data = model(time, **injection_parameters) + np.random.normal(0, sigma, N)

likelihood = bilby.core.likelihood.GaussianLikelihood(
    time, data, model, sigma=sigma)

priors = {}
priors['m'] = bilby.core.prior.Uniform(0, 1, 'm')
priors['c'] = bilby.core.prior.Uniform(-2, 2, 'c')

resultA = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='emcee', nsteps=1000,
    nburn=500, outdir=outdir, label='A')

resultB = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='emcee', nsteps=1000,
    nburn=500, outdir=outdir, label='B')

resultA.plot_walkers()
result = resultA + resultB
result.plot_corner()
print(result)


