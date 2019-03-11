#!/usr/bin/env python
"""
An example of how to use bilby with a (multi-modal) multivariate
Gaussian prior distribution.
"""

from __future__ import division
import bilby
import numpy as np
import matplotlib.pyplot as plt

from bilby.core.likelihood import GaussianLikelihood

# A few simple setup steps
label = 'multivariate_gaussian_prior'
outdir = 'outdir'
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# Now we define the injection parameters which we make simulated data with
injection_parameters = dict(m=0.0, c=0.0)

# For this example, we'll use standard Gaussian noise
sigma = 200.

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_parameters when calling the model function.
sampling_frequency = 10
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency)
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

# Now lets instantiate a version of our GaussianLikelihood, giving it
# the time, data and signal model
likelihood = GaussianLikelihood(time, data, model, sigma=sigma)

# Create a Multivariate Gaussian prior distribution with two modes
names = ['m', 'c']
mus = [[-3., -3.], [5., 5.]]  # means of two modes
corrcoefs = [[[1., 0.], [0., 1.]], [[1., 0.95], [0.95, 1.]]]  # correlation coefficients of the modes
sigmas = [[1., 1.], [2., 2.]]  # standard deviations of the modes
weights = [0.5, 0.5]  # weights of each mode
nmodes = 2
mvg = bilby.core.prior.MultivariateGaussianDist(names, nmodes=2, mus=mus,
                                                corrcoefs=corrcoefs,
                                                sigmas=sigmas, weights=weights)
priors = dict()
priors['m'] = bilby.core.prior.MultivariateGaussian(mvg, 'm')
priors['c'] = bilby.core.prior.MultivariateGaussian(mvg, 'c')

# And run sampler
#result = bilby.run_sampler(
#    likelihood=likelihood, priors=priors, sampler='pymc3',
#    injection_parameters=injection_parameters, outdir=outdir,
#    draws=2000, label=label)
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', nlive=2000,
    outdir=outdir, label=label)
#result = bilby.run_sampler(
#    likelihood=likelihood, priors=priors, sampler='emcee', nsteps=100,
#    nwalkers=100, outdir=outdir, label=label)
result.plot_corner()
