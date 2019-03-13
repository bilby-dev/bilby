#!/usr/bin/env python
"""
An example of how to use bilby with a (multi-modal) multivariate
Gaussian prior distribution.
"""

from __future__ import division
import bilby
import numpy as np

from bilby.core.likelihood import GaussianLikelihood

# A few simple setup steps
label = 'multivariate_gaussian_prior'
outdir = 'outdir'
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# For this example assume a very broad Gaussian likelihood, so that the
# posterior is completely dominated by the prior (and we can check the
# prior sampling looks correct!)
sigma = 300.

# These lines of code generate the fake (noise-free) data
sampling_frequency = 1
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency)
N = len(time)
data = model(time, 0., 0.)  # noiseless data


# Now lets instantiate a version of our GaussianLikelihood, giving it
# the time, data, signal model and standard deviation
likelihood = GaussianLikelihood(time, data, model, sigma=sigma)

# Create a Multivariate Gaussian prior distribution with two modes
names = ['m', 'c']
mus = [[-5., -5.], [5., 5.]]  # means of two modes
corrcoefs = [[[1., -0.7], [-0.7, 1.]], [[1., -0.7], [-0.7, 1.]]]  # correlation coefficients of the modes
sigmas = [[1.5, 1.5], [1.5, 1.5]]  # standard deviations of the modes
weights = [0.5, 0.5]  # weights of each mode
nmodes = 2
mvg = bilby.core.prior.MultivariateGaussianDist(names, nmodes=2, mus=mus,
                                                corrcoefs=corrcoefs,
                                                sigmas=sigmas, weights=weights)
priors = dict()
priors['m'] = bilby.core.prior.MultivariateGaussian(mvg, 'm')
priors['c'] = bilby.core.prior.MultivariateGaussian(mvg, 'c')

# And run sampler
# result = bilby.run_sampler(
#    likelihood=likelihood, priors=priors, sampler='pymc3',
#    outdir=outdir, draws=2000, label=label)

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', nlive=4000,
    outdir=outdir, label=label)

# result = bilby.run_sampler(
#     likelihood=likelihood, priors=priors, sampler='nestle', nlive=4000,
#     outdir=outdir, label=label)

# result = bilby.run_sampler(
#     likelihood=likelihood, priors=priors, sampler='emcee', nsteps=1000,
#     nwalkers=200, nburn=500, outdir=outdir, label=label)
result.plot_corner()