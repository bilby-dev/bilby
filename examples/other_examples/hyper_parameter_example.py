#!/bin/python
"""
An example of how to use tupak to perform paramater estimation for hyper params
"""
from __future__ import division
import tupak
import numpy as np
import matplotlib.pyplot as plt

outdir = 'outdir'


# Define a model to fit to each data set
def model(x, c0, c1):
    return c0 + c1*x


N = 10
x = np.linspace(0, 10, N)
sigma = 1
Nevents = 3
labels = ['a', 'b', 'c']

true_mu_c0 = 5
true_sigma_c0 = 1

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

# Make the sample sets
samples = []
for i in range(Nevents):
    c0 = np.random.normal(true_mu_c0, true_sigma_c0)
    c1 = np.random.uniform(-1, 1)
    injection_parameters = dict(c0=c0, c1=c1)

    data = model(x, **injection_parameters) + np.random.normal(0, sigma, N)
    line = ax1.plot(x, data, '-x', label=labels[i])

    likelihood = tupak.core.likelihood.GaussianLikelihood(x, data, model, sigma)
    priors = dict(c0=tupak.core.prior.Uniform(-10, 10, 'c0'),
                  c1=tupak.core.prior.Uniform(-10, 10, 'c1'))

    result = tupak.core.sampler.run_sampler(
        likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
        outdir=outdir, verbose=False, label='individual_{}'.format(i))
    ax2.hist(result.posterior.c0, color=line[0].get_color(), normed=True,
             alpha=0.5, label=labels[i])
    samples.append(result.posterior)

ax1.set_xlabel('x')
ax1.set_ylabel('y(x)')
ax1.legend()
fig1.savefig('outdir/hyper_parameter_data.png')
ax2.set_xlabel('c0')
ax2.set_ylabel('density')
ax2.legend()
fig2.savefig('outdir/hyper_parameter_combined_posteriors.png')


def hyper_prior(data, mu, sigma):
    return np.exp(- (data['c0'] - mu)**2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)**0.5


def run_prior(data):
    return 1 / 20


hp_likelihood = tupak.hyper.likelihood.HyperparameterLikelihood(
        posteriors=samples, hyper_prior=hyper_prior, sampling_prior=run_prior)

hp_priors = dict(mu=tupak.core.prior.Uniform(-10, 10, 'mu', '$\mu_{c0}$'),
                 sigma=tupak.core.prior.Uniform(0, 10, 'sigma', '$\sigma_{c0}$'))

# And run sampler
result = tupak.core.sampler.run_sampler(
    likelihood=hp_likelihood, priors=hp_priors, sampler='dynesty',
    npoints=1000, outdir=outdir, label='hyper_parameter', verbose=True, clean=True)
result.plot_corner(truth=dict(mu=true_mu_c0, sigma=true_sigma_c0))
