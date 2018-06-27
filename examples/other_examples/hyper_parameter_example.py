#!/bin/python
"""
An example of how to use tupak to perform paramater estimation for hyper params
"""
from __future__ import division
import tupak
import numpy as np
import inspect
import matplotlib.pyplot as plt

outdir = 'outdir'


class GaussianLikelihood(tupak.Likelihood):
    def __init__(self, x, y, function, sigma=None):
        """
        A general Gaussian likelihood for known or unknown noise - the model
        parameters are inferred from the arguments of function

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments are
            will require a prior and will be sampled over (unless a fixed
            value is given).
        sigma: None, float, array_like
            If None, the standard deviation of the noise is unknown and will be
            estimated (note: this requires a prior to be given for sigma). If
            not None, this defined the standard-deviation of the data points.
            This can either be a single float, or an array with length equal
            to that for `x` and `y`.
        """
        self.x = x
        self.y = y
        self.N = len(x)
        self.sigma = sigma
        self.function = function

        # These lines of code infer the parameters from the provided function
        parameters = inspect.getargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        self.function_keys = self.parameters.keys()
        if self.sigma is None:
            self.parameters['sigma'] = None

    def log_likelihood(self):
        sigma = self.parameters.get('sigma', self.sigma)
        model_parameters = {k: self.parameters[k] for k in self.function_keys}
        res = self.y - self.function(self.x, **model_parameters)
        return -0.5 * (np.sum((res / sigma)**2)
                       + self.N*np.log(2*np.pi*sigma**2))


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

    likelihood = GaussianLikelihood(x, data, model, sigma)
    priors = dict(c0=tupak.core.prior.Uniform(-10, 10, 'c0'),
                  c1=tupak.core.prior.Uniform(-10, 10, 'c1'),
                  )

    result = tupak.core.sampler.run_sampler(
        likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
        outdir=outdir, verbose=False, label='individual_{}'.format(i))
    ax2.hist(result.posterior.c0, color=line[0].get_color(), normed=True,
             alpha=0.5, label=labels[i])
    samples.append(result.posterior.c0.values)

ax1.set_xlabel('x')
ax1.set_ylabel('y(x)')
ax1.legend()
fig1.savefig('outdir/hyper_parameter_data.png')
ax2.set_xlabel('c0')
ax2.set_ylabel('density')
ax2.legend()
fig2.savefig('outdir/hyper_parameter_combined_posteriors.png')

# Now run the hyper parameter inference
run_prior = tupak.core.prior.Uniform(minimum=-10, maximum=10, name='mu_c0')
hyper_prior = tupak.core.prior.Gaussian(mu=0, sigma=1, name='hyper')

hp_likelihood = tupak.core.likelihood.HyperparameterLikelihood(
        samples, hyper_prior, run_prior)

hp_priors = dict(
    mu=tupak.core.prior.Uniform(-10, 10, 'mu', '$\mu_{c0}$'),
    sigma=tupak.core.prior.Uniform(0, 10, 'sigma', '$\sigma_{c0}$'))

# And run sampler
result = tupak.core.sampler.run_sampler(
    likelihood=hp_likelihood, priors=hp_priors, sampler='dynesty',
    npoints=1000, outdir=outdir, label='hyper_parameter', verbose=True)
result.plot_corner(truth=dict(mu=true_mu_c0, sigma=true_sigma_c0))
