#!/bin/python
"""
An example of how to use tupak to perform paramater estimation for
non-gravitational wave data. In this case, fitting a linear function to
data with background Gaussian noise with unknown variance.

"""
from __future__ import division
import tupak
import numpy as np
import matplotlib.pyplot as plt
import inspect

# A few simple setup steps
tupak.core.utils.setup_logger()
label = 'linear_regression_unknown_noise'
outdir = 'outdir'


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# New we define the injection parameters which we make simulated data with
injection_parameters = dict(m=0.5, c=0.2)

# For this example, we'll inject standard Gaussian noise
sigma = 1

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_paramsters when calling the model function.
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


# Parameter estimation: we now define a Gaussian Likelihood class relevant for
# our model.

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


# Now lets instantiate a version of our GaussianLikelihood, giving it
# the time, data and signal model
likelihood = GaussianLikelihood(time, data, model)

# From hereon, the syntax is exactly equivalent to other tupak examples
# We make a prior
priors = {}
priors['m'] = tupak.core.prior.Uniform(0, 5, 'm')
priors['c'] = tupak.core.prior.Uniform(-2, 2, 'c')
priors['sigma'] = tupak.core.prior.Uniform(0, 10, 'sigma')

# And run sampler
result = tupak.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=500,
    walks=10, injection_parameters=injection_parameters, outdir=outdir,
    label=label)
result.plot_corner()
