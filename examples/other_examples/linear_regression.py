#!/bin/python
"""
An example of how to use tupak to perform paramater estimation for
non-gravitational wave data. In this case, fitting a linear function to
data with background Gaussian noise

"""
from __future__ import division
import tupak
import numpy as np
import matplotlib.pyplot as plt
import inspect

# A few simple setup steps
label = 'linear_regression'
outdir = 'outdir'


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# New we define the injection parameters which we make simulated data with
injection_parameters = dict(m=0.5, c=0.2)

# For this example, we'll use standard Gaussian noise
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


class GaussianLikelihoodKnownNoise(tupak.Likelihood):
    def __init__(self, x, y, sigma, function):
        """
        A general Gaussian likelihood - the parameters are inferred from the
        arguments of function

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        sigma: float
            The standard deviation of the noise
        function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments are
            will require a prior and will be sampled over (unless a fixed
            value is given).
        """
        self.x = x
        self.y = y
        self.sigma = sigma
        self.N = len(x)
        self.function = function

        # These lines of code infer the parameters from the provided function
        parameters = inspect.getargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)

    def log_likelihood(self):
        res = self.y - self.function(self.x, **self.parameters)
        return -0.5 * (np.sum((res / self.sigma)**2)
                       + self.N*np.log(2*np.pi*self.sigma**2))


# Now lets instantiate a version of our GaussianLikelihood, giving it
# the time, data and signal model
likelihood = GaussianLikelihoodKnownNoise(time, data, sigma, model)

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
