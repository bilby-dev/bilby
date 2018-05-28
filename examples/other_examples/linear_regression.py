#!/bin/python
"""
An example of how to use tupak to perform paramater estimation for
non-gravitational wave data
"""
from __future__ import division
import tupak
import numpy as np
import matplotlib.pyplot as plt

# A few simple setup steps
tupak.utils.setup_logger()
label = 'linear-regression'
outdir = 'outdir'

# Here is minimum requirement for a Likelihood class to run linear regression
# with tupak. In this case, we setup a GaussianLikelihood, which needs to have
# a log_likelihood method. Note, in this case we make use of the `tupak`
# waveform_generator to make the signal (more on this later) But, one could
# make this work without the waveform generator.

# Making simulated data


# First, we define our signal model, in this case a simple linear function
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


class GaussianLikelihood(tupak.likelihood.Likelihood):
    def __init__(self, x, y, sigma, waveform_generator):
        """

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        sigma: float
            The standard deviation of the noise
        waveform_generator: `tupak.waveform_generator.WaveformGenerator`
            An object which can generate the 'waveform', which in this case is
            any arbitrary function
        """
        self.x = x
        self.y = y
        self.sigma = sigma
        self.N = len(x)
        self.waveform_generator = waveform_generator
        self.parameters = waveform_generator.parameters

    def log_likelihood(self):
        res = self.y - self.waveform_generator.time_domain_strain()
        return -0.5 * (np.sum((res / self.sigma)**2)
                       + self.N*np.log(2*np.pi*self.sigma**2))

    def noise_log_likelihood(self):
        return -0.5 * (np.sum((self.y / self.sigma)**2)
                       + self.N*np.log(2*np.pi*self.sigma**2))


# Here, we make a `tupak` waveform_generator. In this case, of course, the
# name doesn't make so much sense. But essentially this is an objects that
# can generate a signal. We give it information on how to make the time series
# and the model() we wrote earlier.

waveform_generator = tupak.waveform_generator.WaveformGenerator(
    time_duration=time_duration, sampling_frequency=sampling_frequency,
    time_domain_source_model=model)

# Now lets instantiate a version of out GravitationalWaveTransient, giving it
# the time, data and waveform_generator
likelihood = GaussianLikelihood(time, data, sigma, waveform_generator)

# From hereon, the syntax is exactly equivalent to other tupak examples
# We make a prior
priors = {}
priors['m'] = tupak.prior.Uniform(0, 5, 'm')
priors['c'] = tupak.prior.Uniform(-2, 2, 'c')

# And run sampler
result = tupak.sampler.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=500,
    walks=10, injection_parameters=injection_parameters, outdir=outdir,
    label=label, plot=True)
print(result)
