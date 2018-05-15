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
tupak.utils.setup_logger(log_level="info")
label = 'test'
outdir = 'outdir'


# Here is minimum requirement for a Likelihood class needed to run tupak. In
# this case, we setup a GaussianLikelihood, which needs to have a
# log_likelihood and noise_log_likelihood method. Note, in this case we make
# use of the `tupak` waveform_generator to make the signal (more on this later)
# But, one could make this work without the waveform generator.

class GaussianLikelihood():
    def __init__(self, x, y, waveform_generator):
        self.x = x
        self.y = y
        self.N = len(x)
        self.waveform_generator = waveform_generator

    def log_likelihood(self):
        sigma = 1
        res = self.y - self.waveform_generator.time_domain_strain()
        return -0.5 * (np.sum((res / sigma)**2)
                       + self.N*np.log(2*np.pi*sigma**2))

    def noise_log_likelihood(self):
        sigma = 1
        return -0.5 * (np.sum((self.y / sigma)**2)
                       + self.N*np.log(2*np.pi*sigma**2))


# Here we define our signal model, in this case a very basic trig. function
def model(time, A, P):
    return A*np.sin(2*np.pi*time/P)


# Here we define the injection parameters which we make simulated data with
injection_parameters = dict(A=1.5, P=10)

# For this example, we'll use standard Gaussian noise
sigma = 1

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_paramsters when calling the model function.
sampling_frequency = 10
time_duration = 100
time = np.arange(0, time_duration, 1/sampling_frequency)
N = len(time)
data = model(time, **injection_parameters) + np.random.normal(0, sigma, N)

# We quickly plot the data to check it looks sensible
fig, ax = plt.subplots()
ax.plot(time, data)
ax.plot(time, model(time, **injection_parameters), '--r')
fig.savefig('{}/data.png'.format(outdir))

# Here, we make a `tupak` waveform_generator. In this case, of course, the
# name doesn't make so much sense. But essentially this is an objects that
# can generate a signal. We give it information on how to make the time series
# and the model() we wrote earlier.
waveform_generator = tupak.waveform_generator.WaveformGenerator(time_duration=time_duration,
                                                                sampling_frequency=sampling_frequency,
                                                                time_domain_source_model=model)


# Now lets instantiate a version of out Likelihood, giving it the time, data
# and waveform_generator
likelihood = GaussianLikelihood(time, data, waveform_generator)

# From hereon, the syntax is exactly equivalent to other tupak examples
# We make a prior
priors = {}
priors['A'] = tupak.prior.Uniform(0, 5, 'A')
priors['P'] = tupak.prior.Uniform(0, 20, 'P')

# And run sampler
result = tupak.sampler.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
    walks=10, injection_parameters=injection_parameters, outdir=outdir,
    label=label, use_ratio=False)
result.plot_corner()
print(result)
