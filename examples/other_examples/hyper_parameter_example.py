#!/bin/python
"""
An example of how to use tupak to perform paramater estimation for hyperparams
"""
from __future__ import division
import tupak
import numpy as np

# A few simple setup steps
tupak.utils.setup_logger()
label = 'hyperparameter'
outdir = 'outdir'


class GaussianLikelihood():
    def __init__(self, x, y, waveform_generator):
        self.x = x
        self.y = y
        self.N = len(x)
        self.waveform_generator = waveform_generator
        self.parameters = waveform_generator.parameters

    def log_likelihood(self):
        sigma = 1
        res = self.y - self.waveform_generator.time_domain_strain()
        return -0.5 * (np.sum((res / sigma)**2)
                       + self.N*np.log(2*np.pi*sigma**2))

    def noise_log_likelihood(self):
        return np.nan


def model(time, m):
    return m * time


sampling_frequency = 10
time_duration = 100
time = np.arange(0, time_duration, 1/sampling_frequency)

true_mu_m = 5
true_sigma_m = 0.1
sigma = 0.1
Nevents = 10
samples = []

# Make the sample sets
for i in range(Nevents):
    m = np.random.normal(true_mu_m, true_sigma_m)
    injection_parameters = dict(m=m)

    N = len(time)
    data = model(time, **injection_parameters) + np.random.normal(0, sigma, N)

    waveform_generator = tupak.waveform_generator.WaveformGenerator(
        time_duration=time_duration, sampling_frequency=sampling_frequency,
        time_domain_source_model=model)

    likelihood = GaussianLikelihood(time, data, waveform_generator)

    priors = dict(m=tupak.prior.Uniform(-10, 10, 'm'))

    # And run sampler
    result = tupak.sampler.run_sampler(
        likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
        injection_parameters=injection_parameters, outdir=outdir, verbose=False,
        label=label + '_{}'.format(i), use_ratio=False, sample='unif')
    result.plot_corner()
    samples.append(result.samples)

# Now run the hyperparameter inference


def run_prior(val):
    if np.all(val > -10) & np.all(val < 10):
        return 1/20.
    else:
        return 0


def hyper_prior(val, mu_m, sigma_m):
    return np.exp(-(mu_m - val)**2 / 2 / sigma_m**2) / np.sqrt(2*np.pi*sigma_m**2)


def log_run_prior(val):
    if np.all(val > -10) & np.all(val < 10):
        return len(val) * - np.log(20)
    else:
        return 0


def log_hyper_prior(val, mu_m, sigma_m):
    res = val - mu_m
    return -0.5 * (np.sum((res / sigma_m)**2)
                   + len(val)*np.log(2*np.pi*sigma_m**2))


hp_likelihood = tupak.likelihood.HyperparameterLikelihood(
        samples, log_hyper_prior, log_run_prior, mu_m=None, sigma_m=None)

hp_priors = dict(
    mu_m=tupak.prior.Uniform(-10, 10, 'mu_m', '$\mu_m$'),
    sigma_m=tupak.prior.Uniform(0, 10, 'sigma_m', '$\sigma_m$'))


# And run sampler
result = tupak.sampler.run_sampler(
    likelihood=hp_likelihood, priors=hp_priors, sampler='dynesty', npoints=1000,
    outdir=outdir, label=label + '_hp', use_ratio=False, sample='unif', verbose=True)
result.plot_corner()

