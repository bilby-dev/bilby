"""
Tutorial to show signal injection and PE
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

import peyote
import corner
from dynesty import plotting as dyplot


class Likelihood:
    def __init__(self, data):
        self.data = data
        self.N = len(data)
        self.parameter_keys = ['mu', 'sigma']

    def logl(self, theta):
        mu = theta[0]
        sigma = theta[1]
        res = (self.data - mu)
        return -0.5 * (np.sum((res / sigma)**2) + self.N*np.log(2*np.pi*sigma))


data = np.random.normal(0.5, 1, 10000)

parameters = dict(
    mu=peyote.parameter.Parameter(
        'mu', prior=peyote.prior.Uniform(lower=-1, upper=1)),
    sigma=peyote.parameter.Parameter(
        'sigma', prior=peyote.prior.Uniform(lower=0, upper=10)))

likelihood = Likelihood(data)
sampler = peyote.sampler.Sampler(
    likelihood=likelihood, parameters=parameters)
res = sampler.run()

fig, axes = dyplot.traceplot(res)
fig.tight_layout()
fig.savefig('single_trace')

fig = corner.corner(res.samples, weights=res.weights)
fig.savefig('test')
