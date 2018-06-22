#!/bin/python
"""

"""
from __future__ import division
import tupak
import numpy as np
import matplotlib.pyplot as plt

# A few simple setup steps
label = 'occam_factor'
outdir = 'outdir'
tupak.utils.check_directory_exists_and_if_not_mkdir(outdir)

sigma = 1

N = 100
time = np.linspace(0, 1, N)
coeffs = [1, 2, 3]
data = np.polyval(coeffs, time) + np.random.normal(0, sigma, N)

fig, ax = plt.subplots()
ax.plot(time, data, 'o', label='data', color='C0')
ax.plot(time, np.polyval(coeffs, time), label='true signal', color='C1')
ax.set_xlabel('time')
ax.set_ylabel('y')
ax.legend()
fig.savefig('{}/{}_data.png'.format(outdir, label))


class Polynomial(tupak.Likelihood):
    def __init__(self, x, y, sigma, n):
        """
        A Gaussian likelihood for polynomial of degree `n`.

        Parameters
        ----------
        x, y: array_like
            The data to analyse.
        sigma: float
            The standard deviation of the noise.
        n: int
            The degree of the polynomial to fit.
        """
        self.x = x
        self.y = y
        self.sigma = sigma
        self.N = len(x)
        self.n = n
        self.keys = ['c{}'.format(k) for k in range(n)]
        self.parameters = {k: None for k in self.keys}

    def polynomial(self, x, parameters):
        coeffs = [parameters[k] for k in self.keys]
        return np.polyval(coeffs, x)

    def log_likelihood(self):
        res = self.y - self.polynomial(self.x, self.parameters)
        return -0.5 * (np.sum((res / self.sigma)**2)
                       + self.N*np.log(2*np.pi*self.sigma**2))


def fit(n):
    likelihood = Polynomial(time, data, sigma, n)
    priors = {}
    for i in range(n):
        k = 'c{}'.format(i)
        priors[k] = tupak.core.prior.Uniform(0, 10, k)

    result = tupak.run_sampler(
        likelihood=likelihood, priors=priors, npoints=100, outdir=outdir,
        label=label)
    return result.log_evidence, result.log_evidence_err, np.log(result.occam_factor(priors))


fig, ax = plt.subplots()

log_evidences = []
log_evidences_err = []
log_occam_factors = []
ns = range(1, 10)
for l in ns:
    e, e_err, o = fit(l)
    log_evidences.append(e)
    log_evidences_err.append(e_err)
    log_occam_factors.append(o)

ax.errorbar(ns, log_evidences-np.max(log_evidences), yerr=log_evidences_err,
            fmt='-o', color='C0')
ax.plot(ns, log_occam_factors-np.max(log_occam_factors),
        '-o', color='C1', alpha=0.5)

fig.savefig('{}/{}_test'.format(outdir, label))
