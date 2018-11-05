from __future__ import absolute_import

from .base_sampler import NestedSampler
import numpy as np

import PyPolyChord
from PyPolyChord.settings import PolyChordSettings
from PyPolyChord.priors import UniformPrior

n_dims = 3
n_derived = 1


def likelihood(theta):
    """ Simple Gaussian Likelihood"""

    sigma = 0.1
    nDims = len(theta)

    r2 = sum(theta**2)

    log_l = -np.log(2*np.pi*sigma*sigma)*nDims/2.0
    log_l += -r2/2/sigma/sigma

    return log_l, [r2]


def prior(hypercube):
    """ Uniform prior from [-1,1]^D. """
    return UniformPrior(-1, 1)(hypercube)


def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])


try:
    import getdist.plots
    import matplotlib.pyplot as plt
    posterior = output.posterior
    g = getdist.plots.getSubplotPlotter()
    g.triangle_plot(posterior, filled=True)
    plt.show()
except ImportError:
    print("Install matplotlib and getdist for plotting examples")


class BBPolychord(NestedSampler):

    default_kwargs = dict()

    def run_sampler(self):
        import pypolychord
        self._verify_kwargs_against_default_kwargs()

        out = pypolychord.run()

        self.result.sampler_output = out
        self.result.samples = out['samples']
        self.result.log_evidence = out['logZ']
        self.result.log_evidence_err = out['logZerr']
        self.result.outputfiles_basename = self.kwargs['outputfiles_basename']
        return self.result
