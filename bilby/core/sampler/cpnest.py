from __future__ import absolute_import
import numpy as np
from pandas import DataFrame
from ..utils import logger
from .base_sampler import NestedSampler


class Cpnest(NestedSampler):
    """ bilby wrapper of cpnest (https://github.com/johnveitch/cpnest)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `cpnest.CPNest`, see documentation
    for that class for further help. Under Keyword Arguments, we list commonly
    used kwargs and the bilby defaults.

    Keyword Arguments
    -----------------
    npoints: int
        The number of live points, note this can also equivalently be given as
        one of [nlive, nlives, n_live_points]
    seed: int (1234)
        Initialised random seed
    Nthreads: int, (1)
        Number of threads to use
    maxmcmc: int (1000)
        The maximum number of MCMC steps to take
    verbose: Bool
        If true, print information information about the convergence during

    """
    default_kwargs = dict(verbose=1, Nthreads=1, Nlive=500, maxmcmc=1000,
                          Poolsize=100, seed=None, balance_samplers=True)

    def _translate_kwargs(self, kwargs):
        if 'Nlive' not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['Nlive'] = kwargs.pop(equiv)
        if 'seed' not in kwargs:
            logger.warning('No seed provided, cpnest will use 1234.')

    def run_sampler(self):
        from cpnest import model as cpmodel, CPNest

        class Model(cpmodel.Model):
            """ A wrapper class to pass our log_likelihood into cpnest """

            def __init__(self, names, bounds):
                self.names = names
                self.bounds = bounds
                self._check_bounds()

            @staticmethod
            def log_likelihood(x, **kwargs):
                theta = [x[n] for n in self.search_parameter_keys]
                return self.log_likelihood(theta)

            @staticmethod
            def log_prior(x, **kwargs):
                theta = [x[n] for n in self.search_parameter_keys]
                return self.log_prior(theta)

            def _check_bounds(self):
                for bound in self.bounds:
                    if not all(np.isfinite(bound)):
                        raise ValueError(
                            'CPNest requires priors to have finite bounds.')

        bounds = [[self.priors[key].minimum, self.priors[key].maximum]
                  for key in self.search_parameter_keys]
        model = Model(self.search_parameter_keys, bounds)
        out = CPNest(model, output=self.outdir, **self.kwargs)
        out.run()

        if self.plot:
            out.plot()

        # Since the output is not just samples, but log_likelihood as well,
        # we turn this into a dataframe here. The index [0] here may be wrong
        self.result.posterior = DataFrame(out.posterior_samples[0])
        self.result.log_evidence = out.NS.state.logZ
        self.result.log_evidence_err = np.nan
        return self.result
