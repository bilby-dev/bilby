from __future__ import absolute_import

import numpy as np
from pandas import DataFrame

from .base_sampler import NestedSampler
from ..utils import logger, check_directory_exists_and_if_not_mkdir


class Cpnest(NestedSampler):
    """ bilby wrapper of cpnest (https://github.com/johnveitch/cpnest)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `cpnest.CPNest`, see documentation
    for that class for further help. Under Other Parameters, we list commonly
    used kwargs and the bilby defaults.

    Other Parameters
    ----------------
    nlive: int
        The number of live points, note this can also equivalently be given as
        one of [npoints, nlives, n_live_points]
    seed: int (1234)
        Initialised random seed
    nthreads: int, (1)
        Number of threads to use
    maxmcmc: int (1000)
        The maximum number of MCMC steps to take
    verbose: Bool (True)
        If true, print information information about the convergence during
    resume: Bool (False)
        Whether or not to resume from a previous run
    output: str
        Where to write the CPNest, by default this is
        {self.outdir}/cpnest_{self.label}/

    """
    default_kwargs = dict(verbose=1, nthreads=1, nlive=500, maxmcmc=1000,
                          seed=None, poolsize=100, nhamiltonian=0, resume=False,
                          output=None)

    def _translate_kwargs(self, kwargs):
        if 'nlive' not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['nlive'] = kwargs.pop(equiv)
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
        out = CPNest(model, **self.kwargs)
        out.run()

        if self.plot:
            out.plot()

        self.result.posterior = DataFrame(out.posterior_samples)
        self.result.posterior.rename(columns=dict(
            logL='log_likelihood', logPrior='log_prior'), inplace=True)
        self.result.log_evidence = out.NS.state.logZ
        self.result.log_evidence_err = np.nan
        return self.result

    def _verify_kwargs_against_default_kwargs(self):
        """
        Set the directory where the output will be written.
        """
        if not self.kwargs['output']:
            self.kwargs['output'] = \
                '{}/cpnest_{}/'.format(self.outdir, self.label)
        if self.kwargs['output'].endswith('/') is False:
            self.kwargs['output'] = '{}/'.format(self.kwargs['output'])
        check_directory_exists_and_if_not_mkdir(self.kwargs['output'])
        NestedSampler._verify_kwargs_against_default_kwargs(self)
