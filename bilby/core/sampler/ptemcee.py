from __future__ import absolute_import

import numpy as np

from ..utils import get_progress_bar, logger
from . import Emcee

try:
    import ptemcee
except ImportError:
    logger.debug('PTEmcee is not installed on this system, you will '
                 'not be able to use the PTEmcee sampler')


class Ptemcee(Emcee):
    """bilby wrapper ptemcee (https://github.com/willvousden/ptemcee)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `ptemcee.Sampler`, see
    documentation for that class for further help. Under Keyword Arguments, we
    list commonly used kwargs and the bilby defaults.

    Keyword Arguments
    -----------------
    nwalkers: int, (100)
        The number of walkers
    nsteps: int, (100)
        The number of steps to take
    nburn: int (50)
        The fixed number of steps to discard as burn-in
    ntemps: int (2)
        The number of temperatures used by ptemcee

    """
    default_kwargs = dict(ntemps=2, nwalkers=500,
                          Tmax=None, betas=None,
                          threads=1, pool=None, a=2.0,
                          loglargs=[], logpargs=[],
                          loglkwargs={}, logpkwargs={},
                          adaptation_lag=10000, adaptation_time=100,
                          random=None, iterations=100, thin=1,
                          storechain=True, adapt=True,
                          swap_ratios=False,
                          )

    def __init__(self, likelihood, priors, outdir='outdir', label='label', use_ratio=False, plot=False,
                 skip_import_verification=False, nburn=None, burn_in_fraction=0.25,
                 burn_in_act=3, **kwargs):
        Emcee.__init__(self, likelihood=likelihood, priors=priors, outdir=outdir, label=label,
                       use_ratio=use_ratio, plot=plot, skip_import_verification=skip_import_verification,
                       nburn=nburn, burn_in_fraction=burn_in_fraction, burn_in_act=burn_in_act, **kwargs)

    @property
    def sampler_function_kwargs(self):
        keys = ['iterations', 'thin', 'storechain', 'adapt', 'swap_ratios']
        return {key: self.kwargs[key] for key in keys}

    @property
    def sampler_init_kwargs(self):
        return {key: value
                for key, value in self.kwargs.items()
                if key not in self.sampler_function_kwargs}

    def run_sampler(self):
        tqdm = get_progress_bar()

        sampler = ptemcee.Sampler(dim=self.ndim, logl=self.log_likelihood,
                                  logp=self.log_prior, **self.sampler_init_kwargs)
        self.pos0 = [[self.get_random_draw_from_prior()
                      for _ in range(self.nwalkers)]
                     for _ in range(self.kwargs['ntemps'])]

        for _ in tqdm(
                sampler.sample(self.pos0, **self.sampler_function_kwargs),
                total=self.nsteps):
            pass

        self.result.nburn = self.nburn
        self.result.sampler_output = np.nan
        self.result.samples = sampler.chain[0, :, self.nburn:, :].reshape(
            (-1, self.ndim))
        self.result.walkers = sampler.chain[0, :, :, :]
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan

        logger.info("Max autocorr time = {}"
                    .format(np.max(sampler.get_autocorr_time())))
        logger.info("Tswap frac = {}"
                    .format(sampler.tswap_acceptance_fraction))
        return self.result
