from __future__ import absolute_import, division, print_function

import numpy as np

from ..utils import get_progress_bar
from . import Emcee
from .base_sampler import SamplerError


class Ptemcee(Emcee):
    """bilby wrapper ptemcee (https://github.com/willvousden/ptemcee)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `ptemcee.Sampler`, see
    documentation for that class for further help. Under Other Parameters, we
    list commonly used kwargs and the bilby defaults.

    Other Parameters
    ----------------
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
        import ptemcee
        tqdm = get_progress_bar()
        sampler = ptemcee.Sampler(dim=self.ndim, logl=self.log_likelihood,
                                  logp=self.log_prior, **self.sampler_init_kwargs)
        self.pos0 = [[self.get_random_draw_from_prior()
                      for _ in range(self.nwalkers)]
                     for _ in range(self.kwargs['ntemps'])]

        log_likelihood_evaluations = []
        log_prior_evaluations = []
        for pos, logpost, loglike in tqdm(
                sampler.sample(self.pos0, **self.sampler_function_kwargs),
                total=self.nsteps):
            log_likelihood_evaluations.append(loglike)
            log_prior_evaluations.append(logpost - loglike)
            pass

        self.calculate_autocorrelation(sampler.chain.reshape((-1, self.ndim)))
        self.result.sampler_output = np.nan
        self.print_nburn_logging_info()
        self.result.nburn = self.nburn
        if self.result.nburn > self.nsteps:
            raise SamplerError(
                "The run has finished, but the chain is not burned in: "
                "`nburn < nsteps`. Try increasing the number of steps.")
        self.result.samples = sampler.chain[0, :, self.nburn:, :].reshape(
            (-1, self.ndim))
        self.result.log_likelihood_evaluations = np.array(
            log_likelihood_evaluations)[self.nburn:, 0, :].reshape((-1))
        self.result.log_prior_evaluations = np.array(
            log_prior_evaluations)[self.nburn:, 0, :].reshape((-1))
        self.result.betas = sampler.betas
        self.result.log_evidence, self.result.log_evidence_err =\
            sampler.log_evidence_estimate(
                sampler.loglikelihood, self.nburn / self.nsteps)
        self.result.walkers = sampler.chain[0, :, :, :]

        return self.result
