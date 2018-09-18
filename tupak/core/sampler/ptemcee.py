import numpy as np
from ..utils import get_progress_bar, logger
from . import Emcee


class Ptemcee(Emcee):
    """tupak wrapper ptemcee (https://github.com/willvousden/ptemcee)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `ptemcee.Sampler`, see
    documentation for that class for further help. Under Keyword Arguments, we
    list commonly used kwargs and the tupak defaults.

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

    def _run_external_sampler(self):
        self.ntemps = self.kwargs.pop('ntemps', 2)
        self.nwalkers = self.kwargs.pop('nwalkers', 100)
        self.nsteps = self.kwargs.pop('nsteps', 100)
        self.nburn = self.kwargs.pop('nburn', 50)
        ptemcee = self.external_sampler
        tqdm = get_progress_bar(self.kwargs.pop('tqdm', 'tqdm'))

        sampler = ptemcee.Sampler(
            ntemps=self.ntemps, nwalkers=self.nwalkers, dim=self.ndim,
            logl=self.log_likelihood, logp=self.log_prior,
            **self.kwargs)
        pos0 = [[self.get_random_draw_from_prior()
                 for _ in range(self.nwalkers)]
                for _ in range(self.ntemps)]

        for _ in tqdm(
                sampler.sample(pos0, iterations=self.nsteps, adapt=True),
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
