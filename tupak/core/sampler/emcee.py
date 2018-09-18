import numpy as np
from pandas import DataFrame
from ..utils import logger, get_progress_bar
from .base_sampler import Sampler


class Emcee(Sampler):
    """tupak wrapper emcee (https://github.com/dfm/emcee)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `emcee.EnsembleSampler`, see
    documentation for that class for further help. Under Keyword Arguments, we
    list commonly used kwargs and the tupak defaults.

    Keyword Arguments
    -----------------
    nwalkers: int, (100)
        The number of walkers
    nsteps: int, (100)
        The number of steps
    nburn: int (None)
        If given, the fixed number of steps to discard as burn-in. Else,
        nburn is estimated from the autocorrelation time
    burn_in_fraction: float, (0.25)
        The fraction of steps to discard as burn-in in the event that the
        autocorrelation time cannot be calculated
    burn_in_act: float
        The number of autocorrelation times to discard as burn-in
    a: float (2)
        The proposal scale factor


    """

    def _run_external_sampler(self):
        self.nwalkers = self.kwargs.get('nwalkers', 100)
        self.nsteps = self.kwargs.get('nsteps', 100)
        self.nburn = self.kwargs.get('nburn', None)
        self.burn_in_fraction = self.kwargs.get('burn_in_fraction', 0.25)
        self.burn_in_act = self.kwargs.get('burn_in_act', 3)
        a = self.kwargs.get('a', 2)
        emcee = self.external_sampler
        tqdm = get_progress_bar(self.kwargs.pop('tqdm', 'tqdm'))

        sampler = emcee.EnsembleSampler(
            nwalkers=self.nwalkers, dim=self.ndim, lnpostfn=self.lnpostfn,
            a=a)

        if 'pos0' in self.kwargs:
            logger.debug("Using given initial positions for walkers")
            pos0 = self.kwargs['pos0']
            if isinstance(pos0, DataFrame):
                pos0 = pos0[self.search_parameter_keys].values
            elif type(pos0) in (list, np.ndarray):
                pos0 = np.squeeze(self.kwargs['pos0'])

            if pos0.shape != (self.nwalkers, self.ndim):
                raise ValueError(
                    'Input pos0 should be of shape ndim, nwalkers')
            logger.debug("Checking input pos0")
            for draw in pos0:
                self.check_draw(draw)
        else:
            logger.debug("Generating initial walker positions from prior")
            pos0 = [self.get_random_draw_from_prior()
                    for _ in range(self.nwalkers)]

        for _ in tqdm(sampler.sample(pos0, iterations=self.nsteps),
                      total=self.nsteps):
            pass

        self.result.sampler_output = np.nan
        self.calculate_autocorrelation(sampler.chain.reshape((-1, self.ndim)))
        self.setup_nburn()
        self.result.nburn = self.nburn
        self.result.samples = sampler.chain[:, self.nburn:, :].reshape(
            (-1, self.ndim))
        self.result.walkers = sampler.chain
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
        return self.result

    def lnpostfn(self, theta):
        p = self.log_prior(theta)
        if np.isinf(p):
            return -np.inf
        else:
            return self.log_likelihood(theta) + p
