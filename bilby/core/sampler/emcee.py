from __future__ import absolute_import, print_function

import numpy as np
from pandas import DataFrame

from ..utils import logger, get_progress_bar
from .base_sampler import MCMCSampler


class Emcee(MCMCSampler):
    """bilby wrapper emcee (https://github.com/dfm/emcee)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `emcee.EnsembleSampler`, see
    documentation for that class for further help. Under Other Parameters, we
    list commonly used kwargs and the bilby defaults.

    Other Parameters
    ----------------
    nwalkers: int, (100)
        The number of walkers
    nsteps: int, (100)
        The number of steps
    nburn: int (None)
        If given, the fixed number of steps to discard as burn-in. These will
        be discarded from the total number of steps set by `nsteps` and
        therefore the value must be greater than `nsteps`. Else, nburn is
        estimated from the autocorrelation time
    burn_in_fraction: float, (0.25)
        The fraction of steps to discard as burn-in in the event that the
        autocorrelation time cannot be calculated
    burn_in_act: float
        The number of autocorrelation times to discard as burn-in
    a: float (2)
        The proposal scale factor


    """

    default_kwargs = dict(nwalkers=500, a=2, args=[], kwargs={},
                          postargs=None, threads=1, pool=None, live_dangerously=False,
                          runtime_sortingfn=None, lnprob0=None, rstate0=None,
                          blobs0=None, iterations=100, thin=1, storechain=True, mh_proposal=None)

    def __init__(self, likelihood, priors, outdir='outdir', label='label', use_ratio=False, plot=False,
                 skip_import_verification=False, pos0=None, nburn=None, burn_in_fraction=0.25,
                 burn_in_act=3, **kwargs):
        MCMCSampler.__init__(self, likelihood=likelihood, priors=priors, outdir=outdir, label=label,
                             use_ratio=use_ratio, plot=plot,
                             skip_import_verification=skip_import_verification,
                             **kwargs)
        self.pos0 = pos0
        self.nburn = nburn
        self.burn_in_fraction = burn_in_fraction
        self.burn_in_act = burn_in_act

    def _translate_kwargs(self, kwargs):
        if 'nwalkers' not in kwargs:
            for equiv in self.nwalkers_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['nwalkers'] = kwargs.pop(equiv)
        if 'iterations' not in kwargs:
            if 'nsteps' in kwargs:
                kwargs['iterations'] = kwargs.pop('nsteps')

    @property
    def sampler_function_kwargs(self):
        keys = ['lnprob0', 'rstate0', 'blobs0', 'iterations', 'thin', 'storechain', 'mh_proposal']
        return {key: self.kwargs[key] for key in keys}

    @property
    def sampler_init_kwargs(self):
        return {key: value
                for key, value in self.kwargs.items()
                if key not in self.sampler_function_kwargs}

    @property
    def nburn(self):
        if type(self.__nburn) in [float, int]:
            return int(self.__nburn)
        elif self.result.max_autocorrelation_time is None:
            return int(self.burn_in_fraction * self.nsteps)
        else:
            return int(self.burn_in_act * self.result.max_autocorrelation_time)

    @nburn.setter
    def nburn(self, nburn):
        if isinstance(nburn, (float, int)):
            if nburn > self.kwargs['iterations'] - 1:
                raise ValueError('Number of burn-in samples must be smaller '
                                 'than the total number of iterations')

        self.__nburn = nburn

    @property
    def nwalkers(self):
        return self.kwargs['nwalkers']

    @property
    def nsteps(self):
        return self.kwargs['iterations']

    @nsteps.setter
    def nsteps(self, nsteps):
        self.kwargs['iterations'] = nsteps

    def run_sampler(self):
        import emcee
        tqdm = get_progress_bar()
        sampler = emcee.EnsembleSampler(dim=self.ndim, lnpostfn=self.lnpostfn, **self.sampler_init_kwargs)
        self._set_pos0()
        for _ in tqdm(sampler.sample(p0=self.pos0, **self.sampler_function_kwargs),
                      total=self.nsteps):
            pass
        self.result.sampler_output = np.nan
        self.calculate_autocorrelation(sampler.chain.reshape((-1, self.ndim)))
        self.print_nburn_logging_info()
        self.result.nburn = self.nburn
        self.result.samples = sampler.chain[:, self.nburn:, :].reshape((-1, self.ndim))
        self.result.walkers = sampler.chain
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
        return self.result

    def _set_pos0(self):
        if self.pos0 is not None:
            logger.debug("Using given initial positions for walkers")
            if isinstance(self.pos0, DataFrame):
                self.pos0 = self.pos0[self.search_parameter_keys].values
            elif type(self.pos0) in (list, np.ndarray):
                self.pos0 = np.squeeze(self.kwargs['pos0'])

            if self.pos0.shape != (self.nwalkers, self.ndim):
                raise ValueError(
                    'Input pos0 should be of shape ndim, nwalkers')
            logger.debug("Checking input pos0")
            for draw in self.pos0:
                self.check_draw(draw)
        else:
            logger.debug("Generating initial walker positions from prior")
            self.pos0 = [self.get_random_draw_from_prior()
                         for _ in range(self.nwalkers)]

    def lnpostfn(self, theta):
        p = self.log_prior(theta)
        if np.isinf(p):
            return -np.inf
        else:
            return self.log_likelihood(theta) + p
