from __future__ import absolute_import, division, print_function

import os
from collections import namedtuple

import numpy as np

from ..utils import (
    logger, get_progress_bar, check_directory_exists_and_if_not_mkdir)
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
    default_kwargs = dict(ntemps=2, nwalkers=500, Tmax=None, betas=None,
                          threads=1, pool=None, a=2.0, loglargs=[], logpargs=[],
                          loglkwargs={}, logpkwargs={}, adaptation_lag=10000,
                          adaptation_time=100, random=None, iterations=100,
                          thin=1, storechain=True, adapt=True, swap_ratios=False,
                          )

    def __init__(self, likelihood, priors, outdir='outdir', label='label',
                 use_ratio=False, plot=False, skip_import_verification=False,
                 nburn=None, burn_in_fraction=0.25, burn_in_act=3, resume=True,
                 **kwargs):
        Emcee.__init__(
            self, likelihood=likelihood, priors=priors, outdir=outdir,
            label=label, use_ratio=use_ratio, plot=plot,
            skip_import_verification=skip_import_verification,
            nburn=nburn, burn_in_fraction=burn_in_fraction,
            burn_in_act=burn_in_act, resume=True, **kwargs)

    @property
    def sampler_function_kwargs(self):
        keys = ['iterations', 'thin', 'storechain', 'adapt', 'swap_ratios']
        return {key: self.kwargs[key] for key in keys}

    @property
    def sampler_init_kwargs(self):
        return {key: value
                for key, value in self.kwargs.items()
                if key not in self.sampler_function_kwargs}

    @property
    def checkpoint_info(self):
        out_dir = os.path.join(self.outdir, 'ptemcee_{}'.format(self.label))
        chain_file = os.path.join(out_dir, 'chain.dat')
        last_pos_file = os.path.join(out_dir, 'last_pos.npy')

        check_directory_exists_and_if_not_mkdir(out_dir)
        if not os.path.isfile(chain_file):
            with open(chain_file, "w") as ff:
                ff.write('walker\t{}\tlog_l\tlog_p\n'.format(
                    '\t'.join(self.search_parameter_keys)))
        template =\
            '{:d}' + '\t{:.9e}' * (len(self.search_parameter_keys) + 2) + '\n'

        CheckpointInfo = namedtuple(
            'CheckpointInfo', ['last_pos_file', 'chain_file', 'template'])

        checkpoint_info = CheckpointInfo(
            last_pos_file=last_pos_file, chain_file=chain_file, template=template)

        return checkpoint_info

    def _draw_pos0_from_prior(self):
        return [[self.get_random_draw_from_prior()
                 for _ in range(self.nwalkers)]
                for _ in range(self.kwargs['ntemps'])]

    @property
    def _old_chain(self):
        try:
            old_chain = self.__old_chain
            n = old_chain.shape[0]
            idx = n - np.mod(n, self.nwalkers)
            return old_chain[:idx]
        except AttributeError:
            return None

    @_old_chain.setter
    def _old_chain(self, old_chain):
        self.__old_chain = old_chain

    @property
    def stored_chain(self):
        return np.genfromtxt(self.checkpoint_info.chain_file, names=True)

    @property
    def stored_samples(self):
        return self.stored_chain[self.search_parameter_keys]

    @property
    def stored_loglike(self):
        return self.stored_chain['log_l']

    @property
    def stored_logprior(self):
        return self.stored_chain['log_p']

    def load_old_chain(self):
        try:
            last_pos = np.load(self.checkpoint_info.last_pos_file)
            self.pos0 = last_pos
            self._old_chain = self.stored_samples
            logger.info(
                'Resuming from {} with {} iterations'.format(
                    self.checkpoint_info.chain_file,
                    self._previous_iterations))
        except Exception:
            logger.info('Unable to resume')
            self._set_pos0()

    def run_sampler(self):
        import ptemcee
        tqdm = get_progress_bar()
        sampler = ptemcee.Sampler(dim=self.ndim, logl=self.log_likelihood,
                                  logp=self.log_prior, **self.sampler_init_kwargs)

        if self.resume:
            self.load_old_chain()
        else:
            self._set_pos0()

        sampler_function_kwargs = self.sampler_function_kwargs
        iterations = sampler_function_kwargs.pop('iterations')
        iterations -= self._previous_iterations

        for pos, logpost, loglike in tqdm(
                sampler.sample(self.pos0, iterations=iterations,
                               **sampler_function_kwargs),
                total=iterations):
            np.save(self.checkpoint_info.last_pos_file, pos)
            with open(self.checkpoint_info.chain_file, "a") as ff:
                loglike = np.squeeze(loglike[:1, :])
                logprior = np.squeeze(logpost[:1, :]) - loglike
                for ii, (point, logl, logp) in enumerate(zip(pos[0, :, :], loglike, logprior)):
                    line = np.concatenate((point, [logl, logp]))
                    ff.write(self.checkpoint_info.template.format(ii, *line))

        self.calculate_autocorrelation(sampler.chain.reshape((-1, self.ndim)))
        self.result.sampler_output = np.nan
        self.print_nburn_logging_info()
        self.result.nburn = self.nburn
        if self.result.nburn > self.nsteps:
            raise SamplerError(
                "The run has finished, but the chain is not burned in: "
                "`nburn < nsteps`. Try increasing the number of steps.")
        walkers = self.stored_samples.view((float, self.ndim))
        walkers = walkers.reshape(self.nwalkers, self.nsteps, self.ndim)
        self.result.walkers = walkers
        self.result.samples = walkers[:, self.nburn:, :].reshape((-1, self.ndim))
        n_samples = self.nwalkers * self.nburn
        self.result.log_likelihood_evaluations = self.stored_loglike[n_samples:]
        self.result.log_prior_evaluations = self.stored_logprior[n_samples:]
        self.result.betas = sampler.betas
        self.result.log_evidence, self.result.log_evidence_err =\
            sampler.log_evidence_estimate(
                sampler.loglikelihood, self.nburn / self.nsteps)

        return self.result
