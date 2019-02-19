from __future__ import absolute_import, print_function

import os

import numpy as np
from pandas import DataFrame
from distutils.version import LooseVersion

from ..utils import (
    logger, get_progress_bar, check_directory_exists_and_if_not_mkdir)
from .base_sampler import MCMCSampler, SamplerError


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
                          postargs=None, pool=None, live_dangerously=False,
                          runtime_sortingfn=None, lnprob0=None, rstate0=None,
                          blobs0=None, iterations=100, thin=1, storechain=True,
                          mh_proposal=None)

    def __init__(self, likelihood, priors, outdir='outdir', label='label',
                 use_ratio=False, plot=False, skip_import_verification=False,
                 pos0=None, nburn=None, burn_in_fraction=0.25, resume=True,
                 burn_in_act=3, **kwargs):
        import emcee
        if LooseVersion(emcee.__version__) > LooseVersion('2.2.1'):
            self.prerelease = True
        else:
            self.prerelease = False
        MCMCSampler.__init__(
            self, likelihood=likelihood, priors=priors, outdir=outdir,
            label=label, use_ratio=use_ratio, plot=plot,
            skip_import_verification=skip_import_verification, **kwargs)
        self.resume = resume
        self.pos0 = pos0
        self.nburn = nburn
        self.burn_in_fraction = burn_in_fraction
        self.burn_in_act = burn_in_act
        self._old_chain = None

    def _translate_kwargs(self, kwargs):
        if 'nwalkers' not in kwargs:
            for equiv in self.nwalkers_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['nwalkers'] = kwargs.pop(equiv)
        if 'iterations' not in kwargs:
            if 'nsteps' in kwargs:
                kwargs['iterations'] = kwargs.pop('nsteps')
        if 'threads' in kwargs:
            if kwargs['threads'] != 1:
                logger.warning("The 'threads' argument cannot be used for "
                               "parallelisation. This run will proceed "
                               "without parallelisation, but consider the use "
                               "of an appropriate Pool object passed to the "
                               "'pool' keyword.")
                kwargs['threads'] = 1

    @property
    def sampler_function_kwargs(self):
        import emcee
        keys = ['lnprob0', 'rstate0', 'blobs0', 'iterations', 'thin', 'storechain', 'mh_proposal']

        # updated function keywords for emcee > v2.2.1
        updatekeys = {'p0': 'initial_state',
                      'lnprob0': 'log_prob0',
                      'storechain': 'store'}

        function_kwargs = {key: self.kwargs[key] for key in keys if key in self.kwargs}
        function_kwargs['p0'] = self.pos0

        if self.prerelease:
            if function_kwargs['mh_proposal'] is not None:
                logger.warning("The 'mh_proposal' option is no longer used "
                               "in emcee v{}, and will be ignored.".format(emcee.__version__))
            del function_kwargs['mh_proposal']

            for key in updatekeys:
                if updatekeys[key] not in function_kwargs:
                    function_kwargs[updatekeys[key]] = function_kwargs.pop(key)
                else:
                    del function_kwargs[key]

        return function_kwargs

    @property
    def sampler_init_kwargs(self):
        init_kwargs = {key: value
                       for key, value in self.kwargs.items()
                       if key not in self.sampler_function_kwargs}

        init_kwargs['lnpostfn'] = self.lnpostfn
        init_kwargs['dim'] = self.ndim

        # updated init keywords for emcee > v2.2.1
        updatekeys = {'dim': 'ndim',
                      'lnpostfn': 'log_prob_fn'}

        if self.prerelease:
            for key in updatekeys:
                if key in init_kwargs:
                    init_kwargs[updatekeys[key]] = init_kwargs.pop(key)

            oldfunckeys = ['p0', 'lnprob0', 'storechain', 'mh_proposal']
            for key in oldfunckeys:
                if key in init_kwargs:
                    del init_kwargs[key]

        return init_kwargs

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

    def __getstate__(self):
        # In order to be picklable with dill, we need to discard the pool
        # object before trying.
        d = self.__dict__
        d["_Sampler__kwargs"]["pool"] = None
        return d

    def run_sampler(self):
        import emcee
        tqdm = get_progress_bar()
        sampler = emcee.EnsembleSampler(**self.sampler_init_kwargs)
        out_dir = os.path.join(self.outdir, 'emcee_{}'.format(self.label))
        out_file = os.path.join(out_dir, 'chain.dat')

        if self.resume:
            self.load_old_chain(out_file)
        else:
            self._set_pos0()

        check_directory_exists_and_if_not_mkdir(out_dir)
        if not os.path.isfile(out_file):
            with open(out_file, "w") as ff:
                ff.write('walker\t{}\tlog_l'.format(
                    '\t'.join(self.search_parameter_keys)))
        template =\
            '{:d}' + '\t{:.9e}' * (len(self.search_parameter_keys) + 2) + '\n'

        for sample in tqdm(sampler.sample(**self.sampler_function_kwargs),
                           total=self.nsteps):
            if self.prerelease:
                points = np.hstack([sample.coords, sample.blobs])
            else:
                points = np.hstack([sample[0], np.array(sample[3])])
            with open(out_file, "a") as ff:
                for ii, point in enumerate(points):
                    ff.write(template.format(ii, *point))

        self.result.sampler_output = np.nan
        blobs_flat = np.array(sampler.blobs).reshape((-1, 2))
        log_likelihoods, log_priors = blobs_flat.T
        if self._old_chain is not None:
            chain = np.vstack([self._old_chain[:, :-2],
                               sampler.chain.reshape((-1, self.ndim))])
            log_ls = np.hstack([self._old_chain[:, -2], log_likelihoods])
            log_ps = np.hstack([self._old_chain[:, -1], log_priors])
            self.nsteps = chain.shape[0] // self.nwalkers
        else:
            chain = sampler.chain.reshape((-1, self.ndim))
            log_ls = log_likelihoods
            log_ps = log_priors
        self.calculate_autocorrelation(chain)
        self.print_nburn_logging_info()
        self.result.nburn = self.nburn
        n_samples = self.nwalkers * self.nburn
        if self.result.nburn > self.nsteps:
            raise SamplerError(
                "The run has finished, but the chain is not burned in: "
                "`nburn < nsteps`. Try increasing the number of steps.")
        self.result.samples = chain[n_samples:, :]
        self.result.log_likelihood_evaluations = log_ls[n_samples:]
        self.result.log_prior_evaluations = log_ps[n_samples:]
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

    def load_old_chain(self, file_name=None):
        if file_name is None:
            out_dir = os.path.join(self.outdir, 'emcee_{}'.format(self.label))
            file_name = os.path.join(out_dir, 'chain.dat')
        if os.path.isfile(file_name):
            old_chain = np.genfromtxt(file_name, skip_header=1)
            self.pos0 = [np.squeeze(old_chain[-(self.nwalkers - ii), 1:-2])
                         for ii in range(self.nwalkers)]
            self._old_chain = old_chain[:-self.nwalkers + 1, 1:]
            logger.info('Resuming from {}'.format(os.path.abspath(file_name)))
        else:
            logger.warning('Failed to resume. {} not found.'.format(file_name))
            self._set_pos0()

    def lnpostfn(self, theta):
        log_prior = self.log_prior(theta)
        if np.isinf(log_prior):
            return -np.inf, [np.nan, np.nan]
        else:
            log_likelihood = self.log_likelihood(theta)
            return log_likelihood + log_prior, [log_likelihood, log_prior]
