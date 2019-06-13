from __future__ import absolute_import, print_function

from collections import namedtuple
import os
import signal
from shutil import copyfile
import sys

import numpy as np
from pandas import DataFrame
from distutils.version import LooseVersion
import dill as pickle

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

    default_kwargs = dict(
        nwalkers=500, a=2, args=[], kwargs={}, postargs=None, pool=None,
        live_dangerously=False, runtime_sortingfn=None, lnprob0=None,
        rstate0=None, blobs0=None, iterations=100, thin=1, storechain=True,
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

        signal.signal(signal.SIGTERM, self.checkpoint_and_exit)
        signal.signal(signal.SIGINT, self.checkpoint_and_exit)

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

    def lnpostfn(self, theta):
        log_prior = self.log_prior(theta)
        if np.isinf(log_prior):
            return -np.inf, [np.nan, np.nan]
        else:
            log_likelihood = self.log_likelihood(theta)
            return log_likelihood + log_prior, [log_likelihood, log_prior]

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

    @property
    def stored_chain(self):
        """ Read the stored zero-temperature chain data in from disk """
        return np.genfromtxt(self.checkpoint_info.chain_file, names=True)

    @property
    def stored_samples(self):
        """ Returns the samples stored on disk """
        return self.stored_chain[self.search_parameter_keys]

    @property
    def stored_loglike(self):
        """ Returns the log-likelihood stored on disk """
        return self.stored_chain['log_l']

    @property
    def stored_logprior(self):
        """ Returns the log-prior stored on disk """
        return self.stored_chain['log_p']

    def _init_chain_file(self):
        with open(self.checkpoint_info.chain_file, "w+") as ff:
            ff.write('walker\t{}\tlog_l\tlog_p\n'.format(
                '\t'.join(self.search_parameter_keys)))

    @property
    def checkpoint_info(self):
        """ Defines various things related to checkpointing and storing data

        Returns
        -------
        checkpoint_info: named_tuple
            An object with attributes `sampler_file`, `chain_file`, and
            `chain_template`. The first two give paths to where the sampler and
            chain data is stored, the last a formatted-str-template with which
            to write the chain data to disk

        """
        out_dir = os.path.join(
            self.outdir, '{}_{}'.format(self.__class__.__name__.lower(),
                                        self.label))
        check_directory_exists_and_if_not_mkdir(out_dir)

        chain_file = os.path.join(out_dir, 'chain.dat')
        sampler_file = os.path.join(out_dir, 'sampler.pickle')
        chain_template =\
            '{:d}' + '\t{:.9e}' * (len(self.search_parameter_keys) + 2) + '\n'

        CheckpointInfo = namedtuple(
            'CheckpointInfo', ['sampler_file', 'chain_file', 'chain_template'])

        checkpoint_info = CheckpointInfo(
            sampler_file=sampler_file, chain_file=chain_file,
            chain_template=chain_template)

        return checkpoint_info

    @property
    def sampler_chain(self):
        nsteps = self._previous_iterations
        return self.sampler.chain[:, :nsteps, :]

    def checkpoint(self):
        """ Writes a pickle file of the sampler to disk using dill """
        logger.info("Checkpointing sampler to file {}"
                    .format(self.checkpoint_info.sampler_file))
        with open(self.checkpoint_info.sampler_file, 'wb') as f:
            # Overwrites the stored sampler chain with one that is truncated
            # to only the completed steps
            self.sampler._chain = self.sampler_chain
            pickle.dump(self._sampler, f)

    def checkpoint_and_exit(self, signum, frame):
        logger.info("Recieved signal {}".format(signum))
        self.checkpoint()
        sys.exit()

    def _initialise_sampler(self):
        import emcee
        self._sampler = emcee.EnsembleSampler(**self.sampler_init_kwargs)
        self._init_chain_file()

    @property
    def sampler(self):
        """ Returns the ptemcee sampler object

        If, alrady initialized, returns the stored _sampler value. Otherwise,
        first checks if there is a pickle file from which to load. If there is
        not, then initialize the sampler and set the initial random draw

        """
        if hasattr(self, '_sampler'):
            pass
        elif self.resume and os.path.isfile(self.checkpoint_info.sampler_file):
            logger.info("Resuming run from checkpoint file {}"
                        .format(self.checkpoint_info.sampler_file))
            with open(self.checkpoint_info.sampler_file, 'rb') as f:
                self._sampler = pickle.load(f)
            self._set_pos0_for_resume()
        else:
            self._initialise_sampler()
            self._set_pos0()
        return self._sampler

    def write_chains_to_file(self, sample):
        chain_file = self.checkpoint_info.chain_file
        temp_chain_file = chain_file + '.temp'
        if os.path.isfile(chain_file):
            copyfile(chain_file, temp_chain_file)

        if self.prerelease:
            points = np.hstack([sample.coords, sample.blobs])
        else:
            points = np.hstack([sample[0], np.array(sample[3])])
        with open(temp_chain_file, "a") as ff:
            for ii, point in enumerate(points):
                ff.write(self.checkpoint_info.chain_template.format(ii, *point))
        os.rename(temp_chain_file, chain_file)

    @property
    def _previous_iterations(self):
        """ Returns the number of iterations that the sampler has saved

        This is used when loading in a sampler from a pickle file to figure out
        how much of the run has already been completed
        """
        return len(self.sampler.blobs)

    def _draw_pos0_from_prior(self):
        return np.array(
            [self.get_random_draw_from_prior() for _ in range(self.nwalkers)])

    @property
    def _pos0_shape(self):
        return (self.nwalkers, self.ndim)

    def _set_pos0(self):
        if self.pos0 is not None:
            logger.debug("Using given initial positions for walkers")
            if isinstance(self.pos0, DataFrame):
                self.pos0 = self.pos0[self.search_parameter_keys].values
            elif type(self.pos0) in (list, np.ndarray):
                self.pos0 = np.squeeze(self.pos0)

            if self.pos0.shape != self._pos0_shape:
                raise ValueError(
                    'Input pos0 should be of shape ndim, nwalkers')
            logger.debug("Checking input pos0")
            for draw in self.pos0:
                self.check_draw(draw)
        else:
            logger.debug("Generating initial walker positions from prior")
            self.pos0 = self._draw_pos0_from_prior()

    def _set_pos0_for_resume(self):
        self.pos0 = self.sampler.chain[:, -1, :]

    def run_sampler(self):
        tqdm = get_progress_bar()
        sampler_function_kwargs = self.sampler_function_kwargs
        iterations = sampler_function_kwargs.pop('iterations')
        iterations -= self._previous_iterations

        sampler_function_kwargs['p0'] = self.pos0

        # main iteration loop
        for sample in tqdm(
                self.sampler.sample(iterations=iterations, **sampler_function_kwargs),
                total=iterations):
            self.write_chains_to_file(sample)
        self.checkpoint()

        self.result.sampler_output = np.nan
        blobs_flat = np.array(self.sampler.blobs).reshape((-1, 2))
        log_likelihoods, log_priors = blobs_flat.T
        chain = self.sampler.chain.reshape((-1, self.ndim))
        log_ls = log_likelihoods
        log_ps = log_priors
        self.calculate_autocorrelation(chain)
        self.print_nburn_logging_info()
        self.calc_likelihood_count()
        self.result.nburn = self.nburn
        n_samples = self.nwalkers * self.nburn
        if self.result.nburn > self.nsteps:
            raise SamplerError(
                "The run has finished, but the chain is not burned in: "
                "`nburn < nsteps`. Try increasing the number of steps.")
        self.result.samples = chain[n_samples:, :]
        self.result.log_likelihood_evaluations = log_ls[n_samples:]
        self.result.log_prior_evaluations = log_ps[n_samples:]
        self.result.walkers = self.sampler.chain
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
        return self.result
