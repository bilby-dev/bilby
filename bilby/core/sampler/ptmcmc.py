from __future__ import absolute_import, print_function

import glob
import shutil

import numpy as np

from .base_sampler import MCMCSampler, SamplerNotInstalledError
from ..utils import logger


class PTMCMCSampler(MCMCSampler):
    """bilby wrapper of PTMCMC (https://github.com/jellis18/PTMCMCSampler/)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `PTMCMCSampler.PTMCMCSampler`, see
    documentation for that class for further help. Under Other Parameters, we
    list commonly used kwargs and the bilby defaults.

    Other Parameters
    ----------------
    Niter: int (2*10**4 + 1)
        The number of mcmc steps
    burn: int (5 * 10**3)
        If given, the fixed number of steps to discard as burn-in
    thin: int (1)
        The number of steps before saving the sample to the chain
    custom_proposals: dict (None)
        Add dictionary of proposals to the array of proposals, this must be in
        the form of a dictionary with the name of the proposal, then a list
        containing the jump function and the weight e.g {'name' : [function ,
        weight]} see
        (https://github.com/rgreen1995/PTMCMCSampler/blob/master/examples/simple.ipynb)
        and
        (http://jellis18.github.io/PTMCMCSampler/PTMCMCSampler.html#ptmcmcsampler-ptmcmcsampler-module)
        for examples and more info.  logl_grad: func (None) Gradient of
        likelihood  if known (default = None) logp_grad: func (None) Gradient
        of prior if known (default = None) verbose: bool (True) Update current
        run-status to the screen

    """

    default_kwargs = {'p0': None, 'Niter': 2 * 10**4 + 1, 'neff': 10**4,
                      'burn': 5 * 10**3, 'verbose': True,
                      'ladder': None, 'Tmin': 1, 'Tmax': None, 'Tskip': 100,
                      'isave': 1000, 'thin': 1, 'covUpdate': 1000,
                      'SCAMweight': 1, 'AMweight': 1, 'DEweight': 1,
                      'HMCweight': 0, 'MALAweight': 0, 'NUTSweight': 0,
                      'HMCstepsize': 0.1, 'HMCsteps': 300,
                      'groups': None, 'custom_proposals': None,
                      'loglargs': {}, 'loglkwargs': {}, 'logpargs': {},
                      'logpkwargs': {}, 'logl_grad': None, 'logp_grad': None,
                      'outDir': None}

    def __init__(self, likelihood, priors, outdir='outdir', label='label',
                 use_ratio=False, plot=False, skip_import_verification=False,
                 pos0=None, burn_in_fraction=0.25, **kwargs):

        MCMCSampler.__init__(self, likelihood=likelihood, priors=priors,
                             outdir=outdir, label=label, use_ratio=use_ratio,
                             plot=plot,
                             skip_import_verification=skip_import_verification,
                             **kwargs)

        self.p0 = self.get_random_draw_from_prior()
        self.likelihood = likelihood
        self.priors = priors

    def _verify_external_sampler(self):
        # PTMCMC is imported with Caps so need to overwrite the parent function
        # which forces `__name__.lower()
        external_sampler_name = self.__class__.__name__
        try:
            self.external_sampler = __import__(external_sampler_name)
        except (ImportError, SystemExit):
            raise SamplerNotInstalledError(
                "Sampler {} is not installed on this system".format(external_sampler_name))

    def _translate_kwargs(self, kwargs):
        if 'Niter' not in kwargs:
            for equiv in self.nwalkers_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['Niter'] = kwargs.pop(equiv)
        if 'burn' not in kwargs:
            for equiv in self.nburn_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['burn'] = kwargs.pop(equiv)

    @property
    def custom_proposals(self):
        return self.kwargs['custom_proposals']

    @property
    def sampler_init_kwargs(self):
        keys = ['groups',
                'loglargs',
                'logp_grad',
                'logpkwargs',
                'loglkwargs',
                'logl_grad',
                'logpargs',
                'outDir',
                'verbose']
        init_kwargs = {key: self.kwargs[key] for key in keys}
        if init_kwargs['outDir'] is None:
            init_kwargs['outDir'] = '{}/ptmcmc_temp_{}/'.format(self.outdir, self.label)
        return init_kwargs

    @property
    def sampler_function_kwargs(self):
        keys = ['Niter',
                'neff',
                'Tmin',
                'HMCweight',
                'covUpdate',
                'SCAMweight',
                'ladder',
                'burn',
                'NUTSweight',
                'AMweight',
                'MALAweight',
                'thin',
                'HMCstepsize',
                'isave',
                'Tskip',
                'HMCsteps',
                'Tmax',
                'DEweight']
        sampler_kwargs = {key: self.kwargs[key] for key in keys}
        return sampler_kwargs

    @staticmethod
    def _import_external_sampler():
        from PTMCMCSampler import PTMCMCSampler
        return PTMCMCSampler

    def run_sampler(self):
        PTMCMCSampler = self._import_external_sampler()
        sampler = PTMCMCSampler.PTSampler(ndim=self.ndim, logp=self.log_prior,
                                          logl=self.log_likelihood, cov=np.eye(self.ndim),
                                          **self.sampler_init_kwargs)
        if self.custom_proposals is not None:
            for proposal in self.custom_proposals:
                logger.info('Adding {} to proposals with weight {}'.format(
                    proposal, self.custom_proposals[proposal][1]))
                sampler.addProposalToCycle(self.custom_proposals[proposal][0],
                                           self.custom_proposals[proposal][1])
        sampler.sample(p0=self.p0, **self.sampler_function_kwargs)
        samples, meta, loglike = self.__read_in_data()

        self.calc_likelihood_count()
        self.result.nburn = self.sampler_function_kwargs['burn']
        self.result.samples = samples[self.result.nburn:]
        self.meta_data['sampler_meta'] = meta
        self.result.log_likelihood_evaluations = loglike[self.result.nburn:]
        self.result.sampler_output = np.nan
        self.result.walkers = np.nan
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
        return self.result

    def __read_in_data(self):
        """ Read the data stored by PTMCMC to disk """
        temp_outDir = self.sampler_init_kwargs['outDir']
        try:
            data = np.loadtxt('{}chain_1.txt'.format(temp_outDir))
        except OSError:
            data = np.loadtxt('{}chain_1.0.txt'.format(temp_outDir))
        jumpfiles = glob.glob('{}/*jump.txt'.format(temp_outDir))
        jumps = map(np.loadtxt, jumpfiles)
        samples = data[:, :-4]
        loglike = data[:, -3]

        jump_accept = {}
        for ct, j in enumerate(jumps):
            label = jumpfiles[ct].split('/')[-1].split('_jump.txt')[0]
            jump_accept[label] = j
        PT_swap = {'swap_accept': data[:, -1]}
        tot_accept = {'tot_accept': data[:, -2]}
        log_post = {'log_post': data[:, -4]}
        meta = {}
        meta['tot_accept'] = tot_accept
        meta['PT_swap'] = PT_swap
        meta['proposals'] = jump_accept
        meta['log_post'] = log_post

        shutil.rmtree(temp_outDir)

        return samples, meta, loglike
