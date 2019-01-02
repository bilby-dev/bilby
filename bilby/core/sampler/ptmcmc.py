from __future__ import absolute_import, print_function

import numpy as np
from pandas import DataFrame

from ..utils import logger, get_progress_bar
from .base_sampler import MCMCSampler


class PTMCMCSampler(MCMCSampler):
    """bilby wrapper PTMCMC (https://github.com/jellis18/PTMCMCSampler/)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `PTMCMCSampler.PTMCMCSampler`, see
    documentation for that class for further help. Under Other Parameters, we
    list commonly used kwargs and the bilby defaults.

    Other Parameters
    ----------------
    ndim - number of dimensions in problem
    logl - log-likelihood function
    logp - log prior function (must be normalized for evidence evaluation)
    cov - Initial covariance matrix of model parameters for jump proposals
    loglargs - any additional arguments (apart from the parameter vector) for log likelihood
    loglkwargs - any additional keyword arguments (apart from the parameter vector) for log likelihood
    logpargs - any additional arguments (apart from the parameter vector) for log like prior
    logpkwargs - any additional keyword arguments (apart from the parameter vector) for log prior
    logl_grad - Gradient of likelihood  if known (default = None)
    logp_grad - Gradient of prior if known (default = None)
    outDir - Full path to output directory for chain files (default = ./chains)
    verbose - Update current run-status to the screen (default=False)
    """

    default_kwargs = {'p0' : None , 'Niter' : 10**4, 'ladder' : None
                    ,'Tmin' :1 , 'Tmax' : None, 'Tskip' : 100 , 'isave' : 1000
                    ,'NUTSweight' : 20 , 'HMCweight' : 20 , 'MALAweight':0
                    ,'burn':10000 , 'HMCstepsize' :0.1 , 'HMCsteps':300
                    ,'neff' : 10**4 , 'burn' :10**4 , 'thin':1
                    ,'covUpdate' : 500, 'SCAMweight':20, 'AMweight':20, 'DEweight':50
                    , 'cov': np.eye(1)
                    , 'loglargs' : {} , 'loglkwargs' : {} , 'logpargs' : {}, 'logpkwargs': {}
                    , 'logl_grad' : None , 'logp_grad'  : None, 'outDir' : './ptmcmc_test' , 'verbose': False}

    def __init__(self, likelihood, priors, outdir='outdir', label='label', use_ratio=False, plot=False,
                 skip_import_verification=False, pos0=None, nburn=None, burn_in_fraction=0.25, **kwargs):

        MCMCSampler.__init__(self, likelihood=likelihood, priors=priors, outdir=outdir, label=label,
                             use_ratio=use_ratio, plot=plot,
                             skip_import_verification=skip_import_verification,
                             **kwargs)

        self.p0 = self.get_random_draw_from_prior()
        self.likelihood = likelihood
        self.priors = priors

    @property
    def kwargs(self):
        """ Ensures that proper keyword arguments are used for the PTMCMC sampler.

        Returns
        -------
        dict: Keyword arguments used for the PTMCMC

        """
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self.__kwargs = self.default_kwargs.copy()
        self.__kwargs.update(kwargs)
        self._verify_kwargs_against_default_kwargs()

    # def _translate_kwargs(self, kwargs):
    #     if 'nwalkers' not in kwargs:
    #         for equiv in self.nwalkers_equiv_kwargs:
    #             if equiv in kwargs:
    #                 kwargs['nwalkers'] = kwargs.pop(equiv)
    #     if 'iterations' not in kwargs:
    #         if 'nsteps' in kwargs:
    #             kwargs['iterations'] = kwargs.pop('nsteps')



    @property
    def sampler_init_kwargs(self):
        keys = [
                'loglargs',
                'logp_grad',
                'logpkwargs',
                'cov',
                'loglkwargs',
                'logl_grad',
                'logpargs',
                'outDir',
                'verbose']
        init_kwargs = {key: self.kwargs[key] for key in keys}
        return init_kwargs

    @property
    def sampler_function_kwargs(self):
        keys = [
                'Niter',
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

    @property
    def nsteps(self):
        return self.kwargs['Niter']

    @nsteps.setter
    def nsteps(self, nsteps):
        self.kwargs['Niter'] = nsteps

    @staticmethod
    def _import_external_sampler():
        from PTMCMCSampler import PTMCMCSampler
        #import acor
        #from mpi4py import MPI
        #return MPI, PTMCMCSampler
        return PTMCMCSampler

    def run_sampler(self):
        #MPI , PTMCMCSampler = self._import_external_sampler()
        PTMCMCSampler = self._import_external_sampler()
        #tqdm = get_progress_bar()
        #sampler = emcee.EnsembleSampler(dim=self.ndim, lnpostfn=self.lnpostfn, **self.sampler_init_kwargs)
        init_kwargs = self.sampler_init_kwargs
        sampler_kwargs = self.sampler_function_kwargs
        sampler = PTMCMCSampler.PTSampler(ndim=self.ndim, logp = self.log_prior,
                                          logl = self.log_likelihood,  **init_kwargs)
        tqdm = get_progress_bar()
        print(self.nsteps)
        sampler.sample(p0 = self.p0 , **sampler_kwargs)
    

        self.result.sampler_output = np.nan
        self.calculate_autocorrelation(sampler.chain.reshape((-1, self.ndim)))
        self.print_nburn_logging_info()
        self.result.nburn = self.nburn
        self.result.samples = sampler.chain[:, self.nburn:, :].reshape((-1, self.ndim))
        self.result.walkers = sampler.chain
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
        return self.result
