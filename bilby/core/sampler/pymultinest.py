from __future__ import absolute_import

import os

import numpy as np

from ..utils import check_directory_exists_and_if_not_mkdir
from ..utils import logger
from .base_sampler import NestedSampler


class Pymultinest(NestedSampler):
    """
    bilby wrapper of pymultinest
    (https://github.com/JohannesBuchner/PyMultiNest)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `pymultinest.run`, see documentation
    for that class for further help. Under Other Parameters, we list commonly
    used kwargs and the bilby defaults.

    Other Parameters
    ----------------
    npoints: int
        The number of live points, note this can also equivalently be given as
        one of [nlive, nlives, n_live_points]
    importance_nested_sampling: bool, (False)
        If true, use importance nested sampling
    sampling_efficiency: float or {'parameter', 'model'}, ('parameter')
        Defines the sampling efficiency
    verbose: Bool
        If true, print information information about the convergence during
    resume: bool
        If true, resume run from checkpoint (if available)

    """
    default_kwargs = dict(importance_nested_sampling=False, resume=True,
                          verbose=True, sampling_efficiency='parameter',
                          n_live_points=500, n_params=None,
                          n_clustering_params=None, wrapped_params=None,
                          multimodal=True, const_efficiency_mode=False,
                          evidence_tolerance=0.5,
                          n_iter_before_update=100, null_log_evidence=-1e90,
                          max_modes=100, mode_tolerance=-1e90,
                          outputfiles_basename=None, seed=-1,
                          context=0, write_output=True, log_zero=-1e100,
                          max_iter=0, init_MPI=False, dump_callback=None)

    def __init__(self, likelihood, priors, outdir='outdir', label='label', use_ratio=False, plot=False,
                 skip_import_verification=False, **kwargs):
        NestedSampler.__init__(self, likelihood=likelihood, priors=priors, outdir=outdir, label=label,
                               use_ratio=use_ratio, plot=plot,
                               skip_import_verification=skip_import_verification,
                               **kwargs)
        self._apply_multinest_boundaries()

    def _translate_kwargs(self, kwargs):
        if 'n_live_points' not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['n_live_points'] = kwargs.pop(equiv)

    def _verify_kwargs_against_default_kwargs(self):
        """
        Test the length of the directory where multinest will write the output.

        This is an issue with MultiNest that we can't solve here.
        https://github.com/JohannesBuchner/PyMultiNest/issues/115
        """
        if not self.kwargs['outputfiles_basename']:
            self.kwargs['outputfiles_basename'] = \
                '{}/pm_{}/'.format(self.outdir, self.label)
        if self.kwargs['outputfiles_basename'].endswith('/') is False:
            self.kwargs['outputfiles_basename'] = '{}/'.format(
                self.kwargs['outputfiles_basename'])
        if len(self.kwargs['outputfiles_basename']) > (100 - 22):
            logger.warning(
                'The length of {} exceeds 78 characters. '
                ' Post-processing will fail because the file names will be cut'
                ' off. Please choose a shorter "outdir" or "label".'
                .format(self.kwargs['outputfiles_basename']))
        check_directory_exists_and_if_not_mkdir(
            self.kwargs['outputfiles_basename'])
        NestedSampler._verify_kwargs_against_default_kwargs(self)

    def _apply_multinest_boundaries(self):
        if self.kwargs['wrapped_params'] is None:
            self.kwargs['wrapped_params'] = []
            for param, value in self.priors.items():
                if value.boundary == 'periodic':
                    self.kwargs['wrapped_params'].append(1)
                else:
                    self.kwargs['wrapped_params'].append(0)

    def run_sampler(self):
        import pymultinest
        self._verify_kwargs_against_default_kwargs()
        out = pymultinest.solve(
            LogLikelihood=self.log_likelihood, Prior=self.prior_transform,
            n_dims=self.ndim, **self.kwargs)

        post_equal_weights = os.path.join(
            self.kwargs['outputfiles_basename'], 'post_equal_weights.dat')
        post_equal_weights_data = np.loadtxt(post_equal_weights)
        self.result.log_likelihood_evaluations = post_equal_weights_data[:, -1]
        self.result.sampler_output = out
        self.result.samples = post_equal_weights_data[:, :-1]
        self.result.log_evidence = out['logZ']
        self.result.log_evidence_err = out['logZerr']
        self.calc_likelihood_count()
        self.result.outputfiles_basename = self.kwargs['outputfiles_basename']
        return self.result
