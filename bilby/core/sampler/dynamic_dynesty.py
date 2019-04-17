from __future__ import absolute_import

import os
import dill as pickle
import signal

import numpy as np
from pandas import DataFrame

from ..utils import logger, check_directory_exists_and_if_not_mkdir
from .base_sampler import Sampler
from .dynesty import Dynesty


class DynamicDynesty(Dynesty):
    """
    bilby wrapper of `dynesty.DynamicNestedSampler`
    (https://dynesty.readthedocs.io/en/latest/)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `dynesty.DynamicNestedSampler`, see
    documentation for that class for further help. Under Other Parameter below,
    we list commonly all kwargs and the bilby defaults.

    Parameters
    ----------
    likelihood: likelihood.Likelihood
        A  object with a log_l method
    priors: bilby.core.prior.PriorDict, dict
        Priors to be used in the search.
        This has attributes for each parameter to be sampled.
    outdir: str, optional
        Name of the output directory
    label: str, optional
        Naming scheme of the output files
    use_ratio: bool, optional
        Switch to set whether or not you want to use the log-likelihood ratio
        or just the log-likelihood
    plot: bool, optional
        Switch to set whether or not you want to create traceplots
    skip_import_verification: bool
        Skips the check if the sampler is installed if true. This is
        only advisable for testing environments

    Other Parameters
    ----------------
    npoints: int, (250)
        The number of live points, note this can also equivalently be given as
        one of [nlive, nlives, n_live_points]
    bound: {'none', 'single', 'multi', 'balls', 'cubes'}, ('multi')
        Method used to select new points
    sample: {'unif', 'rwalk', 'slice', 'rslice', 'hslice'}, ('rwalk')
        Method used to sample uniformly within the likelihood constraints,
        conditioned on the provided bounds
    walks: int
        Number of walks taken if using `sample='rwalk'`, defaults to `ndim * 5`
    verbose: Bool
        If true, print information information about the convergence during
    check_point: bool,
        If true, use check pointing.
    check_point_delta_t: float (600)
        The approximate checkpoint period (in seconds). Should the run be
        interrupted, it can be resumed from the last checkpoint. Set to
        `None` to turn-off check pointing
    n_check_point: int, optional (None)
        The number of steps to take before check pointing (override
        check_point_delta_t).
    resume: bool
        If true, resume run from checkpoint (if available)
    """
    default_kwargs = dict(bound='multi', sample='rwalk',
                          verbose=True,
                          check_point_delta_t=600,
                          first_update=None,
                          npdim=None, rstate=None, queue_size=None, pool=None,
                          use_pool=None,
                          logl_args=None, logl_kwargs=None,
                          ptform_args=None, ptform_kwargs=None,
                          enlarge=None, bootstrap=None, vol_dec=0.5, vol_check=2.0,
                          facc=0.5, slices=5,
                          walks=None, update_interval=0.6,
                          nlive_init=500, maxiter_init=None, maxcall_init=None,
                          dlogz_init=0.01, logl_max_init=np.inf, nlive_batch=500,
                          wt_function=None, wt_kwargs=None, maxiter_batch=None,
                          maxcall_batch=None, maxiter=None, maxcall=None,
                          maxbatch=None, stop_function=None, stop_kwargs=None,
                          use_stop=True, save_bounds=True,
                          print_progress=True, print_func=None, live_points=None,
                          )

    def __init__(self, likelihood, priors, outdir='outdir', label='label', use_ratio=False, plot=False,
                 skip_import_verification=False, check_point=True, n_check_point=None, check_point_delta_t=600,
                 resume=True, **kwargs):
        Dynesty.__init__(self, likelihood=likelihood, priors=priors, outdir=outdir, label=label,
                         use_ratio=use_ratio, plot=plot,
                         skip_import_verification=skip_import_verification,
                         **kwargs)
        self.n_check_point = n_check_point
        self.check_point = check_point
        self.resume = resume
        if self.n_check_point is None:
            # If the log_likelihood_eval_time is not calculable then
            # check_point is set to False.
            if np.isnan(self._log_likelihood_eval_time):
                self.check_point = False
            n_check_point_raw = (check_point_delta_t / self._log_likelihood_eval_time)
            n_check_point_rnd = int(float("{:1.0g}".format(n_check_point_raw)))
            self.n_check_point = n_check_point_rnd

        self.resume_file = '{}/{}_resume.pickle'.format(self.outdir, self.label)

        signal.signal(signal.SIGTERM, self.write_current_state_and_exit)
        signal.signal(signal.SIGINT, self.write_current_state_and_exit)

    @property
    def external_sampler_name(self):
        return 'dynesty'

    @property
    def sampler_function_kwargs(self):
        keys = ['nlive_init', 'maxiter_init', 'maxcall_init', 'dlogz_init',
                'logl_max_init', 'nlive_batch', 'wt_function', 'wt_kwargs',
                'maxiter_batch', 'maxcall_batch', 'maxiter', 'maxcall',
                'maxbatch', 'stop_function', 'stop_kwargs', 'use_stop',
                'save_bounds', 'print_progress', 'print_func', 'live_points']
        return {key: self.kwargs[key] for key in keys}

    def run_sampler(self):
        import dynesty
        self.sampler = dynesty.DynamicNestedSampler(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.sampler_init_kwargs)

        if self.check_point:
            out = self._run_external_sampler_with_checkpointing()
        else:
            out = self._run_external_sampler_without_checkpointing()

        # Flushes the output to force a line break
        if self.kwargs["verbose"]:
            print("")

        # self.result.sampler_output = out
        weights = np.exp(out['logwt'] - out['logz'][-1])
        nested_samples = DataFrame(
            out.samples, columns=self.search_parameter_keys)
        nested_samples['weights'] = weights
        nested_samples['log_likelihood'] = out.logl

        self.result.samples = dynesty.utils.resample_equal(out.samples, weights)
        self.result.nested_samples = nested_samples
        self.result.log_likelihood_evaluations = self.reorder_loglikelihoods(
            unsorted_loglikelihoods=out.logl, unsorted_samples=out.samples,
            sorted_samples=self.result.samples)
        self.result.log_evidence = out.logz[-1]
        self.result.log_evidence_err = out.logzerr[-1]

        if self.plot:
            self.generate_trace_plots(out)

        return self.result

    def _run_external_sampler_with_checkpointing(self):
        logger.debug("Running sampler with checkpointing")
        if self.resume:
            resume = self.read_saved_state(continuing=True)
            if resume:
                logger.info('Resuming from previous run.')

        old_ncall = self.sampler.ncall
        sampler_kwargs = self.sampler_function_kwargs.copy()
        sampler_kwargs['maxcall'] = self.n_check_point
        while True:
            sampler_kwargs['maxcall'] += self.n_check_point
            self.sampler.run_nested(**sampler_kwargs)
            if self.sampler.ncall == old_ncall:
                break
            old_ncall = self.sampler.ncall

            self.write_current_state()

        self._remove_checkpoint()
        return self.sampler.results

    def write_current_state(self):
        """
        """
        check_directory_exists_and_if_not_mkdir(self.outdir)
        with open(self.resume_file, 'wb') as file:
            pickle.dump(self, file)

    def read_saved_state(self, continuing=False):
        """
        """

        logger.debug("Reading resume file {}".format(self.resume_file))
        if os.path.isfile(self.resume_file):
            with open(self.resume_file, 'rb') as file:
                self = pickle.load(file)
        else:
            logger.debug(
                "Failed to read resume file {}".format(self.resume_file))
            return False

    def _verify_kwargs_against_default_kwargs(self):
        Sampler._verify_kwargs_against_default_kwargs(self)
