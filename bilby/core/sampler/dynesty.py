from __future__ import absolute_import

import datetime
import os
import sys
import pickle
import signal

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from ..utils import logger, check_directory_exists_and_if_not_mkdir
from .base_sampler import Sampler, NestedSampler


class Dynesty(NestedSampler):
    """
    bilby wrapper of `dynesty.NestedSampler`
    (https://dynesty.readthedocs.io/en/latest/)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `dynesty.NestedSampler`, see
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
    npoints: int, (1000)
        The number of live points, note this can also equivalently be given as
        one of [nlive, nlives, n_live_points]
    bound: {'none', 'single', 'multi', 'balls', 'cubes'}, ('multi')
        Method used to select new points
    sample: {'unif', 'rwalk', 'slice', 'rslice', 'hslice'}, ('rwalk')
        Method used to sample uniformly within the likelihood constraints,
        conditioned on the provided bounds
    walks: int
        Number of walks taken if using `sample='rwalk'`, defaults to `ndim * 30`
    dlogz: float, (0.1)
        Stopping criteria
    verbose: Bool
        If true, print information information about the convergence during
    check_point: bool,
        If true, use check pointing.
    check_point_plot: bool,
        If true, generate a trace plot along with the check-point
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
                          verbose=True, periodic=None,
                          check_point_delta_t=600, nlive=1000,
                          first_update=None, walks=None,
                          npdim=None, rstate=None, queue_size=None, pool=None,
                          use_pool=None, live_points=None,
                          logl_args=None, logl_kwargs=None,
                          ptform_args=None, ptform_kwargs=None,
                          enlarge=None, bootstrap=None, vol_dec=0.5, vol_check=2.0,
                          facc=0.5, slices=5,
                          update_interval=None, print_func=None,
                          dlogz=0.1, maxiter=None, maxcall=None,
                          logl_max=np.inf, add_live=True, print_progress=True,
                          save_bounds=False)

    def __init__(self, likelihood, priors, outdir='outdir', label='label',
                 use_ratio=False, plot=False, skip_import_verification=False,
                 check_point=True, check_point_plot=True, n_check_point=None,
                 check_point_delta_t=600, resume=True, **kwargs):
        NestedSampler.__init__(self, likelihood=likelihood, priors=priors,
                               outdir=outdir, label=label, use_ratio=use_ratio,
                               plot=plot,
                               skip_import_verification=skip_import_verification,
                               **kwargs)
        self.n_check_point = n_check_point
        self.check_point = check_point
        self.check_point_plot = check_point_plot
        self.resume = resume
        self._periodic = list()
        self._reflective = list()
        self._apply_dynesty_boundaries()
        if self.n_check_point is None:
            # If the log_likelihood_eval_time is not calculable then
            # check_point is set to False.
            if np.isnan(self._log_likelihood_eval_time):
                self.check_point = False
            n_check_point_raw = (check_point_delta_t / self._log_likelihood_eval_time)
            n_check_point_rnd = int(float("{:1.0g}".format(n_check_point_raw)))
            self.n_check_point = n_check_point_rnd

        logger.info("Checkpoint every n_check_point = {}".format(self.n_check_point))

        self.resume_file = '{}/{}_resume.pickle'.format(self.outdir, self.label)
        self.sampling_time = datetime.timedelta()

        signal.signal(signal.SIGTERM, self.write_current_state_and_exit)
        signal.signal(signal.SIGINT, self.write_current_state_and_exit)
        signal.signal(signal.SIGALRM, self.write_current_state_and_exit)

    def __getstate__(self):
        """ For pickle: remove external_sampler, which can be an unpicklable "module" """
        state = self.__dict__.copy()
        del state['external_sampler']
        return state

    @property
    def sampler_function_kwargs(self):
        keys = ['dlogz', 'print_progress', 'print_func', 'maxiter',
                'maxcall', 'logl_max', 'add_live', 'save_bounds']
        return {key: self.kwargs[key] for key in keys}

    @property
    def sampler_init_kwargs(self):
        return {key: value
                for key, value in self.kwargs.items()
                if key not in self.sampler_function_kwargs}

    def _translate_kwargs(self, kwargs):
        if 'nlive' not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['nlive'] = kwargs.pop(equiv)
        if 'print_progress' not in kwargs:
            if 'verbose' in kwargs:
                kwargs['print_progress'] = kwargs.pop('verbose')
        if 'walks' not in kwargs:
            for equiv in self.walks_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['walks'] = kwargs.pop(equiv)

    def _verify_kwargs_against_default_kwargs(self):
        if not self.kwargs['walks']:
            self.kwargs['walks'] = self.ndim * 30
        if not self.kwargs['update_interval']:
            self.kwargs['update_interval'] = int(0.6 * self.kwargs['nlive'])
        if not self.kwargs['print_func']:
            self.kwargs['print_func'] = self._print_func
        Sampler._verify_kwargs_against_default_kwargs(self)

    def _print_func(self, results, niter, ncall, dlogz, *args, **kwargs):
        """ Replacing status update for dynesty.result.print_func """

        # Extract results at the current iteration.
        (worst, ustar, vstar, loglstar, logvol, logwt,
         logz, logzvar, h, nc, worst_it, boundidx, bounditer,
         eff, delta_logz) = results

        # Adjusting outputs for printing.
        if delta_logz > 1e6:
            delta_logz = np.inf
        if 0. <= logzvar <= 1e6:
            logzerr = np.sqrt(logzvar)
        else:
            logzerr = np.nan
        if logz <= -1e6:
            logz = -np.inf
        if loglstar <= -1e6:
            loglstar = -np.inf

        if self.use_ratio:
            key = 'logz ratio'
        else:
            key = 'logz'

        # Constructing output.
        raw_string = "\r {}| {}={:6.3f} +/- {:6.3f} | dlogz: {:6.3f} > {:6.3f}"
        print_str = raw_string.format(
            niter, key, logz, logzerr, delta_logz, dlogz)

        # Printing.
        sys.stdout.write(print_str)
        sys.stdout.flush()

    def _apply_dynesty_boundaries(self):
        if self.kwargs['periodic'] is None:
            logger.debug("Setting periodic boundaries for keys:")
            self.kwargs['periodic'] = []
            self._periodic = list()
            self._reflective = list()
            for ii, key in enumerate(self.search_parameter_keys):
                if self.priors[key].boundary in ['periodic', 'reflective']:
                    self.kwargs['periodic'].append(ii)
                    logger.debug("  {}".format(key))
                    if self.priors[key].boundary == 'periodic':
                        self._periodic.append(ii)
                    else:
                        self._reflective.append(ii)

    def run_sampler(self):
        import dynesty
        if self.kwargs['live_points'] is None:
            self.kwargs['live_points'] = (
                self.get_initial_points_from_prior(
                    self.kwargs['nlive']))
        self.sampler = dynesty.NestedSampler(
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

        check_directory_exists_and_if_not_mkdir(self.outdir)
        dynesty_result = "{}/{}_dynesty.pickle".format(self.outdir, self.label)
        with open(dynesty_result, 'wb') as file:
            pickle.dump(out, file)

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
        self.calc_likelihood_count()
        self.result.log_evidence = out.logz[-1]
        self.result.log_evidence_err = out.logzerr[-1]
        self.result.sampling_time = self.sampling_time

        if self.plot:
            self.generate_trace_plots(out)

        return self.result

    def _run_external_sampler_without_checkpointing(self):
        logger.debug("Running sampler without checkpointing")
        self.sampler.run_nested(**self.sampler_function_kwargs)
        return self.sampler.results

    def _run_external_sampler_with_checkpointing(self):
        logger.debug("Running sampler with checkpointing")
        if self.resume:
            resume = self.read_saved_state(continuing=True)
            if resume:
                logger.info('Resuming from previous run.')

        old_ncall = self.sampler.ncall
        sampler_kwargs = self.sampler_function_kwargs.copy()
        sampler_kwargs['maxcall'] = self.n_check_point
        sampler_kwargs['add_live'] = False
        self.start_time = datetime.datetime.now()
        while True:
            sampler_kwargs['maxcall'] += self.n_check_point
            self.sampler.run_nested(**sampler_kwargs)
            if self.sampler.ncall == old_ncall:
                break
            old_ncall = self.sampler.ncall

            self.write_current_state()

        sampler_kwargs['add_live'] = True
        self.sampler.run_nested(**sampler_kwargs)
        return self.sampler.results

    def _remove_checkpoint(self):
        """Remove checkpointed state"""
        if os.path.isfile(self.resume_file):
            os.remove(self.resume_file)

    def read_saved_state(self, continuing=False):
        """
        Read a saved state of the sampler to disk.

        The required information to reconstruct the state of the run is read
        from a pickle file.
        This currently adds the whole chain to the sampler.
        We then remove the old checkpoint and write all unnecessary items back
        to disk.
        FIXME: Load only the necessary quantities, rather than read/write?

        Parameters
        ----------
        sampler: `dynesty.NestedSampler`
            NestedSampler instance to reconstruct from the saved state.
        continuing: bool
            Whether the run is continuing or terminating, if True, the loaded
            state is mostly written back to disk.
        """

        if os.path.isfile(self.resume_file):
            logger.info("Reading resume file {}".format(self.resume_file))
            try:
                with open(self.resume_file, 'rb') as file:
                    saved = pickle.load(file)
                logger.info(
                    "Succesfuly read resume file {}".format(self.resume_file))
            except EOFError as e:
                logger.warning("Resume file reading failed with error {}".format(e))
                return False

            self.sampler.saved_u = list(saved['unit_cube_samples'])
            self.sampler.saved_v = list(saved['physical_samples'])
            self.sampler.saved_logl = list(saved['sample_likelihoods'])
            self.sampler.saved_logvol = list(saved['sample_log_volume'])
            self.sampler.saved_logwt = list(saved['sample_log_weights'])
            self.sampler.saved_logz = list(saved['cumulative_log_evidence'])
            self.sampler.saved_logzvar = list(saved['cumulative_log_evidence_error'])
            self.sampler.saved_id = list(saved['id'])
            self.sampler.saved_it = list(saved['it'])
            self.sampler.saved_nc = list(saved['nc'])
            self.sampler.saved_boundidx = list(saved['boundidx'])
            self.sampler.saved_bounditer = list(saved['bounditer'])
            self.sampler.saved_scale = list(saved['scale'])
            self.sampler.saved_h = list(saved['cumulative_information'])
            self.sampler.ncall = saved['ncall']
            self.sampler.live_logl = list(saved['live_logl'])
            self.sampler.it = saved['iteration'] + 1
            self.sampler.live_u = saved['live_u']
            self.sampler.live_v = saved['live_v']
            self.sampler.nlive = saved['nlive']
            self.sampler.live_bound = saved['live_bound']
            self.sampler.live_it = saved['live_it']
            self.sampler.added_live = saved['added_live']
            self.sampling_time += datetime.timedelta(seconds=saved['sampling_time'])
            return True

        else:
            logger.debug(
                "No resume file {}".format(self.resume_file))
            return False

    def write_current_state_and_exit(self, signum=None, frame=None):
        logger.warning("Run terminated with signal {}".format(signum))
        self.write_current_state(plot=False)
        sys.exit(130)

    def write_current_state(self, plot=True):
        """
        Write the current state of the sampler to disk.

        The required information to reconstruct the state of the run are written
        to an hdf5 file.
        All but the most recent removed live point in the chain are removed from
        the sampler to reduce memory usage.
        This means it is necessary to not append the first live point to the
        file if updating a previous checkpoint.

        Parameters
        ----------
        sampler: `dynesty.NestedSampler`
            NestedSampler to write to disk.
        """
        check_directory_exists_and_if_not_mkdir(self.outdir)
        logger.info("Writing checkpoint file {}".format(self.resume_file))

        end_time = datetime.datetime.now()
        self.sampling_time += end_time - self.start_time
        self.start_time = end_time

        current_state = dict(
            unit_cube_samples=self.sampler.saved_u,
            physical_samples=self.sampler.saved_v,
            sample_likelihoods=self.sampler.saved_logl,
            sample_log_volume=self.sampler.saved_logvol,
            sample_log_weights=self.sampler.saved_logwt,
            cumulative_log_evidence=self.sampler.saved_logz,
            cumulative_log_evidence_error=self.sampler.saved_logzvar,
            cumulative_information=self.sampler.saved_h,
            id=self.sampler.saved_id,
            it=self.sampler.saved_it,
            nc=self.sampler.saved_nc,
            boundidx=self.sampler.saved_boundidx,
            bounditer=self.sampler.saved_bounditer,
            scale=self.sampler.saved_scale,
            sampling_time=self.sampling_time.total_seconds()
        )

        current_state.update(
            ncall=self.sampler.ncall, live_logl=self.sampler.live_logl,
            iteration=self.sampler.it - 1, live_u=self.sampler.live_u,
            live_v=self.sampler.live_v, nlive=self.sampler.nlive,
            live_bound=self.sampler.live_bound, live_it=self.sampler.live_it,
            added_live=self.sampler.added_live
        )

        try:
            weights = np.exp(current_state['sample_log_weights'] -
                             current_state['cumulative_log_evidence'][-1])
            from dynesty.utils import resample_equal

            current_state['posterior'] = resample_equal(
                np.array(current_state['physical_samples']), weights)
        except ValueError:
            logger.debug("Unable to create posterior")

        with open(self.resume_file, 'wb') as file:
            pickle.dump(current_state, file)

        if plot and self.check_point_plot:
            import dynesty.plotting as dyplot
            labels = [label.replace('_', ' ') for label in self.search_parameter_keys]
            filename = "{}/{}_checkpoint_trace.png".format(self.outdir, self.label)
            try:
                fig = dyplot.traceplot(self.sampler.results, labels=labels)[0]
                fig.tight_layout()
                fig.savefig(filename)
                plt.close('all')
            except (RuntimeError, np.linalg.linalg.LinAlgError) as e:
                logger.warning(e)
                logger.warning('Failed to create dynesty state plot at checkpoint')

    def generate_trace_plots(self, dynesty_results):
        check_directory_exists_and_if_not_mkdir(self.outdir)
        filename = '{}/{}_trace.png'.format(self.outdir, self.label)
        logger.debug("Writing trace plot to {}".format(filename))
        from dynesty import plotting as dyplot
        fig, axes = dyplot.traceplot(dynesty_results,
                                     labels=self.result.parameter_labels)
        fig.tight_layout()
        fig.savefig(filename)

    def _run_test(self):
        import dynesty
        import pandas as pd
        self.sampler = dynesty.NestedSampler(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.sampler_init_kwargs)
        sampler_kwargs = self.sampler_function_kwargs.copy()
        sampler_kwargs['maxiter'] = 2

        self.sampler.run_nested(**sampler_kwargs)
        self.result.samples = pd.DataFrame(
            self.priors.sample(100))[self.search_parameter_keys].values
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
        return self.result

    def prior_transform(self, theta):
        """ Prior transform method that is passed into the external sampler.
        cube we map this back to [0, 1].

        Parameters
        ----------
        theta: list
            List of sampled values on a unit interval

        Returns
        -------
        list: Properly rescaled sampled values

        Notes
        -----
        Since dynesty allows periodic parameters to wander outside the unit,
        here we transform them depending of if they should be periodic or
        reflective. For reflective boundaries, theta < 0 you shift to |theta|
        and when theta > 1 you return 2 - theta. For periodic boundaries,
        if theta < 0, you shift to 1-|theta| and when theta > 1 you shift to
        |theta| - 1 (i.e. wrap around).

        """
        theta[self._periodic] = np.mod(theta[self._periodic], 1)
        theta_ref = theta[self._reflective]
        theta[self._reflective] = np.minimum(
            np.maximum(theta_ref, abs(theta_ref)), 2 - theta_ref)
        return self.priors.rescale(self._search_parameter_keys, theta)
