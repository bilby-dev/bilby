import datetime
import dill
import os
import sys
import pickle
import signal
import time

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from ..utils import (
    logger,
    check_directory_exists_and_if_not_mkdir,
    reflect,
    safe_file_dump,
)
from .base_sampler import Sampler, NestedSampler
from ..result import rejection_sample

from numpy import linalg
from dynesty.utils import unitcheck
import warnings


_likelihood = None
_priors = None
_search_parameter_keys = None
_use_ratio = False


def _initialize_global_variables(
        likelihood, priors, search_parameter_keys, use_ratio
):
    """
    Store a global copy of the likelihood, priors, and search keys for
    multiprocessing.
    """
    global _likelihood
    global _priors
    global _search_parameter_keys
    global _use_ratio
    _likelihood = likelihood
    _priors = priors
    _search_parameter_keys = search_parameter_keys
    _use_ratio = use_ratio


def _prior_transform_wrapper(theta):
    """Wrapper to the prior transformation. Needed for multiprocessing."""
    return _priors.rescale(_search_parameter_keys, theta)


def _log_likelihood_wrapper(theta):
    """Wrapper to the log likelihood. Needed for multiprocessing."""
    if _priors.evaluate_constraints({
        key: theta[ii] for ii, key in enumerate(_search_parameter_keys)
    }):
        params = {key: t for key, t in zip(_search_parameter_keys, theta)}
        _likelihood.parameters.update(params)
        if _use_ratio:
            return _likelihood.log_likelihood_ratio()
        else:
            return _likelihood.log_likelihood()
    else:
        return np.nan_to_num(-np.inf)


class Dynesty(NestedSampler):
    """
    bilby wrapper of `dynesty.NestedSampler`
    (https://dynesty.readthedocs.io/en/latest/)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `dynesty.NestedSampler`, see
    documentation for that class for further help. Under Other Parameter below,
    we list commonly all kwargs and the bilby defaults.

    Parameters
    ==========
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
    ------==========
    npoints: int, (1000)
        The number of live points, note this can also equivalently be given as
        one of [nlive, nlives, n_live_points]
    bound: {'none', 'single', 'multi', 'balls', 'cubes'}, ('multi')
        Method used to select new points
    sample: {'unif', 'rwalk', 'slice', 'rslice', 'hslice'}, ('rwalk')
        Method used to sample uniformly within the likelihood constraints,
        conditioned on the provided bounds
    walks: int
        Number of walks taken if using `sample='rwalk'`, defaults to `ndim * 10`
    dlogz: float, (0.1)
        Stopping criteria
    verbose: Bool
        If true, print information information about the convergence during
    check_point: bool,
        If true, use check pointing.
    check_point_plot: bool,
        If true, generate a trace plot along with the check-point
    check_point_delta_t: float (600)
        The minimum checkpoint period (in seconds). Should the run be
        interrupted, it can be resumed from the last checkpoint.
    n_check_point: int, optional (None)
        The number of steps to take before checking whether to check_point.
    resume: bool
        If true, resume run from checkpoint (if available)
    exit_code: int
        The code which the same exits on if it hasn't finished sampling
    """
    default_kwargs = dict(bound='multi', sample='rwalk',
                          verbose=True, periodic=None, reflective=None,
                          check_point_delta_t=600, nlive=1000,
                          first_update=None, walks=100,
                          npdim=None, rstate=None, queue_size=1, pool=None,
                          use_pool=None, live_points=None,
                          logl_args=None, logl_kwargs=None,
                          ptform_args=None, ptform_kwargs=None,
                          enlarge=1.5, bootstrap=None, vol_dec=0.5, vol_check=8.0,
                          facc=0.2, slices=5,
                          update_interval=None, print_func=None,
                          dlogz=0.1, maxiter=None, maxcall=None,
                          logl_max=np.inf, add_live=True, print_progress=True,
                          save_bounds=False, n_effective=None,
                          maxmcmc=5000, nact=5)

    def __init__(self, likelihood, priors, outdir='outdir', label='label',
                 use_ratio=False, plot=False, skip_import_verification=False,
                 check_point=True, check_point_plot=True, n_check_point=None,
                 check_point_delta_t=600, resume=True, exit_code=130, **kwargs):
        super(Dynesty, self).__init__(likelihood=likelihood, priors=priors,
                                      outdir=outdir, label=label, use_ratio=use_ratio,
                                      plot=plot, skip_import_verification=skip_import_verification,
                                      exit_code=exit_code,
                                      **kwargs)
        self.n_check_point = n_check_point
        self.check_point = check_point
        self.check_point_plot = check_point_plot
        self.resume = resume
        self._periodic = list()
        self._reflective = list()
        self._apply_dynesty_boundaries()

        if self.n_check_point is None:
            self.n_check_point = 1000
        self.check_point_delta_t = check_point_delta_t
        logger.info("Checkpoint every check_point_delta_t = {}s"
                    .format(check_point_delta_t))

        self.resume_file = '{}/{}_resume.pickle'.format(self.outdir, self.label)
        self.sampling_time = datetime.timedelta()

        try:
            signal.signal(signal.SIGTERM, self.write_current_state_and_exit)
            signal.signal(signal.SIGINT, self.write_current_state_and_exit)
            signal.signal(signal.SIGALRM, self.write_current_state_and_exit)
        except AttributeError:
            logger.debug(
                "Setting signal attributes unavailable on this system. "
                "This is likely the case if you are running on a Windows machine"
                " and is no further concern.")

    def __getstate__(self):
        """ For pickle: remove external_sampler, which can be an unpicklable "module" """
        state = self.__dict__.copy()
        if "external_sampler" in state:
            del state['external_sampler']
        return state

    @property
    def sampler_function_kwargs(self):
        keys = ['dlogz', 'print_progress', 'print_func', 'maxiter',
                'maxcall', 'logl_max', 'add_live', 'save_bounds',
                'n_effective']
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
        if "queue_size" not in kwargs:
            for equiv in self.npool_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['queue_size'] = kwargs.pop(equiv)

    def _verify_kwargs_against_default_kwargs(self):
        if not self.kwargs['walks']:
            self.kwargs['walks'] = self.ndim * 10
        if not self.kwargs['update_interval']:
            self.kwargs['update_interval'] = int(0.6 * self.kwargs['nlive'])
        if self.kwargs['print_func'] is None:
            self.kwargs['print_func'] = self._print_func
            self.pbar = tqdm(file=sys.stdout)
        Sampler._verify_kwargs_against_default_kwargs(self)

    def _print_func(self, results, niter, ncall=None, dlogz=None, *args, **kwargs):
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
            key = 'logz-ratio'
        else:
            key = 'logz'

        # Constructing output.
        string = []
        string.append("bound:{:d}".format(bounditer))
        string.append("nc:{:3d}".format(nc))
        string.append("ncall:{:.1e}".format(ncall))
        string.append("eff:{:0.1f}%".format(eff))
        string.append("{}={:0.2f}+/-{:0.2f}".format(key, logz, logzerr))
        string.append("dlogz:{:0.3f}>{:0.2g}".format(delta_logz, dlogz))

        self.pbar.set_postfix_str(" ".join(string), refresh=False)
        self.pbar.update(niter - self.pbar.n)

    def _apply_dynesty_boundaries(self):
        self._periodic = list()
        self._reflective = list()
        for ii, key in enumerate(self.search_parameter_keys):
            if self.priors[key].boundary == 'periodic':
                logger.debug("Setting periodic boundary for {}".format(key))
                self._periodic.append(ii)
            elif self.priors[key].boundary == 'reflective':
                logger.debug("Setting reflective boundary for {}".format(key))
                self._reflective.append(ii)

        # The periodic kwargs passed into dynesty allows the parameters to
        # wander out of the bounds, this includes both periodic and reflective.
        # these are then handled in the prior_transform
        self.kwargs["periodic"] = self._periodic
        self.kwargs["reflective"] = self._reflective

    def _setup_pool(self):
        if self.kwargs["pool"] is not None:
            logger.info("Using user defined pool.")
            self.pool = self.kwargs["pool"]
        elif self.kwargs["queue_size"] > 1:
            logger.info(
                "Setting up multiproccesing pool with {} processes.".format(
                    self.kwargs["queue_size"]
                )
            )
            import multiprocessing
            self.pool = multiprocessing.Pool(
                processes=self.kwargs["queue_size"],
                initializer=_initialize_global_variables,
                initargs=(
                    self.likelihood,
                    self.priors,
                    self._search_parameter_keys,
                    self.use_ratio
                )
            )
        else:
            _initialize_global_variables(
                likelihood=self.likelihood,
                priors=self.priors,
                search_parameter_keys=self._search_parameter_keys,
                use_ratio=self.use_ratio
            )
            self.pool = None
        self.kwargs["pool"] = self.pool

    def _close_pool(self):
        if getattr(self, "pool", None) is not None:
            logger.info("Starting to close worker pool.")
            self.pool.close()
            self.pool.join()
            self.pool = None
            self.kwargs["pool"] = self.pool
            logger.info("Finished closing worker pool.")

    def run_sampler(self):
        import dynesty
        logger.info("Using dynesty version {}".format(dynesty.__version__))

        if self.kwargs.get("sample", "rwalk") == "rwalk":
            logger.info(
                "Using the bilby-implemented rwalk sample method with ACT estimated walks")
            dynesty.dynesty._SAMPLING["rwalk"] = sample_rwalk_bilby
            dynesty.nestedsamplers._SAMPLING["rwalk"] = sample_rwalk_bilby
            if self.kwargs.get("walks", 25) > self.kwargs.get("maxmcmc"):
                raise DynestySetupError("You have maxmcmc > walks (minimum mcmc)")
            if self.kwargs.get("nact", 5) < 1:
                raise DynestySetupError("Unable to run with nact < 1")
        elif self.kwargs.get("sample") == "rwalk_dynesty":
            self._kwargs["sample"] = "rwalk"
            logger.info(
                "Using the dynesty-implemented rwalk sample method")
        elif self.kwargs.get("sample") == "rstagger_dynesty":
            self._kwargs["sample"] = "rstagger"
            logger.info(
                "Using the dynesty-implemented rstagger sample method")

        self._setup_pool()

        if self.resume:
            self.resume = self.read_saved_state(continuing=True)

        if self.resume:
            logger.info('Resume file successfully loaded.')
        else:
            if self.kwargs['live_points'] is None:
                self.kwargs['live_points'] = (
                    self.get_initial_points_from_prior(self.kwargs['nlive'])
                )
            self.sampler = dynesty.NestedSampler(
                loglikelihood=_log_likelihood_wrapper,
                prior_transform=_prior_transform_wrapper,
                ndim=self.ndim, **self.sampler_init_kwargs
            )

        if self.check_point:
            out = self._run_external_sampler_with_checkpointing()
        else:
            out = self._run_external_sampler_without_checkpointing()

        self._close_pool()

        # Flushes the output to force a line break
        if self.kwargs["verbose"]:
            self.pbar.close()
            print("")

        check_directory_exists_and_if_not_mkdir(self.outdir)
        dynesty_result = "{}/{}_dynesty.pickle".format(self.outdir, self.label)
        with open(dynesty_result, 'wb') as file:
            pickle.dump(out, file)

        self._generate_result(out)
        self.calc_likelihood_count()
        self.result.sampling_time = self.sampling_time

        if self.plot:
            self.generate_trace_plots(out)

        return self.result

    def _generate_result(self, out):
        import dynesty
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
        self.result.information_gain = out.information[-1]

    def _run_nested_wrapper(self, kwargs):
        """ Wrapper function to run_nested

        This wrapper catches exceptions related to different versions of
        dynesty accepting different arguments.

        Parameters
        ==========
        kwargs: dict
            The dictionary of kwargs to pass to run_nested

        """
        logger.debug("Calling run_nested with sampler_function_kwargs {}"
                     .format(kwargs))
        try:
            self.sampler.run_nested(**kwargs)
        except TypeError:
            kwargs.pop("n_effective")
            self.sampler.run_nested(**kwargs)

    def _run_external_sampler_without_checkpointing(self):
        logger.debug("Running sampler without checkpointing")
        self._run_nested_wrapper(self.sampler_function_kwargs)
        return self.sampler.results

    def _run_external_sampler_with_checkpointing(self):
        logger.debug("Running sampler with checkpointing")

        old_ncall = self.sampler.ncall
        sampler_kwargs = self.sampler_function_kwargs.copy()
        sampler_kwargs['maxcall'] = self.n_check_point
        sampler_kwargs['add_live'] = True
        self.start_time = datetime.datetime.now()
        while True:
            self._run_nested_wrapper(sampler_kwargs)
            if self.sampler.ncall == old_ncall:
                break
            old_ncall = self.sampler.ncall

            if os.path.isfile(self.resume_file):
                last_checkpoint_s = time.time() - os.path.getmtime(self.resume_file)
            else:
                last_checkpoint_s = (datetime.datetime.now() - self.start_time).total_seconds()
            if last_checkpoint_s > self.check_point_delta_t:
                self.write_current_state()
                self.plot_current_state()
            if self.sampler.added_live:
                self.sampler._remove_live_points()

        sampler_kwargs['add_live'] = True
        self._run_nested_wrapper(sampler_kwargs)
        self.write_current_state()
        self.plot_current_state()
        return self.sampler.results

    def _remove_checkpoint(self):
        """Remove checkpointed state"""
        if os.path.isfile(self.resume_file):
            os.remove(self.resume_file)

    def read_saved_state(self, continuing=False):
        """
        Read a pickled saved state of the sampler to disk.

        If the live points are present and the run is continuing
        they are removed.
        The random state must be reset, as this isn't saved by the pickle.
        `nqueue` is set to a negative number to trigger the queue to be
        refilled before the first iteration.
        The previous run time is set to self.

        Parameters
        ==========
        continuing: bool
            Whether the run is continuing or terminating, if True, the loaded
            state is mostly written back to disk.
        """
        from ... import __version__ as bilby_version
        from dynesty import __version__ as dynesty_version
        versions = dict(bilby=bilby_version, dynesty=dynesty_version)
        if os.path.isfile(self.resume_file):
            logger.info("Reading resume file {}".format(self.resume_file))
            with open(self.resume_file, 'rb') as file:
                sampler = dill.load(file)

                if not hasattr(sampler, "versions"):
                    logger.warning(
                        "The resume file {} is corrupted or the version of "
                        "bilby has changed between runs. This resume file will "
                        "be ignored."
                        .format(self.resume_file)
                    )
                    return False
                version_warning = (
                    "The {code} version has changed between runs. "
                    "This may cause unpredictable behaviour and/or failure. "
                    "Old version = {old}, new version = {new}."

                )
                for code in versions:
                    if not versions[code] == sampler.versions.get(code, None):
                        logger.warning(version_warning.format(
                            code=code,
                            old=sampler.versions.get(code, "None"),
                            new=versions[code]
                        ))
                del sampler.versions
                self.sampler = sampler
                if self.sampler.added_live and continuing:
                    self.sampler._remove_live_points()
                self.sampler.nqueue = -1
                self.sampler.rstate = np.random
                self.start_time = self.sampler.kwargs.pop("start_time")
                self.sampling_time = self.sampler.kwargs.pop("sampling_time")
                self.sampler.pool = self.pool
                if self.pool is not None:
                    self.sampler.M = self.pool.map
                else:
                    self.sampler.M = map
            return True
        else:
            logger.info(
                "Resume file {} does not exist.".format(self.resume_file))
            return False

    def write_current_state_and_exit(self, signum=None, frame=None):
        """
        Make sure that if a pool of jobs is running only the parent tries to
        checkpoint and exit. Only the parent has a 'pool' attribute.
        """
        if self.kwargs["queue_size"] == 1 or getattr(self, "pool", None) is not None:
            if signum == 14:
                logger.info(
                    "Run interrupted by alarm signal {}: checkpoint and exit on {}"
                    .format(signum, self.exit_code))
            else:
                logger.info(
                    "Run interrupted by signal {}: checkpoint and exit on {}"
                    .format(signum, self.exit_code))
            self.write_current_state()
            self._close_pool()
            os._exit(self.exit_code)

    def write_current_state(self):
        """
        Write the current state of the sampler to disk.

        The sampler is pickle dumped using `dill`.
        The sampling time is also stored to get the full CPU time for the run.

        The check of whether the sampler is picklable is to catch an error
        when using pytest. Hopefully, this message won't be triggered during
        normal running.
        """

        from ... import __version__ as bilby_version
        from dynesty import __version__ as dynesty_version
        check_directory_exists_and_if_not_mkdir(self.outdir)
        end_time = datetime.datetime.now()
        if hasattr(self, 'start_time'):
            self.sampling_time += end_time - self.start_time
            self.start_time = end_time
            self.sampler.kwargs["sampling_time"] = self.sampling_time
            self.sampler.kwargs["start_time"] = self.start_time
        self.sampler.versions = dict(
            bilby=bilby_version, dynesty=dynesty_version
        )
        self.sampler.pool = None
        self.sampler.M = map
        if dill.pickles(self.sampler):
            safe_file_dump(self.sampler, self.resume_file, dill)
            logger.info("Written checkpoint file {}".format(self.resume_file))
        else:
            logger.warning(
                "Cannot write pickle resume file! "
                "Job will not resume if interrupted."
            )
        self.sampler.pool = self.pool
        if self.sampler.pool is not None:
            self.sampler.M = self.sampler.pool.map

        self.dump_samples_to_dat()

    def dump_samples_to_dat(self):
        sampler = self.sampler
        ln_weights = sampler.saved_logwt - sampler.saved_logz[-1]

        weights = np.exp(ln_weights)
        samples = rejection_sample(np.array(sampler.saved_v), weights)
        nsamples = len(samples)

        # If we don't have enough samples, don't dump them
        if nsamples < 100:
            return

        filename = "{}/{}_samples.dat".format(self.outdir, self.label)
        logger.info("Writing {} current samples to {}".format(nsamples, filename))

        df = DataFrame(samples, columns=self.search_parameter_keys)
        df.to_csv(filename, index=False, header=True, sep=' ')

    def plot_current_state(self):
        if self.check_point_plot:
            import dynesty.plotting as dyplot
            labels = [label.replace('_', ' ') for label in self.search_parameter_keys]
            try:
                filename = "{}/{}_checkpoint_trace.png".format(self.outdir, self.label)
                fig = dyplot.traceplot(self.sampler.results, labels=labels)[0]
                fig.tight_layout()
                fig.savefig(filename)
            except (RuntimeError, np.linalg.linalg.LinAlgError, ValueError, OverflowError, Exception) as e:
                logger.warning(e)
                logger.warning('Failed to create dynesty state plot at checkpoint')
            finally:
                plt.close("all")
            try:
                filename = "{}/{}_checkpoint_run.png".format(self.outdir, self.label)
                fig, axs = dyplot.runplot(
                    self.sampler.results, logplot=False, use_math_text=False)
                fig.tight_layout()
                plt.savefig(filename)
            except (RuntimeError, np.linalg.linalg.LinAlgError, ValueError) as e:
                logger.warning(e)
                logger.warning('Failed to create dynesty run plot at checkpoint')
            finally:
                plt.close('all')
            try:
                filename = "{}/{}_checkpoint_stats.png".format(self.outdir, self.label)
                fig, axs = plt.subplots(nrows=3, sharex=True)
                for ax, name in zip(axs, ["boundidx", "nc", "scale"]):
                    ax.plot(getattr(self.sampler, "saved_{}".format(name)), color="C0")
                    ax.set_ylabel(name)
                axs[-1].set_xlabel("iteration")
                fig.tight_layout()
                plt.savefig(filename)
            except (RuntimeError, ValueError) as e:
                logger.warning(e)
                logger.warning('Failed to create dynesty stats plot at checkpoint')
            finally:
                plt.close('all')

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
        N = 100
        self.result.samples = pd.DataFrame(
            self.priors.sample(N))[self.search_parameter_keys].values
        self.result.nested_samples = self.result.samples
        self.result.log_likelihood_evaluations = np.ones(N)
        self.result.log_evidence = 1
        self.result.log_evidence_err = 0.1

        return self.result

    def prior_transform(self, theta):
        """ Prior transform method that is passed into the external sampler.
        cube we map this back to [0, 1].

        Parameters
        ==========
        theta: list
            List of sampled values on a unit interval

        Returns
        =======
        list: Properly rescaled sampled values

        """
        return self.priors.rescale(self._search_parameter_keys, theta)

    def calc_likelihood_count(self):
        if self.likelihood_benchmark:
            if hasattr(self, 'sampler'):
                self.result.num_likelihood_evaluations = \
                    getattr(self.sampler, 'ncall', 0)
            else:
                self.result.num_likelihood_evaluations = 0
        else:
            return None


def sample_rwalk_bilby(args):
    """ Modified bilby-implemented version of dynesty.sampling.sample_rwalk """

    # Unzipping.
    (u, loglstar, axes, scale,
     prior_transform, loglikelihood, kwargs) = args
    rstate = np.random

    # Bounds
    nonbounded = kwargs.get('nonbounded', None)
    periodic = kwargs.get('periodic', None)
    reflective = kwargs.get('reflective', None)

    # Setup.
    n = len(u)
    walks = kwargs.get('walks', 25)  # minimum number of steps
    maxmcmc = kwargs.get('maxmcmc', 2000)  # Maximum number of steps
    nact = kwargs.get('nact', 5)  # Number of ACT
    old_act = kwargs.get('old_act', walks)

    # Initialize internal variables
    accept = 0
    reject = 0
    nfail = 0
    act = np.inf
    u_list = []
    v_list = []
    logl_list = []

    ii = 0
    while ii < nact * act:
        ii += 1

        # Propose a direction on the unit n-sphere.
        drhat = rstate.randn(n)
        drhat /= linalg.norm(drhat)

        # Scale based on dimensionality.
        dr = drhat * rstate.rand() ** (1.0 / n)

        # Transform to proposal distribution.
        du = np.dot(axes, dr)
        u_prop = u + scale * du

        # Wrap periodic parameters
        if periodic is not None:
            u_prop[periodic] = np.mod(u_prop[periodic], 1)
        # Reflect
        if reflective is not None:
            u_prop[reflective] = reflect(u_prop[reflective])

        # Check unit cube constraints.
        if unitcheck(u_prop, nonbounded):
            pass
        else:
            nfail += 1
            # Only start appending to the chain once a single jump is made
            if accept > 0:
                u_list.append(u_list[-1])
                v_list.append(v_list[-1])
                logl_list.append(logl_list[-1])
            continue

        # Check proposed point.
        v_prop = prior_transform(np.array(u_prop))
        logl_prop = loglikelihood(np.array(v_prop))
        if logl_prop > loglstar:
            u = u_prop
            v = v_prop
            logl = logl_prop
            accept += 1
            u_list.append(u)
            v_list.append(v)
            logl_list.append(logl)
        else:
            reject += 1
            # Only start appending to the chain once a single jump is made
            if accept > 0:
                u_list.append(u_list[-1])
                v_list.append(v_list[-1])
                logl_list.append(logl_list[-1])

        # If we've taken the minimum number of steps, calculate the ACT
        if accept + reject > walks:
            act = estimate_nmcmc(
                accept_ratio=accept / (accept + reject + nfail),
                old_act=old_act, maxmcmc=maxmcmc)

        # If we've taken too many likelihood evaluations then break
        if accept + reject > maxmcmc:
            warnings.warn(
                "Hit maximum number of walks {} with accept={}, reject={}, "
                "and nfail={} try increasing maxmcmc"
                .format(maxmcmc, accept, reject, nfail))
            break

    # If the act is finite, pick randomly from within the chain
    if np.isfinite(act) and int(.5 * nact * act) < len(u_list):
        idx = np.random.randint(int(.5 * nact * act), len(u_list))
        u = u_list[idx]
        v = v_list[idx]
        logl = logl_list[idx]
    else:
        logger.debug("Unable to find a new point using walk: returning a random point")
        u = np.random.uniform(size=n)
        v = prior_transform(u)
        logl = loglikelihood(v)

    blob = {'accept': accept, 'reject': reject, 'fail': nfail, 'scale': scale}
    kwargs["old_act"] = act

    ncall = accept + reject
    return u, v, logl, ncall, blob


def estimate_nmcmc(accept_ratio, old_act, maxmcmc, safety=5, tau=None):
    """ Estimate autocorrelation length of chain using acceptance fraction

    Using ACL = (2/acc) - 1 multiplied by a safety margin. Code adapated from CPNest:

    - https://github.com/johnveitch/cpnest/blob/master/cpnest/sampler.py
    - http://github.com/farr/Ensemble.jl

    Parameters
    ==========
    accept_ratio: float [0, 1]
        Ratio of the number of accepted points to the total number of points
    old_act: int
        The ACT of the last iteration
    maxmcmc: int
        The maximum length of the MCMC chain to use
    safety: int
        A safety factor applied in the calculation
    tau: int (optional)
        The ACT, if given, otherwise estimated.

    """
    if tau is None:
        tau = maxmcmc / safety

    if accept_ratio == 0.0:
        Nmcmc_exact = (1 + 1 / tau) * old_act
    else:
        Nmcmc_exact = (
            (1. - 1. / tau) * old_act +
            (safety / tau) * (2. / accept_ratio - 1.)
        )
        Nmcmc_exact = float(min(Nmcmc_exact, maxmcmc))
    return max(safety, int(Nmcmc_exact))


class DynestySetupError(Exception):
    pass
