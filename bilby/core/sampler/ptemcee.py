
import os
import datetime
import copy
import signal
import sys
import time
import dill
from collections import namedtuple
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

from ..utils import logger, check_directory_exists_and_if_not_mkdir
from .base_sampler import SamplerError, MCMCSampler


ConvergenceInputs = namedtuple(
    "ConvergenceInputs",
    [
        "autocorr_c",
        "autocorr_tol",
        "autocorr_tau",
        "gradient_tau",
        "gradient_mean_log_posterior",
        "Q_tol",
        "safety",
        "burn_in_nact",
        "burn_in_fixed_discard",
        "mean_logl_frac",
        "thin_by_nact",
        "nsamples",
        "ignore_keys_for_tau",
        "min_tau",
        "niterations_per_check",
    ],
)


class Ptemcee(MCMCSampler):
    """bilby wrapper ptemcee (https://github.com/willvousden/ptemcee)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `ptemcee.Sampler`, see
    documentation for that class for further help. Under Other Parameters, we
    list commonly used kwargs and the bilby defaults.

    Parameters
    ==========
    nsamples: int, (5000)
        The requested number of samples. Note, in cases where the
        autocorrelation parameter is difficult to measure, it is possible to
        end up with more than nsamples.
    burn_in_nact, thin_by_nact: int, (50, 1)
        The number of burn-in autocorrelation times to discard and the thin-by
        factor. Increasing burn_in_nact increases the time required for burn-in.
        Increasing thin_by_nact increases the time required to obtain nsamples.
    burn_in_fixed_discard: int (0)
        A fixed number of samples to discard for burn-in
    mean_logl_frac: float, (0.0.1)
        The maximum fractional change the mean log-likelihood to accept
    autocorr_tol: int, (50)
        The minimum number of autocorrelation times needed to trust the
        estimate of the autocorrelation time.
    autocorr_c: int, (5)
        The step size for the window search used by emcee.autocorr.integrated_time
    safety: int, (1)
        A multiplicitive factor for the estimated autocorrelation. Useful for
        cases where non-convergence can be observed by eye but the automated
        tools are failing.
    autocorr_tau: int, (1)
        The number of autocorrelation times to use in assessing if the
        autocorrelation time is stable.
    gradient_tau: float, (0.1)
        The maximum (smoothed) local gradient of the ACT estimate to allow.
        This ensures the ACT estimate is stable before finishing sampling.
    gradient_mean_log_posterior: float, (0.1)
        The maximum (smoothed) local gradient of the logliklilhood to allow.
        This ensures the ACT estimate is stable before finishing sampling.
    Q_tol: float (1.01)
        The maximum between-chain to within-chain tolerance allowed (akin to
        the Gelman-Rubin statistic).
    min_tau: int, (1)
        A minimum tau (autocorrelation time) to accept.
    check_point_deltaT: float, (600)
        The period with which to checkpoint (in seconds).
    threads: int, (1)
        If threads > 1, a MultiPool object is setup and used.
    exit_code: int, (77)
        The code on which the sampler exits.
    store_walkers: bool (False)
        If true, store the unthinned, unburnt chains in the result. Note, this
        is not recommended for cases where tau is large.
    ignore_keys_for_tau: str
        A pattern used to ignore keys in estimating the autocorrelation time.
    pos0: str, list ("prior")
        If a string, one of "prior" or "minimize". For "prior", the initial
        positions of the sampler are drawn from the sampler. If "minimize",
        a scipy.optimize step is applied to all parameters a number of times.
        The walkers are then initialized from the range of values obtained.
        If a list, for the keys in the list the optimization step is applied,
        otherwise the initial points are drawn from the prior.
    niterations_per_check: int (5)
        The number of iteration steps to take before checking ACT. This
        effectively pre-thins the chains. Larger values reduce the per-eval
        timing due to improved efficiency. But, if it is made too large the
        pre-thinning may be overly agressive effectively wasting compute-time.
        If you see tau=1, then niterations_per_check is likely too large.


    Other Parameters
    ------==========
    nwalkers: int, (200)
        The number of walkers
    nsteps: int, (100)
        The number of steps to take
    ntemps: int (10)
        The number of temperatures used by ptemcee
    Tmax: float
        The maximum temperature

    """

    # Arguments used by ptemcee
    default_kwargs = dict(
        ntemps=10,
        nwalkers=100,
        Tmax=None,
        betas=None,
        a=2.0,
        adaptation_lag=10000,
        adaptation_time=100,
        random=None,
        adapt=False,
        swap_ratios=False,
    )

    def __init__(
        self,
        likelihood,
        priors,
        outdir="outdir",
        label="label",
        use_ratio=False,
        check_point_plot=True,
        skip_import_verification=False,
        resume=True,
        nsamples=5000,
        burn_in_nact=50,
        burn_in_fixed_discard=0,
        mean_logl_frac=0.01,
        thin_by_nact=0.5,
        autocorr_tol=50,
        autocorr_c=5,
        safety=1,
        autocorr_tau=1,
        gradient_tau=0.1,
        gradient_mean_log_posterior=0.1,
        Q_tol=1.02,
        min_tau=1,
        check_point_deltaT=600,
        threads=1,
        exit_code=77,
        plot=False,
        store_walkers=False,
        ignore_keys_for_tau=None,
        pos0="prior",
        niterations_per_check=5,
        log10beta_min=None,
        **kwargs
    ):
        super(Ptemcee, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=use_ratio,
            plot=plot,
            skip_import_verification=skip_import_verification,
            exit_code=exit_code,
            **kwargs
        )

        self.nwalkers = self.sampler_init_kwargs["nwalkers"]
        self.ntemps = self.sampler_init_kwargs["ntemps"]
        self.max_steps = 500

        # Setup up signal handling
        signal.signal(signal.SIGTERM, self.write_current_state_and_exit)
        signal.signal(signal.SIGINT, self.write_current_state_and_exit)
        signal.signal(signal.SIGALRM, self.write_current_state_and_exit)

        # Checkpointing inputs
        self.resume = resume
        self.check_point_deltaT = check_point_deltaT
        self.check_point_plot = check_point_plot
        self.resume_file = "{}/{}_checkpoint_resume.pickle".format(
            self.outdir, self.label
        )

        # Store convergence checking inputs in a named tuple
        convergence_inputs_dict = dict(
            autocorr_c=autocorr_c,
            autocorr_tol=autocorr_tol,
            autocorr_tau=autocorr_tau,
            safety=safety,
            burn_in_nact=burn_in_nact,
            burn_in_fixed_discard=burn_in_fixed_discard,
            mean_logl_frac=mean_logl_frac,
            thin_by_nact=thin_by_nact,
            gradient_tau=gradient_tau,
            gradient_mean_log_posterior=gradient_mean_log_posterior,
            Q_tol=Q_tol,
            nsamples=nsamples,
            ignore_keys_for_tau=ignore_keys_for_tau,
            min_tau=min_tau,
            niterations_per_check=niterations_per_check,
        )
        self.convergence_inputs = ConvergenceInputs(**convergence_inputs_dict)
        logger.info("Using convergence inputs: {}".format(self.convergence_inputs))

        # Check if threads was given as an equivalent arg
        if threads == 1:
            for equiv in self.npool_equiv_kwargs:
                if equiv in kwargs:
                    threads = kwargs.pop(equiv)

        # Store threads
        self.threads = threads

        # Misc inputs
        self.store_walkers = store_walkers
        self.pos0 = pos0

        self._periodic = [
            self.priors[key].boundary == "periodic" for key in self.search_parameter_keys
        ]
        self.priors.sample()
        self._minima = np.array([
            self.priors[key].minimum for key in self.search_parameter_keys
        ])
        self._range = np.array([
            self.priors[key].maximum for key in self.search_parameter_keys
        ]) - self._minima

        self.log10beta_min = log10beta_min
        if self.log10beta_min is not None:
            betas = np.logspace(0, self.log10beta_min, self.ntemps)
            logger.warning("Using betas {}".format(betas))
            self.kwargs["betas"] = betas

    @property
    def sampler_function_kwargs(self):
        """ Kwargs passed to samper.sampler() """
        keys = ["adapt", "swap_ratios"]
        return {key: self.kwargs[key] for key in keys}

    @property
    def sampler_init_kwargs(self):
        """ Kwargs passed to initialize ptemcee.Sampler() """
        return {
            key: value
            for key, value in self.kwargs.items()
            if key not in self.sampler_function_kwargs
        }

    def _translate_kwargs(self, kwargs):
        """ Translate kwargs """
        if "nwalkers" not in kwargs:
            for equiv in self.nwalkers_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["nwalkers"] = kwargs.pop(equiv)

    def get_pos0_from_prior(self):
        """ Draw the initial positions from the prior

        Returns
        =======
        pos0: list
            The initial postitions of the walkers, with shape (ntemps, nwalkers, ndim)

        """
        logger.info("Generating pos0 samples")
        return [
            [
                self.get_random_draw_from_prior()
                for _ in range(self.nwalkers)
            ]
            for _ in range(self.kwargs["ntemps"])
        ]

    def get_pos0_from_minimize(self, minimize_list=None):
        """ Draw the initial positions using an initial minimization step

        See pos0 in the class initialization for details.

        Returns
        =======
        pos0: list
            The initial postitions of the walkers, with shape (ntemps, nwalkers, ndim)

        """

        from scipy.optimize import minimize

        # Set up the minimize list: keys not in this list will have initial
        # positions drawn from the prior
        if minimize_list is None:
            minimize_list = self.search_parameter_keys
            pos0 = np.zeros((self.kwargs["ntemps"], self.kwargs["nwalkers"], self.ndim))
        else:
            pos0 = np.array(self.get_pos0_from_prior())

        logger.info("Attempting to set pos0 for {} from minimize".format(minimize_list))

        likelihood_copy = copy.copy(self.likelihood)

        def neg_log_like(params):
            """ Internal function to minimize """
            likelihood_copy.parameters.update(
                {key: val for key, val in zip(minimize_list, params)}
            )
            try:
                return -likelihood_copy.log_likelihood()
            except RuntimeError:
                return +np.inf

        # Bounds used in the minimization
        bounds = [
            (self.priors[key].minimum, self.priors[key].maximum)
            for key in minimize_list
        ]

        # Run the minimization step several times to get a range of values
        trials = 0
        success = []
        while True:
            draw = self.priors.sample()
            likelihood_copy.parameters.update(draw)
            x0 = [draw[key] for key in minimize_list]
            res = minimize(
                neg_log_like, x0, bounds=bounds, method="L-BFGS-B", tol=1e-15
            )
            if res.success:
                success.append(res.x)
            if trials > 100:
                raise SamplerError("Unable to set pos0 from minimize")
            if len(success) >= 10:
                break

        # Initialize positions from the range of values
        success = np.array(success)
        for i, key in enumerate(minimize_list):
            pos0_min = np.min(success[:, i])
            pos0_max = np.max(success[:, i])
            logger.info(
                "Initialize {} walkers from {}->{}".format(key, pos0_min, pos0_max)
            )
            j = self.search_parameter_keys.index(key)
            pos0[:, :, j] = np.random.uniform(
                pos0_min,
                pos0_max,
                size=(self.kwargs["ntemps"], self.kwargs["nwalkers"]),
            )
        return pos0

    def setup_sampler(self):
        """ Either initialize the sampler or read in the resume file """
        import ptemcee

        if os.path.isfile(self.resume_file) and self.resume is True:
            logger.info("Resume data {} found".format(self.resume_file))
            with open(self.resume_file, "rb") as file:
                data = dill.load(file)

            # Extract the check-point data
            self.sampler = data["sampler"]
            self.iteration = data["iteration"]
            self.chain_array = data["chain_array"]
            self.log_likelihood_array = data["log_likelihood_array"]
            self.log_posterior_array = data["log_posterior_array"]
            self.pos0 = data["pos0"]
            self.beta_list = data["beta_list"]
            self.sampler._betas = np.array(self.beta_list[-1])
            self.tau_list = data["tau_list"]
            self.tau_list_n = data["tau_list_n"]
            self.Q_list = data["Q_list"]
            self.time_per_check = data["time_per_check"]

            # Initialize the pool
            self.sampler.pool = self.pool
            self.sampler.threads = self.threads

            logger.info(
                "Resuming from previous run with time={}".format(self.iteration)
            )

        else:
            # Initialize the PTSampler
            if self.threads == 1:
                self.sampler = ptemcee.Sampler(
                    dim=self.ndim,
                    logl=self.log_likelihood,
                    logp=self.log_prior,
                    **self.sampler_init_kwargs
                )
            else:
                self.sampler = ptemcee.Sampler(
                    dim=self.ndim,
                    logl=do_nothing_function,
                    logp=do_nothing_function,
                    pool=self.pool,
                    threads=self.threads,
                    **self.sampler_init_kwargs
                )

                self.sampler._likeprior = LikePriorEvaluator(
                    self.search_parameter_keys, use_ratio=self.use_ratio
                )

            # Initialize storing results
            self.iteration = 0
            self.chain_array = self.get_zero_chain_array()
            self.log_likelihood_array = self.get_zero_array()
            self.log_posterior_array = self.get_zero_array()
            self.beta_list = []
            self.tau_list = []
            self.tau_list_n = []
            self.Q_list = []
            self.time_per_check = []
            self.pos0 = self.get_pos0()

        return self.sampler

    def get_zero_chain_array(self):
        return np.zeros((self.nwalkers, self.max_steps, self.ndim))

    def get_zero_array(self):
        return np.zeros((self.ntemps, self.nwalkers, self.max_steps))

    def get_pos0(self):
        """ Master logic for setting pos0 """
        if isinstance(self.pos0, str) and self.pos0.lower() == "prior":
            return self.get_pos0_from_prior()
        elif isinstance(self.pos0, str) and self.pos0.lower() == "minimize":
            return self.get_pos0_from_minimize()
        elif isinstance(self.pos0, list):
            return self.get_pos0_from_minimize(minimize_list=self.pos0)
        else:
            raise SamplerError("pos0={} not implemented".format(self.pos0))

    def setup_pool(self):
        """ If threads > 1, setup a MultiPool, else run in serial mode """
        if self.threads > 1:
            import schwimmbad

            logger.info("Creating MultiPool with {} processes".format(self.threads))
            self.pool = schwimmbad.MultiPool(
                self.threads, initializer=init, initargs=(self.likelihood, self.priors)
            )
        else:
            self.pool = None

    def run_sampler(self):
        self.setup_pool()
        sampler = self.setup_sampler()

        t0 = datetime.datetime.now()
        logger.info("Starting to sample")
        while True:
            for (pos0, log_posterior, log_likelihood) in sampler.sample(
                    self.pos0, storechain=False,
                    iterations=self.convergence_inputs.niterations_per_check,
                    **self.sampler_function_kwargs):
                pos0[:, :, self._periodic] = np.mod(
                    pos0[:, :, self._periodic] - self._minima[self._periodic],
                    self._range[self._periodic]
                ) + self._minima[self._periodic]

            if self.iteration == self.chain_array.shape[1]:
                self.chain_array = np.concatenate((
                    self.chain_array, self.get_zero_chain_array()), axis=1)
                self.log_likelihood_array = np.concatenate((
                    self.log_likelihood_array, self.get_zero_array()),
                    axis=2)
                self.log_posterior_array = np.concatenate((
                    self.log_posterior_array, self.get_zero_array()),
                    axis=2)

            self.pos0 = pos0
            self.chain_array[:, self.iteration, :] = pos0[0, :, :]
            self.log_likelihood_array[:, :, self.iteration] = log_likelihood
            self.log_posterior_array[:, :, self.iteration] = log_posterior
            self.mean_log_posterior = np.mean(
                self.log_posterior_array[:, :, :self. iteration], axis=1)

            # Calculate time per iteration
            self.time_per_check.append((datetime.datetime.now() - t0).total_seconds())
            t0 = datetime.datetime.now()

            self.iteration += 1

            # Calculate minimum iteration step to discard
            minimum_iteration = get_minimum_stable_itertion(
                self.mean_log_posterior,
                frac=self.convergence_inputs.mean_logl_frac
            )
            logger.debug("Minimum iteration = {}".format(minimum_iteration))

            # Calculate the maximum discard number
            discard_max = np.max(
                [self.convergence_inputs.burn_in_fixed_discard,
                 minimum_iteration]
            )

            if self.iteration > discard_max + self.nwalkers:
                # If we have taken more than nwalkers steps after the discard
                # then set the discard
                self.discard = discard_max
            else:
                # If haven't discard everything (avoid initialisation bias)
                logger.debug("Too few steps to calculate convergence")
                self.discard = self.iteration

            (
                stop,
                self.nburn,
                self.thin,
                self.tau_int,
                self.nsamples_effective,
            ) = check_iteration(
                self.iteration,
                self.chain_array[:, self.discard:self.iteration, :],
                sampler,
                self.convergence_inputs,
                self.search_parameter_keys,
                self.time_per_check,
                self.beta_list,
                self.tau_list,
                self.tau_list_n,
                self.Q_list,
                self.mean_log_posterior,
            )

            if stop:
                logger.info("Finished sampling")
                break

            # If a checkpoint is due, checkpoint
            if os.path.isfile(self.resume_file):
                last_checkpoint_s = time.time() - os.path.getmtime(self.resume_file)
            else:
                last_checkpoint_s = np.sum(self.time_per_check)

            if last_checkpoint_s > self.check_point_deltaT:
                self.write_current_state(plot=self.check_point_plot)

        # Run a final checkpoint to update the plots and samples
        self.write_current_state(plot=self.check_point_plot)

        # Get 0-likelihood samples and store in the result
        self.result.samples = self.chain_array[
            :, self.discard + self.nburn : self.iteration : self.thin, :
        ].reshape((-1, self.ndim))
        loglikelihood = self.log_likelihood_array[
            0, :, self.discard + self.nburn : self.iteration : self.thin
        ]  # nwalkers, nsteps
        self.result.log_likelihood_evaluations = loglikelihood.reshape((-1))

        if self.store_walkers:
            self.result.walkers = self.sampler.chain
        self.result.nburn = self.nburn
        self.result.discard = self.discard

        log_evidence, log_evidence_err = compute_evidence(
            sampler, self.log_likelihood_array, self.outdir,
            self.label, self.discard, self.nburn,
            self.thin, self.iteration,
        )
        self.result.log_evidence = log_evidence
        self.result.log_evidence_err = log_evidence_err

        self.result.sampling_time = datetime.timedelta(
            seconds=np.sum(self.time_per_check)
        )

        if self.pool:
            self.pool.close()

        return self.result

    def write_current_state_and_exit(self, signum=None, frame=None):
        logger.warning("Run terminated with signal {}".format(signum))
        if getattr(self, "pool", None) or self.threads == 1:
            self.write_current_state(plot=False)
        if getattr(self, "pool", None):
            logger.info("Closing pool")
            self.pool.close()
        logger.info("Exit on signal {}".format(self.exit_code))
        sys.exit(self.exit_code)

    def write_current_state(self, plot=True):
        check_directory_exists_and_if_not_mkdir(self.outdir)
        checkpoint(
            self.iteration,
            self.outdir,
            self.label,
            self.nsamples_effective,
            self.sampler,
            self.discard,
            self.nburn,
            self.thin,
            self.search_parameter_keys,
            self.resume_file,
            self.log_likelihood_array,
            self.log_posterior_array,
            self.chain_array,
            self.pos0,
            self.beta_list,
            self.tau_list,
            self.tau_list_n,
            self.Q_list,
            self.time_per_check,
        )

        if plot:
            try:
                # Generate the walkers plot diagnostic
                plot_walkers(
                    self.chain_array[:, : self.iteration, :],
                    self.nburn,
                    self.thin,
                    self.search_parameter_keys,
                    self.outdir,
                    self.label,
                    self.discard,
                )
            except Exception as e:
                logger.info("Walkers plot failed with exception {}".format(e))

            try:
                # Generate the tau plot diagnostic if DEBUG
                if logger.level < logging.INFO:
                    plot_tau(
                        self.tau_list_n,
                        self.tau_list,
                        self.search_parameter_keys,
                        self.outdir,
                        self.label,
                        self.tau_int,
                        self.convergence_inputs.autocorr_tau,
                    )
            except Exception as e:
                logger.info("tau plot failed with exception {}".format(e))

            try:
                plot_mean_log_posterior(
                    self.mean_log_posterior,
                    self.outdir,
                    self.label,
                )
            except Exception as e:
                logger.info("mean_logl plot failed with exception {}".format(e))


def get_minimum_stable_itertion(mean_array, frac, nsteps_min=10):
    nsteps = mean_array.shape[1]
    if nsteps < nsteps_min:
        return 0

    min_it = 0
    for x in mean_array:
        maxl = np.max(x)
        fracdiff = (maxl - x) / np.abs(maxl)
        idxs = fracdiff < frac
        if np.sum(idxs) > 0:
            min_it = np.max([min_it, np.min(np.arange(len(idxs))[idxs])])
    return min_it


def check_iteration(
    iteration,
    samples,
    sampler,
    convergence_inputs,
    search_parameter_keys,
    time_per_check,
    beta_list,
    tau_list,
    tau_list_n,
    Q_list,
    mean_log_posterior,
):
    """ Per-iteration logic to calculate the convergence check

    Parameters
    ==========
    convergence_inputs: bilby.core.sampler.ptemcee.ConvergenceInputs
        A named tuple of the convergence checking inputs
    search_parameter_keys: list
        A list of the search parameter keys
    time_per_check, tau_list, tau_list_n: list
        Lists used for tracking the run

    Returns
    =======
    stop: bool
        A boolean flag, True if the stoping criteria has been met
    burn: int
        The number of burn-in steps to discard
    thin: int
        The thin-by factor to apply
    tau_int: int
        The integer estimated ACT
    nsamples_effective: int
        The effective number of samples after burning and thinning
    """

    ci = convergence_inputs
    # Note: nsteps is the number of steps in the samples while iterations is
    # the current iteration number. So iteration > nsteps by the number of
    # of discards
    nwalkers, nsteps, ndim = samples.shape

    tau_array = calculate_tau_array(samples, search_parameter_keys, ci)

    # Maximum over parameters, mean over walkers
    tau = np.max(np.mean(tau_array, axis=0))

    # Apply multiplicitive safety factor
    tau = ci.safety * tau

    # Store for convergence checking and plotting
    beta_list.append(list(sampler.betas))
    tau_list.append(list(np.mean(tau_array, axis=0)))
    tau_list_n.append(iteration)

    Q = get_Q_convergence(samples)
    Q_list.append(Q)

    if np.isnan(tau) or np.isinf(tau):
        print_progress(
            iteration, sampler, time_per_check, np.nan, np.nan,
            np.nan, np.nan, np.nan, False, convergence_inputs, Q,
        )
        return False, np.nan, np.nan, np.nan, np.nan

    # Convert to an integer
    tau_int = int(np.ceil(tau))

    # Calculate the effective number of samples available
    nburn = int(ci.burn_in_nact * tau_int)
    thin = int(np.max([1, ci.thin_by_nact * tau_int]))
    samples_per_check = nwalkers / thin
    nsamples_effective = int(nwalkers * (nsteps - nburn) / thin)

    # Calculate convergence boolean
    converged = Q < ci.Q_tol and ci.nsamples < nsamples_effective
    logger.debug("Convergence: Q<Q_tol={}, nsamples<nsamples_effective={}"
                 .format(Q < ci.Q_tol, ci.nsamples < nsamples_effective))

    GRAD_WINDOW_LENGTH = nwalkers + 1
    nsteps_to_check = ci.autocorr_tau * np.max([2 * GRAD_WINDOW_LENGTH, tau_int])
    lower_tau_index = np.max([0, len(tau_list) - nsteps_to_check])
    check_taus = np.array(tau_list[lower_tau_index :])
    if not np.any(np.isnan(check_taus)) and check_taus.shape[0] > GRAD_WINDOW_LENGTH:
        gradient_tau = get_max_gradient(
            check_taus, axis=0, window_length=11)

        if gradient_tau < ci.gradient_tau:
            logger.debug(
                "tau usable as {} < gradient_tau={}"
                .format(gradient_tau, ci.gradient_tau)
            )
            tau_usable = True
        else:
            logger.debug(
                "tau not usable as {} > gradient_tau={}"
                .format(gradient_tau, ci.gradient_tau)
            )
            tau_usable = False

        check_mean_log_posterior = mean_log_posterior[:, -nsteps_to_check:]
        gradient_mean_log_posterior = get_max_gradient(
            check_mean_log_posterior, axis=1, window_length=GRAD_WINDOW_LENGTH,
            smooth=True)

        if gradient_mean_log_posterior < ci.gradient_mean_log_posterior:
            logger.debug(
                "tau usable as {} < gradient_mean_log_posterior={}"
                .format(gradient_mean_log_posterior, ci.gradient_mean_log_posterior)
            )
            tau_usable *= True
        else:
            logger.debug(
                "tau not usable as {} > gradient_mean_log_posterior={}"
                .format(gradient_mean_log_posterior, ci.gradient_mean_log_posterior)
            )
            tau_usable = False

    else:
        logger.debug("ACT is nan")
        gradient_tau = np.nan
        gradient_mean_log_posterior = np.nan
        tau_usable = False

    if nsteps < tau_int * ci.autocorr_tol:
        logger.debug("ACT less than autocorr_tol")
        tau_usable = False
    elif tau_int < ci.min_tau:
        logger.debug("ACT less than min_tau")
        tau_usable = False

    # Print an update on the progress
    print_progress(
        iteration,
        sampler,
        time_per_check,
        nsamples_effective,
        samples_per_check,
        tau_int,
        gradient_tau,
        gradient_mean_log_posterior,
        tau_usable,
        convergence_inputs,
        Q
    )
    stop = converged and tau_usable
    return stop, nburn, thin, tau_int, nsamples_effective


def get_max_gradient(x, axis=0, window_length=11, polyorder=2, smooth=False):
    if smooth:
        x = scipy.signal.savgol_filter(
            x, axis=axis, window_length=window_length, polyorder=3)
    return np.max(scipy.signal.savgol_filter(
        x, axis=axis, window_length=window_length, polyorder=polyorder,
        deriv=1))


def get_Q_convergence(samples):
    nwalkers, nsteps, ndim = samples.shape
    if nsteps > 1:
        W = np.mean(np.var(samples, axis=1), axis=0)
        per_walker_mean = np.mean(samples, axis=1)
        mean = np.mean(per_walker_mean, axis=0)
        B = nsteps / (nwalkers - 1.) * np.sum((per_walker_mean - mean)**2, axis=0)
        Vhat = (nsteps - 1) / nsteps * W + (nwalkers + 1) / (nwalkers * nsteps) * B
        Q_per_dim = np.sqrt(Vhat / W)
        return np.max(Q_per_dim)
    else:
        return np.inf


def print_progress(
    iteration,
    sampler,
    time_per_check,
    nsamples_effective,
    samples_per_check,
    tau_int,
    gradient_tau,
    gradient_mean_log_posterior,
    tau_usable,
    convergence_inputs,
    Q,
):
    # Setup acceptance string
    acceptance = sampler.acceptance_fraction[0, :]
    acceptance_str = "{:1.2f}-{:1.2f}".format(np.min(acceptance), np.max(acceptance))

    # Setup tswap acceptance string
    tswap_acceptance_fraction = sampler.tswap_acceptance_fraction
    tswap_acceptance_str = "{:1.2f}-{:1.2f}".format(
        np.min(tswap_acceptance_fraction), np.max(tswap_acceptance_fraction)
    )

    ave_time_per_check = np.mean(time_per_check[-3:])
    time_left = (convergence_inputs.nsamples - nsamples_effective) * ave_time_per_check / samples_per_check
    if time_left > 0:
        time_left = str(datetime.timedelta(seconds=int(time_left)))
    else:
        time_left = "waiting on convergence"

    sampling_time = datetime.timedelta(seconds=np.sum(time_per_check))

    tau_str = "{}(+{:0.2f},+{:0.2f})".format(
        tau_int, gradient_tau, gradient_mean_log_posterior
    )

    if tau_usable:
        tau_str = "={}".format(tau_str)
    else:
        tau_str = "!{}".format(tau_str)

    Q_str = "{:0.2f}".format(Q)

    evals_per_check = sampler.nwalkers * sampler.ntemps * convergence_inputs.niterations_per_check

    ncalls = "{:1.1e}".format(
        convergence_inputs.niterations_per_check * iteration * sampler.nwalkers * sampler.ntemps)
    eval_timing = "{:1.2f}ms/ev".format(1e3 * ave_time_per_check / evals_per_check)

    try:
        print(
            "{}|{}|nc:{}|a0:{}|swp:{}|n:{}<{}|t{}|q:{}|{}".format(
                iteration,
                str(sampling_time).split(".")[0],
                ncalls,
                acceptance_str,
                tswap_acceptance_str,
                nsamples_effective,
                convergence_inputs.nsamples,
                tau_str,
                Q_str,
                eval_timing,
            ),
            flush=True,
        )
    except OSError as e:
        logger.debug("Failed to print iteration due to :{}".format(e))


def calculate_tau_array(samples, search_parameter_keys, ci):
    """ Compute ACT tau for 0-temperature chains """
    import emcee
    nwalkers, nsteps, ndim = samples.shape
    tau_array = np.zeros((nwalkers, ndim)) + np.inf
    if nsteps > 1:
        for ii in range(nwalkers):
            for jj, key in enumerate(search_parameter_keys):
                if ci.ignore_keys_for_tau and ci.ignore_keys_for_tau in key:
                    continue
                try:
                    tau_array[ii, jj] = emcee.autocorr.integrated_time(
                        samples[ii, :, jj], c=ci.autocorr_c, tol=0)[0]
                except emcee.autocorr.AutocorrError:
                    tau_array[ii, jj] = np.inf
    return tau_array


def checkpoint(
    iteration,
    outdir,
    label,
    nsamples_effective,
    sampler,
    discard,
    nburn,
    thin,
    search_parameter_keys,
    resume_file,
    log_likelihood_array,
    log_posterior_array,
    chain_array,
    pos0,
    beta_list,
    tau_list,
    tau_list_n,
    Q_list,
    time_per_check,
):
    logger.info("Writing checkpoint and diagnostics")
    ndim = sampler.dim

    # Store the samples if possible
    if nsamples_effective > 0:
        filename = "{}/{}_samples.txt".format(outdir, label)
        samples = np.array(chain_array)[:, discard + nburn : iteration : thin, :].reshape(
            (-1, ndim)
        )
        df = pd.DataFrame(samples, columns=search_parameter_keys)
        df.to_csv(filename, index=False, header=True, sep=" ")

    # Pickle the resume artefacts
    sampler_copy = copy.copy(sampler)
    del sampler_copy.pool

    data = dict(
        iteration=iteration,
        sampler=sampler_copy,
        beta_list=beta_list,
        tau_list=tau_list,
        tau_list_n=tau_list_n,
        Q_list=Q_list,
        time_per_check=time_per_check,
        log_likelihood_array=log_likelihood_array,
        log_posterior_array=log_posterior_array,
        chain_array=chain_array,
        pos0=pos0,
    )

    with open(resume_file, "wb") as file:
        dill.dump(data, file, protocol=4)
    del data, sampler_copy
    logger.info("Finished writing checkpoint")


def plot_walkers(walkers, nburn, thin, parameter_labels, outdir, label,
                 discard=0):
    """ Method to plot the trace of the walkers in an ensemble MCMC plot """
    nwalkers, nsteps, ndim = walkers.shape
    if np.isnan(nburn):
        nburn = nsteps
    if np.isnan(thin):
        thin = 1
    idxs = np.arange(nsteps)
    fig, axes = plt.subplots(nrows=ndim, ncols=2, figsize=(8, 3 * ndim))
    scatter_kwargs = dict(lw=0, marker="o", markersize=1, alpha=0.1,)

    # Plot the fixed burn-in
    if discard > 0:
        for i, (ax, axh) in enumerate(axes):
            ax.plot(
                idxs[: discard],
                walkers[:, : discard, i].T,
                color="gray",
                **scatter_kwargs
            )

    # Plot the burn-in
    for i, (ax, axh) in enumerate(axes):
        ax.plot(
            idxs[discard: discard + nburn + 1],
            walkers[:, discard: discard + nburn + 1, i].T,
            color="C1",
            **scatter_kwargs
        )

    # Plot the thinned posterior samples
    for i, (ax, axh) in enumerate(axes):
        ax.plot(
            idxs[discard + nburn::thin],
            walkers[:, discard + nburn::thin, i].T,
            color="C0",
            **scatter_kwargs
        )
        axh.hist(walkers[:, discard + nburn::thin, i].reshape((-1)), bins=50, alpha=0.8)

    for i, (ax, axh) in enumerate(axes):
        axh.set_xlabel(parameter_labels[i])
        ax.set_ylabel(parameter_labels[i])

    fig.tight_layout()
    filename = "{}/{}_checkpoint_trace.png".format(outdir, label)
    fig.savefig(filename)
    plt.close(fig)


def plot_tau(
    tau_list_n, tau_list, search_parameter_keys, outdir, label, tau, autocorr_tau,
):
    fig, ax = plt.subplots()
    for i, key in enumerate(search_parameter_keys):
        ax.plot(tau_list_n, np.array(tau_list)[:, i], label=key)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\langle \tau \rangle$")
    ax.legend()
    fig.tight_layout()
    fig.savefig("{}/{}_checkpoint_tau.png".format(outdir, label))
    plt.close(fig)


def plot_mean_log_posterior(mean_log_posterior, outdir, label):

    ntemps, nsteps = mean_log_posterior.shape
    ymax = np.max(mean_log_posterior)
    ymin = np.min(mean_log_posterior[:, -100:])
    ymax += 0.1 * (ymax - ymin)
    ymin -= 0.1 * (ymax - ymin)

    fig, ax = plt.subplots()
    idxs = np.arange(nsteps)
    ax.plot(idxs, mean_log_posterior.T)
    ax.set(xlabel="Iteration", ylabel=r"$\langle\mathrm{log-posterior}\rangle$",
           ylim=(ymin, ymax))
    fig.tight_layout()
    fig.savefig("{}/{}_checkpoint_meanlogposterior.png".format(outdir, label))
    plt.close(fig)


def compute_evidence(sampler, log_likelihood_array, outdir, label, discard, nburn, thin,
                     iteration, make_plots=True):
    """ Computes the evidence using thermodynamic integration """
    betas = sampler.betas
    # We compute the evidence without the burnin samples, but we do not thin
    lnlike = log_likelihood_array[:, :, discard + nburn : iteration]
    mean_lnlikes = np.mean(np.mean(lnlike, axis=1), axis=1)

    mean_lnlikes = mean_lnlikes[::-1]
    betas = betas[::-1]

    if any(np.isinf(mean_lnlikes)):
        logger.warning(
            "mean_lnlikes contains inf: recalculating without"
            " the {} infs".format(len(betas[np.isinf(mean_lnlikes)]))
        )
        idxs = np.isinf(mean_lnlikes)
        mean_lnlikes = mean_lnlikes[~idxs]
        betas = betas[~idxs]

    lnZ = np.trapz(mean_lnlikes, betas)
    z1 = np.trapz(mean_lnlikes, betas)
    z2 = np.trapz(mean_lnlikes[::-1][::2][::-1], betas[::-1][::2][::-1])
    lnZerr = np.abs(z1 - z2)

    if make_plots:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 8))
        ax1.semilogx(betas, mean_lnlikes, "-o")
        ax1.set_xlabel(r"$\beta$")
        ax1.set_ylabel(r"$\langle \log(\mathcal{L}) \rangle$")
        min_betas = []
        evidence = []
        for i in range(int(len(betas) / 2.0)):
            min_betas.append(betas[i])
            evidence.append(np.trapz(mean_lnlikes[i:], betas[i:]))

        ax2.semilogx(min_betas, evidence, "-o")
        ax2.set_ylabel(
            r"$\int_{\beta_{min}}^{\beta=1}" + r"\langle \log(\mathcal{L})\rangle d\beta$",
            size=16,
        )
        ax2.set_xlabel(r"$\beta_{min}$")
        plt.tight_layout()
        fig.savefig("{}/{}_beta_lnl.png".format(outdir, label))

    return lnZ, lnZerr


def do_nothing_function():
    """ This is a do-nothing function, we overwrite the likelihood and prior elsewhere """
    pass


likelihood = None
priors = None


def init(likelihood_in, priors_in):
    global likelihood
    global priors
    likelihood = likelihood_in
    priors = priors_in


class LikePriorEvaluator(object):
    """
    This class is copied and modified from ptemcee.LikePriorEvaluator, see
    https://github.com/willvousden/ptemcee for the original version

    We overwrite the logl and logp methods in order to improve the performance
    when using a MultiPool object: essentially reducing the amount of data
    transfer overhead.

    """

    def __init__(self, search_parameter_keys, use_ratio=False):
        self.search_parameter_keys = search_parameter_keys
        self.use_ratio = use_ratio
        self.periodic_set = False

    def _setup_periodic(self):
        self._periodic = [
            priors[key].boundary == "periodic" for key in self.search_parameter_keys
        ]
        priors.sample()
        self._minima = np.array([
            priors[key].minimum for key in self.search_parameter_keys
        ])
        self._range = np.array([
            priors[key].maximum for key in self.search_parameter_keys
        ]) - self._minima
        self.periodic_set = True

    def _wrap_periodic(self, array):
        if not self.periodic_set:
            self._setup_periodic()
        array[self._periodic] = np.mod(
            array[self._periodic] - self._minima[self._periodic],
            self._range[self._periodic]
        ) + self._minima[self._periodic]
        return array

    def logl(self, v_array):
        parameters = {key: v for key, v in zip(self.search_parameter_keys, v_array)}
        if priors.evaluate_constraints(parameters) > 0:
            likelihood.parameters.update(parameters)
            if self.use_ratio:
                return likelihood.log_likelihood() - likelihood.noise_log_likelihood()
            else:
                return likelihood.log_likelihood()
        else:
            return np.nan_to_num(-np.inf)

    def logp(self, v_array):
        params = {key: t for key, t in zip(self.search_parameter_keys, v_array)}
        return priors.ln_prob(params)

    def __call__(self, x):
        lp = self.logp(x)
        if np.isnan(lp):
            raise ValueError("Prior function returned NaN.")

        if lp == float("-inf"):
            # Can't return -inf, since this messes with beta=0 behaviour.
            ll = 0
        else:
            ll = self.logl(x)
            if np.isnan(ll).any():
                raise ValueError("Log likelihood function returned NaN.")

        return ll, lp
