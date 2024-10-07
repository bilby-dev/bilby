import datetime
import importlib
import os
import time

import numpy as np

from .base_sampler import NestedSampler, _TemporaryFileSamplerMixin, signal_wrapper


class Pymultinest(_TemporaryFileSamplerMixin, NestedSampler):
    """
    bilby wrapper of pymultinest
    (https://github.com/JohannesBuchner/PyMultiNest)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `pymultinest.run`, see documentation
    for that class for further help. Under Other Parameters, we list commonly
    used kwargs and the bilby defaults.

    Parameters
    ==========
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

    sampler_name = "pymultinest"
    abbreviation = "pm"
    default_kwargs = dict(
        importance_nested_sampling=False,
        resume=True,
        verbose=True,
        sampling_efficiency="parameter",
        n_live_points=500,
        n_params=None,
        n_clustering_params=None,
        wrapped_params=None,
        multimodal=True,
        const_efficiency_mode=False,
        evidence_tolerance=0.5,
        n_iter_before_update=100,
        null_log_evidence=-1e90,
        max_modes=100,
        mode_tolerance=-1e90,
        outputfiles_basename=None,
        seed=-1,
        context=0,
        write_output=True,
        log_zero=-1e100,
        max_iter=0,
        init_MPI=False,
        dump_callback=None,
    )
    short_name = "pm"
    hard_exit = True
    sampling_seed_key = "seed"

    def __init__(
        self,
        likelihood,
        priors,
        outdir="outdir",
        label="label",
        use_ratio=False,
        plot=False,
        exit_code=77,
        skip_import_verification=False,
        temporary_directory=True,
        **kwargs
    ):
        super(Pymultinest, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=use_ratio,
            plot=plot,
            skip_import_verification=skip_import_verification,
            exit_code=exit_code,
            temporary_directory=temporary_directory,
            **kwargs
        )
        self._apply_multinest_boundaries()

    def _translate_kwargs(self, kwargs):
        kwargs = super()._translate_kwargs(kwargs)
        if "n_live_points" not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["n_live_points"] = kwargs.pop(equiv)

    def _verify_kwargs_against_default_kwargs(self):
        """Check the kwargs"""

        self.outputfiles_basename = self.kwargs.pop("outputfiles_basename", None)

        # for PyMultiNest >=2.9 the n_params kwarg cannot be None
        if self.kwargs["n_params"] is None:
            self.kwargs["n_params"] = self.ndim
        if self.kwargs["dump_callback"] is None:
            self.kwargs["dump_callback"] = self._dump_callback
        NestedSampler._verify_kwargs_against_default_kwargs(self)

    def _dump_callback(self, *args, **kwargs):
        if self.use_temporary_directory:
            self._copy_temporary_directory_contents_to_proper_path()
        self._calculate_and_save_sampling_time()

    def _apply_multinest_boundaries(self):
        if self.kwargs["wrapped_params"] is None:
            self.kwargs["wrapped_params"] = list()
            for param in self.search_parameter_keys:
                if self.priors[param].boundary == "periodic":
                    self.kwargs["wrapped_params"].append(1)
                else:
                    self.kwargs["wrapped_params"].append(0)

    @signal_wrapper
    def run_sampler(self):
        import pymultinest

        self._verify_kwargs_against_default_kwargs()

        self._setup_run_directory()
        self._check_and_load_sampling_time_file()
        if not self.kwargs["resume"]:
            self.total_sampling_time = 0.0

        # Overwrite pymultinest's signal handling function
        pm_run = importlib.import_module("pymultinest.run")
        pm_run.interrupt_handler = self.write_current_state_and_exit

        self.start_time = time.time()
        out = pymultinest.solve(
            LogLikelihood=self.log_likelihood,
            Prior=self.prior_transform,
            n_dims=self.ndim,
            **self.kwargs
        )
        self._calculate_and_save_sampling_time()

        self._clean_up_run_directory()

        post_equal_weights = os.path.join(
            self.outputfiles_basename, "post_equal_weights.dat"
        )
        post_equal_weights_data = np.loadtxt(post_equal_weights)
        self.result.log_likelihood_evaluations = post_equal_weights_data[:, -1]
        self.result.sampler_output = out
        self.result.samples = post_equal_weights_data[:, :-1]
        self.result.log_evidence = out["logZ"]
        self.result.log_evidence_err = out["logZerr"]
        self.calc_likelihood_count()
        self.result.outputfiles_basename = self.outputfiles_basename
        self.result.sampling_time = datetime.timedelta(seconds=self.total_sampling_time)
        self.result.nested_samples = self._nested_samples
        return self.result

    @property
    def _nested_samples(self):
        """
        Extract nested samples from the pymultinest files.
        This requires combining the "dead" points from `ev.dat` and the "live"
        points from `phys_live.points`.
        The prior volume associated with the current live points is the simple
        estimate of `remaining_prior_volume / N`.
        """
        import pandas as pd

        dir_ = self.kwargs["outputfiles_basename"]
        dead_points = np.genfromtxt(dir_ + "/ev.dat")
        live_points = np.genfromtxt(dir_ + "/phys_live.points")

        nlive = self.kwargs["n_live_points"]
        final_log_prior_volume = -len(dead_points) / nlive - np.log(nlive)
        live_points = np.insert(live_points, -1, final_log_prior_volume, axis=-1)

        nested_samples = pd.DataFrame(
            np.vstack([dead_points, live_points]).copy(),
            columns=self.search_parameter_keys
            + ["log_likelihood", "log_prior_volume", "mode"],
        )
        return nested_samples
