import datetime
import inspect
import time

import numpy as np
from pandas import DataFrame

from ..utils import logger
from .base_sampler import NestedSampler, _TemporaryFileSamplerMixin, signal_wrapper


class Ultranest(_TemporaryFileSamplerMixin, NestedSampler):
    """
    bilby wrapper of ultranest
    (https://johannesbuchner.github.io/UltraNest/index.html)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `ultranest.ReactiveNestedSampler.run`
    or `ultranest.NestedSampler.run`, see documentation for those classes for
    further help. Under Other Parameters, we list commonly used kwargs and the
    bilby defaults. If the number of live points is specified the
    `ultranest.NestedSampler` will be used, otherwise the
    `ultranest.ReactiveNestedSampler` will be used.

    Parameters
    ==========
    num_live_points: int
        The number of live points, note this can also equivalently be given as
        one of [nlive, nlives, n_live_points, num_live_points]. If not given
        then the `ultranest.ReactiveNestedSampler` will be used, which does not
        require the number of live points to be specified.
    show_status: Bool
        If true, print information information about the convergence during
    resume: bool
        If true, resume run from checkpoint (if available)
    step_sampler:
        An UltraNest step sampler object. This defaults to None, so the default
        stepping behaviour is used.
    """

    sampler_name = "ultranest"
    abbreviation = "ultra"
    default_kwargs = dict(
        resume=True,
        show_status=True,
        num_live_points=None,
        wrapped_params=None,
        log_dir=None,
        derived_param_names=[],
        run_num=None,
        vectorized=False,
        num_test_samples=2,
        draw_multiple=True,
        num_bootstraps=30,
        update_interval_iter=None,
        update_interval_ncall=None,
        log_interval=None,
        dlogz=None,
        max_iters=None,
        update_interval_volume_fraction=0.2,
        viz_callback=None,
        dKL=0.5,
        frac_remain=0.01,
        Lepsilon=0.001,
        min_ess=400,
        max_ncalls=None,
        max_num_improvement_loops=-1,
        min_num_live_points=400,
        cluster_num_live_points=40,
        step_sampler=None,
    )

    short_name = "ultra"

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
        callback_interval=10,
        **kwargs,
    ):
        super(Ultranest, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=use_ratio,
            plot=plot,
            skip_import_verification=skip_import_verification,
            exit_code=exit_code,
            temporary_directory=temporary_directory,
            **kwargs,
        )
        self._apply_ultranest_boundaries()

        if self.use_temporary_directory:
            # set callback interval, so copying of results does not thrash the
            # disk (ultranest will call viz_callback quite a lot)
            self.callback_interval = callback_interval

    def _translate_kwargs(self, kwargs):
        kwargs = super()._translate_kwargs(kwargs)
        if "num_live_points" not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["num_live_points"] = kwargs.pop(equiv)
        if "verbose" in kwargs and "show_status" not in kwargs:
            kwargs["show_status"] = kwargs.pop("verbose")
        resume = kwargs.get("resume", False)
        if resume is True:
            kwargs["resume"] = "overwrite"
        elif resume is False:
            kwargs["resume"] = "overwrite"

    def _verify_kwargs_against_default_kwargs(self):
        """Check the kwargs"""

        self.outputfiles_basename = self.kwargs.pop("log_dir", None)
        if self.kwargs["viz_callback"] is None:
            self.kwargs["viz_callback"] = self._viz_callback

        NestedSampler._verify_kwargs_against_default_kwargs(self)

    def _viz_callback(self, *args, **kwargs):
        if self.use_temporary_directory:
            if not (self._viz_callback_counter % self.callback_interval):
                self._copy_temporary_directory_contents_to_proper_path()
                self._calculate_and_save_sampling_time()
            self._viz_callback_counter += 1

    def _apply_ultranest_boundaries(self):
        if (
            self.kwargs["wrapped_params"] is None
            or len(self.kwargs.get("wrapped_params", [])) == 0
        ):
            self.kwargs["wrapped_params"] = []
            for param, value in self.priors.items():
                if param in self.search_parameter_keys:
                    if value.boundary == "periodic":
                        self.kwargs["wrapped_params"].append(1)
                    else:
                        self.kwargs["wrapped_params"].append(0)

    def _copy_temporary_directory_contents_to_proper_path(self):
        """
        Copy the temporary back to the proper path.
        Do not delete the temporary directory.
        """
        if inspect.stack()[1].function != "_viz_callback":
            super(Ultranest, self)._copy_temporary_directory_contents_to_proper_path()

    @property
    def sampler_function_kwargs(self):
        if self.kwargs.get("num_live_points", None) is not None:
            keys = [
                "update_interval_iter",
                "update_interval_ncall",
                "log_interval",
                "dlogz",
                "max_iters",
            ]
        else:
            keys = [
                "update_interval_volume_fraction",
                "update_interval_ncall",
                "log_interval",
                "show_status",
                "viz_callback",
                "dlogz",
                "dKL",
                "frac_remain",
                "Lepsilon",
                "min_ess",
                "max_iters",
                "max_ncalls",
                "max_num_improvement_loops",
                "min_num_live_points",
                "cluster_num_live_points",
            ]

        function_kwargs = {key: self.kwargs[key] for key in keys if key in self.kwargs}

        return function_kwargs

    @property
    def sampler_init_kwargs(self):
        keys = [
            "derived_param_names",
            "resume",
            "run_num",
            "vectorized",
            "log_dir",
            "wrapped_params",
        ]
        if self.kwargs.get("num_live_points", None) is not None:
            keys += ["num_live_points"]
        else:
            keys += ["num_test_samples", "draw_multiple", "num_bootstraps"]

        init_kwargs = {key: self.kwargs[key] for key in keys if key in self.kwargs}

        return init_kwargs

    @signal_wrapper
    def run_sampler(self):
        import ultranest
        import ultranest.stepsampler

        if self.kwargs["dlogz"] is None:
            # remove dlogz, so ultranest defaults (which are different for
            # NestedSampler and ReactiveNestedSampler) are used
            self.kwargs.pop("dlogz")

        self._verify_kwargs_against_default_kwargs()

        stepsampler = self.kwargs.pop("step_sampler", None)

        self._setup_run_directory()
        self.kwargs["log_dir"] = self.kwargs["outputfiles_basename"]
        self._check_and_load_sampling_time_file()

        # use reactive nested sampler when no live points are given
        if self.kwargs.get("num_live_points", None) is not None:
            integrator = ultranest.integrator.NestedSampler
        else:
            integrator = ultranest.integrator.ReactiveNestedSampler

        sampler = integrator(
            self.search_parameter_keys,
            self.log_likelihood,
            transform=self.prior_transform,
            **self.sampler_init_kwargs,
        )

        if stepsampler is not None:
            if isinstance(stepsampler, ultranest.stepsampler.StepSampler):
                sampler.stepsampler = stepsampler
            else:
                logger.warning(
                    "The supplied step sampler is not the correct type. "
                    "The default step sampling will be used instead."
                )

        if self.use_temporary_directory:
            self._viz_callback_counter = 1

        self.start_time = time.time()
        results = sampler.run(**self.sampler_function_kwargs)
        self._calculate_and_save_sampling_time()

        self._clean_up_run_directory()

        self._generate_result(results)
        self.calc_likelihood_count()

        return self.result

    def _generate_result(self, out):
        # extract results
        from ..utils.random import rng

        data = np.array(out["weighted_samples"]["points"])
        weights = np.array(out["weighted_samples"]["weights"])

        scaledweights = weights / weights.max()
        mask = rng.uniform(0, 1, len(scaledweights)) < scaledweights

        nested_samples = DataFrame(data, columns=self.search_parameter_keys)
        nested_samples["weights"] = weights
        nested_samples["log_likelihood"] = out["weighted_samples"]["logl"]
        self.result.log_likelihood_evaluations = np.array(
            out["weighted_samples"]["logl"]
        )[mask]
        self.result.sampler_output = out
        self.result.samples = data[mask, :]
        self.result.nested_samples = nested_samples
        self.result.log_evidence = out["logz"]
        self.result.log_evidence_err = out["logzerr"]
        if self.kwargs["num_live_points"] is not None:
            self.result.information_gain = (
                np.power(out["logzerr"], 2) * self.kwargs["num_live_points"]
            )

        self.result.outputfiles_basename = self.outputfiles_basename
        self.result.sampling_time = datetime.timedelta(seconds=self.total_sampling_time)

    def log_likelihood(self, theta):
        log_l = super(Ultranest, self).log_likelihood(theta=theta)
        return np.nan_to_num(log_l)
