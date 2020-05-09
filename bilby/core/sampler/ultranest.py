from __future__ import absolute_import

import os
import shutil
import signal
import tempfile

import numpy as np
from pandas import DataFrame

from ..utils import check_directory_exists_and_if_not_mkdir, logger
from .base_sampler import NestedSampler


class Ultranest(NestedSampler):
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

    Other Parameters
    ----------------
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
        update_interval_iter_fraction=0.2,
        viz_callback="auto",
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
            **kwargs,
        )
        self._apply_ultranest_boundaries()

        signal.signal(signal.SIGTERM, self.write_current_state_and_exit)
        signal.signal(signal.SIGINT, self.write_current_state_and_exit)
        signal.signal(signal.SIGALRM, self.write_current_state_and_exit)

    def _translate_kwargs(self, kwargs):
        if "num_live_points" not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["num_live_points"] = kwargs.pop(equiv)

        if "verbose" in kwargs and "show_status" not in kwargs:
            kwargs["show_status"] = kwargs.pop("verbose")

    def _verify_kwargs_against_default_kwargs(self):
        """ Check the kwargs """

        self.outputfiles_basename = self.kwargs.pop("log_dir", None)

        NestedSampler._verify_kwargs_against_default_kwargs(self)

    def _apply_ultranest_boundaries(self):
        if (
            self.kwargs["wrapped_params"] is None
            or len(self.kwargs.get("wrapped_params", [])) == 0
        ):
            self.kwargs["wrapped_params"] = []
            for param, value in self.priors.items():
                if value.boundary == "periodic":
                    self.kwargs["wrapped_params"].append(1)
                else:
                    self.kwargs["wrapped_params"].append(0)

    @property
    def outputfiles_basename(self):
        return self._outputfiles_basename

    @outputfiles_basename.setter
    def outputfiles_basename(self, outputfiles_basename):
        if outputfiles_basename is None:
            outputfiles_basename = "{}/ultra_{}".format(self.outdir, self.label)
        if outputfiles_basename.endswith("/") is True:
            outputfiles_basename = outputfiles_basename.rstrip("/")
        check_directory_exists_and_if_not_mkdir(self.outdir)
        self._outputfiles_basename = outputfiles_basename

    @property
    def temporary_outputfiles_basename(self):
        return self._temporary_outputfiles_basename

    @temporary_outputfiles_basename.setter
    def temporary_outputfiles_basename(self, temporary_outputfiles_basename):
        if temporary_outputfiles_basename.endswith("/") is False:
            temporary_outputfiles_basename = "{}/".format(
                temporary_outputfiles_basename
            )
        self._temporary_outputfiles_basename = temporary_outputfiles_basename
        if os.path.exists(self.outputfiles_basename):
            shutil.copytree(
                self.outputfiles_basename, self.temporary_outputfiles_basename
            )
            if os.path.islink(self.outputfiles_basename):
                os.unlink(self.outputfiles_basename)
            else:
                shutil.rmtree(self.outputfiles_basename)

    def write_current_state_and_exit(self, signum=None, frame=None):
        """ Write current state and exit on exit_code """
        logger.info(
            "Run interrupted by signal {}: checkpoint and exit on {}".format(
                signum, self.exit_code
            )
        )
        # self.copy_temporary_directory_to_proper_path()
        os._exit(self.exit_code)

    def copy_temporary_directory_to_proper_path(self):
        logger.info(
            "Overwriting {} with {}".format(
                self.outputfiles_basename, self.temporary_outputfiles_basename
            )
        )

        # First remove anything in the outputfiles_basename for overwriting
        if os.path.exists(self.outputfiles_basename):
            if os.path.islink(self.outputfiles_basename):
                os.unlink(self.outputfiles_basename)
            else:
                shutil.rmtree(self.outputfiles_basename, ignore_errors=True)

        shutil.copytree(self.temporary_outputfiles_basename, self.outputfiles_basename)

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
                "update_interval_iter_fraction",
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

    def run_sampler(self):
        import ultranest
        import ultranest.stepsampler

        if self.kwargs["dlogz"] is None:
            # remove dlogz, so ultranest defaults (which are different for
            # NestedSampler and ReactiveNestedSampler) are used
            self.kwargs.pop("dlogz")

        self._verify_kwargs_against_default_kwargs()

        stepsampler = self.kwargs.pop("step_sampler", None)

        temporary_outputfiles_basename = tempfile.TemporaryDirectory().name
        self.temporary_outputfiles_basename = temporary_outputfiles_basename
        logger.info("Using temporary file {}".format(temporary_outputfiles_basename))

        check_directory_exists_and_if_not_mkdir(temporary_outputfiles_basename)
        self.kwargs["log_dir"] = self.temporary_outputfiles_basename

        # Symlink the temporary directory with the target directory: ensures data is stored on exit
        os.symlink(
            os.path.abspath(self.temporary_outputfiles_basename),
            os.path.abspath(self.outputfiles_basename),
            target_is_directory=True,
        )

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

        results = sampler.run(**self.sampler_function_kwargs)

        self.copy_temporary_directory_to_proper_path()

        # Clean up
        shutil.rmtree(temporary_outputfiles_basename)

        self._generate_result(results)
        self.calc_likelihood_count()

        return self.result

    def _generate_result(self, out):
        # extract results (samples stored in "v" will change to "points",
        # weights stored in "w" will change to "weights")
        datakey = "v" if "v" in out["weighted_samples"] else "points"
        weightskey = "w" if "w" in out["weighted_samples"] else "weights"
        data = np.array(out["weighted_samples"][datakey])
        weights = np.array(out["weighted_samples"][weightskey])

        scaledweights = weights / weights.max()
        mask = np.random.rand(len(scaledweights)) < scaledweights

        nested_samples = DataFrame(data, columns=self.search_parameter_keys)
        nested_samples["weights"] = weights
        nested_samples["log_likelihood"] = out["weighted_samples"]["L"]
        self.result.log_likelihood_evaluations = np.array(out["weighted_samples"]["L"])[
            mask
        ]
        self.result.sampler_output = out
        self.result.samples = data[mask, :]
        self.result.nested_samples = nested_samples
        self.result.log_evidence = out["logz"]
        self.result.log_evidence_err = out["logzerr"]

        self.result.outputfiles_basename = self.outputfiles_basename
