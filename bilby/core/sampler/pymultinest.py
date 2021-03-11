import importlib
import os
import shutil
import distutils.dir_util
import signal
import time
import datetime
import sys

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
            **kwargs
        )
        self._apply_multinest_boundaries()
        self.exit_code = exit_code
        using_mpi = len([key for key in os.environ if "MPI" in key])
        if using_mpi and temporary_directory:
            logger.info(
                "Temporary directory incompatible with MPI, "
                "will run in original directory"
            )
        self.use_temporary_directory = temporary_directory and not using_mpi

        signal.signal(signal.SIGTERM, self.write_current_state_and_exit)
        signal.signal(signal.SIGINT, self.write_current_state_and_exit)
        signal.signal(signal.SIGALRM, self.write_current_state_and_exit)

    def _translate_kwargs(self, kwargs):
        if "n_live_points" not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["n_live_points"] = kwargs.pop(equiv)

    def _verify_kwargs_against_default_kwargs(self):
        """ Check the kwargs """

        self.outputfiles_basename = self.kwargs.pop("outputfiles_basename", None)

        # for PyMultiNest >=2.9 the n_params kwarg cannot be None
        if self.kwargs["n_params"] is None:
            self.kwargs["n_params"] = self.ndim
        if self.kwargs['dump_callback'] is None:
            self.kwargs['dump_callback'] = self._dump_callback
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

    @property
    def outputfiles_basename(self):
        return self._outputfiles_basename

    @outputfiles_basename.setter
    def outputfiles_basename(self, outputfiles_basename):
        if outputfiles_basename is None:
            outputfiles_basename = "{}/pm_{}/".format(self.outdir, self.label)
        if not outputfiles_basename.endswith("/"):
            outputfiles_basename += "/"
        check_directory_exists_and_if_not_mkdir(self.outdir)
        self._outputfiles_basename = outputfiles_basename

    @property
    def temporary_outputfiles_basename(self):
        return self._temporary_outputfiles_basename

    @temporary_outputfiles_basename.setter
    def temporary_outputfiles_basename(self, temporary_outputfiles_basename):
        if not temporary_outputfiles_basename.endswith("/"):
            temporary_outputfiles_basename = "{}/".format(
                temporary_outputfiles_basename
            )
        self._temporary_outputfiles_basename = temporary_outputfiles_basename
        if os.path.exists(self.outputfiles_basename):
            shutil.copytree(
                self.outputfiles_basename, self.temporary_outputfiles_basename
            )

    def write_current_state_and_exit(self, signum=None, frame=None):
        """ Write current state and exit on exit_code """
        logger.info(
            "Run interrupted by signal {}: checkpoint and exit on {}".format(
                signum, self.exit_code
            )
        )
        self._calculate_and_save_sampling_time()
        if self.use_temporary_directory:
            self._move_temporary_directory_to_proper_path()
        sys.exit(self.exit_code)

    def _copy_temporary_directory_contents_to_proper_path(self):
        """
        Copy the temporary back to the proper path.
        Do not delete the temporary directory.
        """
        logger.info(
            "Overwriting {} with {}".format(
                self.outputfiles_basename, self.temporary_outputfiles_basename
            )
        )
        if self.outputfiles_basename.endswith('/'):
            outputfiles_basename_stripped = self.outputfiles_basename[:-1]
        else:
            outputfiles_basename_stripped = self.outputfiles_basename
        distutils.dir_util.copy_tree(self.temporary_outputfiles_basename, outputfiles_basename_stripped)

    def _move_temporary_directory_to_proper_path(self):
        """
        Copy the temporary back to the proper path

        Anything in the temporary directory at this point is removed
        """
        self._copy_temporary_directory_contents_to_proper_path()
        shutil.rmtree(self.temporary_outputfiles_basename)

    def run_sampler(self):
        import pymultinest

        self._verify_kwargs_against_default_kwargs()

        self._setup_run_directory()
        self._check_and_load_sampling_time_file()

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
        return self.result

    def _check_and_load_sampling_time_file(self):
        self.time_file_path = self.kwargs["outputfiles_basename"] + '/sampling_time.dat'
        if os.path.exists(self.time_file_path):
            with open(self.time_file_path, 'r') as time_file:
                self.total_sampling_time = float(time_file.readline())
        else:
            self.total_sampling_time = 0

    def _calculate_and_save_sampling_time(self):
        current_time = time.time()
        new_sampling_time = current_time - self.start_time
        self.total_sampling_time += new_sampling_time
        self.start_time = current_time
        with open(self.time_file_path, 'w') as time_file:
            time_file.write(str(self.total_sampling_time))

    def _clean_up_run_directory(self):
        if self.use_temporary_directory:
            self._move_temporary_directory_to_proper_path()
            self.kwargs["outputfiles_basename"] = self.outputfiles_basename
