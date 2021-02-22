import os
import shutil
import distutils.dir_util
import signal
import time
import datetime
import sys

import numpy as np
import pandas as pd

from ..utils import check_directory_exists_and_if_not_mkdir, logger
from .base_sampler import NestedSampler


class _DNest4Model(object):

    def __init__(self, log_likelihood_func, from_prior_func, widths, centers, highs, lows):
        """Initialize the DNest4 model.
        Args:
            log_likelihood_func: function
                The loglikelihood function to use during the Nested Sampling run.
            from_prior_func: function
                The function to use when randomly selecting parameter vectors from the prior space.
            widths: array_like
                The approximate widths of the prior distrbutions.
            centers: array_like
                The approximate center points of the prior distributions.
        """
        self._log_likelihood = log_likelihood_func
        self._from_prior = from_prior_func
        self._widths = widths
        self._centers = centers
        self._highs = highs
        self._lows = lows
        self._n_dim = len(widths)
        return

    def log_likelihood(self, coords):
        """The model's log_likelihood function"""
        return self._log_likelihood(coords)

    def from_prior(self):
        """The model's function to select random points from the prior space."""
        return self._from_prior()

    def perturb(self, coords):
        """The perturb function to perform Monte Carlo trial moves."""
        idx = np.random.randint(self._n_dim)

        coords[idx] += (self._widths[idx] * (np.random.uniform(size=1) - 0.5))
        cw = self._widths[idx]
        cc = self._centers[idx]

        coords[idx] = self.wrap(coords[idx], (cc - 0.5 * cw), cc + 0.5 * cw)

        return 0.0

    @staticmethod
    def wrap(x, minimum, maximum):
        if maximum <= minimum:
            raise ValueError("maximum {} <= minimum {}, when trying to wrap coordinates".format(maximum, minimum))
        return (x - minimum) % (maximum - minimum) + minimum


class DNest4(NestedSampler):

    """
    Bilby wrapper of DNest4

    Parameters
    ==========
    TBD

    Other Parameters
    ------==========
    num_particles: int
        The number of points to use in the Nested Sampling active population.
    max_num_levels: int
        The max number of diffusive likelihood levels that DNest4 should initialize
        during the Diffusive Nested Sampling run.
    backend: str
        The python DNest4 backend for storing the output.
        Options are: 'memory' and 'csv'. If 'memory' the
        DNest4 outputs are stored in memory during the run. If 'csv' the
        DNest4 outputs are written out to files with a CSV format during
        the run.
        CSV backend may not be functional right now (October 2020)
    num_steps: int
        The number of MCMC iterations to run
    new_level_interval: int
        The number of moves to run before creating a new diffusive likelihood level
    lam: float
        Set the backtracking scale length
    beta: float
        Set the strength of effect to force the histogram to equal bin counts
    seed: int
        Set the seed for the C++ random number generator
    verbose: Bool
        If True, prints information during run
    """

    default_kwargs = dict(max_num_levels=20, num_steps=500,
                          new_level_interval=10000, num_per_step=10000,
                          thread_steps=1, num_particles=1000, lam=10.0,
                          beta=100, seed=None, verbose=True, outputfiles_basename=None,
                          backend='memory')

    def __init__(self, likelihood, priors, outdir="outdir", label="label", use_ratio=False, plot=False,
                 exit_code=77, skip_import_verification=False, temporary_directory=True, **kwargs):
        super(DNest4, self).__init__(
            likelihood=likelihood, priors=priors, outdir=outdir, label=label,
            use_ratio=use_ratio, plot=plot, skip_import_verification=skip_import_verification,
            exit_code=exit_code, **kwargs)

        self.num_particles = self.kwargs["num_particles"]
        self.max_num_levels = self.kwargs["max_num_levels"]
        self._verbose = self.kwargs["verbose"]
        self._backend = self.kwargs["backend"]
        self.use_temporary_directory = temporary_directory

        self.start_time = np.nan
        self.sampler = None
        self._information = np.nan
        self._last_live_sample_info = np.nan
        self._outputfiles_basename = None
        self._temporary_outputfiles_basename = None

        signal.signal(signal.SIGTERM, self.write_current_state_and_exit)
        signal.signal(signal.SIGINT, self.write_current_state_and_exit)
        signal.signal(signal.SIGALRM, self.write_current_state_and_exit)

        # Get the estimates of the prior distributions' widths and centers.
        widths = []
        centers = []
        highs = []
        lows = []

        samples = self.priors.sample(size=10000)

        for key in self.search_parameter_keys:
            pts = samples[key]
            low = pts.min()
            high = pts.max()
            width = high - low
            center = (high + low) / 2.0
            widths.append(width)
            centers.append(center)

            highs.append(high)
            lows.append(low)

        self._widths = np.array(widths)
        self._centers = np.array(centers)
        self._highs = np.array(highs)
        self._lows = np.array(lows)

        self._dnest4_model = _DNest4Model(self.log_likelihood, self.get_random_draw_from_prior, self._widths,
                                          self._centers, self._highs, self._lows)

    def _set_backend(self):
        import dnest4
        if self._backend == 'csv':
            return dnest4.backends.CSVBackend("{}/dnest4{}/".format(self.outdir, self.label), sep=" ")
        else:
            return dnest4.backends.MemoryBackend()

    def _set_dnest4_kwargs(self):
        dnest4_keys = ["num_steps", "new_level_interval", "lam", "beta", "seed"]
        self.dnest4_kwargs = {key: self.kwargs[key] for key in dnest4_keys}

    def run_sampler(self):
        import dnest4

        self._set_dnest4_kwargs()
        backend = self._set_backend()

        self._verify_kwargs_against_default_kwargs()
        self._setup_run_directory()
        self._check_and_load_sampling_time_file()
        self.start_time = time.time()

        self.sampler = dnest4.DNest4Sampler(self._dnest4_model, backend=backend)
        out = self.sampler.sample(self.max_num_levels,
                                  num_particles=self.num_particles,
                                  **self.dnest4_kwargs)

        for i, sample in enumerate(out):
            if self._verbose and ((i + 1) % 100 == 0):
                stats = self.sampler.postprocess()
                logger.info("Iteration: {0} log(Z): {1}".format(i + 1, stats['log_Z']))

        self._calculate_and_save_sampling_time()
        self._clean_up_run_directory()

        stats = self.sampler.postprocess(resample=1)
        self.result.log_evidence = stats['log_Z']
        self._information = stats['H']
        self.result.log_evidence_err = np.sqrt(self._information / self.num_particles)

        if self._backend == 'memory':
            self._last_live_sample_info = pd.DataFrame(self.sampler.backend.sample_info[-1])
            self.result.log_likelihood_evaluations = self._last_live_sample_info['log_likelihood']
            self.result.samples = np.array(self.sampler.backend.posterior_samples)
        else:
            sample_info_path = './' + self.kwargs["outputfiles_basename"] + '/sample_info.txt'
            sample_info = np.genfromtxt(sample_info_path, comments='#', names=True)
            self.result.log_likelihood_evaluations = sample_info['log_likelihood']
            self.result.samples = np.array(self.sampler.backend.posterior_samples)

        self.result.sampler_output = out
        self.result.outputfiles_basename = self.outputfiles_basename
        self.result.sampling_time = datetime.timedelta(seconds=self.total_sampling_time)

        self.calc_likelihood_count()

        return self.result

    def _translate_kwargs(self, kwargs):
        if 'num_steps' not in kwargs:
            for equiv in self.walks_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['num_steps'] = kwargs.pop(equiv)

    def _verify_kwargs_against_default_kwargs(self):
        self.outputfiles_basename = self.kwargs.pop("outputfiles_basename", None)
        super(DNest4, self)._verify_kwargs_against_default_kwargs()

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

        with open(self.time_file_path, 'w') as time_file:
            time_file.write(str(self.total_sampling_time))

        self.start_time = current_time

    def _clean_up_run_directory(self):
        if self.use_temporary_directory:
            self._move_temporary_directory_to_proper_path()
            self.kwargs["outputfiles_basename"] = self.outputfiles_basename

    @property
    def outputfiles_basename(self):
        return self._outputfiles_basename

    @outputfiles_basename.setter
    def outputfiles_basename(self, outputfiles_basename):
        if outputfiles_basename is None:
            outputfiles_basename = "{}/dnest4{}/".format(self.outdir, self.label)
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

    def _move_temporary_directory_to_proper_path(self):
        """
        Move the temporary back to the proper path

        Anything in the proper path at this point is removed including links
        """
        self._copy_temporary_directory_contents_to_proper_path()
        shutil.rmtree(self.temporary_outputfiles_basename)

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
