import datetime
import time

import numpy as np

from ..utils import logger
from .base_sampler import NestedSampler, _TemporaryFileSamplerMixin, signal_wrapper


class _DNest4Model(object):
    def __init__(
        self, log_likelihood_func, from_prior_func, widths, centers, highs, lows
    ):
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
        from ..utils.random import rng

        idx = rng.integers(self._n_dim)

        coords[idx] += self._widths[idx] * (rng.uniform(size=1) - 0.5)
        cw = self._widths[idx]
        cc = self._centers[idx]

        coords[idx] = self.wrap(coords[idx], (cc - 0.5 * cw), cc + 0.5 * cw)

        return 0.0

    @staticmethod
    def wrap(x, minimum, maximum):
        if maximum <= minimum:
            raise ValueError(
                f"maximum {maximum} <= minimum {minimum}, when trying to wrap coordinates"
            )
        return (x - minimum) % (maximum - minimum) + minimum


class DNest4(_TemporaryFileSamplerMixin, NestedSampler):

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

    sampler_name = "d4nest"
    default_kwargs = dict(
        max_num_levels=20,
        num_steps=500,
        new_level_interval=10000,
        num_per_step=10000,
        thread_steps=1,
        num_particles=1000,
        lam=10.0,
        beta=100,
        seed=None,
        verbose=True,
        outputfiles_basename=None,
        backend="memory",
    )
    short_name = "dn4"
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
        **kwargs,
    ):
        super(DNest4, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=use_ratio,
            plot=plot,
            skip_import_verification=skip_import_verification,
            temporary_directory=temporary_directory,
            exit_code=exit_code,
            **kwargs,
        )

        self.num_particles = self.kwargs["num_particles"]
        self.max_num_levels = self.kwargs["max_num_levels"]
        self._verbose = self.kwargs["verbose"]
        self._backend = self.kwargs["backend"]

        self.start_time = np.nan
        self.sampler = None
        self._information = np.nan

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

        self._dnest4_model = _DNest4Model(
            self.log_likelihood,
            self.get_random_draw_from_prior,
            self._widths,
            self._centers,
            self._highs,
            self._lows,
        )

    def _set_backend(self):
        import dnest4

        if self._backend == "csv":
            return dnest4.backends.CSVBackend(
                f"{self.outdir}/dnest4{self.label}/", sep=" "
            )
        else:
            return dnest4.backends.MemoryBackend()

    def _set_dnest4_kwargs(self):
        dnest4_keys = ["num_steps", "new_level_interval", "lam", "beta", "seed"]
        self.dnest4_kwargs = {key: self.kwargs[key] for key in dnest4_keys}

    @signal_wrapper
    def run_sampler(self):
        import dnest4

        self._set_dnest4_kwargs()
        backend = self._set_backend()

        self._verify_kwargs_against_default_kwargs()
        self._setup_run_directory()
        self._check_and_load_sampling_time_file()
        self.start_time = time.time()

        self.sampler = dnest4.DNest4Sampler(self._dnest4_model, backend=backend)
        out = self.sampler.sample(
            self.max_num_levels, num_particles=self.num_particles, **self.dnest4_kwargs
        )

        for i, sample in enumerate(out):
            if self._verbose and ((i + 1) % 100 == 0):
                stats = self.sampler.postprocess()
                logger.info(f"Iteration: {i + 1} log(Z): {stats['log_Z']}")

        self._calculate_and_save_sampling_time()
        self._clean_up_run_directory()

        stats = self.sampler.postprocess(resample=1)
        self.result.log_evidence = stats["log_Z"]
        self._information = stats["H"]
        self.result.log_evidence_err = np.sqrt(self._information / self.num_particles)
        self.result.samples = np.array(self.sampler.backend.posterior_samples)

        self.result.sampler_output = out
        self.result.outputfiles_basename = self.outputfiles_basename
        self.result.sampling_time = datetime.timedelta(seconds=self.total_sampling_time)

        self.calc_likelihood_count()

        return self.result

    def _translate_kwargs(self, kwargs):
        kwargs = super()._translate_kwargs(kwargs)
        if "num_steps" not in kwargs:
            for equiv in self.walks_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["num_steps"] = kwargs.pop(equiv)

    def _verify_kwargs_against_default_kwargs(self):
        self.outputfiles_basename = self.kwargs.pop("outputfiles_basename", None)
        super(DNest4, self)._verify_kwargs_against_default_kwargs()
