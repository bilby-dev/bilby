import os
import signal
import shutil
import sys
from collections import namedtuple
from shutil import copyfile

import numpy as np
from pandas import DataFrame

from ..utils import logger, check_directory_exists_and_if_not_mkdir
from .base_sampler import MCMCSampler, SamplerError


class Zeus(MCMCSampler):
    """bilby wrapper for Zeus (https://zeus-mcmc.readthedocs.io/)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `zeus.EnsembleSampler`, see
    documentation for that class for further help. Under Other Parameters, we
    list commonly used kwargs and the bilby defaults.

    Parameters
    ==========
    nwalkers: int, (500)
        The number of walkers
    nsteps: int, (100)
        The number of steps
    nburn: int (None)
        If given, the fixed number of steps to discard as burn-in. These will
        be discarded from the total number of steps set by `nsteps` and
        therefore the value must be greater than `nsteps`. Else, nburn is
        estimated from the autocorrelation time
    burn_in_fraction: float, (0.25)
        The fraction of steps to discard as burn-in in the event that the
        autocorrelation time cannot be calculated
    burn_in_act: float
        The number of autocorrelation times to discard as burn-in

    """

    default_kwargs = dict(
        nwalkers=500,
        args=[],
        kwargs={},
        pool=None,
        log_prob0=None,
        start=None,
        blobs0=None,
        iterations=100,
        thin=1,
    )

    def __init__(
        self,
        likelihood,
        priors,
        outdir="outdir",
        label="label",
        use_ratio=False,
        plot=False,
        skip_import_verification=False,
        pos0=None,
        nburn=None,
        burn_in_fraction=0.25,
        resume=True,
        burn_in_act=3,
        **kwargs
    ):
        import zeus

        self.zeus = zeus

        super(Zeus, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=use_ratio,
            plot=plot,
            skip_import_verification=skip_import_verification,
            **kwargs
        )
        self.resume = resume
        self.pos0 = pos0
        self.nburn = nburn
        self.burn_in_fraction = burn_in_fraction
        self.burn_in_act = burn_in_act

        signal.signal(signal.SIGTERM, self.checkpoint_and_exit)
        signal.signal(signal.SIGINT, self.checkpoint_and_exit)

    def _translate_kwargs(self, kwargs):
        if "nwalkers" not in kwargs:
            for equiv in self.nwalkers_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["nwalkers"] = kwargs.pop(equiv)
        if "iterations" not in kwargs:
            if "nsteps" in kwargs:
                kwargs["iterations"] = kwargs.pop("nsteps")

        # check if using emcee-style arguments
        if "start" not in kwargs:
            if "rstate0" in kwargs:
                kwargs["start"] = kwargs.pop("rstate0")
        if "log_prob0" not in kwargs:
            if "lnprob0" in kwargs:
                kwargs["log_prob0"] = kwargs.pop("lnprob0")

        if "threads" in kwargs:
            if kwargs["threads"] != 1:
                logger.warning(
                    "The 'threads' argument cannot be used for "
                    "parallelisation. This run will proceed "
                    "without parallelisation, but consider the use "
                    "of an appropriate Pool object passed to the "
                    "'pool' keyword."
                )
                kwargs["threads"] = 1

    @property
    def sampler_function_kwargs(self):
        keys = ["log_prob0", "start", "blobs0", "iterations", "thin", "progress"]

        function_kwargs = {key: self.kwargs[key] for key in keys if key in self.kwargs}

        return function_kwargs

    @property
    def sampler_init_kwargs(self):
        init_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key not in self.sampler_function_kwargs
        }

        init_kwargs["logprob_fn"] = self.lnpostfn
        init_kwargs["ndim"] = self.ndim

        return init_kwargs

    def lnpostfn(self, theta):
        log_prior = self.log_prior(theta)
        if np.isinf(log_prior):
            return -np.inf, [np.nan, np.nan]
        else:
            log_likelihood = self.log_likelihood(theta)
            return log_likelihood + log_prior, [log_likelihood, log_prior]

    @property
    def nburn(self):
        if type(self.__nburn) in [float, int]:
            return int(self.__nburn)
        elif self.result.max_autocorrelation_time is None:
            return int(self.burn_in_fraction * self.nsteps)
        else:
            return int(self.burn_in_act * self.result.max_autocorrelation_time)

    @nburn.setter
    def nburn(self, nburn):
        if isinstance(nburn, (float, int)):
            if nburn > self.kwargs["iterations"] - 1:
                raise ValueError(
                    "Number of burn-in samples must be smaller "
                    "than the total number of iterations"
                )

        self.__nburn = nburn

    @property
    def nwalkers(self):
        return self.kwargs["nwalkers"]

    @property
    def nsteps(self):
        return self.kwargs["iterations"]

    @nsteps.setter
    def nsteps(self, nsteps):
        self.kwargs["iterations"] = nsteps

    @property
    def stored_chain(self):
        """Read the stored zero-temperature chain data in from disk"""
        return np.genfromtxt(self.checkpoint_info.chain_file, names=True)

    @property
    def stored_samples(self):
        """Returns the samples stored on disk"""
        return self.stored_chain[self.search_parameter_keys]

    @property
    def stored_loglike(self):
        """Returns the log-likelihood stored on disk"""
        return self.stored_chain["log_l"]

    @property
    def stored_logprior(self):
        """Returns the log-prior stored on disk"""
        return self.stored_chain["log_p"]

    def _init_chain_file(self):
        with open(self.checkpoint_info.chain_file, "w+") as ff:
            ff.write(
                "walker\t{}\tlog_l\tlog_p\n".format(
                    "\t".join(self.search_parameter_keys)
                )
            )

    @property
    def checkpoint_info(self):
        """Defines various things related to checkpointing and storing data

        Returns
        =======
        checkpoint_info: named_tuple
            An object with attributes `sampler_file`, `chain_file`, and
            `chain_template`. The first two give paths to where the sampler and
            chain data is stored, the last a formatted-str-template with which
            to write the chain data to disk

        """
        out_dir = os.path.join(
            self.outdir, "{}_{}".format(self.__class__.__name__.lower(), self.label)
        )
        check_directory_exists_and_if_not_mkdir(out_dir)

        chain_file = os.path.join(out_dir, "chain.dat")
        sampler_file = os.path.join(out_dir, "sampler.pickle")
        chain_template = (
            "{:d}" + "\t{:.9e}" * (len(self.search_parameter_keys) + 2) + "\n"
        )

        CheckpointInfo = namedtuple(
            "CheckpointInfo", ["sampler_file", "chain_file", "chain_template"]
        )

        checkpoint_info = CheckpointInfo(
            sampler_file=sampler_file,
            chain_file=chain_file,
            chain_template=chain_template,
        )

        return checkpoint_info

    @property
    def sampler_chain(self):
        nsteps = self._previous_iterations
        return self.sampler.chain[:, :nsteps, :]

    def checkpoint(self):
        """Writes a pickle file of the sampler to disk using dill"""
        import dill

        logger.info(
            "Checkpointing sampler to file {}".format(self.checkpoint_info.sampler_file)
        )
        with open(self.checkpoint_info.sampler_file, "wb") as f:
            # Overwrites the stored sampler chain with one that is truncated
            # to only the completed steps
            self.sampler._chain = self.sampler_chain
            dill.dump(self._sampler, f)

    def checkpoint_and_exit(self, signum, frame):
        logger.info("Received signal {}".format(signum))
        self.checkpoint()
        sys.exit()

    def _initialise_sampler(self):
        self._sampler = self.zeus.EnsembleSampler(**self.sampler_init_kwargs)
        self._init_chain_file()

    @property
    def sampler(self):
        """Returns the Zeus sampler object

        If, already initialized, returns the stored _sampler value. Otherwise,
        first checks if there is a pickle file from which to load. If there is
        not, then initialize the sampler and set the initial random draw

        """
        if hasattr(self, "_sampler"):
            pass
        elif self.resume and os.path.isfile(self.checkpoint_info.sampler_file):
            import dill

            logger.info(
                "Resuming run from checkpoint file {}".format(
                    self.checkpoint_info.sampler_file
                )
            )
            with open(self.checkpoint_info.sampler_file, "rb") as f:
                self._sampler = dill.load(f)
            self._set_pos0_for_resume()
        else:
            self._initialise_sampler()
            self._set_pos0()
        return self._sampler

    def write_chains_to_file(self, sample):
        chain_file = self.checkpoint_info.chain_file
        temp_chain_file = chain_file + ".temp"
        if os.path.isfile(chain_file):
            copyfile(chain_file, temp_chain_file)

        points = np.hstack([sample[0], np.array(sample[2])])

        with open(temp_chain_file, "a") as ff:
            for ii, point in enumerate(points):
                ff.write(self.checkpoint_info.chain_template.format(ii, *point))
        shutil.move(temp_chain_file, chain_file)

    @property
    def _previous_iterations(self):
        """Returns the number of iterations that the sampler has saved

        This is used when loading in a sampler from a pickle file to figure out
        how much of the run has already been completed
        """
        try:
            return len(self.sampler.get_blobs())
        except AttributeError:
            return 0

    def _draw_pos0_from_prior(self):
        return np.array(
            [self.get_random_draw_from_prior() for _ in range(self.nwalkers)]
        )

    @property
    def _pos0_shape(self):
        return (self.nwalkers, self.ndim)

    def _set_pos0(self):
        if self.pos0 is not None:
            logger.debug("Using given initial positions for walkers")
            if isinstance(self.pos0, DataFrame):
                self.pos0 = self.pos0[self.search_parameter_keys].values
            elif type(self.pos0) in (list, np.ndarray):
                self.pos0 = np.squeeze(self.pos0)

            if self.pos0.shape != self._pos0_shape:
                raise ValueError("Input pos0 should be of shape ndim, nwalkers")
            logger.debug("Checking input pos0")
            for draw in self.pos0:
                self.check_draw(draw)
        else:
            logger.debug("Generating initial walker positions from prior")
            self.pos0 = self._draw_pos0_from_prior()

    def _set_pos0_for_resume(self):
        self.pos0 = self.sampler.get_last_sample()

    def run_sampler(self):
        sampler_function_kwargs = self.sampler_function_kwargs
        iterations = sampler_function_kwargs.pop("iterations")
        iterations -= self._previous_iterations

        sampler_function_kwargs["start"] = self.pos0

        # main iteration loop
        for sample in self.sampler.sample(
            iterations=iterations, **sampler_function_kwargs
        ):
            self.write_chains_to_file(sample)
        self.checkpoint()

        self.result.sampler_output = np.nan
        self.calculate_autocorrelation(self.sampler.chain.reshape((-1, self.ndim)))
        self.print_nburn_logging_info()

        self._generate_result()

        self.result.samples = self.sampler.get_chain(flat=True, discard=self.nburn)
        self.result.walkers = self.sampler.chain
        return self.result

    def _generate_result(self):
        self.result.nburn = self.nburn
        self.calc_likelihood_count()
        if self.result.nburn > self.nsteps:
            raise SamplerError(
                "The run has finished, but the chain is not burned in: "
                "`nburn < nsteps` ({} < {}). Try increasing the "
                "number of steps.".format(self.result.nburn, self.nsteps)
            )
        blobs = np.array(self.sampler.get_blobs(flat=True, discard=self.nburn)).reshape((-1, 2))
        log_likelihoods, log_priors = blobs.T
        self.result.log_likelihood_evaluations = log_likelihoods
        self.result.log_prior_evaluations = log_priors
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
