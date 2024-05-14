import os
import shutil
from shutil import copyfile

import numpy as np

from .base_sampler import SamplerError, signal_wrapper
from .emcee import Emcee
from .ptemcee import LikePriorEvaluator

_evaluator = LikePriorEvaluator()


class Zeus(Emcee):
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

    sampler_name = "zeus"
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
        **kwargs,
    ):
        super(Zeus, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=use_ratio,
            plot=plot,
            skip_import_verification=skip_import_verification,
            pos0=pos0,
            nburn=nburn,
            burn_in_fraction=burn_in_fraction,
            resume=resume,
            burn_in_act=burn_in_act,
            **kwargs,
        )

    def _translate_kwargs(self, kwargs):
        super(Zeus, self)._translate_kwargs(kwargs=kwargs)

        # check if using emcee-style arguments
        if "start" not in kwargs:
            if "rstate0" in kwargs:
                kwargs["start"] = kwargs.pop("rstate0")
        if "log_prob0" not in kwargs:
            if "lnprob0" in kwargs:
                kwargs["log_prob0"] = kwargs.pop("lnprob0")

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

        init_kwargs["logprob_fn"] = _evaluator.call_emcee
        init_kwargs["ndim"] = self.ndim

        return init_kwargs

    def write_current_state(self):
        self._sampler.distribute = map
        super(Zeus, self).write_current_state()
        self._sampler.distribute = getattr(self._sampler.pool, "map", map)

    def _initialise_sampler(self):
        from zeus import EnsembleSampler

        self._sampler = EnsembleSampler(**self.sampler_init_kwargs)
        self._init_chain_file()

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

    def _set_pos0_for_resume(self):
        self.pos0 = self.sampler.get_last_sample()

    @signal_wrapper
    def run_sampler(self):
        self._setup_pool()
        sampler_function_kwargs = self.sampler_function_kwargs
        iterations = sampler_function_kwargs.pop("iterations")
        iterations -= self._previous_iterations

        sampler_function_kwargs["start"] = self.pos0

        # main iteration loop
        for sample in self.sampler.sample(
            iterations=iterations, **sampler_function_kwargs
        ):
            self.write_chains_to_file(sample)
        self._close_pool()
        self.write_current_state()

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
                f"`nburn < nsteps` ({self.result.nburn} < {self.nsteps})."
                " Try increasing the number of steps."
            )
        blobs = np.array(self.sampler.get_blobs(flat=True, discard=self.nburn)).reshape(
            (-1, 2)
        )
        log_likelihoods, log_priors = blobs.T
        self.result.log_likelihood_evaluations = log_likelihoods
        self.result.log_prior_evaluations = log_priors
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
