import multiprocessing
import os
import sys
import threading
import time
from signal import SIGINT

multiprocessing.set_start_method("fork")  # noqa

import unittest
import pytest
from parameterized import parameterized
import shutil

import bilby
import numpy as np


_sampler_kwargs = dict(
    bilby_mcmc=dict(nsamples=200, printdt=1),
    cpnest=dict(nlive=100),
    dnest4=dict(
        max_num_levels=2,
        num_steps=10,
        new_level_interval=10,
        num_per_step=10,
        thread_steps=1,
        num_particles=50,
        max_pool=1,
    ),
    dynesty=dict(nlive=10, sample="acceptance-walk", nact=5, proposals=["diff"]),
    dynamic_dynesty=dict(
        nlive_init=10,
        nlive_batch=10,
        dlogz_init=1.0,
        maxbatch=0,
        maxcall=100,
        sample="act-walk",
    ),
    emcee=dict(iterations=1000, nwalkers=10),
    kombine=dict(iterations=200, nwalkers=10, autoburnin=False),
    nessai=dict(
        nlive=100,
        poolsize=100,
        max_iteration=500,
    ),
    nestle=dict(nlive=100),
    ptemcee=dict(
        nsamples=100,
        nwalkers=50,
        burn_in_act=1,
        ntemps=1,
        frac_threshold=0.5,
    ),
    PTMCMCSampler=dict(Niter=101, burn=100, covUpdate=100, isave=100),
    pymc=dict(draws=50, tune=50, n_init=250),
    pymultinest=dict(nlive=100),
    ultranest=dict(nlive=100, temporary_directory=False),
    zeus=dict(nwalkers=10, iterations=100)
)

sampler_imports = dict(
    bilby_mcmc="bilby",
    dynamic_dynesty="dynesty"
)

no_pool_test = ["dnest4", "pymultinest", "nestle", "ptmcmcsampler", "ultranest", "pymc"]

loaded_samplers = {k: v.load() for k, v in bilby.core.sampler.IMPLEMENTED_SAMPLERS.items()}


def slow_func(x, m, c):
    time.sleep(0.01)
    return m * x + c


def model(x, m, c):
    return m * x + c


class TestRunningSamplers(unittest.TestCase):
    def setUp(self):
        bilby.core.utils.random.seed(42)
        bilby.core.utils.command_line_args.bilby_test_mode = False
        rng = bilby.core.utils.random.rng
        self.x = np.linspace(0, 1, 11)
        self.injection_parameters = dict(m=0.5, c=0.2)
        self.sigma = 0.1
        self.y = model(self.x, **self.injection_parameters) + rng.normal(
            0, self.sigma, len(self.x)
        )
        self.likelihood = bilby.likelihood.GaussianLikelihood(
            self.x, self.y, model, self.sigma
        )

        self.priors = bilby.core.prior.PriorDict()
        self.priors["m"] = bilby.core.prior.Uniform(0, 5, boundary="periodic")
        self.priors["c"] = bilby.core.prior.Uniform(-2, 2, boundary="reflective")
        self._remove_tree()
        bilby.core.utils.check_directory_exists_and_if_not_mkdir("outdir")

    @staticmethod
    def conversion_function(parameters, likelihood, prior):
        converted = parameters.copy()
        if "derived" not in converted:
            converted["derived"] = converted["m"] * converted["c"]
        return converted

    def tearDown(self):
        del self.likelihood
        del self.priors
        bilby.core.utils.command_line_args.bilby_test_mode = False
        self._remove_tree()

    def _remove_tree(self):
        try:
            shutil.rmtree("outdir")
        except OSError:
            pass

    @parameterized.expand(_sampler_kwargs.keys())
    def test_run_sampler_single(self, sampler):
        self._run_sampler(sampler, pool_size=1)

    @parameterized.expand(_sampler_kwargs.keys())
    def test_run_sampler_pool(self, sampler):
        self._run_sampler(sampler, pool_size=2)

    def _run_sampler(self, sampler, pool_size, **extra_kwargs):
        pytest.importorskip(sampler_imports.get(sampler, sampler))
        if pool_size > 1 and sampler.lower() in no_pool_test:
            pytest.skip(f"{sampler} cannot be parallelized")
        bilby.core.utils.check_directory_exists_and_if_not_mkdir("outdir")
        kwargs = _sampler_kwargs[sampler]
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler=sampler,
            save="hdf5",
            npool=pool_size,
            conversion_function=self.conversion_function,
            **kwargs,
            **extra_kwargs,
        )
        assert "derived" in res.posterior
        if sampler != "dnest4":
            assert res.log_likelihood_evaluations is not None

    @parameterized.expand(_sampler_kwargs.keys())
    def test_interrupt_sampler_single(self, sampler):
        self._run_with_signal_handling(sampler, pool_size=1)

    @parameterized.expand(_sampler_kwargs.keys())
    def test_interrupt_sampler_pool(self, sampler):
        self._run_with_signal_handling(sampler, pool_size=2)

    def _run_with_signal_handling(self, sampler, pool_size=1):
        pytest.importorskip(sampler_imports.get(sampler, sampler))
        if loaded_samplers[sampler.lower()].hard_exit:
            pytest.skip(f"{sampler} hard exits, can't test signal handling.")
        if pool_size > 1 and sampler.lower() in no_pool_test:
            pytest.skip(f"{sampler} cannot be parallelized")
        if sys.version_info.minor == 8 and sampler.lower == "cpnest":
            pytest.skip("Pool interrupting broken for cpnest with py3.8")
        pid = os.getpid()
        print(sampler)

        def trigger_signal():
            # You could do something more robust, e.g. wait until port is listening
            time.sleep(4)
            os.kill(pid, SIGINT)

        thread = threading.Thread(target=trigger_signal)
        thread.daemon = True
        thread.start()

        self.likelihood._func = slow_func

        with self.assertRaises((SystemExit, KeyboardInterrupt)):
            try:
                while True:
                    self._run_sampler(sampler=sampler, pool_size=pool_size, exit_code=5)
            except SystemExit as error:
                self.assertEqual(error.code, 5)
                raise


if __name__ == "__main__":
    unittest.main()
