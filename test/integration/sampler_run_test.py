import unittest
import pytest
import shutil

import bilby
import numpy as np


class TestRunningSamplers(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        bilby.core.utils.command_line_args.bilby_test_mode = False
        self.x = np.linspace(0, 1, 11)
        self.model = lambda x, m, c: m * x + c
        self.injection_parameters = dict(m=0.5, c=0.2)
        self.sigma = 0.1
        self.y = self.model(self.x, **self.injection_parameters) + np.random.normal(
            0, self.sigma, len(self.x)
        )
        self.likelihood = bilby.likelihood.GaussianLikelihood(
            self.x, self.y, self.model, self.sigma
        )

        self.priors = bilby.core.prior.PriorDict()
        self.priors["m"] = bilby.core.prior.Uniform(0, 5, boundary="periodic")
        self.priors["c"] = bilby.core.prior.Uniform(-2, 2, boundary="reflective")
        self.kwargs = dict(
            save=False,
            conversion_function=self.conversion_function,
            verbose=True,
        )
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
        shutil.rmtree("outdir")

    def test_run_cpnest(self):
        pytest.importorskip("cpnest")
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="cpnest",
            nlive=100,
            resume=False,
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

    def test_run_dnest4(self):
        pytest.importorskip("dnest4")
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="dnest4",
            max_num_levels=2,
            num_steps=10,
            new_level_interval=10,
            num_per_step=10,
            thread_steps=1,
            num_particles=50,
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

    def test_run_dynesty(self):
        pytest.importorskip("dynesty")
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="dynesty",
            nlive=100,
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

    def test_run_dynamic_dynesty(self):
        pytest.importorskip("dynesty")
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="dynamic_dynesty",
            nlive_init=100,
            nlive_batch=100,
            dlogz_init=1.0,
            maxbatch=0,
            maxcall=100,
            bound="single",
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

    def test_run_emcee(self):
        pytest.importorskip("emcee")
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="emcee",
            iterations=1000,
            nwalkers=10,
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

    def test_run_kombine(self):
        pytest.importorskip("kombine")
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="kombine",
            iterations=2000,
            nwalkers=20,
            autoburnin=False,
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

    def test_run_nestle(self):
        pytest.importorskip("nestle")
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="nestle",
            nlive=100,
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

    def test_run_nessai(self):
        pytest.importorskip("nessai")
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="nessai",
            nlive=100,
            poolsize=1000,
            max_iteration=1000,
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

    def test_run_pypolychord(self):
        pytest.importorskip("pypolychord")
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="pypolychord",
            nlive=100,
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

    def test_run_ptemcee(self):
        pytest.importorskip("ptemcee")
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="ptemcee",
            nsamples=100,
            nwalkers=50,
            burn_in_act=1,
            ntemps=1,
            frac_threshold=0.5,
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

    @pytest.mark.xfail(
        raises=AttributeError,
        reason="Dependency issue with pymc3 causes attribute error on import",
    )
    def test_run_pymc3(self):
        pytest.importorskip("pymc3")
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="pymc3",
            draws=50,
            tune=50,
            n_init=250,
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

    def test_run_pymultinest(self):
        pytest.importorskip("pymultinest")
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="pymultinest",
            nlive=100,
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

    def test_run_PTMCMCSampler(self):
        pytest.importorskip("PTMCMCSampler")
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="PTMCMCsampler",
            Niter=101,
            burn=2,
            isave=100,
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

    def test_run_ultranest(self):
        pytest.importorskip("ultranest")
        # run using NestedSampler (with nlive specified)
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="ultranest",
            nlive=100,
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

        # run using ReactiveNestedSampler (with no nlive given)
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="ultranest",
            **self.kwargs,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None

    def test_run_bilby_mcmc(self):
        res = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="bilby_mcmc",
            nsamples=200,
            **self.kwargs,
            printdt=1,
        )
        assert "derived" in res.posterior
        assert res.log_likelihood_evaluations is not None


if __name__ == "__main__":
    unittest.main()
