import unittest
import pytest
import shutil
import sys

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
        self.priors["m"] = bilby.core.prior.Uniform(0, 5, boundary="reflective")
        self.priors["c"] = bilby.core.prior.Uniform(-2, 2, boundary="reflective")
        bilby.core.utils.check_directory_exists_and_if_not_mkdir("outdir")

    def tearDown(self):
        del self.likelihood
        del self.priors
        bilby.core.utils.command_line_args.bilby_test_mode = False
        shutil.rmtree("outdir")

    def test_run_cpnest(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="cpnest",
            nlive=100,
            save=False,
            resume=False,
        )

    def test_run_dnest4(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="dnest4",
            nlive=100,
            save=False,
        )

    def test_run_dynesty(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="dynesty",
            nlive=100,
            save=False,
        )

    def test_run_dynamic_dynesty(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="dynamic_dynesty",
            nlive_init=100,
            nlive_batch=100,
            dlogz_init=1.0,
            maxbatch=0,
            maxcall=100,
            bound="single",
            save=False,
        )

    def test_run_emcee(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="emcee",
            iterations=1000,
            nwalkers=10,
            save=False,
        )

    def test_run_kombine(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="kombine",
            iterations=1000,
            nwalkers=100,
            save=False,
            autoburnin=True,
        )

    def test_run_nestle(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="nestle",
            nlive=100,
            save=False,
        )

    def test_run_nessai(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="nessai",
            nlive=100,
            poolsize=1000,
            max_iteration=1000,
            save=False,
        )

    def test_run_pypolychord(self):
        pytest.importorskip("pypolychord")
        _ = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="pypolychord",
            nlive=100,
            save=False,
        )

    def test_run_ptemcee(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="ptemcee",
            nsamples=100,
            nwalkers=50,
            burn_in_act=1,
            ntemps=1,
            frac_threshold=0.5,
            save=False,
        )

    @pytest.mark.skipif(sys.version_info[1] <= 6, reason="pymc3 is broken in py36")
    def test_run_pymc3(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="pymc3",
            draws=50,
            tune=50,
            n_init=1000,
            save=False,
        )

    def test_run_pymultinest(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="pymultinest",
            nlive=100,
            save=False,
        )

    def test_run_PTMCMCSampler(self):
        _ = bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="PTMCMCsampler",
            Niter=101,
            burn=2,
            isave=100,
            save=False,
        )

    def test_run_ultranest(self):
        # run using NestedSampler (with nlive specified)
        _ = bilby.run_sampler(
            likelihood=self.likelihood, priors=self.priors,
            sampler="ultranest", nlive=100, save=False,
        )

        # run using ReactiveNestedSampler (with no nlive given)
        _ = bilby.run_sampler(
            likelihood=self.likelihood, priors=self.priors,
            sampler='ultranest', save=False,
        )


if __name__ == "__main__":
    unittest.main()
