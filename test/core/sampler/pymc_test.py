import unittest
from unittest.mock import MagicMock

import bilby
import bilby.core.sampler.pymc


class TestPyMC(unittest.TestCase):
    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.sampler = bilby.core.sampler.pymc.Pymc(
            self.likelihood,
            self.priors,
            outdir="outdir",
            label="label",
            use_ratio=False,
            plot=False,
            skip_import_verification=True,
        )

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler

    def test_default_kwargs(self):
        expected = dict(
            draws=500,
            step=None,
            init="auto",
            n_init=200000,
            initvals=None,
            trace=None,
            chains=2,
            cores=1,
            tune=500,
            progressbar=True,
            model=None,
            nuts_kwargs=None,
            step_kwargs=None,
            random_seed=None,
            discard_tuned_samples=True,
            compute_convergence_checks=True,
        )
        expected.update(self.sampler.default_nuts_kwargs)
        expected.update(self.sampler.default_step_kwargs)
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(
            draws=500,
            step=None,
            init="auto",
            n_init=200000,
            initvals=None,
            trace=None,
            chains=2,
            cores=1,
            tune=500,
            progressbar=True,
            model=None,
            nuts_kwargs=None,
            step_kwargs=None,
            random_seed=None,
            discard_tuned_samples=True,
            compute_convergence_checks=True,
        )
        expected.update(self.sampler.default_nuts_kwargs)
        expected.update(self.sampler.default_step_kwargs)
        self.sampler.kwargs["draws"] = 123
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["draws"]
            new_kwargs[equiv] = 500
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


if __name__ == "__main__":
    unittest.main()
