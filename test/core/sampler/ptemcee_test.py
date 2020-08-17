import unittest

from mock import MagicMock

import bilby


class TestPTEmcee(unittest.TestCase):
    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.sampler = bilby.core.sampler.Ptemcee(
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
            ntemps=20,
            nwalkers=200,
            Tmax=None,
            betas=None,
            a=2.0,
            adaptation_lag=10000,
            adaptation_time=100,
            random=None,
            adapt=True,
            swap_ratios=False,
        )
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(
            ntemps=20,
            nwalkers=200,
            Tmax=None,
            betas=None,
            a=2.0,
            adaptation_lag=10000,
            adaptation_time=100,
            random=None,
            adapt=True,
            swap_ratios=False,
        )
        for equiv in bilby.core.sampler.base_sampler.MCMCSampler.nwalkers_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["nwalkers"]
            new_kwargs[equiv] = 200
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


if __name__ == "__main__":
    unittest.main()
