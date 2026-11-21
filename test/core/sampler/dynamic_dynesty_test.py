import unittest

import bilby
import bilby.core.sampler.dynamic_dynesty


class TestDynamicDynesty(unittest.TestCase):
    def setUp(self):
        self.likelihood = bilby.core.likelihood.Likelihood()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.init_sampler()

    def init_sampler(self, **kwargs):
        self.sampler = bilby.core.sampler.dynamic_dynesty.DynamicDynesty(
            self.likelihood,
            self.priors,
            outdir="outdir",
            label="label",
            use_ratio=False,
            plot=False,
            skip_import_verification=True,
            **kwargs,
        )

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler

    def test_default_kwargs(self):
        """Only test the kwargs where we specify different defaults to dynesty"""
        expected = dict(
            sample="act-walk",
            bound="default",
            facc=0.2,
            save_bounds=False,
            update_interval=600,
        )
        for key in expected:
            self.assertEqual(expected[key], self.sampler.kwargs[key])

        self.assertEqual("none", self.sampler.sampler_init_kwargs["bound"])

    def test_default_kwargs_unif_sample(self):
        self.init_sampler(sample="unif")
        self.assertEqual("multi", self.sampler.sampler_init_kwargs["bound"])


if __name__ == "__main__":
    unittest.main()
