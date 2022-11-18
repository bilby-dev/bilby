import unittest
from copy import deepcopy
from unittest.mock import MagicMock

import bilby


class TestDynesty(unittest.TestCase):
    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.sampler = bilby.core.sampler.Dynesty(
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
        """Only test the kwargs where we specify different defaults to dynesty"""
        expected = dict(sample="rwalk", facc=0.2, save_bounds=False, dlogz=0.1)
        for key in expected:
            self.assertEqual(expected[key], self.sampler.kwargs[key])

    def test_translate_kwargs(self):
        expected = 1000
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = deepcopy(self.sampler.kwargs)
            del new_kwargs["nlive"]
            new_kwargs[equiv] = expected
            self.sampler._translate_kwargs(new_kwargs)
            self.assertEqual(new_kwargs["nlive"], expected)

    def test_prior_boundary(self):
        self.priors["a"] = bilby.core.prior.Prior(boundary="periodic")
        self.priors["b"] = bilby.core.prior.Prior(boundary="reflective")
        self.priors["c"] = bilby.core.prior.Prior(boundary=None)
        self.priors["d"] = bilby.core.prior.Prior(boundary="reflective")
        self.priors["e"] = bilby.core.prior.Prior(boundary="periodic")
        self.sampler = bilby.core.sampler.Dynesty(
            self.likelihood,
            self.priors,
            outdir="outdir",
            label="label",
            use_ratio=False,
            plot=False,
            skip_import_verification=True,
        )
        self.assertEqual([0, 4], self.sampler.kwargs["periodic"])
        self.assertEqual([1, 3], self.sampler.kwargs["reflective"])


if __name__ == "__main__":
    unittest.main()
