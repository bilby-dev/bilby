import unittest
from unittest.mock import MagicMock

import bilby
import bilby.core.sampler.cpnest


class TestCPNest(unittest.TestCase):
    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.sampler = bilby.core.sampler.cpnest.Cpnest(
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
            verbose=3,
            nthreads=1,
            nlive=500,
            maxmcmc=1000,
            seed=None,
            poolsize=100,
            nhamiltonian=0,
            resume=True,
            output="outdir/cpnest_label/",
            proposals=None,
            n_periodic_checkpoint=8000,
        )
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(
            verbose=3,
            nthreads=1,
            nlive=250,
            maxmcmc=1000,
            seed=None,
            poolsize=100,
            nhamiltonian=0,
            resume=True,
            output="outdir/cpnest_label/",
            proposals=None,
            n_periodic_checkpoint=8000,
        )
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["nlive"]
            new_kwargs[equiv] = 250
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


if __name__ == "__main__":
    unittest.main()
