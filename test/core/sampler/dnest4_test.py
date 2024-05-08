import unittest
from unittest.mock import MagicMock

import bilby
import bilby.core.sampler.dnest4


class TestDnest4(unittest.TestCase):
    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.sampler = bilby.core.sampler.dnest4.DNest4(
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
            max_num_levels=20, num_steps=500,
            new_level_interval=10000, num_per_step=10000,
            thread_steps=1, num_particles=1000, lam=10.0,
            beta=100, seed=None, verbose=True, backend='memory'
        )
        for key in self.sampler.kwargs.keys():
            print(
                "key={}, expected={}, actual={}".format(
                    key, expected[key], self.sampler.kwargs[key]
                )
            )
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(
            max_num_levels=20, num_steps=500,
            new_level_interval=10000, num_per_step=10000,
            thread_steps=1, num_particles=1000, lam=10.0,
            beta=100, seed=None, verbose=True, backend='memory'
        )

        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["num_particles"]
            new_kwargs[equiv] = 1000
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


if __name__ == "__main__":
    unittest.main()
