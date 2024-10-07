import unittest
from unittest.mock import MagicMock

import bilby
import bilby.core.sampler.nestle


class TestNestle(unittest.TestCase):
    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.sampler = bilby.core.sampler.nestle.Nestle(
            self.likelihood,
            self.priors,
            outdir="outdir",
            label="label",
            use_ratio=False,
            plot=False,
            skip_import_verification=True,
            verbose=False,
        )

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler

    def test_default_kwargs(self):
        expected = dict(
            verbose=False,
            method="multi",
            npoints=500,
            update_interval=None,
            npdim=None,
            maxiter=None,
            maxcall=None,
            dlogz=None,
            decline_factor=None,
            rstate=None,
            callback=None,
            steps=20,
            enlarge=1.2,
        )
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(
            verbose=False,
            method="multi",
            npoints=345,
            update_interval=None,
            npdim=None,
            maxiter=None,
            maxcall=None,
            dlogz=None,
            decline_factor=None,
            rstate=None,
            callback=None,
            steps=20,
            enlarge=1.2,
        )
        self.sampler.kwargs["npoints"] = 123
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["npoints"]
            new_kwargs[equiv] = 345
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


if __name__ == "__main__":
    unittest.main()
