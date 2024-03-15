import unittest
from unittest.mock import MagicMock

import bilby
import bilby.core.sampler.kombine


class TestKombine(unittest.TestCase):
    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.sampler = bilby.core.sampler.kombine.Kombine(
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
            nwalkers=500,
            args=[],
            pool=None,
            transd=False,
            lnpost0=None,
            blob0=None,
            iterations=500,
            storechain=True,
            processes=1,
            update_interval=None,
            kde=None,
            kde_size=None,
            spaces=None,
            freeze_transd=False,
            test_steps=16,
            critical_pval=0.05,
            max_steps=None,
            burnin_verbose=False,
        )
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(
            nwalkers=400,
            args=[],
            pool=None,
            transd=False,
            lnpost0=None,
            blob0=None,
            iterations=500,
            storechain=True,
            processes=1,
            update_interval=None,
            kde=None,
            kde_size=None,
            spaces=None,
            freeze_transd=False,
            test_steps=16,
            critical_pval=0.05,
            max_steps=None,
            burnin_verbose=False,
        )
        for equiv in bilby.core.sampler.base_sampler.MCMCSampler.nwalkers_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["nwalkers"]
            new_kwargs[equiv] = 400
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


if __name__ == "__main__":
    unittest.main()
