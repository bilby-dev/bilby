import unittest
from types import SimpleNamespace

import bilby
import bilby.core.sampler.emcee
import numpy as np


class TestEmcee(unittest.TestCase):
    def setUp(self):
        self.likelihood = bilby.core.likelihood.Likelihood()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.sampler = bilby.core.sampler.emcee.Emcee(
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
            a=2,
            args=[],
            kwargs={},
            postargs=None,
            pool=None,
            live_dangerously=False,
            runtime_sortingfn=None,
            lnprob0=None,
            rstate0=None,
            blobs0=None,
            iterations=100,
            thin=1,
            storechain=True,
            mh_proposal=None,
        )
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(
            nwalkers=100,
            a=2,
            args=[],
            kwargs={},
            postargs=None,
            pool=None,
            live_dangerously=False,
            runtime_sortingfn=None,
            lnprob0=None,
            rstate0=None,
            blobs0=None,
            iterations=100,
            thin=1,
            storechain=True,
            mh_proposal=None,
        )
        for equiv in bilby.core.sampler.base_sampler.MCMCSampler.nwalkers_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["nwalkers"]
            new_kwargs[equiv] = 100
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)

    def test_expected_output_files(self):
        expected_filenames = [
            "outdir/emcee_output_test/chain.dat",
            "outdir/emcee_output_test/sampler.pickle",
        ]
        expected_dirs = ["outdir/emcee_output_test"]
        filenames, dirs = self.sampler.get_expected_outputs(
            outdir="outdir", label="output_test"
        )
        self.assertListEqual(expected_filenames, filenames)
        self.assertListEqual(expected_dirs, dirs)

    def test_generate_result_flattens_blobs_in_sample_order(self):
        chain = np.arange(12).reshape(2, 3, 2)
        blobs = np.empty((3, 2, 2))
        blobs[:, :, 0] = chain[:, :, 0].T
        blobs[:, :, 1] = chain[:, :, 1].T
        self.sampler._sampler = SimpleNamespace(chain=chain, blobs=blobs)
        self.sampler.nburn = 1
        self.sampler.result.samples = chain[:, self.sampler.nburn :, :].reshape(-1, 2)

        self.sampler._generate_result()

        np.testing.assert_array_equal(
            self.sampler.result.log_likelihood_evaluations,
            self.sampler.result.samples[:, 0],
        )
        np.testing.assert_array_equal(
            self.sampler.result.log_prior_evaluations,
            self.sampler.result.samples[:, 1],
        )


if __name__ == "__main__":
    unittest.main()
