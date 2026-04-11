import os
import unittest

import bilby
import bilby.core.sampler.emcee


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


def test_get_expected_outputs():
    label = "par0"
    outdir = os.path.join("some", "bilby_pipe", "dir")
    filenames, directories = bilby.core.sampler.emcee.Emcee.get_expected_outputs(
        outdir=outdir, label=label
    )
    assert len(filenames) == 2
    assert len(directories) == 1
    run_dir = os.path.join(outdir, f"emcee_{label}")
    assert run_dir in directories
    assert os.path.join(run_dir, "chain.dat") in filenames
    assert os.path.join(run_dir, "sampler.pickle") in filenames


if __name__ == "__main__":
    unittest.main()
