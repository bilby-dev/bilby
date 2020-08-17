import unittest

import numpy as np
from mock import MagicMock

import bilby


class TestPolyChord(unittest.TestCase):
    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.sampler = bilby.core.sampler.PyPolyChord(
            self.likelihood,
            self.priors,
            outdir="outdir",
            label="polychord",
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
            use_polychord_defaults=False,
            nlive=self.sampler.ndim * 25,
            num_repeats=self.sampler.ndim * 5,
            nprior=-1,
            do_clustering=True,
            feedback=1,
            precision_criterion=0.001,
            logzero=-1e30,
            max_ndead=-1,
            boost_posterior=0.0,
            posteriors=True,
            equals=True,
            cluster_posteriors=True,
            write_resume=True,
            write_paramnames=False,
            read_resume=True,
            write_stats=True,
            write_live=True,
            write_dead=True,
            write_prior=True,
            compression_factor=np.exp(-1),
            base_dir="outdir",
            file_root="polychord",
            seed=-1,
            grade_dims=list([self.sampler.ndim]),
            grade_frac=list([1.0] * len([self.sampler.ndim])),
            nlives={},
        )
        self.sampler._setup_dynamic_defaults()
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(
            use_polychord_defaults=False,
            nlive=123,
            num_repeats=self.sampler.ndim * 5,
            nprior=-1,
            do_clustering=True,
            feedback=1,
            precision_criterion=0.001,
            logzero=-1e30,
            max_ndead=-1,
            boost_posterior=0.0,
            posteriors=True,
            equals=True,
            cluster_posteriors=True,
            write_resume=True,
            write_paramnames=False,
            read_resume=True,
            write_stats=True,
            write_live=True,
            write_dead=True,
            write_prior=True,
            compression_factor=np.exp(-1),
            base_dir="outdir",
            file_root="polychord",
            seed=-1,
            grade_dims=list([self.sampler.ndim]),
            grade_frac=list([1.0] * len([self.sampler.ndim])),
            nlives={},
        )
        self.sampler._setup_dynamic_defaults()
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["nlive"]
            new_kwargs[equiv] = 123
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


if __name__ == "__main__":
    unittest.main()
