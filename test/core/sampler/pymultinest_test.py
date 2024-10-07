import unittest
from unittest.mock import MagicMock

import bilby
import bilby.core.sampler.pymultinest


class TestPymultinest(unittest.TestCase):
    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.priors["a"] = bilby.core.prior.Prior(boundary="periodic")
        self.priors["b"] = bilby.core.prior.Prior(boundary="reflective")
        self.sampler = bilby.core.sampler.pymultinest.Pymultinest(
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
        expected = dict(importance_nested_sampling=False, resume=True,
                        verbose=True, sampling_efficiency='parameter',
                        n_live_points=500, n_params=2,
                        n_clustering_params=None, wrapped_params=None,
                        multimodal=True, const_efficiency_mode=False,
                        evidence_tolerance=0.5,
                        n_iter_before_update=100, null_log_evidence=-1e90,
                        max_modes=100, mode_tolerance=-1e90, seed=-1,
                        context=0, write_output=True, log_zero=-1e100,
                        max_iter=0, init_MPI=False, dump_callback='dumper')
        self.sampler.kwargs['dump_callback'] = 'dumper'  # Check like the dynesty print_func
        self.assertListEqual([1, 0], self.sampler.kwargs['wrapped_params'])  # Check this separately
        self.sampler.kwargs['wrapped_params'] = None  # The dict comparison can't handle lists
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs(self):
        expected = dict(importance_nested_sampling=False, resume=True,
                        verbose=True, sampling_efficiency='parameter',
                        n_live_points=123, n_params=2,
                        n_clustering_params=None, wrapped_params=None,
                        multimodal=True, const_efficiency_mode=False,
                        evidence_tolerance=0.5,
                        n_iter_before_update=100, null_log_evidence=-1e90,
                        max_modes=100, mode_tolerance=-1e90, seed=-1,
                        context=0, write_output=True, log_zero=-1e100,
                        max_iter=0, init_MPI=False, dump_callback='dumper')

        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["n_live_points"]
            new_kwargs[
                "wrapped_params"
            ] = None  # The dict comparison can't handle lists
            new_kwargs['dump_callback'] = 'dumper'  # Check this like Dynesty print_func
            new_kwargs[equiv] = 123
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)


if __name__ == "__main__":
    unittest.main()
