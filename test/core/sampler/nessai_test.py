import unittest
from unittest.mock import MagicMock, patch, mock_open

import bilby


class TestNessai(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.sampler = bilby.core.sampler.Nessai(
            self.likelihood,
            self.priors,
            outdir="outdir",
            label="label",
            use_ratio=False,
            plot=False,
            skip_import_verification=True,
            sampling_seed=150914,
        )
        self.expected = self.sampler.default_kwargs
        self.expected['output'] = 'outdir/label_nessai/'
        self.expected['seed'] = 150914

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler
        del self.expected

    def test_translate_kwargs_nlive(self):
        expected = self.expected.copy()
        # nlive in the default kwargs is not a fixed but depends on the
        # version of nessai, so get the value here and use it when setting
        # the equivalent kwargs.
        nlive = expected["nlive"]
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["nlive"]
            new_kwargs[equiv] = nlive
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs_npool(self):
        expected = self.expected.copy()
        expected["n_pool"] = None
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npool_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["n_pool"]
            new_kwargs[equiv] = None
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs_seed(self):
        assert self.expected["seed"] == 150914

    def test_npool_max_threads(self):
        expected = self.expected.copy()
        expected["n_pool"] = None
        new_kwargs = self.sampler.kwargs.copy()
        new_kwargs["n_pool"] = 1
        self.sampler.kwargs = new_kwargs
        self.assertDictEqual(expected, self.sampler.kwargs)

    @patch("builtins.open", mock_open(read_data='{"nlive": 4000}'))
    def test_update_from_config_file(self):
        expected = self.expected.copy()
        expected["nlive"] = 4000
        new_kwargs = self.expected.copy()
        new_kwargs["config_file"] = "config_file.json"
        self.sampler.kwargs = new_kwargs
        self.assertDictEqual(expected, self.sampler.kwargs)


if __name__ == "__main__":
    unittest.main()
