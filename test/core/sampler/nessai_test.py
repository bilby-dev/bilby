import unittest
from unittest.mock import MagicMock, patch, mock_open

import bilby
import bilby.core.sampler.nessai
import os


class TestNessai(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.likelihood = MagicMock()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.sampler = bilby.core.sampler.nessai.Nessai(
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
        self.expected["n_pool"] = 1  # Because npool=1 by default
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
        expected["n_pool"] = 2
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npool_equiv_kwargs:
            new_kwargs = self.sampler.kwargs.copy()
            del new_kwargs["n_pool"]
            new_kwargs[equiv] = 2
            self.sampler.kwargs = new_kwargs
            self.assertDictEqual(expected, self.sampler.kwargs)

    def test_split_kwargs(self):
        kwargs, run_kwargs = self.sampler.split_kwargs()
        assert "save" not in run_kwargs
        assert "plot" in run_kwargs

    def test_translate_kwargs_no_npool(self):
        expected = self.expected.copy()
        expected["n_pool"] = 3
        new_kwargs = self.sampler.kwargs.copy()
        del new_kwargs["n_pool"]
        self.sampler._npool = 3
        self.sampler.kwargs = new_kwargs
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_translate_kwargs_seed(self):
        assert self.expected["seed"] == 150914

    @patch("builtins.open", mock_open(read_data='{"nlive": 4000}'))
    def test_update_from_config_file(self):
        expected = self.expected.copy()
        expected["nlive"] = 4000
        new_kwargs = self.expected.copy()
        new_kwargs["config_file"] = "config_file.json"
        self.sampler.kwargs = new_kwargs
        self.assertDictEqual(expected, self.sampler.kwargs)


def test_get_expected_outputs():
    label = "par0"
    outdir = os.path.join("some", "bilby_pipe", "dir")
    filenames, directories = bilby.core.sampler.nessai.Nessai.get_expected_outputs(
        outdir=outdir, label=label
    )
    assert len(filenames) == 0
    assert len(directories) == 3
    base_dir = os.path.join(outdir, f"{label}_nessai", "")
    assert base_dir in directories
    assert os.path.join(base_dir, "proposal", "") in directories
    assert os.path.join(base_dir, "diagnostics", "") in directories


if __name__ == "__main__":
    unittest.main()
