import copy
import os
import unittest

import numpy as np
from mock import MagicMock

import bilby
from bilby.core import prior


class TestSampler(unittest.TestCase):
    def setUp(self):
        likelihood = bilby.core.likelihood.Likelihood()
        likelihood.parameters = dict(a=1, b=2, c=3)
        delta_prior = prior.DeltaFunction(peak=0)
        delta_prior.rescale = MagicMock(return_value=prior.DeltaFunction(peak=1))
        delta_prior.prob = MagicMock(return_value=1)
        delta_prior.sample = MagicMock(return_value=0)
        uniform_prior = prior.Uniform(0, 1)
        uniform_prior.rescale = MagicMock(return_value=prior.Uniform(0, 2))
        uniform_prior.prob = MagicMock(return_value=1)
        uniform_prior.sample = MagicMock(return_value=0.5)

        priors = dict(a=delta_prior, b="string", c=uniform_prior)
        likelihood.log_likelihood_ratio = MagicMock(return_value=1)
        likelihood.log_likelihood = MagicMock(return_value=2)
        test_directory = "test_directory"
        if os.path.isdir(test_directory):
            os.rmdir(test_directory)
        self.sampler = bilby.core.sampler.Sampler(
            likelihood=likelihood,
            priors=priors,
            outdir=test_directory,
            use_ratio=False,
            skip_import_verification=True,
        )

    def tearDown(self):
        del self.sampler

    def test_search_parameter_keys(self):
        expected_search_parameter_keys = ["c"]
        self.assertListEqual(
            self.sampler.search_parameter_keys, expected_search_parameter_keys
        )

    def test_fixed_parameter_keys(self):
        expected_fixed_parameter_keys = ["a"]
        self.assertListEqual(
            self.sampler.fixed_parameter_keys, expected_fixed_parameter_keys
        )

    def test_ndim(self):
        self.assertEqual(self.sampler.ndim, 1)

    def test_kwargs(self):
        self.assertDictEqual(self.sampler.kwargs, {})

    def test_label(self):
        self.assertEqual(self.sampler.label, "label")

    def test_prior_transform_transforms_search_parameter_keys(self):
        self.sampler.prior_transform([0])
        expected_prior = prior.Uniform(0, 1)
        self.assertListEqual(
            [self.sampler.priors["c"].minimum, self.sampler.priors["c"].maximum],
            [expected_prior.minimum, expected_prior.maximum],
        )

    def test_prior_transform_does_not_transform_fixed_parameter_keys(self):
        self.sampler.prior_transform([0])
        self.assertEqual(
            self.sampler.priors["a"].peak, prior.DeltaFunction(peak=0).peak
        )

    def test_log_prior(self):
        self.assertEqual(self.sampler.log_prior({1}), 0.0)

    def test_log_likelihood_with_use_ratio(self):
        self.sampler.use_ratio = True
        self.assertEqual(self.sampler.log_likelihood([0]), 1)

    def test_log_likelihood_without_use_ratio(self):
        self.sampler.use_ratio = False
        self.assertEqual(self.sampler.log_likelihood([0]), 2)

    def test_log_likelihood_correctly_sets_parameters(self):
        expected_dict = dict(a=0, b=2, c=0)
        _ = self.sampler.log_likelihood([0])
        self.assertDictEqual(self.sampler.likelihood.parameters, expected_dict)

    def test_get_random_draw(self):
        self.assertEqual(self.sampler.get_random_draw_from_prior(), np.array([0.5]))

    def test_base_run_sampler(self):
        sampler_copy = copy.copy(self.sampler)
        self.sampler.run_sampler()
        self.assertDictEqual(sampler_copy.__dict__, self.sampler.__dict__)


if __name__ == "__main__":
    unittest.main()
