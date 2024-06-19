import copy
import os
import shutil
import unittest
from unittest.mock import MagicMock
from parameterized import parameterized

import numpy as np

import bilby
from bilby.core import prior

loaded_samplers = {k: v.load() for k, v in bilby.core.sampler.IMPLEMENTED_SAMPLERS.items()}


class TestSampler(unittest.TestCase):
    def setUp(self, soft_init=False):
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
            soft_init=soft_init
        )

    def tearDown(self):
        del self.sampler

    def test_softinit(self):
        self.setUp(soft_init=True)
        self.assertTrue(hasattr(self.sampler, "_log_likelihood_eval_time"))

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

    @parameterized.expand(["sampling_seed", "seed", "random_seed"])
    def test_translate_kwargs(self, key):
        self.sampler.sampling_seed_key = key
        for k in self.sampler.sampling_seed_equiv_kwargs:
            kwargs = {k: 1234}
            updated_kwargs = self.sampler._translate_kwargs(kwargs)
            self.assertDictEqual(updated_kwargs, {key: 1234})
        self.sampler.sampling_seed_key = None

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

    def test_bad_value_nan(self):
        self.sampler._check_bad_value(val=np.nan, warning=False, theta=None, label=None)

    def test_bad_value_np_abs_nan(self):
        self.sampler._check_bad_value(
            val=np.abs(np.nan), warning=False, theta=None, label=None
        )

    def test_bad_value_abs_nan(self):
        self.sampler._check_bad_value(
            val=abs(np.nan), warning=False, theta=None, label=None
        )

    def test_bad_value_pos_inf(self):
        self.sampler._check_bad_value(val=np.inf, warning=False, theta=None, label=None)

    def test_bad_value_neg_inf(self):
        self.sampler._check_bad_value(
            val=-np.inf, warning=False, theta=None, label=None
        )

    def test_bad_value_pos_inf_nan_to_num(self):
        self.sampler._check_bad_value(
            val=np.nan_to_num(np.inf), warning=False, theta=None, label=None
        )

    def test_bad_value_neg_inf_nan_to_num(self):
        self.sampler._check_bad_value(
            val=np.nan_to_num(-np.inf), warning=False, theta=None, label=None
        )


def test_get_expected_outputs():
    outdir = os.path.join("some", "bilby_pipe", "dir")
    label = "par0"
    filenames, directories = bilby.core.sampler.Sampler.get_expected_outputs(
        outdir=outdir, label=label
    )
    assert len(filenames) == 0
    assert len(directories) == 1
    assert directories[0] == os.path.join(outdir, f"sampler_{label}", "")


def test_get_expected_outputs_abbreviation():
    outdir = os.path.join("some", "bilby_pipe", "dir")
    label = "par0"
    bilby.core.sampler.Sampler.abbreviation = "abbr"
    filenames, directories = bilby.core.sampler.Sampler.get_expected_outputs(
        outdir=outdir, label=label
    )
    assert len(filenames) == 0
    assert len(directories) == 1
    assert directories[0] == os.path.join(outdir, f"abbr_{label}", "")
    bilby.core.sampler.Sampler.abbreviation = None


samplers = [
    "bilby_mcmc",
    "dynamic_dynesty",
    "dynesty",
    "emcee",
    "kombine",
    "ptemcee",
    "zeus",
]


class GenericSamplerTest(unittest.TestCase):
    def setUp(self):
        self.likelihood = bilby.core.likelihood.Likelihood(dict())
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )

    def tearDown(self):
        if os.path.isdir("outdir"):
            shutil.rmtree("outdir")

    @parameterized.expand(samplers)
    def test_pool_creates_properly_no_pool(self, sampler_name):
        sampler = loaded_samplers[sampler_name](self.likelihood, self.priors)
        sampler._setup_pool()
        if sampler_name == "kombine":
            from kombine import SerialPool

            self.assertIsInstance(sampler.pool, SerialPool)
            pass
        else:
            self.assertIsNone(sampler.pool)

    @parameterized.expand(samplers)
    def test_pool_creates_properly_pool(self, sampler):
        sampler = loaded_samplers[sampler](
            self.likelihood, self.priors, npool=2
        )
        sampler._setup_pool()
        if hasattr(sampler, "setup_sampler"):
            sampler.setup_sampler()
        self.assertEqual(sampler.pool._processes, 2)
        sampler._close_pool()


class ReorderLikelihoodsTest(unittest.TestCase):
    def setUp(self):
        self.unsorted_ln_likelihoods = np.array([1, 5, 2, 5, 1])
        self.unsorted_samples = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [0, 1]])
        self.sorted_samples = np.array([[0, 1], [0, 1], [1, 0], [1, 1], [0, 0]])
        self.sorted_ln_likelihoods = np.array([1, 1, 2, 5, 5])

    def tearDown(self):
        pass

    def test_ordering(self):
        func = bilby.core.sampler.base_sampler.NestedSampler.reorder_loglikelihoods
        sorted_ln_likelihoods = func(
            self.unsorted_ln_likelihoods, self.unsorted_samples, self.sorted_samples
        )
        self.assertTrue(
            np.array_equal(sorted_ln_likelihoods, self.sorted_ln_likelihoods)
        )


if __name__ == "__main__":
    unittest.main()
