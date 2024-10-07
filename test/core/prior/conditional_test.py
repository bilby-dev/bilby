import os
import shutil
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import pickle

import bilby


class TestConditionalPrior(unittest.TestCase):
    def setUp(self):
        self.condition_func_call_counter = 0

        def condition_func(reference_parameters, test_variable_1, test_variable_2):
            self.condition_func_call_counter += 1
            return {key: value + 1 for key, value in reference_parameters.items()}

        self.condition_func = condition_func
        self.minimum = 0
        self.maximum = 5
        self.test_variable_1 = 0
        self.test_variable_2 = 1
        self.prior = bilby.core.prior.ConditionalBasePrior(
            condition_func=condition_func, minimum=self.minimum, maximum=self.maximum
        )

    def tearDown(self):
        del self.condition_func
        del self.condition_func_call_counter
        del self.minimum
        del self.maximum
        del self.test_variable_1
        del self.test_variable_2
        del self.prior

    def test_reference_params(self):
        self.assertDictEqual(
            dict(minimum=self.minimum, maximum=self.maximum),
            self.prior.reference_params,
        )

    def test_required_variables(self):
        self.assertListEqual(
            ["test_variable_1", "test_variable_2"],
            sorted(self.prior.required_variables),
        )

    def test_required_variables_no_condition_func(self):
        self.prior = bilby.core.prior.ConditionalBasePrior(
            condition_func=None, minimum=self.minimum, maximum=self.maximum
        )
        self.assertListEqual([], self.prior.required_variables)

    def test_get_instantiation_dict(self):
        expected = dict(
            minimum=0,
            maximum=5,
            name=None,
            latex_label=None,
            unit=None,
            boundary=None,
            condition_func=self.condition_func,
        )
        actual = self.prior.get_instantiation_dict()
        for key, value in expected.items():
            if key == "condition_func":
                continue
            self.assertEqual(value, actual[key])

    def test_update_conditions_correct_variables(self):
        self.prior.update_conditions(
            test_variable_1=self.test_variable_1, test_variable_2=self.test_variable_2
        )
        self.assertEqual(1, self.condition_func_call_counter)
        self.assertEqual(self.minimum + 1, self.prior.minimum)
        self.assertEqual(self.maximum + 1, self.prior.maximum)

    def test_update_conditions_no_variables(self):
        self.prior.update_conditions(
            test_variable_1=self.test_variable_1, test_variable_2=self.test_variable_2
        )
        self.prior.update_conditions()
        self.assertEqual(1, self.condition_func_call_counter)
        self.assertEqual(self.minimum + 1, self.prior.minimum)
        self.assertEqual(self.maximum + 1, self.prior.maximum)

    def test_update_conditions_illegal_variables(self):
        with self.assertRaises(bilby.core.prior.IllegalRequiredVariablesException):
            self.prior.update_conditions(test_parameter_1=self.test_variable_1)

    def test_sample_calls_update_conditions(self):
        with mock.patch.object(self.prior, "update_conditions") as m:
            self.prior.sample(
                1,
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )
            m.assert_called_with(
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )

    def test_rescale_calls_update_conditions(self):
        with mock.patch.object(self.prior, "update_conditions") as m:
            self.prior.rescale(
                1,
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )
            m.assert_called_with(
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )

    def test_prob_calls_update_conditions(self):
        with mock.patch.object(self.prior, "update_conditions") as m:
            self.prior.prob(
                1,
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )
            m.assert_called_with(
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )

    def test_rescale_ln_prob_update_conditions(self):
        with mock.patch.object(self.prior, "update_conditions") as m:
            self.prior.ln_prob(
                1,
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )
            calls = [
                mock.call(
                    test_parameter_1=self.test_variable_1,
                    test_parameter_2=self.test_variable_2,
                ),
                mock.call(),
            ]
            m.assert_has_calls(calls)

    def test_cdf_calls_update_conditions(self):
        self.prior = bilby.core.prior.ConditionalUniform(
            condition_func=self.condition_func, minimum=self.minimum, maximum=self.maximum
        )
        with mock.patch.object(self.prior, "update_conditions") as m:
            self.prior.cdf(
                1,
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )
            m.assert_called_with(
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )

    def test_reset_to_reference_parameters(self):
        self.prior.minimum = 10
        self.prior.maximum = 20
        self.prior.reset_to_reference_parameters()
        self.assertEqual(self.prior.reference_params["minimum"], self.prior.minimum)
        self.assertEqual(self.prior.reference_params["maximum"], self.prior.maximum)

    def test_cond_prior_instantiation_no_boundary_prior(self):
        prior = bilby.core.prior.ConditionalFermiDirac(
            condition_func=None, sigma=1, mu=1
        )
        self.assertIsNone(prior.boundary)


class TestConditionalPriorDict(unittest.TestCase):
    def setUp(self):
        def condition_func_1(reference_parameters, var_0):
            return dict(minimum=reference_parameters["minimum"], maximum=var_0)

        def condition_func_2(reference_parameters, var_0, var_1):
            return dict(minimum=reference_parameters["minimum"], maximum=var_1)

        def condition_func_3(reference_parameters, var_1, var_2):
            return dict(minimum=reference_parameters["minimum"], maximum=var_2)

        self.minimum = 0
        self.maximum = 1
        self.prior_0 = bilby.core.prior.Uniform(
            minimum=self.minimum, maximum=self.maximum
        )
        self.prior_1 = bilby.core.prior.ConditionalUniform(
            condition_func=condition_func_1, minimum=self.minimum, maximum=self.maximum
        )
        self.prior_2 = bilby.core.prior.ConditionalUniform(
            condition_func=condition_func_2, minimum=self.minimum, maximum=self.maximum
        )
        self.prior_3 = bilby.core.prior.ConditionalUniform(
            condition_func=condition_func_3, minimum=self.minimum, maximum=self.maximum
        )
        self.conditional_priors = bilby.core.prior.ConditionalPriorDict(
            dict(
                var_3=self.prior_3,
                var_2=self.prior_2,
                var_0=self.prior_0,
                var_1=self.prior_1,
            )
        )
        self.conditional_priors_manually_set_items = (
            bilby.core.prior.ConditionalPriorDict()
        )
        self.test_sample = dict(var_0=0.7, var_1=0.6, var_2=0.5, var_3=0.4)
        self.test_value = 1 / np.prod([self.test_sample[f"var_{ii}"] for ii in range(3)])
        for key, value in dict(
            var_0=self.prior_0,
            var_1=self.prior_1,
            var_2=self.prior_2,
            var_3=self.prior_3,
        ).items():
            self.conditional_priors_manually_set_items[key] = value

    def tearDown(self):
        del self.minimum
        del self.maximum
        del self.prior_0
        del self.prior_1
        del self.prior_2
        del self.prior_3
        del self.conditional_priors
        del self.conditional_priors_manually_set_items
        del self.test_sample

    def test_conditions_resolved_upon_instantiation(self):
        self.assertListEqual(
            ["var_0", "var_1", "var_2", "var_3"], self.conditional_priors.sorted_keys
        )

    def test_conditions_resolved_setting_items(self):
        self.assertListEqual(
            ["var_0", "var_1", "var_2", "var_3"],
            self.conditional_priors_manually_set_items.sorted_keys,
        )

    def test_unconditional_keys_upon_instantiation(self):
        self.assertListEqual(["var_0"], self.conditional_priors.unconditional_keys)

    def test_unconditional_keys_setting_items(self):
        self.assertListEqual(
            ["var_0"], self.conditional_priors_manually_set_items.unconditional_keys
        )

    def test_conditional_keys_upon_instantiation(self):
        self.assertListEqual(
            ["var_1", "var_2", "var_3"], self.conditional_priors.conditional_keys
        )

    def test_conditional_keys_setting_items(self):
        self.assertListEqual(
            ["var_1", "var_2", "var_3"],
            self.conditional_priors_manually_set_items.conditional_keys,
        )

    def test_prob(self):
        self.assertEqual(self.test_value, self.conditional_priors.prob(sample=self.test_sample))

    def test_prob_illegal_conditions(self):
        del self.conditional_priors["var_0"]
        with self.assertRaises(bilby.core.prior.IllegalConditionsException):
            self.conditional_priors.prob(sample=self.test_sample)

    def test_ln_prob(self):
        self.assertEqual(np.log(self.test_value), self.conditional_priors.ln_prob(sample=self.test_sample))

    def test_ln_prob_illegal_conditions(self):
        del self.conditional_priors["var_0"]
        with self.assertRaises(bilby.core.prior.IllegalConditionsException):
            self.conditional_priors.ln_prob(sample=self.test_sample)

    def test_sample_subset_all_keys(self):
        bilby.core.utils.random.seed(5)
        self.assertDictEqual(
            dict(
                var_0=0.8050029237453802,
                var_1=0.6503946979510289,
                var_2=0.33516501262044845,
                var_3=0.09579062316418356,
            ),
            self.conditional_priors.sample_subset(
                keys=["var_0", "var_1", "var_2", "var_3"]
            ),
        )

    def test_sample_illegal_subset(self):
        with self.assertRaises(bilby.core.prior.IllegalConditionsException):
            self.conditional_priors.sample_subset(keys=["var_1"])

    def test_sample_multiple(self):
        def condition_func(reference_params, a):
            return dict(
                minimum=reference_params["minimum"],
                maximum=reference_params["maximum"],
                alpha=reference_params["alpha"] * a,
            )

        priors = bilby.core.prior.ConditionalPriorDict()
        priors["a"] = bilby.core.prior.Uniform(minimum=0, maximum=1)
        priors["b"] = bilby.core.prior.ConditionalPowerLaw(
            condition_func=condition_func, minimum=1, maximum=2, alpha=-2
        )
        print(priors.sample(2))

    def test_rescale(self):
        self.conditional_priors = bilby.core.prior.ConditionalPriorDict(
            dict(
                var_3=self.prior_3,
                var_2=self.prior_2,
                var_0=self.prior_0,
                var_1=self.prior_1,
            )
        )
        ref_variables = self.test_sample.values()
        res = self.conditional_priors.rescale(
            keys=self.test_sample.keys(), theta=ref_variables
        )
        expected = [self.test_sample["var_0"]]
        for ii in range(1, 4):
            expected.append(expected[-1] * self.test_sample[f"var_{ii}"])
        self.assertListEqual(expected, res)

    def test_rescale_with_joint_prior(self):
        """
        Add a joint prior into the conditional prior dictionary and check that
        the returned list is flat.
        """

        # set multivariate Gaussian distribution
        names = ["mvgvar_0", "mvgvar_1"]
        mu = [[0.79, -0.83]]
        cov = [[[0.03, 0.], [0., 0.04]]]
        mvg = bilby.core.prior.MultivariateGaussianDist(names, mus=mu, covs=cov)

        priordict = bilby.core.prior.ConditionalPriorDict(
            dict(
                var_3=self.prior_3,
                var_2=self.prior_2,
                var_0=self.prior_0,
                var_1=self.prior_1,
                mvgvar_0=bilby.core.prior.MultivariateGaussian(mvg, "mvgvar_0"),
                mvgvar_1=bilby.core.prior.MultivariateGaussian(mvg, "mvgvar_1"),
            )
        )

        ref_variables = list(self.test_sample.values()) + [0.4, 0.1]
        keys = list(self.test_sample.keys()) + names
        res = priordict.rescale(keys=keys, theta=ref_variables)

        self.assertIsInstance(res, list)
        self.assertEqual(np.shape(res), (6,))
        self.assertListEqual([isinstance(r, float) for r in res], 6 * [True])

        # check conditional values are still as expected
        expected = [self.test_sample["var_0"]]
        for ii in range(1, 4):
            expected.append(expected[-1] * self.test_sample[f"var_{ii}"])
        self.assertListEqual(expected, res[0:4])

    def test_cdf(self):
        """
        Test that the CDF method is the inverse of the rescale method.

        Note that the format of inputs/outputs is different between the two methods.
        """
        sample = self.conditional_priors.sample()
        self.assertEqual(
            self.conditional_priors.rescale(
                sample.keys(),
                self.conditional_priors.cdf(sample=sample).values()
            ), list(sample.values())
        )

    def test_rescale_illegal_conditions(self):
        del self.conditional_priors["var_0"]
        with self.assertRaises(bilby.core.prior.IllegalConditionsException):
            self.conditional_priors.rescale(
                keys=list(self.test_sample.keys()),
                theta=list(self.test_sample.values()),
            )

    def test_combined_conditions(self):
        def d_condition_func(reference_params, a, b, c):
            return dict(
                minimum=reference_params["minimum"], maximum=reference_params["maximum"]
            )

        def a_condition_func(reference_params, b, c):
            return dict(
                minimum=reference_params["minimum"], maximum=reference_params["maximum"]
            )

        priors = bilby.core.prior.ConditionalPriorDict()

        priors["a"] = bilby.core.prior.ConditionalUniform(
            condition_func=a_condition_func, minimum=0, maximum=1
        )

        priors["b"] = bilby.core.prior.LogUniform(minimum=1, maximum=10)

        priors["d"] = bilby.core.prior.ConditionalUniform(
            condition_func=d_condition_func, minimum=0.0, maximum=1.0
        )

        priors["c"] = bilby.core.prior.LogUniform(minimum=1, maximum=10)
        priors.sample()
        res = priors.rescale(["a", "b", "d", "c"], [0.5, 0.5, 0.5, 0.5])
        print(res)

    def test_subset_sampling(self):
        def _tp_conditional_uniform(ref_params, period):
            min_ref, max_ref = ref_params["minimum"], ref_params["maximum"]
            max_ref = np.minimum(max_ref, min_ref + period)
            return {"minimum": min_ref, "maximum": max_ref}

        p0 = 68400.0
        prior = bilby.core.prior.ConditionalPriorDict(
            {
                "tp": bilby.core.prior.ConditionalUniform(
                    condition_func=_tp_conditional_uniform, minimum=0, maximum=2 * p0
                )
            }
        )

        # ---------- 0. Sanity check: sample full prior
        prior["period"] = p0
        samples2d = prior.sample(1000)
        assert samples2d["tp"].max() < p0

        # ---------- 1. Subset sampling with external delta-prior
        print("Test 1: Subset-sampling conditionals for fixed 'externals':")
        prior["period"] = p0
        samples1d = prior.sample_subset(["tp"], 1000)
        self.assertLess(samples1d["tp"].max(), p0)

        # ---------- 2. Subset sampling with external uniform prior
        prior["period"] = bilby.core.prior.Uniform(minimum=p0, maximum=2 * p0)
        print("Test 2: Subset-sampling conditionals for 'external' uncertainties:")
        with self.assertRaises(bilby.core.prior.IllegalConditionsException):
            prior.sample_subset(["tp"], 1000)


class TestDirichletPrior(unittest.TestCase):

    def setUp(self):
        self.priors = bilby.core.prior.DirichletPriorDict(5)

    def tearDown(self):
        if os.path.isdir("priors"):
            shutil.rmtree("priors")

    def test_samples_sum_to_less_than_one(self):
        """
        Test that the samples sum to less than one as required for the
        Dirichlet distribution.
        """
        samples = pd.DataFrame(self.priors.sample(10000)).values
        self.assertLess(max(np.sum(samples, axis=1)), 1)

    def test_read_write_file(self):
        self.priors.to_file(outdir="priors", label="test")
        test = bilby.core.prior.PriorDict(filename="priors/test.prior")
        self.assertEqual(self.priors, test)

    def test_read_write_json(self):
        self.priors.to_json(outdir="priors", label="test")
        test = bilby.core.prior.PriorDict.from_json(filename="priors/test_prior.json")
        self.assertEqual(self.priors, test)

    def test_pickle(self):
        """Assert can be pickled (needed for use with bilby_pipe)"""
        pickle.dumps(self.priors)


if __name__ == "__main__":
    unittest.main()
