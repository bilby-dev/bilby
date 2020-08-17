import unittest

import mock

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

    def test_rescale_prob_update_conditions(self):
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
            return reference_parameters

        def condition_func_2(reference_parameters, var_0, var_1):
            return reference_parameters

        def condition_func_3(reference_parameters, var_1, var_2):
            return reference_parameters

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
        self.test_sample = dict(var_0=0.3, var_1=0.4, var_2=0.5, var_3=0.4)
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
        self.assertEqual(1, self.conditional_priors.prob(sample=self.test_sample))

    def test_prob_illegal_conditions(self):
        del self.conditional_priors["var_0"]
        with self.assertRaises(bilby.core.prior.IllegalConditionsException):
            self.conditional_priors.prob(sample=self.test_sample)

    def test_ln_prob(self):
        self.assertEqual(0, self.conditional_priors.ln_prob(sample=self.test_sample))

    def test_ln_prob_illegal_conditions(self):
        del self.conditional_priors["var_0"]
        with self.assertRaises(bilby.core.prior.IllegalConditionsException):
            self.conditional_priors.ln_prob(sample=self.test_sample)

    def test_sample_subset_all_keys(self):
        with mock.patch("numpy.random.uniform") as m:
            m.return_value = 0.5
            self.assertDictEqual(
                dict(var_0=0.5, var_1=0.5, var_2=0.5, var_3=0.5),
                self.conditional_priors.sample_subset(
                    keys=["var_0", "var_1", "var_2", "var_3"]
                ),
            )

    def test_sample_illegal_subset(self):
        with mock.patch("numpy.random.uniform") as m:
            m.return_value = 0.5
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
        def condition_func_1_rescale(reference_parameters, var_0):
            if var_0 == 0.5:
                return dict(minimum=reference_parameters["minimum"], maximum=1)
            return reference_parameters

        def condition_func_2_rescale(reference_parameters, var_0, var_1):
            if var_0 == 0.5 and var_1 == 0.5:
                return dict(minimum=reference_parameters["minimum"], maximum=1)
            return reference_parameters

        def condition_func_3_rescale(reference_parameters, var_1, var_2):
            if var_1 == 0.5 and var_2 == 0.5:
                return dict(minimum=reference_parameters["minimum"], maximum=1)
            return reference_parameters

        self.prior_0 = bilby.core.prior.Uniform(minimum=self.minimum, maximum=1)
        self.prior_1 = bilby.core.prior.ConditionalUniform(
            condition_func=condition_func_1_rescale, minimum=self.minimum, maximum=2
        )
        self.prior_2 = bilby.core.prior.ConditionalUniform(
            condition_func=condition_func_2_rescale, minimum=self.minimum, maximum=2
        )
        self.prior_3 = bilby.core.prior.ConditionalUniform(
            condition_func=condition_func_3_rescale, minimum=self.minimum, maximum=2
        )
        self.conditional_priors = bilby.core.prior.ConditionalPriorDict(
            dict(
                var_3=self.prior_3,
                var_2=self.prior_2,
                var_0=self.prior_0,
                var_1=self.prior_1,
            )
        )
        ref_variables = [0.5, 0.5, 0.5, 0.5]
        res = self.conditional_priors.rescale(
            keys=list(self.test_sample.keys()), theta=ref_variables
        )
        self.assertListEqual(ref_variables, res)

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


if __name__ == "__main__":
    unittest.main()
