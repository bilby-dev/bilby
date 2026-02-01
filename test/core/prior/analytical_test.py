import unittest

import bilby
import numpy as np
import pytest


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TestDiscreteValuesPrior(unittest.TestCase):
    def test_single_sample(self):
        values = [1.1, 1.2, 1.3]
        discrete_value_prior = bilby.core.prior.DiscreteValues(values)
        in_prior = True
        for _ in range(1000):
            s = discrete_value_prior.sample(xp=self.xp)
            if s not in values:
                in_prior = False
        self.assertTrue(in_prior)

    def test_array_sample(self):
        values = [1.0, 1.1, 1.2, 1.3]
        nvalues = 4
        discrete_value_prior = bilby.core.prior.DiscreteValues(values)
        N = 100000
        s = discrete_value_prior.sample(N, xp=self.xp)
        zeros = np.sum(s == 1.0)
        ones = np.sum(s == 1.1)
        twos = np.sum(s == 1.2)
        threes = np.sum(s == 1.3)
        self.assertEqual(zeros + ones + twos + threes, N)
        self.assertAlmostEqual(zeros / N, 1 / nvalues, places=int(np.log10(np.sqrt(N))))
        self.assertAlmostEqual(ones / N, 1 / nvalues, places=int(np.log10(np.sqrt(N))))
        self.assertAlmostEqual(twos / N, 1 / nvalues, places=int(np.log10(np.sqrt(N))))
        self.assertAlmostEqual(threes / N, 1 / nvalues, places=int(np.log10(np.sqrt(N))))

    def test_single_probability(self):
        N = 3
        values = [1.1, 2.2, 300.0]
        discrete_value_prior = bilby.core.prior.DiscreteValues(values)
        self.assertEqual(discrete_value_prior.prob(self.xp.array(1.1)), 1 / N)
        self.assertEqual(discrete_value_prior.prob(self.xp.array(2.2)), 1 / N)
        self.assertEqual(discrete_value_prior.prob(self.xp.array(300.0)), 1 / N)
        self.assertEqual(discrete_value_prior.prob(self.xp.array(0.5)), 0)
        self.assertEqual(discrete_value_prior.prob(self.xp.array(200)), 0)

    def test_single_probability_unsorted(self):
        N = 3
        values = [1.1, 300, 2.2]
        discrete_value_prior = bilby.core.prior.DiscreteValues(values)
        self.assertEqual(discrete_value_prior.prob(self.xp.array(1.1)), 1 / N)
        self.assertEqual(discrete_value_prior.prob(self.xp.array(2.2)), 1 / N)
        self.assertEqual(discrete_value_prior.prob(self.xp.array(300.0)), 1 / N)
        self.assertEqual(discrete_value_prior.prob(self.xp.array(0.5)), 0)
        self.assertEqual(discrete_value_prior.prob(self.xp.array(200)), 0)
        self.assertEqual(
            discrete_value_prior.prob(self.xp.array(0.5)).__array_namespace__(),
            self.xp,
        )

    def test_array_probability(self):
        N = 3
        values = [1.1, 2.2, 300.0]
        discrete_value_prior = bilby.core.prior.DiscreteValues(values)
        self.assertTrue(
            np.all(
                discrete_value_prior.prob(self.xp.array([1.1, 2.2, 2.2, 300.0, 200.0]))
                == np.array([1 / N, 1 / N, 1 / N, 1 / N, 0])
            )
        )

    def test_single_lnprobability(self):
        N = 3
        values = [1.1, 2.2, 300.0]
        discrete_value_prior = bilby.core.prior.DiscreteValues(values)
        self.assertEqual(discrete_value_prior.ln_prob(self.xp.array(1.1)), -np.log(N))
        self.assertEqual(discrete_value_prior.ln_prob(self.xp.array(2.2)), -np.log(N))
        self.assertEqual(discrete_value_prior.ln_prob(self.xp.array(300)), -np.log(N))
        self.assertEqual(discrete_value_prior.ln_prob(self.xp.array(150)), -np.inf)
        self.assertEqual(
            discrete_value_prior.ln_prob(self.xp.array(0.5)).__array_namespace__(),
            self.xp,
        )

    def test_array_lnprobability(self):
        N = 3
        values = [1.1, 2.2, 300.0]
        discrete_value_prior = bilby.core.prior.DiscreteValues(values)
        self.assertTrue(
            np.all(
                discrete_value_prior.ln_prob(self.xp.array([1.1, 2.2, 2.2, 300, 150]))
                == np.array([-np.log(N), -np.log(N), -np.log(N), -np.log(N), -np.inf])
            )
        )


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TestCategoricalPrior(unittest.TestCase):
    def test_single_sample(self):
        categorical_prior = bilby.core.prior.Categorical(3)
        in_prior = True
        for _ in range(1000):
            s = categorical_prior.sample(xp=self.xp)
            if s not in [0, 1, 2]:
                in_prior = False
        self.assertTrue(in_prior)

    def test_array_sample(self):
        ncat = 4
        categorical_prior = bilby.core.prior.Categorical(ncat)
        N = 100000
        s = categorical_prior.sample(N, xp=self.xp)
        zeros = np.sum(s == 0)
        ones = np.sum(s == 1)
        twos = np.sum(s == 2)
        threes = np.sum(s == 3)
        self.assertEqual(zeros + ones + twos + threes, N)
        self.assertAlmostEqual(zeros / N, 1 / ncat, places=int(np.log10(np.sqrt(N))))
        self.assertAlmostEqual(ones / N, 1 / ncat, places=int(np.log10(np.sqrt(N))))
        self.assertAlmostEqual(twos / N, 1 / ncat, places=int(np.log10(np.sqrt(N))))
        self.assertAlmostEqual(threes / N, 1 / ncat, places=int(np.log10(np.sqrt(N))))
        self.assertEqual(s.__array_namespace__(), self.xp)

    def test_single_probability(self):
        N = 3
        categorical_prior = bilby.core.prior.Categorical(N)
        self.assertEqual(categorical_prior.prob(self.xp.array(0)), 1 / N)
        self.assertEqual(categorical_prior.prob(self.xp.array(1)), 1 / N)
        self.assertEqual(categorical_prior.prob(self.xp.array(2)), 1 / N)
        self.assertEqual(categorical_prior.prob(self.xp.array(0.5)), 0)
        self.assertEqual(
            categorical_prior.prob(self.xp.array(0.5)).__array_namespace__(),
            self.xp,
        )

    def test_array_probability(self):
        N = 3
        categorical_prior = bilby.core.prior.Categorical(N)
        self.assertTrue(np.all(
            categorical_prior.prob(self.xp.array([0, 1, 1, 2, 3]))
            == np.array([1 / N, 1 / N, 1 / N, 1 / N, 0])
        ))

    def test_single_lnprobability(self):
        N = 3
        categorical_prior = bilby.core.prior.Categorical(N)
        self.assertEqual(categorical_prior.ln_prob(self.xp.array(0)), -np.log(N))
        self.assertEqual(categorical_prior.ln_prob(self.xp.array(1)), -np.log(N))
        self.assertEqual(categorical_prior.ln_prob(self.xp.array(2)), -np.log(N))
        self.assertEqual(categorical_prior.ln_prob(self.xp.array(0.5)), -np.inf)
        self.assertEqual(
            categorical_prior.ln_prob(self.xp.array(0.5)).__array_namespace__(),
            self.xp,
        )

    def test_array_lnprobability(self):
        N = 3
        categorical_prior = bilby.core.prior.Categorical(N)
        self.assertTrue(np.all(
            categorical_prior.ln_prob(self.xp.array([0, 1, 1, 2, 3]))
            == np.array([-np.log(N), -np.log(N), -np.log(N), -np.log(N), -np.inf])
        ))


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TestWeightedCategoricalPrior(unittest.TestCase):
    def test_single_sample(self):
        categorical_prior = bilby.core.prior.WeightedCategorical(3, [1, 2, 3])
        in_prior = True
        for _ in range(1000):
            s = categorical_prior.sample(xp=self.xp)
            if s not in [0, 1, 2]:
                in_prior = False
        self.assertTrue(in_prior)

    def test_fail_init(self):
        with self.assertRaises(ValueError):
            bilby.core.prior.WeightedCategorical(3, [[1, 2], [2, 3], [3, 4]])
        with self.assertRaises(ValueError):
            bilby.core.prior.WeightedCategorical(3, [1, 2, 3, 4])

    def test_array_sample(self):
        ncat = 4
        weights = np.arange(1, ncat + 1)
        categorical_prior = bilby.core.prior.WeightedCategorical(ncat, weights=weights)
        N = 100000
        s = categorical_prior.sample(N, xp=self.xp)
        cases = 0
        for i in self.xp.array(categorical_prior.values):
            case = np.sum(s == i)
            cases += case
            self.assertAlmostEqual(case / N, categorical_prior.prob(i), places=int(np.log10(np.sqrt(N))))
            self.assertAlmostEqual(case / N, weights[i] / np.sum(weights), places=int(np.log10(np.sqrt(N))))
        self.assertEqual(cases, N)
        self.assertEqual(s.__array_namespace__(), self.xp)

    def test_single_probability(self):
        N = 3
        weights = np.arange(1, N + 1)
        categorical_prior = bilby.core.prior.WeightedCategorical(N, weights=weights)
        for i in self.xp.array(categorical_prior.values):
            self.assertEqual(categorical_prior.prob(i), weights[i] / np.sum(weights))
        prob = categorical_prior.prob(self.xp.array(0.5))
        self.assertEqual(prob, 0)
        self.assertEqual(prob.__array_namespace__(), self.xp)

    def test_array_probability(self):
        N = 3
        test_cases = self.xp.array([0, 1, 1, 2, 3])
        weights = np.arange(1, N + 1)
        categorical_prior = bilby.core.prior.WeightedCategorical(N, weights=weights)
        probs = np.arange(1, N + 2) / np.sum(weights)
        probs[-1] = 0
        new = categorical_prior.prob(test_cases)
        self.assertTrue(np.all(new == probs[test_cases]))
        self.assertEqual(new.__array_namespace__(), self.xp)

    def test_single_lnprobability(self):
        N = 3
        weights = np.arange(1, N + 1)
        categorical_prior = bilby.core.prior.WeightedCategorical(N, weights=weights)
        for i in self.xp.array(categorical_prior.values):
            self.assertEqual(
                categorical_prior.ln_prob(self.xp.array(i)),
                np.log(weights[i] / np.sum(weights)),
            )
        prob = categorical_prior.prob(self.xp.array(0.5))
        self.assertEqual(prob, 0)
        self.assertEqual(prob.__array_namespace__(), self.xp)

    def test_array_lnprobability(self):
        N = 3
        test_cases = [0, 1, 1, 2, 3]
        weights = np.arange(1, N + 1)

        categorical_prior = bilby.core.prior.WeightedCategorical(N, weights=weights)
        ln_probs = np.log(np.arange(1, N + 2) / np.sum(weights))
        ln_probs[-1] = -np.inf

        new = categorical_prior.ln_prob(self.xp.array(test_cases))
        self.assertTrue(np.all(new == ln_probs[test_cases]))
        self.assertEqual(new.__array_namespace__(), self.xp)

    def test_cdf(self):
        """
        Test that the CDF method is the inverse of the rescale method.

        Note that the format of inputs/outputs is different between the two methods.
        """
        N = 3
        weights = np.arange(1, N + 1)

        categorical_prior = bilby.core.prior.WeightedCategorical(N, weights=weights)
        sample = categorical_prior.sample(size=10)
        original = self.xp.asarray(sample)
        new = self.xp.array(categorical_prior.rescale(
            categorical_prior.cdf(sample)
        ))
        np.testing.assert_array_equal(original, new)
        self.assertEqual(type(new), type(original))


if __name__ == "__main__":
    unittest.main()
