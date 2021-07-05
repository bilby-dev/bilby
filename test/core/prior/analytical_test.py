import unittest

import numpy as np
import bilby


class TestCategoricalPrior(unittest.TestCase):
    def test_single_sample(self):
        categorical_prior = bilby.core.prior.Categorical(3)
        in_prior = True
        for _ in range(1000):
            s = categorical_prior.sample()
            if s not in [0, 1, 2]:
                in_prior = False
        self.assertTrue(in_prior)

    def test_array_sample(self):
        ncat = 4
        categorical_prior = bilby.core.prior.Categorical(ncat)
        N = 100000
        s = categorical_prior.sample(N)
        zeros = np.sum(s == 0)
        ones = np.sum(s == 1)
        twos = np.sum(s == 2)
        threes = np.sum(s == 3)
        self.assertEqual(zeros + ones + twos + threes, N)
        self.assertAlmostEqual(zeros / N, 1 / ncat, places=int(np.log10(np.sqrt(N))))
        self.assertAlmostEqual(ones / N, 1 / ncat, places=int(np.log10(np.sqrt(N))))
        self.assertAlmostEqual(twos / N, 1 / ncat, places=int(np.log10(np.sqrt(N))))
        self.assertAlmostEqual(threes / N, 1 / ncat, places=int(np.log10(np.sqrt(N))))

    def test_single_probability(self):
        N = 3
        categorical_prior = bilby.core.prior.Categorical(N)
        self.assertEqual(categorical_prior.prob(0), 1 / N)
        self.assertEqual(categorical_prior.prob(1), 1 / N)
        self.assertEqual(categorical_prior.prob(2), 1 / N)
        self.assertEqual(categorical_prior.prob(0.5), 0)

    def test_array_probability(self):
        N = 3
        categorical_prior = bilby.core.prior.Categorical(N)
        self.assertTrue(
            np.all(
                categorical_prior.prob([0, 1, 1, 2, 3])
                == np.array([1 / N, 1 / N, 1 / N, 1 / N, 0])
            )
        )

    def test_single_lnprobability(self):
        N = 3
        categorical_prior = bilby.core.prior.Categorical(N)
        self.assertEqual(categorical_prior.ln_prob(0), -np.log(N))
        self.assertEqual(categorical_prior.ln_prob(1), -np.log(N))
        self.assertEqual(categorical_prior.ln_prob(2), -np.log(N))
        self.assertEqual(categorical_prior.ln_prob(0.5), -np.inf)

    def test_array_lnprobability(self):
        N = 3
        categorical_prior = bilby.core.prior.Categorical(N)
        self.assertTrue(
            np.all(
                categorical_prior.ln_prob([0, 1, 1, 2, 3])
                == np.array([-np.log(N), -np.log(N), -np.log(N), -np.log(N), -np.inf])
            )
        )


if __name__ == "__main__":
    unittest.main()
