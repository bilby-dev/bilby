from __future__ import absolute_import
import tupak
from tupak.core import prior
from tupak.core.result import Result
import unittest
from mock import MagicMock
import numpy as np
import inspect
import os
import copy


class TestGaussianLikelihood(unittest.TestCase):

    def setUp(self):
        self.N = 100
        self.sigma = 0.1
        self.x = np.linspace(0, 1, self.N)
        self.y = 2 * self.x + 1 + np.random.normal(0, self.sigma, self.N)

        def test_function(x, m, c):
            return m * x + c
        self.function = test_function

    def tearDown(self):
        del self.N
        del self.sigma
        del self.x
        del self.y
        del self.function

    def test_known_sigma(self):
        likelihood = tupak.core.likelihood.GaussianLikelihood(
            self.x, self.y, self.function, self.sigma)
        likelihood.parameters['m'] = 2
        likelihood.parameters['c'] = 0
        likelihood.log_likelihood()
        self.assertEqual(likelihood.sigma, self.sigma)

    def test_known_array_sigma(self):
        sigma_array = np.ones(self.N) * self.sigma
        likelihood = tupak.core.likelihood.GaussianLikelihood(
            self.x, self.y, self.function, sigma_array)
        likelihood.parameters['m'] = 2
        likelihood.parameters['c'] = 0
        likelihood.log_likelihood()
        self.assertTrue(type(likelihood.sigma) == type(sigma_array))
        self.assertTrue(all(likelihood.sigma == sigma_array))

    def test_unknown_float_sigma(self):
        likelihood = tupak.core.likelihood.GaussianLikelihood(
            self.x, self.y, self.function, sigma=None)
        likelihood.parameters['m'] = 2
        likelihood.parameters['c'] = 0
        self.assertTrue(likelihood.sigma is None)
        with self.assertRaises(TypeError):
            likelihood.log_likelihood()
        likelihood.parameters['sigma'] = 1
        likelihood.log_likelihood()
        self.assertTrue(likelihood.sigma is None)

    def test_y(self):
        likelihood = tupak.core.likelihood.GaussianLikelihood(
            self.x, self.y, self.function, self.sigma)
        self.assertTrue(all(likelihood.y == self.y))

    def test_x(self):
        likelihood = tupak.core.likelihood.GaussianLikelihood(
            self.x, self.y, self.function, self.sigma)
        self.assertTrue(all(likelihood.x == self.x))

    def test_N(self):
        likelihood = tupak.core.likelihood.GaussianLikelihood(
            self.x, self.y, self.function, self.sigma)
        self.assertTrue(likelihood.N == len(self.x))


class TestExponentialLikelihood(unittest.TestCase):
    
    def setUp(self):
        self.N = 100
        self.mu = 5
        self.x = np.linspace(0, 1, self.N)
        self.y = np.random.exponential(self.mu, self.N)
        self.yneg = np.copy(self.y)
        self.yneg[0] = -1.

        def test_function(x, c):
            return c

        def test_function_array(x, c):
            return c*np.ones(len(x))

        self.function = test_function
        self.function_array = test_function_array

    def tearDown(self):
        del self.N
        del self.mu
        del self.x
        del self.y
        del self.yneg
        del self.function
        del self.function_array

    def test_negative_data(self):
        with self.assertRaises(ValueError):
            tupak.core.likelihood.ExponentialLikelihood(self.x, self.yneg, self.function)

    def test_negative_function(self):
        likelihood = tupak.core.likelihood.ExponentialLikelihood(
            self.x, self.y, self.function)
        likelihood.parameters['c'] = -1
        self.assertEqual(likelihood.log_likelihood(), -np.inf)

    def test_negative_array_function(self):
        likelihood = tupak.core.likelihood.ExponentialLikelihood(
            self.x, self.y, self.function_array)
        likelihood.parameters['c'] = -1
        self.assertEqual(likelihood.log_likelihood(), -np.inf)

    def test_y(self):
        likelihood = tupak.core.likelihood.ExponentialLikelihood(
            self.x, self.y, self.function)
        self.assertTrue(all(likelihood.y == self.y))

    def test_x(self):
        likelihood = tupak.core.likelihood.ExponentialLikelihood(
            self.x, self.y, self.function)
        self.assertTrue(all(likelihood.x == self.x))

    def test_N(self):
        likelihood = tupak.core.likelihood.ExponentialLikelihood(
            self.x, self.y, self.function)
        self.assertTrue(likelihood.N == len(self.x))


if __name__ == '__main__':
    unittest.main()
