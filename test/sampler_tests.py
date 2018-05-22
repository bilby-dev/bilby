from context import tupak
from tupak import prior
from tupak import likelihood
import unittest
from mock import Mock, MagicMock
import numpy as np


class TestSamplerInstantiation(unittest.TestCase):

    def setUp(self):
        self.likelihood = likelihood.Likelihood()
        self.likelihood.parameters = MagicMock(return_value=dict(a=1, b=2, c=3))
        delta_prior = prior.DeltaFunction(0)
        delta_prior.peak = MagicMock(return_value=0)
        delta_prior.rescale = MagicMock(return_value=delta_prior)
        delta_prior.prob = MagicMock(return_value=1)
        uniform_prior = prior.Uniform(0, 1)
        uniform_prior.minimum = MagicMock(return_value=0)
        uniform_prior.maximum = MagicMock(return_value=1)
        uniform_prior.rescale = MagicMock(return_value=uniform_prior)
        uniform_prior.prob = MagicMock(return_value=1)

        self.priors = dict(a=delta_prior, b='string', c=uniform_prior)
        self.likelihood.log_likelihood_ratio = MagicMock(return_value=1)
        self.likelihood.log_likelihood = MagicMock(return_value=2)

    def tearDown(self):
        del self.likelihood
        del self.priors

    def test_default_instantiation(self):
        sampler = tupak.sampler.Sampler(self.likelihood, self.priors)
