import numpy as np
import unittest

import bilby
from bilby.core.prior.slabspike import SlabSpikePrior
from bilby.core.prior.analytical import Uniform, PowerLaw, LogUniform, TruncatedGaussian, \
    Beta, Gaussian, Cosine, Sine, HalfGaussian, LogNormal, Exponential, StudentT, Logistic, \
    Cauchy, Gamma, ChiSquared


class TestSlabSpikePrior(unittest.TestCase):

    def setUp(self):
        self.minimum = 0
        self.maximum = 1
        self.spike_loc = 0.5
        self.spike_height = 0.3
        self.slab = bilby.core.prior.Prior(minimum=self.minimum, maximum=self.maximum)
        self.prior = SlabSpikePrior(
            slab=self.slab, spike_location=self.spike_loc, spike_height=self.spike_height)

    def tearDown(self):
        del self.minimum
        del self.maximum
        del self.spike_loc
        del self.spike_height
        del self.prior
        del self.slab

    def test_slab_fraction(self):
        expected = 1 - self.spike_height
        self.assertEqual(expected, self.prior.slab_fraction)

    def test_spike_loc(self):
        self.assertEqual(self.spike_loc, self.prior.spike_location)

    def test_set_spike_loc_none(self):
        self.prior.spike_location = None
        self.assertEqual(self.prior.minimum, self.prior.spike_location)

    def test_set_spike_loc_outside_domain(self):
        with self.assertRaises(ValueError):
            self.prior.spike_location = 1.5

    def test_set_spike_loc_maximum(self):
        self.prior.spike_location = self.maximum
        self.assertEqual(self.maximum, self.prior.spike_location)

    def test_class_name(self):
        expected = "SlabSpikePrior"
        self.assertEqual(expected, self.prior.__class__.__name__)
        self.assertEqual(expected, self.prior.__class__.__qualname__)

    def test_set_spike_height_outside_domain(self):
        with self.assertRaises(ValueError):
            self.prior.spike_height = 1.5

    def test_set_spike_height_domain_edge(self):
        self.prior.spike_height = 0
        self.prior.spike_height = 1


class TestSlabSpikeClasses(unittest.TestCase):

    def setUp(self):
        self.minimum = 0.4
        self.maximum = 2.4
        self.spike_loc = 1.5
        self.spike_height = 0.3

        self.slabs = [
            Uniform(minimum=self.minimum, maximum=self.maximum),
            PowerLaw(minimum=self.minimum, maximum=self.maximum, alpha=2),
            LogUniform(minimum=self.minimum, maximum=self.maximum),
            TruncatedGaussian(minimum=self.minimum, maximum=self.maximum, mu=0, sigma=1),
            Beta(minimum=self.minimum, maximum=self.maximum, alpha=1, beta=1),
            Gaussian(mu=0, sigma=1),
            Cosine(),
            Sine(),
            HalfGaussian(sigma=1),
            LogNormal(mu=1, sigma=2),
            Exponential(mu=2),
            StudentT(df=2),
            Logistic(mu=2, scale=1),
            Cauchy(alpha=1, beta=2),
            Gamma(k=1, theta=1.),
            ChiSquared(nu=2)]
        self.slab_spikes = [SlabSpikePrior(slab, spike_height=self.spike_height, spike_location=self.spike_loc)
                            for slab in self.slabs]
        self.test_nodes_finite_support = np.linspace(self.minimum, self.maximum, 1000)
        self.test_nodes_infinite_support = np.linspace(-10, 10, 1000)
        self.test_nodes = [self.test_nodes_finite_support
                           if np.isinf(slab.minimum) or np.isinf(slab.maximum)
                           else self.test_nodes_finite_support for slab in self.slabs]

    def tearDown(self):
        del self.minimum
        del self.maximum
        del self.spike_loc
        del self.spike_height
        del self.slabs
        del self.test_nodes_finite_support
        del self.test_nodes_infinite_support

    def test_prob_on_slab(self):
        for slab, slab_spike, test_nodes in zip(self.slabs, self.slab_spikes, self.test_nodes):
            expected = slab.prob(test_nodes) * slab_spike.slab_fraction
            actual = slab_spike.prob(test_nodes)
            self.assertTrue(np.allclose(expected, actual, rtol=1e-5))

    def test_prob_on_spike(self):
        for slab_spike in self.slab_spikes:
            self.assertEqual(np.inf, slab_spike.prob(self.spike_loc))

    def test_ln_prob_on_slab(self):
        for slab, slab_spike, test_nodes in zip(self.slabs, self.slab_spikes, self.test_nodes):
            expected = slab.ln_prob(test_nodes) + np.log(slab_spike.slab_fraction)
            actual = slab_spike.ln_prob(test_nodes)
            self.assertTrue(np.array_equal(expected, actual))

    def test_ln_prob_on_spike(self):
        for slab_spike in self.slab_spikes:
            self.assertEqual(np.inf, slab_spike.ln_prob(self.spike_loc))

    def test_inverse_cdf_below_spike_with_spike_at_minimum(self):
        for slab in self.slabs:
            slab_spike = SlabSpikePrior(slab=slab, spike_height=0.4, spike_location=slab.minimum)
            self.assertEqual(0, slab_spike.inverse_cdf_below_spike)

    def test_inverse_cdf_below_spike_with_spike_at_maximum(self):
        for slab in self.slabs:
            slab_spike = SlabSpikePrior(slab=slab, spike_height=0.4, spike_location=slab.maximum)
            expected = 1 - slab_spike.spike_height
            actual = slab_spike.inverse_cdf_below_spike
            self.assertEqual(expected, actual)

    def test_inverse_cdf_below_spike_arbitrary_position(self):
        pass

    def test_cdf_below_spike(self):
        for slab, slab_spike, test_nodes in zip(self.slabs, self.slab_spikes, self.test_nodes):
            test_nodes = test_nodes[np.where(test_nodes < self.spike_loc)]
            expected = slab.cdf(test_nodes) * slab_spike.slab_fraction
            actual = slab_spike.cdf(test_nodes)
            self.assertTrue(np.allclose(expected, actual, rtol=1e-5))

    def test_cdf_at_spike(self):
        for slab, slab_spike in zip(self.slabs, self.slab_spikes):
            expected = slab.cdf(self.spike_loc) * slab_spike.slab_fraction
            actual = slab_spike.cdf(self.spike_loc)
            self.assertTrue(np.allclose(expected, actual, rtol=1e-5))

    def test_cdf_above_spike(self):
        for slab, slab_spike, test_nodes in zip(self.slabs, self.slab_spikes, self.test_nodes):
            test_nodes = test_nodes[np.where(test_nodes > self.spike_loc)]
            expected = slab.cdf(test_nodes) * slab_spike.slab_fraction + self.spike_height
            actual = slab_spike.cdf(test_nodes)
            self.assertTrue(np.array_equal(expected, actual))

    def test_cdf_at_minimum(self):
        for slab_spike in self.slab_spikes:
            expected = 0
            actual = slab_spike.cdf(slab_spike.minimum)
            self.assertEqual(expected, actual)

    def test_cdf_at_maximum(self):
        for slab_spike in self.slab_spikes:
            expected = 1
            actual = slab_spike.cdf(slab_spike.maximum)
            self.assertEqual(expected, actual)

    def test_rescale_no_spike(self):
        for slab in self.slabs:
            slab_spike = SlabSpikePrior(slab=slab, spike_height=0, spike_location=slab.minimum)
            vals = np.linspace(0, 1, 1000)
            expected = slab.rescale(vals)
            actual = slab_spike.rescale(vals)
            print(slab)
            self.assertTrue(np.allclose(expected, actual, rtol=1e-5))

    def test_rescale_below_spike(self):
        for slab, slab_spike in zip(self.slabs, self.slab_spikes):
            vals = np.linspace(0, slab_spike.inverse_cdf_below_spike, 1000)
            expected = slab.rescale(vals / slab_spike.slab_fraction)
            actual = slab_spike.rescale(vals)
            self.assertTrue(np.allclose(expected, actual, rtol=1e-5))

    def test_rescale_at_spike(self):
        for slab, slab_spike in zip(self.slabs, self.slab_spikes):
            vals = np.linspace(slab_spike.inverse_cdf_below_spike,
                               slab_spike.inverse_cdf_below_spike + slab_spike.spike_height, 1000)
            expected = np.ones(len(vals)) * slab.rescale(vals[0] / slab_spike.slab_fraction)
            actual = slab_spike.rescale(vals)
            self.assertTrue(np.allclose(expected, actual, rtol=1e-5))

    def test_rescale_above_spike(self):
        for slab, slab_spike in zip(self.slabs, self.slab_spikes):
            vals = np.linspace(slab_spike.inverse_cdf_below_spike + self.spike_height, 1, 1000)
            expected = np.ones(len(vals)) * slab.rescale(
                (vals - self.spike_height) / slab_spike.slab_fraction)
            actual = slab_spike.rescale(vals)
            self.assertTrue(np.allclose(expected, actual, rtol=1e-5))
