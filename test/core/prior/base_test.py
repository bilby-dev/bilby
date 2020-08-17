import unittest

import numpy as np
from mock import Mock

import bilby


class TestPriorInstantiationWithoutOptionalPriors(unittest.TestCase):
    def setUp(self):
        self.prior = bilby.core.prior.Prior()

    def tearDown(self):
        del self.prior

    def test_name(self):
        self.assertIsNone(self.prior.name)

    def test_latex_label(self):
        self.assertIsNone(self.prior.latex_label)

    def test_is_fixed(self):
        self.assertFalse(self.prior.is_fixed)

    def test_class_instance(self):
        self.assertIsInstance(self.prior, bilby.core.prior.Prior)

    def test_magic_call_is_the_same_as_sampling(self):
        self.prior.sample = Mock(return_value=0.5)
        self.assertEqual(self.prior.sample(), self.prior())

    def test_base_rescale_method(self):
        self.assertIsNone(self.prior.rescale(1))

    def test_base_repr(self):
        """
        We compare that the strings contain all of the same characters in not
        necessarily the same order as python2 doesn't conserve the order of the
        arguments.
        """
        self.prior = bilby.core.prior.Prior(
            name="test_name",
            latex_label="test_label",
            minimum=0,
            maximum=1,
            check_range_nonzero=True,
            boundary=None,
        )
        expected_string = (
            "Prior(name='test_name', latex_label='test_label', unit=None, minimum=0, maximum=1, "
            "check_range_nonzero=True, boundary=None)"
        )
        self.assertTrue(sorted(expected_string) == sorted(self.prior.__repr__()))

    def test_base_prob(self):
        self.assertTrue(np.isnan(self.prior.prob(5)))

    def test_base_ln_prob(self):
        self.prior.prob = lambda val: val
        self.assertEqual(np.log(5), self.prior.ln_prob(5))

    def test_is_in_prior(self):
        self.prior.minimum = 0
        self.prior.maximum = 1
        val_below = self.prior.minimum - 0.1
        val_at_minimum = self.prior.minimum
        val_in_prior = (self.prior.minimum + self.prior.maximum) / 2.0
        val_at_maximum = self.prior.maximum
        val_above = self.prior.maximum + 0.1
        self.assertTrue(self.prior.is_in_prior_range(val_at_minimum))
        self.assertTrue(self.prior.is_in_prior_range(val_at_maximum))
        self.assertTrue(self.prior.is_in_prior_range(val_in_prior))
        self.assertFalse(self.prior.is_in_prior_range(val_below))
        self.assertFalse(self.prior.is_in_prior_range(val_above))

    def test_boundary_is_none(self):
        self.assertIsNone(self.prior.boundary)


class TestPriorName(unittest.TestCase):
    def setUp(self):
        self.test_name = "test_name"
        self.prior = bilby.core.prior.Prior(self.test_name)

    def tearDown(self):
        del self.prior
        del self.test_name

    def test_name_assignment(self):
        self.prior.name = "other_name"
        self.assertEqual(self.prior.name, "other_name")


class TestPriorLatexLabel(unittest.TestCase):
    def setUp(self):
        self.test_name = "test_name"
        self.prior = bilby.core.prior.Prior(self.test_name)

    def tearDown(self):
        del self.test_name
        del self.prior

    def test_label_assignment(self):
        test_label = "test_label"
        self.prior.latex_label = "test_label"
        self.assertEqual(test_label, self.prior.latex_label)

    def test_default_label_assignment(self):
        self.prior.name = "chirp_mass"
        self.prior.latex_label = None
        self.assertEqual(self.prior.latex_label, "$\mathcal{M}$")

    def test_default_label_assignment_default(self):
        self.assertTrue(self.prior.latex_label, self.prior.name)


class TestPriorIsFixed(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        del self.prior

    def test_is_fixed_parent_class(self):
        self.prior = bilby.core.prior.Prior()
        self.assertFalse(self.prior.is_fixed)

    def test_is_fixed_delta_function_class(self):
        self.prior = bilby.core.prior.DeltaFunction(peak=0)
        self.assertTrue(self.prior.is_fixed)

    def test_is_fixed_uniform_class(self):
        self.prior = bilby.core.prior.Uniform(minimum=0, maximum=10)
        self.assertFalse(self.prior.is_fixed)


class TestPriorBoundary(unittest.TestCase):
    def setUp(self):
        self.prior = bilby.core.prior.Prior(boundary=None)

    def tearDown(self):
        del self.prior

    def test_set_boundary_valid(self):
        self.prior.boundary = "periodic"
        self.assertEqual(self.prior.boundary, "periodic")

    def test_set_boundary_invalid(self):
        with self.assertRaises(ValueError):
            self.prior.boundary = "else"


if __name__ == "__main__":
    unittest.main()
