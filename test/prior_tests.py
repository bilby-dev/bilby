from context import peyote
import unittest
import numpy as np


class TestPriorInstantiationWithoutOptionalPriors(unittest.TestCase):

    def setUp(self):
        self.prior = peyote.prior.Prior()

    def tearDown(self):
        del self.prior

    def test_name(self):
        self.assertIsNone(self.prior.name)

    def test_latex_label(self):
        self.assertIsNone(self.prior.latex_label)

    def test_is_fixed(self):
        self.assertFalse(self.prior.is_fixed)

    def test_class_instance(self):
        self.assertIsInstance(self.prior, peyote.prior.Prior)


class TestPriorName(unittest.TestCase):

    def setUp(self):
        self.test_name = 'test_name'
        self.prior = peyote.prior.Prior(self.test_name)

    def tearDown(self):
        del self.prior
        del self.test_name

    def test_name_assignment(self):
        self.prior.name = "other_name"
        self.assertEqual(self.prior.name, "other_name")


class TestPriorLatexLabel(unittest.TestCase):
    def setUp(self):
        self.test_name = 'test_name'
        self.prior = peyote.prior.Prior(self.test_name)

    def tearDown(self):
        del self.test_name
        del self.prior

    def test_label_assignment(self):
        test_label = 'test_label'
        self.prior.latex_label = 'test_label'
        self.assertEqual(test_label, self.prior.latex_label)

    def test_default_label_assignment(self):
        self.prior.name = 'mchirp'
        self.prior.latex_label = None
        self.assertEqual(self.prior.latex_label, '$\mathcal{M}$')

    def test_default_label_assignment_default(self):
        self.assertTrue(self.prior.latex_label, self.prior.name)


class TestPriorIsFixed(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_is_fixed_parent_class(self):
        self.prior = peyote.prior.Prior()
        self.assertFalse(self.prior.is_fixed)

    def test_is_fixed_delta_function_class(self):
        self.prior = peyote.prior.DeltaFunction(peak=0)
        self.assertTrue(self.prior.is_fixed)

    def test_is_fixed_uniform_class(self):
        self.prior = peyote.prior.Uniform(minimum=0, maximum=10)
        self.assertFalse(self.prior.is_fixed)


class TestFixMethod(unittest.TestCase):

    def setUp(self):
        self.test_name = 'test_name'
        self.prior = peyote.prior.Prior(self.test_name)

    def tearDown(self):
        del self.prior

    def test_is_fixed_attribute_after_fixing(self):
        arbitrary_float = 11.3
        fixed_prior = peyote.prior.fix(self.prior, arbitrary_float)
        self.assertTrue(fixed_prior.is_fixed)

    def test_value_attribute_after_fixing(self):
        arbitrary_float = 11.3
        fixed_prior = peyote.prior.fix(self.prior, arbitrary_float)
        self.assertEqual(fixed_prior.peak, arbitrary_float)

    def test_prior_attribute_after_fixing(self):
        arbitrary_float = 11.3
        fixed_prior = peyote.prior.fix(self.prior, arbitrary_float)
        self.assertIsInstance(fixed_prior, peyote.prior.DeltaFunction)

    def test_raising_value_error_if_value_is_none(self):
        self.assertRaises(ValueError, peyote.prior.fix, self.prior, np.nan)


if __name__ == '__main__':
    unittest.main()
