import sys
sys.path.append("../")
from peyote import parameter as parameter
from peyote import prior as prior
import unittest
import numpy as np


class TestParameterInstantiationWithoutOptionalParameters(unittest.TestCase):

    def setUp(self):
        self.test_name = 'test_name'
        self.parameter = parameter.Parameter(self.test_name)

    def tearDown(self):
        del self.parameter

    def test_name(self):
        self.assertEqual(self.parameter.name, self.test_name)

    def test_prior(self):
        self.assertIsNone(self.parameter.prior)

    def test_value(self):
        self.assertTrue(np.isnan(self.parameter.value))

    def test_latex_label(self):
        self.assertEqual(self.parameter.latex_label, self.test_name)

    def test_is_fixed(self):
        self.assertFalse(self.parameter.is_fixed)


class TestParameterName(unittest.TestCase):

    def setUp(self):
        self.test_name = 'test_name'
        self.parameter = parameter.Parameter(self.test_name)

    def tearDown(self):
        del self.parameter

    def test_name_assignment(self):
        self.parameter.name = "other_name"
        self.assertEqual(self.parameter.name, "other_name")


class TestParameterPrior(unittest.TestCase):

    def setUp(self):
        self.test_name = 'test_name'
        self.parameter = parameter.Parameter(self.test_name)

    def tearDown(self):
        del self.parameter

    def test_prior_assignment(self):
        test_prior = prior.Uniform(0, 100)
        self.parameter.prior = test_prior
        self.assertDictEqual(test_prior.__dict__, self.parameter.prior.__dict__)

    def test_default_assignment(self):
        test_prior = prior.PowerLaw(alpha=0, bounds=(5, 100))
        self.parameter.name = 'mchirp'
        self.parameter.prior = None
        self.assertDictEqual(test_prior.__dict__, self.parameter.prior.__dict__)


class TestParameterValue(unittest.TestCase):
    def setUp(self):
        self.test_name = 'test_name'
        self.parameter = parameter.Parameter(self.test_name)

    def tearDown(self):
        del self.parameter

    def test_prior_assignment(self):
        test_value = 15
        self.parameter.value = test_value
        self.assertEqual(test_value, self.parameter.value)

    def test_default_value_assignment(self):
        self.parameter.name = 'a1'
        self.parameter.value = None
        self.assertEqual(self.parameter.value, 0)

    def test_default_value_assignment_default(self):
        self.parameter.value = None
        self.assertTrue(np.isnan(self.parameter.value))


class TestParameterLatexLabel(unittest.TestCase):
    def setUp(self):
        self.test_name = 'test_name'
        self.parameter = parameter.Parameter(self.test_name)

    def tearDown(self):
        del self.parameter

    def test_label_assignment(self):
        test_label = 'test_label'
        self.parameter.latex_label = 'test_label'
        self.assertEqual(test_label, self.parameter.latex_label)

    def test_default_label_assignment(self):
        self.parameter.name = 'mchirp'
        self.parameter.latex_label = None
        self.assertEqual(self.parameter.latex_label, '$\mathcal{M}$')

    def test_default_label_assignment_default(self):
        self.assertTrue(self.parameter.latex_label, self.parameter.name)


class TestParameterIsFixed(unittest.TestCase):
    def setUp(self):
        self.test_name = 'test_name'
        self.parameter = parameter.Parameter(self.test_name)

    def tearDown(self):
        del self.parameter

    def test_is_fixed_assignment(self):
        self.parameter.is_fixed = True
        self.assertTrue(self.parameter.is_fixed)

    def test_default_is_fixed_assignment(self):
        self.assertFalse(self.parameter.is_fixed)


class TestFixMethod(unittest.TestCase):

    def setUp(self):
        self.test_name = 'test_name'
        self.parameter = parameter.Parameter(self.test_name)

    def tearDown(self):
        del self.parameter

    def test_is_fixed_attribute_after_fixing(self):
        arbitrary_float = 11.3
        self.parameter.fix(arbitrary_float)
        self.assertTrue(self.parameter.is_fixed)

    def test_value_attribute_after_fixing(self):
        arbitrary_float = 11.3
        self.parameter.fix(arbitrary_float)
        self.assertEqual(self.parameter.value, arbitrary_float)

    def test_prior_attribute_after_fixing(self):
        arbitrary_float = 11.3
        self.parameter.fix(arbitrary_float)
        self.assertIsNone(self.parameter.prior)

    def test_raising_value_error_if_value_is_none(self):
        self.parameter.value = np.nan
        self.assertRaises(ValueError, self.parameter.fix)

    def test_fixing_existing_value(self):
        arbitrary_float = 11.3
        self.parameter.value = arbitrary_float
        self.parameter.fix()
        self.assertEqual(self.parameter.value, arbitrary_float)

    def test_is_fixed_after_fixing_existing_value(self):
        arbitrary_float = 11.3
        self.parameter.value = arbitrary_float
        self.parameter.fix()
        self.assertTrue(self.parameter.is_fixed)

    def test_prior_after_fixing_existing_value(self):
        arbitrary_float = 11.3
        self.parameter.value = arbitrary_float
        self.parameter.fix()
        self.assertIsNone(self.parameter.prior)
