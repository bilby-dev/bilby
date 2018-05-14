from context import tupak
import unittest
import numpy as np


class TestPriorInstantiationWithoutOptionalPriors(unittest.TestCase):

    def setUp(self):
        self.prior = tupak.prior.Prior()

    def tearDown(self):
        del self.prior

    def test_name(self):
        self.assertIsNone(self.prior.name)

    def test_latex_label(self):
        self.assertIsNone(self.prior.latex_label)

    def test_is_fixed(self):
        self.assertFalse(self.prior.is_fixed)

    def test_class_instance(self):
        self.assertIsInstance(self.prior, tupak.prior.Prior)


class TestPriorName(unittest.TestCase):

    def setUp(self):
        self.test_name = 'test_name'
        self.prior = tupak.prior.Prior(self.test_name)

    def tearDown(self):
        del self.prior
        del self.test_name

    def test_name_assignment(self):
        self.prior.name = "other_name"
        self.assertEqual(self.prior.name, "other_name")


class TestPriorLatexLabel(unittest.TestCase):
    def setUp(self):
        self.test_name = 'test_name'
        self.prior = tupak.prior.Prior(self.test_name)

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
        self.prior = tupak.prior.Prior()
        self.assertFalse(self.prior.is_fixed)

    def test_is_fixed_delta_function_class(self):
        self.prior = tupak.prior.DeltaFunction(peak=0)
        self.assertTrue(self.prior.is_fixed)

    def test_is_fixed_uniform_class(self):
        self.prior = tupak.prior.Uniform(minimum=0, maximum=10)
        self.assertFalse(self.prior.is_fixed)


class TestFixMethod(unittest.TestCase):

    def setUp(self):
        self.test_name = 'test_name'
        self.prior = tupak.prior.Prior(self.test_name)

    def tearDown(self):
        del self.prior

    def test_is_fixed_attribute_after_fixing(self):
        arbitrary_float = 11.3
        fixed_prior = tupak.prior.fix(self.prior, arbitrary_float)
        self.assertTrue(fixed_prior.is_fixed)

    def test_value_attribute_after_fixing(self):
        arbitrary_float = 11.3
        fixed_prior = tupak.prior.fix(self.prior, arbitrary_float)
        self.assertEqual(fixed_prior.peak, arbitrary_float)

    def test_prior_attribute_after_fixing(self):
        arbitrary_float = 11.3
        fixed_prior = tupak.prior.fix(self.prior, arbitrary_float)
        self.assertIsInstance(fixed_prior, tupak.prior.DeltaFunction)

    def test_raising_value_error_if_value_is_none(self):
        self.assertRaises(ValueError, tupak.prior.fix, self.prior, np.nan)


class TestPriorClasses(unittest.TestCase):

    def setUp(self):

        self.priors = [
            tupak.prior.DeltaFunction(name='test', peak=1),
            tupak.prior.Gaussian(name='test', mu=0, sigma=1),
            tupak.prior.PowerLaw(name='test', alpha=0, minimum=0, maximum=1),
            tupak.prior.PowerLaw(name='test', alpha=-1, minimum=1, maximum=1e2),
            tupak.prior.Uniform(name='test', minimum=0, maximum=1),
            tupak.prior.UniformComovingVolume(name='test', minimum=2e2, maximum=5e3),
            tupak.prior.Sine(name='test'),
            tupak.prior.Cosine(name='test'),
            tupak.prior.Interped(name='test', xx=np.linspace(0, 10, 1000), yy=np.linspace(0, 10, 1000)**4,
                                 minimum=3, maximum=5),
            tupak.prior.TruncatedGaussian(name='test', mu=1, sigma=0.4, minimum=-1, maximum=1)
        ]

    def test_rescaling(self):
        for prior in self.priors:
            """Test the the rescaling works as expected."""
            minimum_sample = prior.rescale(0)
            self.assertAlmostEqual(minimum_sample, prior.minimum)
            maximum_sample = prior.rescale(1)
            self.assertAlmostEqual(maximum_sample, prior.maximum)
            many_samples = prior.rescale(np.random.uniform(0, 1, 1000))
            self.assertTrue(all((many_samples >= prior.minimum) & (many_samples <= prior.maximum)))
            self.assertRaises(ValueError, lambda: prior.rescale(-1))

    def test_sampling(self):
        """Test that sampling from the prior always returns values within its domain."""
        for prior in self.priors:
            single_sample = prior.sample()
            self.assertTrue((single_sample >= prior.minimum) & (single_sample <= prior.maximum))
            many_samples = prior.sample(1000)
            self.assertTrue(all((many_samples >= prior.minimum) & (many_samples <= prior.maximum)))

    def test_prob(self):
        """Test that the prior probability is non-negative in domain of validity and zero outside."""
        for prior in self.priors:
            # skip delta function prior in this case
            if isinstance(prior, tupak.prior.DeltaFunction):
                continue
            if prior.maximum != np.inf:
                outside_domain = np.linspace(prior.maximum + 1, prior.maximum + 1e4, 1000)
                self.assertTrue(all(prior.prob(outside_domain) == 0))
            if prior.minimum != -np.inf:
                outside_domain = np.linspace(prior.minimum - 1e4, prior.minimum - 1, 1000)
                self.assertTrue(all(prior.prob(outside_domain) == 0))
            if prior.minimum == -np.inf:
                prior.minimum = -1e5
            if prior.maximum == np.inf:
                prior.maximum = 1e5
            domain = np.linspace(prior.minimum, prior.maximum, 1000)
            self.assertTrue(all(prior.prob(domain) >= 0))
            surround_domain = np.linspace(prior.minimum - 1, prior.maximum + 1, 1000)
            prior.prob(surround_domain)


if __name__ == '__main__':
    unittest.main()
