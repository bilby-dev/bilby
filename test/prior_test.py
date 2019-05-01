from __future__ import absolute_import, division
import bilby
import unittest
from mock import Mock
import numpy as np
import os
from collections import OrderedDict


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
        self.prior = bilby.core.prior.Prior(name='test_name', latex_label='test_label', minimum=0, maximum=1,
                                            boundary=None)
        expected_string = "Prior(name='test_name', latex_label='test_label', unit=None, minimum=0, maximum=1, " \
                          "boundary=None)"
        self.assertEqual(expected_string, self.prior.__repr__())

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
        val_in_prior = (self.prior.minimum + self.prior.maximum) / 2.
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
        self.test_name = 'test_name'
        self.prior = bilby.core.prior.Prior(self.test_name)

    def tearDown(self):
        del self.prior
        del self.test_name

    def test_name_assignment(self):
        self.prior.name = "other_name"
        self.assertEqual(self.prior.name, "other_name")


class TestPriorLatexLabel(unittest.TestCase):
    def setUp(self):
        self.test_name = 'test_name'
        self.prior = bilby.core.prior.Prior(self.test_name)

    def tearDown(self):
        del self.test_name
        del self.prior

    def test_label_assignment(self):
        test_label = 'test_label'
        self.prior.latex_label = 'test_label'
        self.assertEqual(test_label, self.prior.latex_label)

    def test_default_label_assignment(self):
        self.prior.name = 'chirp_mass'
        self.prior.latex_label = None
        self.assertEqual(self.prior.latex_label, '$\mathcal{M}$')

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
        self.prior.boundary = 'periodic'
        self.assertEqual(self.prior.boundary, 'periodic')

    def test_set_boundary_invalid(self):
        with self.assertRaises(ValueError):
            self.prior.boundary = 'else'


class TestPriorClasses(unittest.TestCase):

    def setUp(self):

        # set multivariate Gaussian
        mvg = bilby.core.prior.MultivariateGaussianDist(names=['testa', 'testb'],
                                                        mus=[1, 1],
                                                        covs=np.array([[2., 0.5], [0.5, 2.]]),
                                                        weights=1.)
        mvn = bilby.core.prior.MultivariateGaussianDist(names=['testa', 'testb'],
                                                        mus=[1, 1],
                                                        covs=np.array([[2., 0.5], [0.5, 2.]]),
                                                        weights=1.)

        self.priors = [
            bilby.core.prior.DeltaFunction(name='test', unit='unit', peak=1),
            bilby.core.prior.Gaussian(name='test', unit='unit', mu=0, sigma=1),
            bilby.core.prior.Normal(name='test', unit='unit', mu=0, sigma=1),
            bilby.core.prior.PowerLaw(name='test', unit='unit', alpha=0, minimum=0, maximum=1),
            bilby.core.prior.PowerLaw(name='test', unit='unit', alpha=-1, minimum=0.5, maximum=1),
            bilby.core.prior.PowerLaw(name='test', unit='unit', alpha=2, minimum=1, maximum=1e2),
            bilby.core.prior.Uniform(name='test', unit='unit', minimum=0, maximum=1),
            bilby.core.prior.LogUniform(name='test', unit='unit', minimum=5e0, maximum=1e2),
            bilby.gw.prior.UniformComovingVolume(name='redshift', minimum=0.1, maximum=1.0),
            bilby.gw.prior.UniformSourceFrame(name='redshift', minimum=0.1, maximum=1.0),
            bilby.core.prior.Sine(name='test', unit='unit'),
            bilby.core.prior.Cosine(name='test', unit='unit'),
            bilby.core.prior.Interped(name='test', unit='unit', xx=np.linspace(0, 10, 1000),
                                      yy=np.linspace(0, 10, 1000) ** 4,
                                      minimum=3, maximum=5),
            bilby.core.prior.TruncatedGaussian(name='test', unit='unit', mu=1, sigma=0.4, minimum=-1, maximum=1),
            bilby.core.prior.TruncatedNormal(name='test', unit='unit', mu=1, sigma=0.4, minimum=-1, maximum=1),
            bilby.core.prior.HalfGaussian(name='test', unit='unit', sigma=1),
            bilby.core.prior.HalfNormal(name='test', unit='unit', sigma=1),
            bilby.core.prior.LogGaussian(name='test', unit='unit', mu=0, sigma=1),
            bilby.core.prior.LogNormal(name='test', unit='unit', mu=0, sigma=1),
            bilby.core.prior.Exponential(name='test', unit='unit', mu=1),
            bilby.core.prior.StudentT(name='test', unit='unit', df=3, mu=0, scale=1),
            bilby.core.prior.Beta(name='test', unit='unit', alpha=2.0, beta=2.0),
            bilby.core.prior.Logistic(name='test', unit='unit', mu=0, scale=1),
            bilby.core.prior.Cauchy(name='test', unit='unit', alpha=0, beta=1),
            bilby.core.prior.Lorentzian(name='test', unit='unit', alpha=0, beta=1),
            bilby.core.prior.Gamma(name='test', unit='unit', k=1, theta=1),
            bilby.core.prior.ChiSquared(name='test', unit='unit', nu=2),
            bilby.gw.prior.AlignedSpin(name='test', unit='unit'),
            bilby.core.prior.MultivariateGaussian(mvg=mvg, name='testa', unit='unit'),
            bilby.core.prior.MultivariateGaussian(mvg=mvg, name='testb', unit='unit'),
            bilby.core.prior.MultivariateNormal(mvg=mvn, name='testa', unit='unit'),
            bilby.core.prior.MultivariateNormal(mvg=mvn, name='testb', unit='unit')
        ]

    def tearDown(self):
        del self.priors

    def test_minimum_rescaling(self):
        """Test the the rescaling works as expected."""
        for prior in self.priors:
            if isinstance(prior, bilby.core.prior.MultivariateGaussian):
                minimum_sample = prior.rescale(0)
                if prior.mvg.filled_rescale():
                    self.assertAlmostEqual(minimum_sample[0], prior.minimum)
                    self.assertAlmostEqual(minimum_sample[1], prior.minimum)
            else:
                minimum_sample = prior.rescale(0)
                self.assertAlmostEqual(minimum_sample, prior.minimum)

    def test_maximum_rescaling(self):
        """Test the the rescaling works as expected."""
        for prior in self.priors:
            if isinstance(prior, bilby.core.prior.MultivariateGaussian):
                maximum_sample = prior.rescale(0)
                if prior.mvg.filled_rescale():
                    self.assertAlmostEqual(maximum_sample[0], prior.maximum)
                    self.assertAlmostEqual(maximum_sample[1], prior.maximum)
            else:
                maximum_sample = prior.rescale(1)
                self.assertAlmostEqual(maximum_sample, prior.maximum)

    def test_many_sample_rescaling(self):
        """Test the the rescaling works as expected."""
        for prior in self.priors:
            many_samples = prior.rescale(np.random.uniform(0, 1, 1000))
            if isinstance(prior, bilby.core.prior.MultivariateGaussian):
                if not prior.mvg.filled_rescale():
                    continue
            self.assertTrue(all((many_samples >= prior.minimum) & (many_samples <= prior.maximum)))

    def test_out_of_bounds_rescaling(self):
        """Test the the rescaling works as expected."""
        for prior in self.priors:
            self.assertRaises(ValueError, lambda: prior.rescale(-1))

    def test_sampling_single(self):
        """Test that sampling from the prior always returns values within its domain."""
        for prior in self.priors:
            single_sample = prior.sample()
            self.assertTrue((single_sample >= prior.minimum) & (single_sample <= prior.maximum))

    def test_sampling_many(self):
        """Test that sampling from the prior always returns values within its domain."""
        for prior in self.priors:
            many_samples = prior.sample(1000)
            self.assertTrue(all((many_samples >= prior.minimum) & (many_samples <= prior.maximum)))

    def test_probability_above_domain(self):
        """Test that the prior probability is non-negative in domain of validity and zero outside."""
        for prior in self.priors:
            if prior.maximum != np.inf:
                outside_domain = np.linspace(prior.maximum + 1, prior.maximum + 1e4, 1000)
                self.assertTrue(all(prior.prob(outside_domain) == 0))

    def test_probability_below_domain(self):
        """Test that the prior probability is non-negative in domain of validity and zero outside."""
        for prior in self.priors:
            if prior.minimum != -np.inf:
                outside_domain = np.linspace(prior.minimum - 1e4, prior.minimum - 1, 1000)
                self.assertTrue(all(prior.prob(outside_domain) == 0))

    def test_prob_and_ln_prob(self):
        for prior in self.priors:
            sample = prior.sample()
            if not isinstance(prior, bilby.core.prior.MultivariateGaussian):
                # due to the way that the Multivariate Gaussian prior must sequentially call
                # the prob and ln_prob functions, it must be ignored in this test.
                self.assertAlmostEqual(np.log(prior.prob(sample)), prior.ln_prob(sample), 12)

    def test_log_normal_fail(self):
        with self.assertRaises(ValueError):
            bilby.core.prior.LogNormal(name='test', unit='unit', mu=0, sigma=-1)

    def test_studentt_fail(self):
        with self.assertRaises(ValueError):
            bilby.core.prior.StudentT(name='test', unit='unit', df=3, mu=0, scale=-1)
        with self.assertRaises(ValueError):
            bilby.core.prior.StudentT(name='test', unit='unit', df=0, mu=0, scale=1)

    def test_beta_fail(self):
        with self.assertRaises(ValueError):
            bilby.core.prior.Beta(name='test', unit='unit', alpha=-2.0, beta=2.0),

        with self.assertRaises(ValueError):
            bilby.core.prior.Beta(name='test', unit='unit', alpha=2.0, beta=-2.0),

    def test_multivariate_gaussian_fail(self):
        with self.assertRaises(ValueError):
            # bounds is wrong length
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'],
                                                      bounds=[(-1., 1.)])
        with self.assertRaises(ValueError):
            # bounds has lower value greater than upper
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'],
                                                      bounds=[(-1., 1.), (1., -1)])
        with self.assertRaises(TypeError):
            # bound is not a list/tuple
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'],
                                                      bounds=[(-1., 1.), 2])
        with self.assertRaises(ValueError):
            # bound contains too many values
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'],
                                                      bounds=[(-1., 1., 4), 2])
        with self.assertRaises(ValueError):
            # means is not a list
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'], mus=1.)
        with self.assertRaises(ValueError):
            # sigmas is not a list
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'], sigmas=1.)
        with self.assertRaises(TypeError):
            # covariances is not a list
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'], covs=1.)
        with self.assertRaises(TypeError):
            # correlation coefficients is not a list
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'], corrcoefs=1.)
        with self.assertRaises(ValueError):
            # wrong number of weights
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'], weights=[0.5, 0.5])
        with self.assertRaises(ValueError):
            # not enough modes set
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'], mus=[[1., 2.]],
                                                      nmodes=2)
        with self.assertRaises(ValueError):
            # covariance is the wrong shape
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'],
                                                      covs=np.array([[[1., 1.],
                                                                      [1., 1.]]]))
        with self.assertRaises(ValueError):
            # covariance is the wrong shape
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'],
                                                      covs=np.array([[[1., 1.]]]))
        with self.assertRaises(ValueError):
            # correlation coefficient matrix is the wrong shape
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'], sigmas=[1., 1.],
                                                      corrcoefs=np.array([[[[1., 1.],
                                                                            [1., 1.]]]]))
        with self.assertRaises(ValueError):
            # correlation coefficient matrix is the wrong shape
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'], sigmas=[1., 1.],
                                                      corrcoefs=np.array([[[1., 1.]]]))
        with self.assertRaises(ValueError):
            # correlation coefficient has non-unity diagonal value
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'], sigmas=[1., 1.],
                                                      corrcoefs=np.array([[1., 1.],
                                                                          [1., 2.]]))
        with self.assertRaises(ValueError):
            # correlation coefficient matrix is not symmetric
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'], sigmas=[1., 2.],
                                                      corrcoefs=np.array([[1., -1.2],
                                                                          [-0.3, 1.]]))
        with self.assertRaises(ValueError):
            # correlation coefficient matrix is not positive definite
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'], sigmas=[1., 2.],
                                                      corrcoefs=np.array([[1., -1.3],
                                                                          [-1.3, 1.]]))
        with self.assertRaises(ValueError):
            # wrong number of sigmas
            bilby.core.prior.MultivariateGaussianDist(['a', 'b'], sigmas=[1., 2., 3.],
                                                      corrcoefs=np.array([[1., 0.3],
                                                                          [0.3, 1.]]))

    def test_multivariate_gaussian_covariance(self):
        """Test that the correlation coefficient/covariance matrices are correct"""
        cov = np.array([[4., 0], [0., 9.]])
        mvg = bilby.core.prior.MultivariateGaussianDist(['a', 'b'], covs=cov)
        self.assertEqual(mvg.nmodes, 1)
        self.assertTrue(np.allclose(mvg.covs[0], cov))
        self.assertTrue(np.allclose(mvg.sigmas[0], np.sqrt(np.diag(cov))))
        self.assertTrue(np.allclose(mvg.corrcoefs[0], np.eye(2)))

        corrcoef = np.array([[1., 0.5], [0.5, 1.]])
        sigma = [2., 2.]
        mvg = bilby.core.prior.MultivariateGaussianDist(['a', 'b'],
                                                        corrcoefs=corrcoef,
                                                        sigmas=sigma)
        self.assertTrue(np.allclose(mvg.corrcoefs[0], corrcoef))
        self.assertTrue(np.allclose(mvg.sigmas[0], sigma))
        self.assertTrue(np.allclose(np.diag(mvg.covs[0]), np.square(sigma)))
        self.assertTrue(np.allclose(np.diag(np.fliplr(mvg.covs[0])), 2.*np.ones(2)))

    def test_fermidirac_fail(self):
        with self.assertRaises(ValueError):
            bilby.core.prior.FermiDirac(name='test', unit='unit', sigma=1.)

        with self.assertRaises(ValueError):
            bilby.core.prior.FermiDirac(name='test', unit='unit', sigma=1., mu=-1)

    def test_probability_in_domain(self):
        """Test that the prior probability is non-negative in domain of validity and zero outside."""
        for prior in self.priors:
            if prior.minimum == -np.inf:
                prior.minimum = -1e5
            if prior.maximum == np.inf:
                prior.maximum = 1e5
            domain = np.linspace(prior.minimum, prior.maximum, 1000)
            self.assertTrue(all(prior.prob(domain) >= 0))

    def test_probability_surrounding_domain(self):
        """Test that the prior probability is non-negative in domain of validity and zero outside."""
        for prior in self.priors:
            # skip delta function prior in this case
            if isinstance(prior, bilby.core.prior.DeltaFunction):
                continue
            surround_domain = np.linspace(prior.minimum - 1, prior.maximum + 1, 1000)
            prior.prob(surround_domain)

    def test_normalized(self):
        """Test that each of the priors are normalised, this needs care for delta function and Gaussian priors"""
        for prior in self.priors:
            if isinstance(prior, bilby.core.prior.DeltaFunction):
                continue
            if isinstance(prior, bilby.core.prior.Cauchy):
                continue
            if isinstance(prior, bilby.core.prior.MultivariateGaussian):
                continue
            elif isinstance(prior, bilby.core.prior.Gaussian):
                domain = np.linspace(-1e2, 1e2, 1000)
            elif isinstance(prior, bilby.core.prior.Cauchy):
                domain = np.linspace(-1e2, 1e2, 1000)
            elif isinstance(prior, bilby.core.prior.StudentT):
                domain = np.linspace(-1e2, 1e2, 1000)
            elif isinstance(prior, bilby.core.prior.HalfGaussian):
                domain = np.linspace(0., 1e2, 1000)
            elif isinstance(prior, bilby.core.prior.Gamma):
                domain = np.linspace(0., 1e2, 5000)
            elif isinstance(prior, bilby.core.prior.LogNormal):
                domain = np.linspace(0., 1e2, 1000)
            elif isinstance(prior, bilby.core.prior.Exponential):
                domain = np.linspace(0., 1e2, 5000)
            elif isinstance(prior, bilby.core.prior.Logistic):
                domain = np.linspace(-1e2, 1e2, 1000)
            elif isinstance(prior, bilby.core.prior.FermiDirac):
                domain = np.linspace(0., 1e2, 1000)
            else:
                domain = np.linspace(prior.minimum, prior.maximum, 1000)
            self.assertAlmostEqual(np.trapz(prior.prob(domain), domain), 1, 3)

    def test_unit_setting(self):
        for prior in self.priors:
            if isinstance(prior, bilby.gw.prior.Cosmological):
                self.assertEqual(None, prior.unit)
            else:
                self.assertEqual('unit', prior.unit)

    def test_eq_different_classes(self):
        for i in range(len(self.priors)):
            for j in range(len(self.priors)):
                if i == j:
                    self.assertEqual(self.priors[i], self.priors[j])
                else:
                    self.assertNotEqual(self.priors[i], self.priors[j])

    def test_eq_other_condition(self):
        prior_1 = bilby.core.prior.PowerLaw(name='test', unit='unit', alpha=0, minimum=0, maximum=1)
        prior_2 = bilby.core.prior.PowerLaw(name='test', unit='unit', alpha=0, minimum=0, maximum=1.5)
        self.assertNotEqual(prior_1, prior_2)

    def test_eq_different_keys(self):
        prior_1 = bilby.core.prior.PowerLaw(name='test', unit='unit', alpha=0, minimum=0, maximum=1)
        prior_2 = bilby.core.prior.PowerLaw(name='test', unit='unit', alpha=0, minimum=0, maximum=1)
        prior_2.other_key = 5
        self.assertNotEqual(prior_1, prior_2)

    def test_np_array_eq(self):
        prior_1 = bilby.core.prior.PowerLaw(name='test', unit='unit', alpha=0, minimum=0, maximum=1)
        prior_2 = bilby.core.prior.PowerLaw(name='test', unit='unit', alpha=0, minimum=0, maximum=1)
        prior_1.array_attribute = np.array([1, 2, 3])
        prior_2.array_attribute = np.array([2, 2, 3])
        self.assertNotEqual(prior_1, prior_2)

    def test_repr(self):
        for prior in self.priors:
            if isinstance(prior, bilby.core.prior.Interped):
                continue  # we cannot test this because of the numpy arrays
            elif isinstance(prior, bilby.core.prior.MultivariateGaussian):
                continue  # we cannot test this because of the internal objects
            elif isinstance(prior, bilby.gw.prior.UniformComovingVolume):
                repr_prior_string = 'bilby.gw.prior.' + repr(prior)
            else:
                repr_prior_string = 'bilby.core.prior.' + repr(prior)
            repr_prior = eval(repr_prior_string)
            self.assertEqual(prior, repr_prior)

    def test_set_maximum_setting(self):
        for prior in self.priors:
            if isinstance(prior, (
                    bilby.core.prior.DeltaFunction, bilby.core.prior.Gaussian,
                    bilby.core.prior.HalfGaussian, bilby.core.prior.LogNormal,
                    bilby.core.prior.Exponential, bilby.core.prior.StudentT,
                    bilby.core.prior.Logistic, bilby.core.prior.Cauchy,
                    bilby.core.prior.Gamma, bilby.core.prior.MultivariateGaussian,
                    bilby.core.prior.FermiDirac)):
                continue
            prior.maximum = (prior.maximum + prior.minimum) / 2
            self.assertTrue(max(prior.sample(10000)) < prior.maximum)

    def test_set_minimum_setting(self):
        for prior in self.priors:
            if isinstance(prior, (
                    bilby.core.prior.DeltaFunction, bilby.core.prior.Gaussian,
                    bilby.core.prior.HalfGaussian, bilby.core.prior.LogNormal,
                    bilby.core.prior.Exponential, bilby.core.prior.StudentT,
                    bilby.core.prior.Logistic, bilby.core.prior.Cauchy,
                    bilby.core.prior.Gamma, bilby.core.prior.MultivariateGaussian,
                    bilby.core.prior.FermiDirac)):
                continue
            prior.minimum = (prior.maximum + prior.minimum) / 2
            self.assertTrue(min(prior.sample(10000)) > prior.minimum)


class TestPriorDict(unittest.TestCase):

    def setUp(self):
        self.first_prior = bilby.core.prior.Uniform(name='a', minimum=0, maximum=1, unit='kg', boundary=None)
        self.second_prior = bilby.core.prior.PowerLaw(name='b', alpha=3, minimum=1, maximum=2, unit='m/s',
                                                      boundary=None)
        self.third_prior = bilby.core.prior.DeltaFunction(name='c', peak=42, unit='m')
        self.priors = dict(mass=self.first_prior,
                           speed=self.second_prior,
                           length=self.third_prior)
        self.prior_set_from_dict = bilby.core.prior.PriorDict(dictionary=self.priors)
        self.default_prior_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               'prior_files/binary_black_holes.prior')
        self.prior_set_from_file = bilby.core.prior.PriorDict(filename=self.default_prior_file)

    def tearDown(self):
        del self.first_prior
        del self.second_prior
        del self.third_prior
        del self.priors
        del self.prior_set_from_dict
        del self.default_prior_file
        del self.prior_set_from_file

    def test_copy(self):
        priors = bilby.core.prior.PriorDict(self.priors)
        self.assertEqual(priors, priors.copy())

    def test_prior_set(self):
        priors_dict = bilby.core.prior.PriorDict(self.priors)
        priors_set = bilby.core.prior.PriorSet(self.priors)
        self.assertEqual(priors_dict, priors_set)

    def test_prior_set_is_ordered_dict(self):
        self.assertIsInstance(self.prior_set_from_dict, OrderedDict)

    def test_prior_set_has_correct_length(self):
        self.assertEqual(3, len(self.prior_set_from_dict))

    def test_prior_set_has_expected_priors(self):
        self.assertDictEqual(self.priors, dict(self.prior_set_from_dict))

    def test_read_from_file(self):
        expected = dict(
            mass_1=bilby.core.prior.Uniform(
                name='mass_1', minimum=5, maximum=100, unit='$M_{\\odot}$', boundary=None),
            mass_2=bilby.core.prior.Uniform(
                name='mass_2', minimum=5, maximum=100, unit='$M_{\\odot}$', boundary=None),
            mass_ratio=bilby.core.prior.Constraint(name='mass_ratio', minimum=0.125, maximum=1, latex_label='$q$',
                                                   unit=None),
            a_1=bilby.core.prior.Uniform(name='a_1', minimum=0, maximum=0.8, boundary='reflective'),
            a_2=bilby.core.prior.Uniform(name='a_2', minimum=0, maximum=0.8, boundary='reflective'),
            tilt_1=bilby.core.prior.Sine(name='tilt_1', boundary='reflective'),
            tilt_2=bilby.core.prior.Sine(name='tilt_2', boundary='reflective'),
            phi_12=bilby.core.prior.Uniform(
                name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic'),
            phi_jl=bilby.core.prior.Uniform(
                name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic'),
            luminosity_distance=bilby.gw.prior.UniformSourceFrame(
                name='luminosity_distance', minimum=1e2,
                maximum=5e3, unit='Mpc', boundary=None),
            dec=bilby.core.prior.Cosine(name='dec', boundary='reflective'),
            ra=bilby.core.prior.Uniform(
                name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic'),
            theta_jn=bilby.core.prior.Sine(name='theta_jn', boundary='reflective'),
            psi=bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic'),
            phase=bilby.core.prior.Uniform(
                name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
            )
        self.assertDictEqual(expected, self.prior_set_from_file)

    def test_to_file(self):
        expected = ["length = DeltaFunction(peak=42, name='c', latex_label='c', unit='m')\n",
                    "speed = PowerLaw(alpha=3, minimum=1, maximum=2, name='b', latex_label='b', "
                    "unit='m/s', boundary=None)\n",
                    "mass = Uniform(minimum=0, maximum=1, name='a', latex_label='a', "
                    "unit='kg', boundary=None)\n"]
        self.prior_set_from_dict.to_file(outdir='prior_files', label='to_file_test')
        with open('prior_files/to_file_test.prior') as f:
            for i, line in enumerate(f.readlines()):
                self.assertTrue(line in expected)

    def test_from_dict_with_string(self):
        string_prior = "bilby.core.prior.PowerLaw(name='b', alpha=3, minimum=1, maximum=2, unit='m/s', " \
                       "boundary=None)"
        self.priors['speed'] = string_prior
        from_dict = bilby.core.prior.PriorDict(dictionary=self.priors)
        self.assertDictEqual(self.prior_set_from_dict, from_dict)

    def test_convert_floats_to_delta_functions(self):
        self.prior_set_from_dict['d'] = 5
        self.prior_set_from_dict['e'] = 7.3
        self.prior_set_from_dict['f'] = 'unconvertable'
        self.prior_set_from_dict.convert_floats_to_delta_functions()
        expected = dict(mass=bilby.core.prior.Uniform(name='a', minimum=0, maximum=1, unit='kg',
                                                      boundary=None),
                        speed=bilby.core.prior.PowerLaw(name='b', alpha=3, minimum=1, maximum=2, unit='m/s',
                                                        boundary=None),
                        length=bilby.core.prior.DeltaFunction(name='c', peak=42, unit='m'),
                        d=bilby.core.prior.DeltaFunction(peak=5),
                        e=bilby.core.prior.DeltaFunction(peak=7.3),
                        f='unconvertable')
        self.assertDictEqual(expected, self.prior_set_from_dict)

    def test_prior_set_from_dict_but_using_a_string(self):
        prior_set = bilby.core.prior.PriorDict(dictionary=self.default_prior_file)
        expected = bilby.core.prior.PriorDict(
            dict(
                mass_1=bilby.core.prior.Uniform(
                    name='mass_1', minimum=5, maximum=100, unit='$M_{\\odot}$', boundary=None),
                mass_2=bilby.core.prior.Uniform(
                    name='mass_2', minimum=5, maximum=100, unit='$M_{\\odot}$', boundary=None),
                mass_ratio=bilby.core.prior.Constraint(name='mass_ratio', minimum=0.125, maximum=1, latex_label='$q$',
                                                       unit=None),
                a_1=bilby.core.prior.Uniform(name='a_1', minimum=0, maximum=0.8, boundary='reflective'),
                a_2=bilby.core.prior.Uniform(name='a_2', minimum=0, maximum=0.8, boundary='reflective'),
                tilt_1=bilby.core.prior.Sine(name='tilt_1', boundary='reflective'),
                tilt_2=bilby.core.prior.Sine(name='tilt_2', boundary='reflective'),
                phi_12=bilby.core.prior.Uniform(
                    name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic'),
                phi_jl=bilby.core.prior.Uniform(
                    name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic'),
                luminosity_distance=bilby.gw.prior.UniformSourceFrame(
                    name='luminosity_distance', minimum=1e2,
                    maximum=5e3, unit='Mpc', boundary=None),
                dec=bilby.core.prior.Cosine(name='dec', boundary='reflective'),
                ra=bilby.core.prior.Uniform(
                    name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic'),
                theta_jn=bilby.core.prior.Sine(name='theta_jn', boundary='reflective'),
                psi=bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic'),
                phase=bilby.core.prior.Uniform(
                    name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
            )
        )
        all_keys = set(prior_set.keys()).union(set(expected.keys()))
        for key in all_keys:
            self.assertEqual(expected[key], prior_set[key])

    def test_dict_argument_is_not_string_or_dict(self):
        with self.assertRaises(ValueError):
            bilby.core.prior.PriorDict(dictionary=list())

    def test_sample_subset_correct_size(self):
        size = 7
        samples = self.prior_set_from_dict.sample_subset(keys=self.prior_set_from_dict.keys(), size=size)
        self.assertEqual(len(self.prior_set_from_dict), len(samples))
        for key in samples:
            self.assertEqual(size, len(samples[key]))

    def test_sample_subset_correct_size_when_non_priors_in_dict(self):
        self.prior_set_from_dict['asdf'] = 'not_a_prior'
        samples = self.prior_set_from_dict.sample_subset(keys=self.prior_set_from_dict.keys())
        self.assertEqual(len(self.prior_set_from_dict) - 1, len(samples))

    def test_sample_subset_with_actual_subset(self):
        size = 3
        samples = self.prior_set_from_dict.sample_subset(keys=['length'], size=size)
        expected = dict(length=np.array([42., 42., 42.]))
        self.assertTrue(np.array_equal(expected['length'], samples['length']))

    def test_sample(self):
        size = 7
        np.random.seed(42)
        samples1 = self.prior_set_from_dict.sample_subset(keys=self.prior_set_from_dict.keys(), size=size)
        np.random.seed(42)
        samples2 = self.prior_set_from_dict.sample(size=size)
        self.assertEqual(set(samples1.keys()), set(samples2.keys()))
        for key in samples1:
            self.assertTrue(np.array_equal(samples1[key], samples2[key]))

    def test_prob(self):
        samples = self.prior_set_from_dict.sample_subset(keys=['mass', 'speed'])
        expected = self.first_prior.prob(samples['mass']) * self.second_prior.prob(samples['speed'])
        self.assertEqual(expected, self.prior_set_from_dict.prob(samples))

    def test_ln_prob(self):
        samples = self.prior_set_from_dict.sample_subset(keys=['mass', 'speed'])
        expected = self.first_prior.ln_prob(samples['mass']) + self.second_prior.ln_prob(samples['speed'])
        self.assertEqual(expected, self.prior_set_from_dict.ln_prob(samples))

    def test_rescale(self):
        theta = [0.5, 0.5, 0.5]
        expected = [self.first_prior.rescale(0.5),
                    self.second_prior.rescale(0.5),
                    self.third_prior.rescale(0.5)]
        self.assertListEqual(sorted(expected), sorted(self.prior_set_from_dict.rescale(
            keys=self.prior_set_from_dict.keys(), theta=theta)))

    def test_redundancy(self):
        for key in self.prior_set_from_dict.keys():
            self.assertFalse(self.prior_set_from_dict.test_redundancy(key=key))


class TestFillPrior(unittest.TestCase):

    def setUp(self):
        self.likelihood = Mock()
        self.likelihood.parameters = dict(a=0, b=0, c=0, d=0, asdf=0, ra=1)
        self.likelihood.non_standard_sampling_parameter_keys = dict(t=8)
        self.priors = dict(a=1, b=1.1, c='string', d=bilby.core.prior.Uniform(0, 1))
        self.priors = bilby.core.prior.PriorDict(dictionary=self.priors)
        self.default_prior_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               'prior_files/binary_black_holes.prior')
        self.priors.fill_priors(self.likelihood, self.default_prior_file)

    def tearDown(self):
        del self.likelihood
        del self.priors

    def test_prior_instances_are_not_changed_by_parsing(self):
        self.assertIsInstance(self.priors['d'], bilby.core.prior.Uniform)

    def test_parsing_ints_to_delta_priors_class(self):
        self.assertIsInstance(self.priors['a'], bilby.core.prior.DeltaFunction)

    def test_parsing_ints_to_delta_priors_with_right_value(self):
        self.assertEqual(self.priors['a'].peak, 1)

    def test_parsing_floats_to_delta_priors_class(self):
        self.assertIsInstance(self.priors['b'], bilby.core.prior.DeltaFunction)

    def test_parsing_floats_to_delta_priors_with_right_value(self):
        self.assertAlmostEqual(self.priors['b'].peak, 1.1, 1e-8)

    def test_without_available_default_priors_no_prior_is_set(self):
        with self.assertRaises(KeyError):
            print(self.priors['asdf'])

    def test_with_available_default_priors_a_default_prior_is_set(self):
        self.assertIsInstance(self.priors['ra'], bilby.core.prior.Uniform)


class TestCreateDefaultPrior(unittest.TestCase):

    def test_none_behaviour(self):
        self.assertIsNone(bilby.core.prior.create_default_prior(name='name', default_priors_file=None))

    def test_bbh_params(self):
        prior_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'prior_files/binary_black_holes.prior')
        prior_set = bilby.core.prior.PriorDict(filename=prior_file)
        for prior in prior_set:
            self.assertEqual(prior_set[prior], bilby.core.prior.create_default_prior(name=prior,
                                                                                     default_priors_file=prior_file))

    def test_unknown_prior(self):
        prior_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'prior_files/binary_black_holes.prior')
        self.assertIsNone(bilby.core.prior.create_default_prior(name='name', default_priors_file=prior_file))


if __name__ == '__main__':
    unittest.main()
