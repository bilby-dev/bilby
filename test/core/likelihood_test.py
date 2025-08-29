import unittest
from unittest import mock

import numpy as np

import bilby.core.likelihood
from bilby.core.likelihood import (
    Likelihood,
    GaussianLikelihood,
    PoissonLikelihood,
    StudentTLikelihood,
    Analytical1DLikelihood,
    ExponentialLikelihood,
    AnalyticalMultidimensionalCovariantGaussian,
    AnalyticalMultidimensionalBimodalCovariantGaussian,
    JointLikelihood,
)


class TestLikelihoodBase(unittest.TestCase):
    def setUp(self):
        self.likelihood = Likelihood()

    def tearDown(self):
        del self.likelihood

    def test_repr(self):
        self.likelihood = Likelihood(parameters=["a", "b"])
        expected = "Likelihood"
        self.assertEqual(expected, repr(self.likelihood))

    def test_base_log_likelihood(self):
        self.assertTrue(np.isnan(self.likelihood.log_likelihood()))

    def test_base_noise_log_likelihood(self):
        self.assertTrue(np.isnan(self.likelihood.noise_log_likelihood()))

    def test_base_log_likelihood_ratio(self):
        self.assertTrue(np.isnan(self.likelihood.log_likelihood_ratio()))

    def test_meta_data_unset(self):
        self.assertEqual(self.likelihood.meta_data, None)

    def test_meta_data_set_fail(self):
        with self.assertRaises(ValueError):
            self.likelihood.meta_data = 10

    def test_meta_data(self):
        meta_data = dict(x=1, y=2)
        self.likelihood.meta_data = meta_data
        self.assertEqual(self.likelihood.meta_data, meta_data)


class TestAnalytical1DLikelihood(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(start=0, stop=100, step=1)
        self.y = np.arange(start=0, stop=100, step=1)

        def test_func(x, parameter1, parameter2):
            return parameter1 * x + parameter2

        self.func = test_func
        self.parameter1_value = 4
        self.parameter2_value = 7
        self.analytical_1d_likelihood = Analytical1DLikelihood(
            x=self.x, y=self.y, func=self.func
        )
        self.analytical_1d_likelihood.parameters["parameter1"] = self.parameter1_value
        self.analytical_1d_likelihood.parameters["parameter2"] = self.parameter2_value

    def tearDown(self):
        del self.x
        del self.y
        del self.func
        del self.analytical_1d_likelihood
        del self.parameter1_value
        del self.parameter2_value

    def test_init_x(self):
        self.assertTrue(np.array_equal(self.x, self.analytical_1d_likelihood.x))

    def test_set_x_to_array(self):
        new_x = np.arange(start=0, stop=50, step=2)
        self.analytical_1d_likelihood.x = new_x
        self.assertTrue(np.array_equal(new_x, self.analytical_1d_likelihood.x))

    def test_set_x_to_int(self):
        new_x = 5
        self.analytical_1d_likelihood.x = new_x
        expected_x = np.array([new_x])
        self.assertTrue(np.array_equal(expected_x, self.analytical_1d_likelihood.x))

    def test_set_x_to_float(self):
        new_x = 5.3
        self.analytical_1d_likelihood.x = new_x
        expected_x = np.array([new_x])
        self.assertTrue(np.array_equal(expected_x, self.analytical_1d_likelihood.x))

    def test_init_y(self):
        self.assertTrue(np.array_equal(self.y, self.analytical_1d_likelihood.y))

    def test_set_y_to_array(self):
        new_y = np.arange(start=0, stop=50, step=2)
        self.analytical_1d_likelihood.y = new_y
        self.assertTrue(np.array_equal(new_y, self.analytical_1d_likelihood.y))

    def test_set_y_to_int(self):
        new_y = 5
        self.analytical_1d_likelihood.y = new_y
        expected_y = np.array([new_y])
        self.assertTrue(np.array_equal(expected_y, self.analytical_1d_likelihood.y))

    def test_set_y_to_float(self):
        new_y = 5.3
        self.analytical_1d_likelihood.y = new_y
        expected_y = np.array([new_y])
        self.assertTrue(np.array_equal(expected_y, self.analytical_1d_likelihood.y))

    def test_init_func(self):
        self.assertEqual(self.func, self.analytical_1d_likelihood.func)

    def test_set_func(self):
        def new_func(x):
            return x

        with self.assertRaises(AttributeError):
            # noinspection PyPropertyAccess
            self.analytical_1d_likelihood.func = new_func

    def test_parameters(self):
        expected_parameters = dict(
            parameter1=self.parameter1_value, parameter2=self.parameter2_value
        )
        self.assertDictEqual(
            expected_parameters, self.analytical_1d_likelihood.parameters
        )

    def test_n(self):
        self.assertEqual(len(self.x), self.analytical_1d_likelihood.n)

    def test_set_n(self):
        with self.assertRaises(AttributeError):
            # noinspection PyPropertyAccess
            self.analytical_1d_likelihood.n = 2

    def test_model_parameters(self):
        sigma = 5
        self.analytical_1d_likelihood.sigma = sigma
        self.analytical_1d_likelihood.parameters["sigma"] = sigma
        expected_model_parameters = dict(
            parameter1=self.parameter1_value, parameter2=self.parameter2_value
        )
        self.assertDictEqual(
            expected_model_parameters, self.analytical_1d_likelihood.model_parameters()
        )

    def test_repr(self):
        expected = "Analytical1DLikelihood(x={}, y={}, func={})".format(
            self.x, self.y, self.func.__name__
        )
        self.assertEqual(expected, repr(self.analytical_1d_likelihood))


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
        likelihood = GaussianLikelihood(self.x, self.y, self.function, self.sigma)
        likelihood.parameters["m"] = 2
        likelihood.parameters["c"] = 0
        likelihood.log_likelihood()
        self.assertEqual(likelihood.sigma, self.sigma)

    def test_known_array_sigma(self):
        sigma_array = np.ones(self.N) * self.sigma
        likelihood = GaussianLikelihood(self.x, self.y, self.function, sigma_array)
        likelihood.parameters["m"] = 2
        likelihood.parameters["c"] = 0
        likelihood.log_likelihood()
        self.assertTrue(type(likelihood.sigma) == type(sigma_array))  # noqa: E721
        self.assertTrue(all(likelihood.sigma == sigma_array))

    def test_set_sigma_None(self):
        likelihood = GaussianLikelihood(self.x, self.y, self.function, sigma=None)
        likelihood.parameters["m"] = 2
        likelihood.parameters["c"] = 0
        self.assertTrue(likelihood.sigma is None)
        with self.assertRaises(TypeError):
            likelihood.log_likelihood()

    def test_sigma_float(self):
        likelihood = GaussianLikelihood(self.x, self.y, self.function, sigma=None)
        likelihood.parameters["m"] = 2
        likelihood.parameters["c"] = 0
        likelihood.parameters["sigma"] = 1
        likelihood.log_likelihood()
        self.assertEqual(likelihood.sigma, 1)

    def test_sigma_other(self):
        likelihood = GaussianLikelihood(self.x, self.y, self.function, sigma=None)
        with self.assertRaises(ValueError):
            likelihood.sigma = "test"

    def test_repr(self):
        likelihood = GaussianLikelihood(self.x, self.y, self.function, sigma=self.sigma)
        expected = "GaussianLikelihood(x={}, y={}, func={}, sigma={})".format(
            self.x, self.y, self.function.__name__, self.sigma
        )
        self.assertEqual(expected, repr(likelihood))


class TestStudentTLikelihood(unittest.TestCase):
    def setUp(self):
        self.N = 100
        self.nu = self.N - 2
        self.sigma = 1
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
        likelihood = StudentTLikelihood(
            self.x, self.y, self.function, self.nu, self.sigma
        )
        likelihood.parameters["m"] = 2
        likelihood.parameters["c"] = 0
        likelihood.log_likelihood()
        self.assertEqual(likelihood.sigma, self.sigma)

    def test_set_nu_none(self):
        likelihood = StudentTLikelihood(self.x, self.y, self.function, nu=None)
        likelihood.parameters["m"] = 2
        likelihood.parameters["c"] = 0
        self.assertTrue(likelihood.nu is None)

    def test_log_likelihood_nu_none(self):
        likelihood = StudentTLikelihood(self.x, self.y, self.function, nu=None)
        likelihood.parameters["m"] = 2
        likelihood.parameters["c"] = 0
        with self.assertRaises((ValueError, TypeError)):
            # ValueError in Python2, TypeError in Python3
            likelihood.log_likelihood()

    def test_log_likelihood_nu_zero(self):
        likelihood = StudentTLikelihood(self.x, self.y, self.function, nu=0)
        likelihood.parameters["m"] = 2
        likelihood.parameters["c"] = 0
        with self.assertRaises(ValueError):
            likelihood.log_likelihood()

    def test_log_likelihood_nu_negative(self):
        likelihood = StudentTLikelihood(self.x, self.y, self.function, nu=-1)
        likelihood.parameters["m"] = 2
        likelihood.parameters["c"] = 0
        with self.assertRaises(ValueError):
            likelihood.log_likelihood()

    def test_setting_nu_positive_does_not_change_class_attribute(self):
        likelihood = StudentTLikelihood(self.x, self.y, self.function, nu=None)
        likelihood.parameters["m"] = 2
        likelihood.parameters["c"] = 0
        likelihood.parameters["nu"] = 98
        self.assertEqual(likelihood.nu, 98)

    def test_lam(self):
        likelihood = StudentTLikelihood(self.x, self.y, self.function, nu=0, sigma=0.5)

        self.assertAlmostEqual(4.0, likelihood.lam)

    def test_repr(self):
        nu = 0
        sigma = 0.5
        likelihood = StudentTLikelihood(
            self.x, self.y, self.function, nu=nu, sigma=sigma
        )
        expected = "StudentTLikelihood(x={}, y={}, func={}, nu={}, sigma={})".format(
            self.x, self.y, self.function.__name__, nu, sigma
        )
        self.assertEqual(expected, repr(likelihood))


class TestPoissonLikelihood(unittest.TestCase):
    def setUp(self):
        self.N = 100
        self.mu = 5
        self.x = np.linspace(0, 1, self.N)
        self.y = np.random.poisson(self.mu, self.N)
        self.yfloat = np.copy(self.y) * 1.0
        self.yneg = np.copy(self.y)
        self.yneg[0] = -1

        def test_function(x, c):
            return c

        def test_function_array(x, c):
            return np.ones(len(x)) * c

        self.function = test_function
        self.function_array = test_function_array
        self.poisson_likelihood = PoissonLikelihood(self.x, self.y, self.function)

    def tearDown(self):
        del self.N
        del self.mu
        del self.x
        del self.y
        del self.yfloat
        del self.yneg
        del self.function
        del self.function_array
        del self.poisson_likelihood

    def test_init_y_non_integer(self):
        with self.assertRaises(ValueError):
            PoissonLikelihood(self.x, self.yfloat, self.function)

    def test_init__y_negative(self):
        with self.assertRaises(ValueError):
            PoissonLikelihood(self.x, self.yneg, self.function)

    def test_neg_rate(self):
        self.poisson_likelihood.parameters["c"] = -2
        with self.assertRaises(ValueError):
            self.poisson_likelihood.log_likelihood()

    def test_neg_rate_array(self):
        likelihood = PoissonLikelihood(self.x, self.y, self.function_array)
        likelihood.parameters["c"] = -2
        with self.assertRaises(ValueError):
            likelihood.log_likelihood()

    def test_init_y(self):
        self.assertTrue(np.array_equal(self.y, self.poisson_likelihood.y))

    def test_set_y_to_array(self):
        new_y = np.arange(start=0, stop=50, step=2)
        self.poisson_likelihood.y = new_y
        self.assertTrue(np.array_equal(new_y, self.poisson_likelihood.y))

    def test_set_y_to_positive_int(self):
        new_y = 5
        self.poisson_likelihood.y = new_y
        expected_y = np.array([new_y])
        self.assertTrue(np.array_equal(expected_y, self.poisson_likelihood.y))

    def test_set_y_to_negative_int(self):
        with self.assertRaises(ValueError):
            self.poisson_likelihood.y = -5

    def test_set_y_to_float(self):
        with self.assertRaises(ValueError):
            self.poisson_likelihood.y = 5.3

    def test_log_likelihood_wrong_func_return_type(self):
        poisson_likelihood = PoissonLikelihood(
            x=self.x, y=self.y, func=lambda x: "test"
        )
        with self.assertRaises(ValueError):
            poisson_likelihood.log_likelihood()

    def test_log_likelihood_negative_func_return_element(self):
        poisson_likelihood = PoissonLikelihood(
            x=self.x, y=self.y, func=lambda x: np.array([3, 6, -2])
        )
        with self.assertRaises(ValueError):
            poisson_likelihood.log_likelihood()

    def test_log_likelihood_zero_func_return_element(self):
        poisson_likelihood = PoissonLikelihood(
            x=self.x, y=self.y, func=lambda x: np.array([3, 6, 0])
        )
        self.assertEqual(-np.inf, poisson_likelihood.log_likelihood())

    def test_log_likelihood_dummy(self):
        """ Merely tests if it goes into the right if else bracket """
        poisson_likelihood = PoissonLikelihood(
            x=self.x, y=self.y, func=lambda x: np.linspace(1, 100, self.N)
        )
        with mock.patch("numpy.sum") as m:
            m.return_value = 1
            self.assertEqual(1, poisson_likelihood.log_likelihood())

    def test_repr(self):
        likelihood = PoissonLikelihood(self.x, self.y, self.function)
        expected = "PoissonLikelihood(x={}, y={}, func={})".format(
            self.x, self.y, self.function.__name__
        )
        self.assertEqual(expected, repr(likelihood))


class TestExponentialLikelihood(unittest.TestCase):
    def setUp(self):
        self.N = 100
        self.mu = 5
        self.x = np.linspace(0, 1, self.N)
        self.y = np.random.exponential(self.mu, self.N)
        self.yneg = np.copy(self.y)
        self.yneg[0] = -1.0

        def test_function(x, c):
            return c

        def test_function_array(x, c):
            return c * np.ones(len(x))

        self.function = test_function
        self.function_array = test_function_array
        self.exponential_likelihood = ExponentialLikelihood(
            x=self.x, y=self.y, func=self.function
        )

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
            ExponentialLikelihood(self.x, self.yneg, self.function)

    def test_negative_function(self):
        likelihood = ExponentialLikelihood(self.x, self.y, self.function)
        likelihood.parameters["c"] = -1
        self.assertEqual(likelihood.log_likelihood(), -np.inf)

    def test_negative_array_function(self):
        likelihood = ExponentialLikelihood(self.x, self.y, self.function_array)
        likelihood.parameters["c"] = -1
        self.assertEqual(likelihood.log_likelihood(), -np.inf)

    def test_init_y(self):
        self.assertTrue(np.array_equal(self.y, self.exponential_likelihood.y))

    def test_set_y_to_array(self):
        new_y = np.arange(start=0, stop=50, step=2)
        self.exponential_likelihood.y = new_y
        self.assertTrue(np.array_equal(new_y, self.exponential_likelihood.y))

    def test_set_y_to_positive_int(self):
        new_y = 5
        self.exponential_likelihood.y = new_y
        expected_y = np.array([new_y])
        self.assertTrue(np.array_equal(expected_y, self.exponential_likelihood.y))

    def test_set_y_to_negative_int(self):
        with self.assertRaises(ValueError):
            self.exponential_likelihood.y = -5

    def test_set_y_to_positive_float(self):
        new_y = 5.3
        self.exponential_likelihood.y = new_y
        self.assertTrue(np.array_equal(np.array([5.3]), self.exponential_likelihood.y))

    def test_set_y_to_negative_float(self):
        with self.assertRaises(ValueError):
            self.exponential_likelihood.y = -5.3

    def test_set_y_to_nd_array_with_negative_element(self):
        with self.assertRaises(ValueError):
            self.exponential_likelihood.y = np.array([4.3, -1.2, 4])

    def test_log_likelihood_default(self):
        """ Merely tests that it ends up at the right place in the code """
        exponential_likelihood = ExponentialLikelihood(
            x=self.x, y=self.y, func=lambda x: np.array([4.2])
        )
        with mock.patch("numpy.sum") as m:
            m.return_value = 3
            self.assertEqual(-3, exponential_likelihood.log_likelihood())

    def test_repr(self):
        expected = "ExponentialLikelihood(x={}, y={}, func={})".format(
            self.x, self.y, self.function.__name__
        )
        self.assertEqual(expected, repr(self.exponential_likelihood))


class TestAnalyticalMultidimensionalCovariantGaussian(unittest.TestCase):
    def setUp(self):
        self.cov = [[1, 0, 0], [0, 4, 0], [0, 0, 9]]
        self.sigma = [1, 2, 3]
        self.mean = [10, 11, 12]
        self.likelihood = AnalyticalMultidimensionalCovariantGaussian(
            mean=self.mean, cov=self.cov
        )
        self.likelihood.parameters.update({f"x{ii}": 0 for ii in range(len(self.sigma))})

    def tearDown(self):
        del self.cov
        del self.sigma
        del self.mean
        del self.likelihood

    def test_cov(self):
        self.assertTrue(np.array_equal(self.cov, self.likelihood.cov))

    def test_mean(self):
        self.assertTrue(np.array_equal(self.mean, self.likelihood.mean))

    def test_sigma(self):
        self.assertTrue(np.array_equal(self.sigma, self.likelihood.sigma))

    def test_parameters(self):
        self.assertDictEqual(dict(x0=0, x1=0, x2=0), self.likelihood.parameters)

    def test_dim(self):
        self.assertEqual(3, self.likelihood.dim)

    def test_log_likelihood(self):
        likelihood = AnalyticalMultidimensionalCovariantGaussian(mean=[0], cov=[1])
        self.assertEqual(-np.log(2 * np.pi) / 2, likelihood.log_likelihood(dict(x0=0)))


class TestAnalyticalMultidimensionalBimodalCovariantGaussian(unittest.TestCase):
    def setUp(self):
        self.cov = [[1, 0, 0], [0, 4, 0], [0, 0, 9]]
        self.sigma = [1, 2, 3]
        self.mean_1 = [10, 11, 12]
        self.mean_2 = [20, 21, 22]
        self.likelihood = AnalyticalMultidimensionalBimodalCovariantGaussian(
            mean_1=self.mean_1, mean_2=self.mean_2, cov=self.cov
        )
        self.likelihood.parameters.update({f"x{ii}": 0 for ii in range(len(self.sigma))})

    def tearDown(self):
        del self.cov
        del self.sigma
        del self.mean_1
        del self.mean_2
        del self.likelihood

    def test_cov(self):
        self.assertTrue(np.array_equal(self.cov, self.likelihood.cov))

    def test_mean_1(self):
        self.assertTrue(np.array_equal(self.mean_1, self.likelihood.mean_1))

    def test_mean_2(self):
        self.assertTrue(np.array_equal(self.mean_2, self.likelihood.mean_2))

    def test_sigma(self):
        self.assertTrue(np.array_equal(self.sigma, self.likelihood.sigma))

    def test_parameters(self):
        self.assertDictEqual(dict(x0=0, x1=0, x2=0), self.likelihood.parameters)

    def test_dim(self):
        self.assertEqual(3, self.likelihood.dim)

    def test_log_likelihood(self):
        likelihood = AnalyticalMultidimensionalBimodalCovariantGaussian(
            mean_1=[0], mean_2=[0], cov=[1]
        )
        self.assertEqual(-np.log(2 * np.pi) / 2, likelihood.log_likelihood(dict(x0=0)))


class TestJointLikelihood(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1, 2, 3])
        self.y = np.array([1, 2, 3])
        self.first_likelihood = GaussianLikelihood(
            x=self.x,
            y=self.y,
            func=lambda x, param1, param2: (param1 + param2) * x,
            sigma=1,
        )
        self.second_likelihood = PoissonLikelihood(
            x=self.x, y=self.y, func=lambda x, param2, param3: (param2 + param3) * x
        )
        self.third_likelihood = ExponentialLikelihood(
            x=self.x, y=self.y, func=lambda x, param4, param5: (param4 + param5) * x
        )
        self.joint_likelihood = JointLikelihood(
            self.first_likelihood, self.second_likelihood, self.third_likelihood
        )

        # self.first_parameters = dict(param1=1, param2=2)
        # self.second_parmaeters = dict(param2=2, param3=3)
        # self.third_parmaeters = dict(param4=4, param5=5)
        self.first_likelihood.parameters["param1"] = 1
        self.first_likelihood.parameters["param2"] = 2
        self.second_likelihood.parameters["param2"] = 2
        self.second_likelihood.parameters["param3"] = 3
        self.third_likelihood.parameters["param4"] = 4
        self.third_likelihood.parameters["param5"] = 5

        self.joint_likelihood.parameters["param1"] = 1
        self.joint_likelihood.parameters["param2"] = 2
        self.joint_likelihood.parameters["param3"] = 3
        self.joint_likelihood.parameters["param4"] = 4
        self.joint_likelihood.parameters["param5"] = 5

    def tearDown(self):
        del self.x
        del self.y
        del self.first_likelihood
        del self.second_likelihood
        del self.third_likelihood
        del self.joint_likelihood

    def test_parameters_consistent_from_init(self):
        expected = dict(param1=1, param2=2, param3=3, param4=4, param5=5,)
        self.assertDictEqual(expected, self.joint_likelihood.parameters)

    def test_log_likelihood_correctly_sums(self):
        expected = (
            self.first_likelihood.log_likelihood()
            + self.second_likelihood.log_likelihood()
            + self.third_likelihood.log_likelihood()
        )
        self.assertEqual(expected, self.joint_likelihood.log_likelihood())

    def test_log_likelihood_checks_parameter_updates(self):
        self.first_likelihood.parameters["param2"] = 7
        self.second_likelihood.parameters["param2"] = 7
        self.joint_likelihood.parameters["param2"] = 7
        expected = (
            self.first_likelihood.log_likelihood()
            + self.second_likelihood.log_likelihood()
            + self.third_likelihood.log_likelihood()
        )
        self.assertEqual(expected, self.joint_likelihood.log_likelihood())

    def test_list_element_parameters_are_updated(self):
        self.joint_likelihood.parameters["param2"] = 7
        self.assertEqual(
            self.joint_likelihood.parameters["param2"],
            self.joint_likelihood.likelihoods[0].parameters["param2"],
        )
        self.assertEqual(
            self.joint_likelihood.parameters["param2"],
            self.joint_likelihood.likelihoods[1].parameters["param2"],
        )

    def test_log_noise_likelihood(self):
        self.first_likelihood.noise_log_likelihood = mock.MagicMock(return_value=1)
        self.second_likelihood.noise_log_likelihood = mock.MagicMock(return_value=2)
        self.third_likelihood.noise_log_likelihood = mock.MagicMock(return_value=3)
        self.joint_likelihood = JointLikelihood(
            self.first_likelihood, self.second_likelihood, self.third_likelihood
        )
        expected = (
            self.first_likelihood.noise_log_likelihood()
            + self.second_likelihood.noise_log_likelihood()
            + self.third_likelihood.noise_log_likelihood()
        )
        self.assertEqual(expected, self.joint_likelihood.noise_log_likelihood())

    def test_init_with_list_of_likelihoods(self):
        with self.assertRaises(ValueError):
            JointLikelihood(
                [self.first_likelihood, self.second_likelihood, self.third_likelihood]
            )

    def test_setting_single_likelihood(self):
        self.joint_likelihood.likelihoods = self.first_likelihood
        self.assertEqual(
            self.first_likelihood.log_likelihood(),
            self.joint_likelihood.log_likelihood(),
        )

    def test_setting_likelihood_other(self):
        with self.assertRaises(ValueError):
            self.joint_likelihood.likelihoods = "test"

    # Appending is not supported
    # def test_appending(self):
    #     joint_likelihood = bilby.core.likelihood.JointLikelihood(self.first_likelihood, self.second_likelihood)
    #     joint_likelihood.likelihoods.append(self.third_likelihood)
    #     self.assertDictEqual(self.joint_likelihood.parameters, joint_likelihood.parameters)


class TestGPLikelihood(unittest.TestCase):

    def setUp(self) -> None:
        self.t = [1, 2, 3]
        self.y = [4, 5, 6]
        self.yerr = [0.4, 0.5, 0.6]
        self.kernel = mock.MagicMock()
        self.mean_model = mock.MagicMock()
        self.gp_mock = mock.MagicMock()
        self.gp_mock.compute = mock.MagicMock()
        self.parameter_dict = dict(a=1, b=2)
        self.gp_mock.get_parameter_dict = mock.MagicMock(return_value=dict(self.parameter_dict))
        self.gp_class = mock.MagicMock(return_value=self.gp_mock)
        self.celerite_likelihood = bilby.core.likelihood._GPLikelihood(
            kernel=self.kernel, mean_model=self.mean_model, t=self.t, y=self.y, yerr=self.yerr, gp_class=self.gp_class)

    def tearDown(self) -> None:
        del self.t
        del self.y
        del self.yerr
        del self.kernel
        del self.mean_model
        del self.parameter_dict
        del self.gp_class
        del self.gp_mock
        del self.celerite_likelihood

    def test_t(self):
        self.assertIsInstance(self.celerite_likelihood.t, np.ndarray)
        self.assertTrue(np.array_equal(self.t, self.celerite_likelihood.t))

    def test_y(self):
        self.assertIsInstance(self.celerite_likelihood.y, np.ndarray)
        self.assertTrue(np.array_equal(self.y, self.celerite_likelihood.y))

    def test_yerr(self):
        self.assertIsInstance(self.celerite_likelihood.yerr, np.ndarray)
        self.assertTrue(np.array_equal(self.yerr, self.celerite_likelihood.yerr))

    def test_gp_class(self):
        self.assertEqual(self.gp_class, self.celerite_likelihood.GPClass)

    def test_gp_instantiation(self):
        self.celerite_likelihood.GPClass.assert_called_once_with(
            kernel=self.kernel, mean=self.mean_model, fit_mean=True, fit_white_noise=True)

    def test_gp_mock(self):
        self.celerite_likelihood.gp.compute.assert_called_once_with(
            self.celerite_likelihood.t, yerr=self.celerite_likelihood.yerr)

    def test_parameters(self):
        self.assertDictEqual(self.parameter_dict, self.celerite_likelihood.parameters)

    def test_set_parameters_no_exceptions(self):
        self.celerite_likelihood.gp.set_parameter = mock.MagicMock()
        self.celerite_likelihood.mean_model.set_parameter = mock.MagicMock()
        expected_a = 5
        self.celerite_likelihood.set_parameters(dict(a=expected_a))
        self.celerite_likelihood.gp.set_parameter.assert_called_once_with(name="a", value=5)
        self.assertEqual(expected_a, self.celerite_likelihood.parameters["a"])


class TestFunctionMeanModel(unittest.TestCase):

    def test_function_to_celerite_mean_model(self):
        def func(x, a, b, c):
            return a * x ** 2 + b * x + c

        mean_model = bilby.core.likelihood.function_to_celerite_mean_model(func=func)
        self.assertListEqual(["a", "b", "c"], list(mean_model.parameter_names))

    def test_function_to_george_mean_model(self):
        def func(x, a, b, c):
            return a * x ** 2 + b * x + c

        mean_model = bilby.core.likelihood.function_to_celerite_mean_model(func=func)
        self.assertListEqual(["a", "b", "c"], list(mean_model.parameter_names))


class TestCeleriteLikelihoodEvaluation(unittest.TestCase):

    def setUp(self) -> None:
        import celerite

        def func(x, a):
            return a * x

        self.t = [0, 1, 2]
        self.y = [0, 1, 2]
        self.yerr = [0.4, 0.5, 0.6]
        self.parameters = {"kernel:log_S0": 0, "kernel:log_Q": 0, "kernel:log_omega0": 0, "mean:a": 1}
        self.kernel = celerite.terms.SHOTerm(log_S0=0, log_Q=0, log_omega0=0)

        self.MeanModel = bilby.likelihood.function_to_celerite_mean_model(func=func)
        self.mean_model = self.MeanModel(a=1)
        self.celerite_likelihood = bilby.core.likelihood.CeleriteLikelihood(
            kernel=self.kernel, mean_model=self.mean_model, t=self.t, y=self.y, yerr=self.yerr)
        self.celerite_likelihood.parameters = self.parameters

    def tearDown(self) -> None:
        del self.t
        del self.y
        del self.yerr
        del self.parameters
        del self.kernel
        del self.MeanModel
        del self.mean_model
        del self.celerite_likelihood

    def test_log_l_evalutation(self):
        log_l = self.celerite_likelihood.log_likelihood()
        expected = -2.0390312696885102
        self.assertAlmostEqual(expected, log_l, places=5)

    def test_set_parameters(self):
        combined_params = {"kernel:log_S0": 2, "kernel:log_Q": 2, "kernel:log_omega0": 2, "mean:a": 2}
        self.celerite_likelihood.set_parameters(combined_params)
        self.assertDictEqual(combined_params, self.celerite_likelihood.parameters)


class TestGeorgeLikelihoodEvaluation(unittest.TestCase):

    def setUp(self) -> None:
        import george

        def func(x, a):
            return a * x

        self.t = [0, 1, 2]
        self.y = [0, 1, 2]
        self.yerr = [0.4, 0.5, 0.6]
        self.parameters = {}
        self.kernel = 2.0 * george.kernels.Matern32Kernel(metric=5.0)

        self.MeanModel = bilby.likelihood.function_to_celerite_mean_model(func=func)
        self.mean_model = self.MeanModel(a=1)
        self.george_likelihood = bilby.core.likelihood.GeorgeLikelihood(
            kernel=self.kernel, mean_model=self.mean_model, t=self.t, y=self.y, yerr=self.yerr)

    def tearDown(self) -> None:
        pass

    def test_likelihood_value(self):
        log_l = self.george_likelihood.log_likelihood()
        expected = -3.2212751203208403
        self.assertAlmostEqual(expected, log_l, places=5)

    def test_set_parameters(self):
        combined_params = {"kernel:k1:log_constant": 2, "kernel:k2:metric:log_M_0_0": 2, "mean:a": 2}
        self.george_likelihood.set_parameters(combined_params)
        self.assertDictEqual(combined_params, self.george_likelihood.parameters)


if __name__ == "__main__":
    unittest.main()
