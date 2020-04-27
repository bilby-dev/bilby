from bilby.gw import calibration
import unittest
import numpy as np


class TestBaseClass(unittest.TestCase):
    def setUp(self):
        self.model = calibration.Recalibrate()

    def tearDown(self):
        del self.model

    def test_repr(self):
        expected = "Recalibrate(prefix={})".format("'recalib_'")
        actual = repr(self.model)
        self.assertEqual(expected, actual)

    def test_calibration_factor(self):
        frequency_array = np.linspace(20, 1024, 1000)
        cal_factor = self.model.get_calibration_factor(frequency_array)
        assert np.alltrue(cal_factor.real == np.ones_like(frequency_array))


class TestCubicSpline(unittest.TestCase):
    def setUp(self):
        self.prefix = "recalib_"
        self.minimum_frequency = 20
        self.maximum_frequency = 1024
        self.n_points = 5
        self.model = calibration.CubicSpline(
            prefix=self.prefix,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            n_points=self.n_points,
        )
        self.parameters = {
            "recalib_{}_{}".format(param, ii): 0.0
            for ii in range(5)
            for param in ["amplitude", "phase"]
        }

    def tearDown(self):
        del self.prefix
        del self.minimum_frequency
        del self.maximum_frequency
        del self.n_points
        del self.model
        del self.parameters

    def test_calibration_factor(self):
        frequency_array = np.linspace(20, 1024, 1000)
        cal_factor = self.model.get_calibration_factor(
            frequency_array, **self.parameters
        )
        assert np.alltrue(cal_factor.real == np.ones_like(frequency_array))

    def test_repr(self):
        expected = "CubicSpline(prefix='{}', minimum_frequency={}, maximum_frequency={}, n_points={})".format(
            self.prefix, self.minimum_frequency, self.maximum_frequency, self.n_points
        )
        actual = repr(self.model)
        self.assertEqual(expected, actual)


class TestCubicSplineRequiresFourNodes(unittest.TestCase):
    def test_cannot_instantiate_with_too_few_nodes(self):
        for ii in range(6):
            if ii < 4:
                with self.assertRaises(ValueError):
                    calibration.CubicSpline("test", 1, 10, ii)
            else:
                calibration.CubicSpline("test", 1, 10, ii)


if __name__ == "__main__":
    unittest.main()
