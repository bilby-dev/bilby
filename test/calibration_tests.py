from tupak.gw import calibration
import unittest
import numpy as np


class TestBaseClass(unittest.TestCase):

    def setUp(self):
        self.model = calibration.Recalibrate()

    def tearDown(self):
        del self.model

    def test_calibration_factor(self):
        frequency_array = np.linspace(20, 1024, 1000)
        cal_factor = self.model.get_calibration_factor(frequency_array)
        assert np.alltrue(cal_factor.real == np.ones_like(frequency_array))


class TestCubicSpline(unittest.TestCase):

    def setUp(self):
        self.model = calibration.CubicSpline(
            prefix='recalib_', minimum_frequency=20, maximum_frequency=1024,
            n_points=5)
        self.parameters = {'recalib_{}_{}'.format(param, ii): 0.0
                           for ii in range(5)
                           for param in ['amplitude', 'phase']}

    def tearDown(self):
        del self.model
        del self.parameters

    def test_calibration_factor(self):
        frequency_array = np.linspace(20, 1024, 1000)
        cal_factor = self.model.get_calibration_factor(frequency_array,
                                                       **self.parameters)
        assert np.alltrue(cal_factor.real == np.ones_like(frequency_array))


class TestCubicSplineRequiresFourNodes(unittest.TestCase):

    def test_cannot_instantiate_with_too_few_nodes(self):
        for ii in range(6):
            if ii < 4:
                with self.assertRaises(ValueError):
                    calibration.CubicSpline('test', 1, 10, ii)
            else:
                calibration.CubicSpline('test', 1, 10, ii)


if __name__ == '__main__':
    unittest.main()
