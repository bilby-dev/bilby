from __future__ import division, absolute_import
import unittest
import mock
import tupak
import numpy as np


class TestBasicConversions(unittest.TestCase):

    def setUp(self):
        self.mass_1 = 1.4
        self.mass_2 = 1.3
        self.mass_ratio = 13/14
        self.total_mass = 2.7
        self.chirp_mass = (1.4 * 1.3)**0.6 / 2.7**0.2
        self.symmetric_mass_ratio = (1.4 * 1.3) / 2.7**2
        self.cos_angle = -1
        self.angle = np.pi
        self.lambda_1 = 300
        self.lambda_2 = 300 * (14 / 13)**5
        self.lambda_tilde = 8 / 13 * (
            (1 + 7 * self.symmetric_mass_ratio
             - 31 * self.symmetric_mass_ratio**2)
            * (self.lambda_1 + self.lambda_2)
            + (1 - 4 * self.symmetric_mass_ratio)**0.5
            * (1 + 9 * self.symmetric_mass_ratio
               - 11 * self.symmetric_mass_ratio**2)
            * (self.lambda_1 - self.lambda_2)
        )
        self.delta_lambda = 1 / 2 * (
                (1 - 4 * self.symmetric_mass_ratio)**0.5
                * (1 - 13272 / 1319 * self.symmetric_mass_ratio
                   + 8944 / 1319 * self.symmetric_mass_ratio**2)
                * (self.lambda_1 + self.lambda_2)
                + (1 - 15910 / 1319 * self.symmetric_mass_ratio
                   + 32850 / 1319 * self.symmetric_mass_ratio**2
                   + 3380 / 1319 * self.symmetric_mass_ratio**3)
                * (self.lambda_1 - self.lambda_2)
        )

    def tearDown(self):
        del self.mass_1
        del self.mass_2
        del self.mass_ratio
        del self.total_mass
        del self.chirp_mass
        del self.symmetric_mass_ratio

    def test_total_mass_and_mass_ratio_to_component_masses(self):
        mass_1, mass_2 = tupak.gw.conversion.total_mass_and_mass_ratio_to_component_masses(self.mass_ratio, self.total_mass)
        self.assertTrue(all([abs(mass_1 - self.mass_1) < 1e-5,
                             abs(mass_2 - self.mass_2) < 1e-5]))

    def test_symmetric_mass_ratio_to_mass_ratio(self):
        mass_ratio = tupak.gw.conversion.symmetric_mass_ratio_to_mass_ratio(self.symmetric_mass_ratio)
        self.assertAlmostEqual(self.mass_ratio, mass_ratio)

    def test_chirp_mass_and_total_mass_to_symmetric_mass_ratio(self):
        symmetric_mass_ratio = tupak.gw.conversion.chirp_mass_and_total_mass_to_symmetric_mass_ratio(self.chirp_mass, self.total_mass)
        self.assertAlmostEqual(self.symmetric_mass_ratio, symmetric_mass_ratio)

    def test_chirp_mass_and_mass_ratio_to_total_mass(self):
        total_mass = tupak.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(self.chirp_mass, self.mass_ratio)
        self.assertAlmostEqual(self.total_mass, total_mass)

    def test_component_masses_to_chirp_mass(self):
        chirp_mass = tupak.gw.conversion.component_masses_to_chirp_mass(self.mass_1, self.mass_2)
        self.assertAlmostEqual(self.chirp_mass, chirp_mass)

    def test_component_masses_to_total_mass(self):
        total_mass = tupak.gw.conversion.component_masses_to_total_mass(self.mass_1, self.mass_2)
        self.assertAlmostEqual(self.total_mass, total_mass)

    def test_component_masses_to_symmetric_mass_ratio(self):
        symmetric_mass_ratio = tupak.gw.conversion.component_masses_to_symmetric_mass_ratio(self.mass_1, self.mass_2)
        self.assertAlmostEqual(self.symmetric_mass_ratio, symmetric_mass_ratio)

    def test_component_masses_to_mass_ratio(self):
        mass_ratio = tupak.gw.conversion.component_masses_to_mass_ratio(self.mass_1, self.mass_2)
        self.assertAlmostEqual(self.mass_ratio, mass_ratio)

    def test_mass_1_and_chirp_mass_to_mass_ratio(self):
        mass_ratio = tupak.gw.conversion.mass_1_and_chirp_mass_to_mass_ratio(self.mass_1, self.chirp_mass)
        self.assertAlmostEqual(self.mass_ratio, mass_ratio)

    def test_lambda_tilde_to_lambda_1_lambda_2(self):
        lambda_1, lambda_2 =\
            tupak.gw.conversion.lambda_tilde_to_lambda_1_lambda_2(
                self.lambda_tilde, self.mass_1, self.mass_2)
        self.assertTrue(all([abs(self.lambda_1 - lambda_1) < 1e-5,
                             abs(self.lambda_2 - lambda_2) < 1e-5]))

    def test_lambda_tilde_delta_lambda_to_lambda_1_lambda_2(self):
        lambda_1, lambda_2 =\
            tupak.gw.conversion.lambda_tilde_delta_lambda_to_lambda_1_lambda_2(
                self.lambda_tilde, self.delta_lambda, self.mass_1, self.mass_2)
        self.assertTrue(all([abs(self.lambda_1 - lambda_1) < 1e-5,
                             abs(self.lambda_2 - lambda_2) < 1e-5]))


class TestConvertToLALBBHParams(unittest.TestCase):

    def setUp(self):
        self.search_keys = []
        self.parameters = dict()
        self.remove = True

    def tearDown(self):
        del self.search_keys
        del self.parameters
        del self.remove

    def test_cos_angle_to_angle_conversion(self):
        with mock.patch('numpy.arccos') as m:
            m.return_value = 42
            self.search_keys.append('cos_tilt_1')
            self.parameters['cos_tilt_1'] = 1
            self.parameters, _ = tupak.gw.conversion.convert_to_lal_binary_black_hole_parameters(self.parameters, self.search_keys)
            self.assertEqual(42, self.parameters['tilt_1'])

    def test_cos_angle_to_angle_conversion_removal(self):
        with mock.patch('numpy.arccos') as m:
            m.return_value = 42
            self.search_keys.append('cos_tilt_1')
            self.parameters['cos_tilt_1'] = 1
            self.parameters, _ = tupak.gw.conversion.convert_to_lal_binary_black_hole_parameters(self.parameters, self.search_keys, remove=True)
            self.assertDictEqual(self.parameters, dict(tilt_1=42, cos_tilt_1=1))


if __name__ == '__main__':
    unittest.main()
