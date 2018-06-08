from __future__ import division, absolute_import
import unittest
import mock
import tupak
import numpy as np


class TestBasicConversions(unittest.TestCase):

    def setUp(self):
        self.mass_1 = 20.
        self.mass_2 = 10.
        self.mass_ratio = 0.5
        self.total_mass = 30.
        self.chirp_mass = 200.**0.6 / 30.**0.2
        self.symmetric_mass_ratio = 2./9.
        self.cos_angle = -1
        self.angle = np.pi

    def tearDown(self):
        del self.mass_1
        del self.mass_2
        del self.mass_ratio
        del self.total_mass
        del self.chirp_mass
        del self.symmetric_mass_ratio

    def test_total_mass_and_mass_ratio_to_component_masses(self):
        mass_1, mass_2 = tupak.conversion.total_mass_and_mass_ratio_to_component_masses(self.mass_1, self.mass_2)
        self.assertTupleEqual((mass_1, mass_2), (self.mass_1, self.mass_2))

    def test_symmetric_mass_ratio_to_mass_ratio(self):
        mass_ratio = tupak.conversion.symmetric_mass_ratio_to_mass_ratio(self.symmetric_mass_ratio)
        self.assertAlmostEqual(self.mass_ratio, mass_ratio)

    def test_chirp_mass_and_total_mass_to_symmetric_mass_ratio(self):
        symmetric_mass_ratio = tupak.conversion.chirp_mass_and_total_mass_to_symmetric_mass_ratio(self.chirp_mass, self.total_mass)
        self.assertAlmostEqual(self.symmetric_mass_ratio, symmetric_mass_ratio)

    def test_chirp_mass_and_mass_ratio_to_total_mass(self):
        total_mass = tupak.conversion.chirp_mass_and_mass_ratio_to_total_mass(self.chirp_mass, self.mass_ratio)
        self.assertAlmostEqual(self.total_mass, total_mass)

    def test_component_masses_to_chirp_mass(self):
        chirp_mass = tupak.conversion.component_masses_to_chirp_mass(self.mass_1, self.mass_2)
        self.assertAlmostEqual(self.chirp_mass, chirp_mass)

    def test_component_masses_to_total_mass(self):
        total_mass = tupak.conversion.component_masses_to_total_mass(self.mass_1, self.mass_2)
        self.assertAlmostEqual(self.total_mass, total_mass)

    def test_component_masses_to_symmetric_mass_ratio(self):
        symmetric_mass_ratio = tupak.conversion.component_masses_to_symmetric_mass_ratio(self.mass_1, self.mass_2)
        self.assertAlmostEqual(self.symmetric_mass_ratio, symmetric_mass_ratio)

    def test_component_masses_to_mass_ratio(self):
        mass_ratio = tupak.conversion.component_masses_to_mass_ratio(self.mass_1, self.mass_2)
        self.assertAlmostEqual(self.mass_ratio, mass_ratio)

    def test_cos_angle_to_angle(self):
        angle = tupak.conversion.cos_angle_to_angle(self.cos_angle)
        self.assertAlmostEqual(self.angle, angle)

    def test_angle_to_cos_angle(self):
        cos_angle = tupak.conversion.angle_to_cos_angle(self.angle)
        self.assertAlmostEqual(self.cos_angle, cos_angle)
