import unittest
from source import *


class TestSourceInstantiation(unittest.TestCase):
    def setUp(self):
        self.source = Source("test_name")

    def tearDown(self):
        del self.source

    def test_name(self):
        self.assertEqual(self.source.name, "test_name")


class TestGlitchInstantiation(unittest.TestCase):

    def setUp(self):
        self.source = Glitch("test_name")

    def tearDown(self):
        del self.source

    def test_name(self):
        self.assertEqual(self.source.name, "test_name")


class TestAstrophysicalInstantiation(unittest.TestCase):

    def setUp(self):
        name = "test_name"
        right_ascension, declination, luminosity_distance = 1, 2, 3
        self.source = AstrophysicalSource(name, right_ascension, declination, luminosity_distance)

    def tearDown(self):
        del self.source

    def test_name(self):
        self.assertEqual(self.source.name, "test_name")

    def test_right_ascension(self):
        self.assertEqual(self.source.right_ascension, 1)

    def test_declination(self):
        self.assertEqual(self.source.declination, 2)

    def test_luminosity_distance(self):
        self.assertEqual(self.source.luminosity_distance, 3)


class TestCompactBinaryCoalescence(unittest.TestCase):

    def setUp(self):
        name = "test_name"
        right_ascension = 1
        declination = 2
        luminosity_distance = 3
        mass_1 = 4
        mass_2 = 5
        spin_1 = [6, 7, 8]
        spin_2 = [9, 10, 11]
        coalescence_time = 12
        inclination_angle = 13
        waveform_phase = 14
        polarisation_angle = 15
        eccentricity = 16
        self.source = CompactBinaryCoalescence(name, right_ascension, declination, luminosity_distance, mass_1, mass_2,
                                               spin_1, spin_2, coalescence_time, inclination_angle, waveform_phase,
                                               polarisation_angle, eccentricity)

    def tearDown(self):
        del self.source

    def test_name(self):
        self.assertEqual(self.source.name, "test_name")

    def test_right_ascension(self):
        self.assertEqual(self.source.right_ascension, 1)

    def test_declination(self):
        self.assertEqual(self.source.declination, 2)

    def test_luminosity_distance(self):
        self.assertEqual(self.source.luminosity_distance, 3)

    def test_mass_1(self):
        self.assertEqual(self.source.mass_1, 4)

    def test_mass_2(self):
        self.assertEqual(self.source.mass_2, 5)

    def test_spin_1(self):
        self.assertEqual(self.source.spin_1, [6, 7, 8])

    def test_spin_2(self):
        self.assertEqual(self.source.spin_2, [9, 10, 11])

    def test_coalescence_time(self):
        self.assertEqual(self.source.coalescence_time, 12)

    def test_inclination_angle(self):
        self.assertEqual(self.source.inclination_angle, 13)

    def test_waveform_phase(self):
        self.assertEqual(self.source.waveform_phase, 14)

    def test_polarisation_angle(self):
        self.assertEqual(self.source.polarisation_angle, 15)

    def test_eccentricity(self):
        self.assertEqual(self.source.eccentricity, 16)


if __name__ == '__main__':
    unittest.main()
