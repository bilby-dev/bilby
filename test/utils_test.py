from __future__ import absolute_import, division

import unittest
import numpy as np
from astropy import constants

import bilby
from bilby.core import utils


class TestConstants(unittest.TestCase):

    def test_speed_of_light(self):
        self.assertTrue(bilby.core.utils.speed_of_light, constants.c.value)

    def test_parsec(self):
        self.assertTrue(bilby.core.utils.parsec, constants.pc.value)

    def test_solar_mass(self):
        self.assertTrue(bilby.core.utils.solar_mass, constants.M_sun.value)

    def test_radius_of_earth(self):
        self.assertTrue(bilby.core.utils.radius_of_earth, constants.R_earth.value)


class TestFFT(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_nfft_frequencies(self):
        f = 2.1
        sampling_frequency = 10
        times = np.arange(0, 100, 1/sampling_frequency)
        tds = np.sin(2*np.pi*times * f + 0.4)
        fds, freqs = bilby.core.utils.nfft(tds, sampling_frequency)
        self.assertTrue(np.abs((f-freqs[np.argmax(np.abs(fds))])/f < 1e-15))

    def test_nfft_infft(self):
        sampling_frequency = 10
        tds = np.random.normal(0, 1, 10)
        fds, _ = bilby.core.utils.nfft(tds, sampling_frequency)
        tds2 = bilby.core.utils.infft(fds, sampling_frequency)
        self.assertTrue(np.all(np.abs((tds - tds2) / tds) < 1e-12))


class TestInferParameters(unittest.TestCase):

    def setUp(self):
        def source_function(freqs, a, b, *args, **kwargs):
            return None

        class TestClass:
            def test_method(self, a, b, *args, **kwargs):
                pass

        self.source1 = source_function
        test_obj = TestClass()
        self.source2 = test_obj.test_method

    def tearDown(self):
        del self.source1
        del self.source2

    def test_args_kwargs_handling(self):
        expected = ['a', 'b']
        actual = utils.infer_parameters_from_function(self.source1)
        self.assertListEqual(expected, actual)

    def test_self_handling(self):
        expected = ['a', 'b']
        actual = utils.infer_args_from_method(self.source2)
        self.assertListEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
