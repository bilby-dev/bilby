from __future__ import absolute_import, division
import unittest
import os

import numpy as np
from astropy import constants
import lal
import matplotlib.pyplot as plt

import bilby
from bilby.core import utils


class TestConstants(unittest.TestCase):
    def test_speed_of_light(self):
        self.assertEqual(utils.speed_of_light, lal.C_SI)
        self.assertLess(
            abs(utils.speed_of_light - constants.c.value) / utils.speed_of_light, 1e-16
        )

    def test_parsec(self):
        self.assertEqual(utils.parsec, lal.PC_SI)
        self.assertLess(abs(utils.parsec - constants.pc.value) / utils.parsec, 1e-11)

    def test_solar_mass(self):
        self.assertEqual(utils.solar_mass, lal.MSUN_SI)
        self.assertLess(
            abs(utils.solar_mass - constants.M_sun.value) / utils.solar_mass, 1e-4
        )

    def test_radius_of_earth(self):
        self.assertEqual(bilby.core.utils.radius_of_earth, lal.REARTH_SI)
        self.assertLess(
            abs(utils.radius_of_earth - constants.R_earth.value)
            / utils.radius_of_earth,
            1e-5,
        )

    def test_gravitational_constant(self):
        self.assertEqual(bilby.core.utils.gravitational_constant, lal.G_SI)


class TestFFT(unittest.TestCase):
    def setUp(self):
        self.sampling_frequency = 10

    def tearDown(self):
        del self.sampling_frequency

    def test_nfft_sine_function(self):
        injected_frequency = 2.7324
        duration = 100
        times = utils.create_time_series(self.sampling_frequency, duration)

        time_domain_strain = np.sin(2 * np.pi * times * injected_frequency + 0.4)

        frequency_domain_strain, frequencies = bilby.core.utils.nfft(
            time_domain_strain, self.sampling_frequency
        )
        frequency_at_peak = frequencies[np.argmax(np.abs(frequency_domain_strain))]
        self.assertAlmostEqual(injected_frequency, frequency_at_peak, places=1)

    def test_nfft_infft(self):
        time_domain_strain = np.random.normal(0, 1, 10)
        frequency_domain_strain, _ = bilby.core.utils.nfft(
            time_domain_strain, self.sampling_frequency
        )
        new_time_domain_strain = bilby.core.utils.infft(
            frequency_domain_strain, self.sampling_frequency
        )
        self.assertTrue(np.allclose(time_domain_strain, new_time_domain_strain))


class TestInferParameters(unittest.TestCase):
    def setUp(self):
        def source_function(freqs, a, b, *args, **kwargs):
            return None

        class TestClass:
            def test_method(self, a, b, *args, **kwargs):
                pass

        class TestClass2:
            def test_method(self, freqs, a, b, *args, **kwargs):
                pass

        self.source1 = source_function
        test_obj = TestClass()
        self.source2 = test_obj.test_method
        test_obj2 = TestClass2()
        self.source3 = test_obj2.test_method

    def tearDown(self):
        del self.source1
        del self.source2

    def test_args_kwargs_handling(self):
        expected = ["a", "b"]
        actual = utils.infer_parameters_from_function(self.source1)
        self.assertListEqual(expected, actual)

    def test_self_handling(self):
        expected = ["a", "b"]
        actual = utils.infer_args_from_method(self.source2)
        self.assertListEqual(expected, actual)

    def test_self_handling_method_as_function(self):
        expected = ["a", "b"]
        actual = utils.infer_parameters_from_function(self.source3)
        self.assertListEqual(expected, actual)


class TestTimeAndFrequencyArrays(unittest.TestCase):
    def setUp(self):
        self.start_time = 1.3
        self.sampling_frequency = 5
        self.duration = 1.6
        self.frequency_array = utils.create_frequency_series(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.time_array = utils.create_time_series(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            starting_time=self.start_time,
        )

    def tearDown(self):
        del self.start_time
        del self.sampling_frequency
        del self.duration
        del self.frequency_array
        del self.time_array

    def test_create_time_array(self):
        expected_time_array = np.array([1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7])
        time_array = utils.create_time_series(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            starting_time=self.start_time,
        )
        self.assertTrue(np.allclose(expected_time_array, time_array))

    def test_create_frequency_array(self):
        expected_frequency_array = np.array([0.0, 0.625, 1.25, 1.875, 2.5])
        frequency_array = utils.create_frequency_series(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.assertTrue(np.allclose(expected_frequency_array, frequency_array))

    def test_get_sampling_frequency_from_time_array(self):
        (
            new_sampling_freq,
            _,
        ) = utils.get_sampling_frequency_and_duration_from_time_array(self.time_array)
        self.assertEqual(self.sampling_frequency, new_sampling_freq)

    def test_get_sampling_frequency_from_time_array_unequally_sampled(self):
        self.time_array[-1] += 0.0001
        with self.assertRaises(ValueError):
            _, _ = utils.get_sampling_frequency_and_duration_from_time_array(
                self.time_array
            )

    def test_get_duration_from_time_array(self):
        _, new_duration = utils.get_sampling_frequency_and_duration_from_time_array(
            self.time_array
        )
        self.assertEqual(self.duration, new_duration)

    def test_get_start_time_from_time_array(self):
        new_start_time = self.time_array[0]
        self.assertEqual(self.start_time, new_start_time)

    def test_get_sampling_frequency_from_frequency_array(self):
        (
            new_sampling_freq,
            _,
        ) = utils.get_sampling_frequency_and_duration_from_frequency_array(
            self.frequency_array
        )
        self.assertEqual(self.sampling_frequency, new_sampling_freq)

    def test_get_sampling_frequency_from_frequency_array_unequally_sampled(self):
        self.frequency_array[-1] += 0.0001
        with self.assertRaises(ValueError):
            _, _ = utils.get_sampling_frequency_and_duration_from_frequency_array(
                self.frequency_array
            )

    def test_get_duration_from_frequency_array(self):
        (
            _,
            new_duration,
        ) = utils.get_sampling_frequency_and_duration_from_frequency_array(
            self.frequency_array
        )
        self.assertEqual(self.duration, new_duration)

    def test_consistency_time_array_to_time_array(self):
        (
            new_sampling_frequency,
            new_duration,
        ) = utils.get_sampling_frequency_and_duration_from_time_array(self.time_array)
        new_start_time = self.time_array[0]
        new_time_array = utils.create_time_series(
            sampling_frequency=new_sampling_frequency,
            duration=new_duration,
            starting_time=new_start_time,
        )
        self.assertTrue(np.allclose(self.time_array, new_time_array))

    def test_consistency_frequency_array_to_frequency_array(self):
        (
            new_sampling_frequency,
            new_duration,
        ) = utils.get_sampling_frequency_and_duration_from_frequency_array(
            self.frequency_array
        )
        new_frequency_array = utils.create_frequency_series(
            sampling_frequency=new_sampling_frequency, duration=new_duration
        )
        self.assertTrue(np.allclose(self.frequency_array, new_frequency_array))

    def test_illegal_sampling_frequency_and_duration(self):
        with self.assertRaises(utils.IllegalDurationAndSamplingFrequencyException):
            _ = utils.create_time_series(
                sampling_frequency=7.7, duration=1.3, starting_time=0
            )


class TestReflect(unittest.TestCase):
    def test_in_range(self):
        xprime = np.array([0.1, 0.5, 0.9])
        x = np.array([0.1, 0.5, 0.9])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)

    def test_in_one_to_two(self):
        xprime = np.array([1.1, 1.5, 1.9])
        x = np.array([0.9, 0.5, 0.1])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)

    def test_in_two_to_three(self):
        xprime = np.array([2.1, 2.5, 2.9])
        x = np.array([0.1, 0.5, 0.9])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)

    def test_in_minus_one_to_zero(self):
        xprime = np.array([-0.9, -0.5, -0.1])
        x = np.array([0.9, 0.5, 0.1])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)

    def test_in_minus_two_to_minus_one(self):
        xprime = np.array([-1.9, -1.5, -1.1])
        x = np.array([0.1, 0.5, 0.9])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)


class TestLatexPlotFormat(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 1)
        self.y = np.sin(self.x)
        self.filename = "test_plot.png"

    def tearDown(self):
        if os.path.isfile(self.filename):
            os.remove(self.filename)

    def test_default(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)
        plot()
        self.assertTrue(os.path.isfile(self.filename))

    def test_mathedefault_one(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)
        plot(BILBY_MATHDEFAULT=1)
        self.assertTrue(os.path.isfile(self.filename))

    def test_mathedefault_zero(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)
        plot(BILBY_MATHDEFAULT=0)
        self.assertTrue(os.path.isfile(self.filename))

    def test_matplotlib_style(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)

        plot(BILBY_STYLE="fivethirtyeight")
        self.assertTrue(os.path.isfile(self.filename))

    def test_user_style(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)

        plot(BILBY_STYLE="test/test.mplstyle")
        self.assertTrue(os.path.isfile(self.filename))


if __name__ == "__main__":
    unittest.main()
