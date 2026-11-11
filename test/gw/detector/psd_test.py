import os
import unittest
from unittest import mock

import numpy as np

import bilby
from bilby.gw.detector.psd import power_spectral_density_to_autocovariance


class TestPowerSpectralDensityWithoutFiles(unittest.TestCase):
    def setUp(self):
        self.frequency_array = np.array([1.0, 2.0, 3.0])
        self.psd_array = np.array([16.0, 25.0, 36.0])
        self.asd_array = np.array([4.0, 5.0, 6.0])

    def tearDown(self):
        del self.frequency_array
        del self.psd_array
        del self.asd_array

    def test_init_with_asd_array(self):
        psd = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.frequency_array, asd_array=self.asd_array
        )
        self.assertTrue(np.array_equal(self.frequency_array, psd.frequency_array))
        self.assertTrue(np.array_equal(self.asd_array, psd.asd_array))
        self.assertTrue(np.array_equal(self.psd_array, psd.psd_array))

    def test_init_with_psd_array(self):
        psd = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.frequency_array, psd_array=self.psd_array
        )
        self.assertTrue(np.array_equal(self.frequency_array, psd.frequency_array))
        self.assertTrue(np.array_equal(self.asd_array, psd.asd_array))
        self.assertTrue(np.array_equal(self.psd_array, psd.psd_array))

    def test_setting_asd_array_after_init(self):
        psd = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.frequency_array
        )
        psd.asd_array = self.asd_array
        self.assertTrue(np.array_equal(self.frequency_array, psd.frequency_array))
        self.assertTrue(np.array_equal(self.asd_array, psd.asd_array))
        self.assertTrue(np.array_equal(self.psd_array, psd.psd_array))

    def test_setting_psd_array_after_init(self):
        psd = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.frequency_array
        )
        psd.psd_array = self.psd_array
        self.assertTrue(np.array_equal(self.frequency_array, psd.frequency_array))
        self.assertTrue(np.array_equal(self.asd_array, psd.asd_array))
        self.assertTrue(np.array_equal(self.psd_array, psd.psd_array))

    def test_power_spectral_density_interpolated_from_asd_array(self):
        expected = np.array([25.0])
        psd = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.frequency_array, asd_array=self.asd_array
        )
        self.assertEqual(expected, psd.power_spectral_density_interpolated(2))

    def test_power_spectral_density_interpolated_from_psd_array(self):
        expected = np.array([25.0])
        psd = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.frequency_array, psd_array=self.psd_array
        )
        self.assertEqual(expected, psd.power_spectral_density_interpolated(2))

    def test_from_amplitude_spectral_density_array(self):
        actual = bilby.gw.detector.PowerSpectralDensity.from_amplitude_spectral_density_array(
            frequency_array=self.frequency_array, asd_array=self.asd_array
        )
        self.assertTrue(np.array_equal(self.psd_array, actual.psd_array))
        self.assertTrue(np.array_equal(self.asd_array, actual.asd_array))

    def test_from_power_spectral_density_array(self):
        actual = (
            bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_array(
                frequency_array=self.frequency_array, psd_array=self.psd_array
            )
        )
        self.assertTrue(np.array_equal(self.psd_array, actual.psd_array))
        self.assertTrue(np.array_equal(self.asd_array, actual.asd_array))

    def test_repr(self):
        psd = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.frequency_array, psd_array=self.psd_array
        )
        expected = "PowerSpectralDensity(frequency_array={}, psd_array={}, asd_array={})".format(
            self.frequency_array, self.psd_array, self.asd_array
        )
        self.assertEqual(expected, repr(psd))


class TestPowerSpectralDensityWithFiles(unittest.TestCase):
    def setUp(self):
        self.dir = os.path.join(os.path.dirname(__file__), "noise_curves")
        os.mkdir(self.dir)
        self.asd_file = os.path.join(
            os.path.dirname(__file__), "noise_curves", "asd_test_file.txt"
        )
        self.psd_file = os.path.join(
            os.path.dirname(__file__), "noise_curves", "psd_test_file.txt"
        )
        with open(self.asd_file, "w") as f:
            f.write("1.\t1.0e-21\n2.\t2.0e-21\n3.\t3.0e-21")
        with open(self.psd_file, "w") as f:
            f.write("1.\t1.0e-42\n2.\t4.0e-42\n3.\t9.0e-42")
        self.frequency_array = np.array([1.0, 2.0, 3.0])
        self.asd_array = np.array([1.0e-21, 2.0e-21, 3.0e-21])
        self.psd_array = np.array([1.0e-42, 4.0e-42, 9.0e-42])

    def tearDown(self):
        os.remove(self.asd_file)
        os.remove(self.psd_file)
        os.rmdir(self.dir)
        del self.dir
        del self.asd_array
        del self.psd_array
        del self.asd_file
        del self.psd_file

    def test_init_with_psd_file(self):
        psd = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.frequency_array, psd_file=self.psd_file
        )
        self.assertEqual(self.psd_file, psd.psd_file)
        self.assertTrue(np.array_equal(self.psd_array, psd.psd_array))
        self.assertTrue(np.allclose(self.asd_array, psd.asd_array, atol=1e-30))

    def test_init_with_asd_file(self):
        psd = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.frequency_array, asd_file=self.asd_file
        )
        self.assertEqual(self.asd_file, psd.asd_file)
        self.assertTrue(np.allclose(self.psd_array, psd.psd_array, atol=1e-60))
        self.assertTrue(np.array_equal(self.asd_array, psd.asd_array))

    def test_setting_psd_array_after_init(self):
        psd = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.frequency_array
        )
        psd.psd_file = self.psd_file
        self.assertEqual(self.psd_file, psd.psd_file)
        self.assertTrue(np.array_equal(self.psd_array, psd.psd_array))
        self.assertTrue(np.allclose(self.asd_array, psd.asd_array, atol=1e-30))

    def test_init_with_asd_array_after_init(self):
        psd = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.frequency_array
        )
        psd.asd_file = self.asd_file
        self.assertEqual(self.asd_file, psd.asd_file)
        self.assertTrue(np.allclose(self.psd_array, psd.psd_array, atol=1e-60))
        self.assertTrue(np.array_equal(self.asd_array, psd.asd_array))

    def test_power_spectral_density_interpolated_from_asd_file(self):
        expected = np.array([4.0e-42])
        psd = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.frequency_array, asd_file=self.asd_file
        )
        self.assertTrue(
            np.allclose(
                expected, psd.power_spectral_density_interpolated(2), atol=1e-60
            )
        )

    def test_power_spectral_density_interpolated_from_psd_file(self):
        expected = np.array([4.0e-42])
        psd = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.frequency_array, psd_file=self.psd_file
        )
        self.assertAlmostEqual(expected, psd.power_spectral_density_interpolated(2))

    def test_from_amplitude_spectral_density_file(self):
        psd = (
            bilby.gw.detector.PowerSpectralDensity.from_amplitude_spectral_density_file(
                asd_file=self.asd_file
            )
        )
        self.assertEqual(self.asd_file, psd.asd_file)
        self.assertTrue(np.allclose(self.psd_array, psd.psd_array, atol=1e-60))
        self.assertTrue(np.array_equal(self.asd_array, psd.asd_array))

    def test_from_power_spectral_density_file(self):
        psd = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_file(
            psd_file=self.psd_file
        )
        self.assertEqual(self.psd_file, psd.psd_file)
        self.assertTrue(np.array_equal(self.psd_array, psd.psd_array))
        self.assertTrue(np.allclose(self.asd_array, psd.asd_array, atol=1e-30))

    def test_from_aligo(self):
        psd = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        expected_filename = "aLIGO_ZERO_DET_high_P_psd.txt"
        expected = bilby.gw.detector.PowerSpectralDensity(psd_file=expected_filename)
        actual_filename = psd.psd_file.split("/")[-1]
        self.assertEqual(expected_filename, actual_filename)
        self.assertTrue(np.allclose(expected.psd_array, psd.psd_array, atol=1e-60))
        self.assertTrue(np.array_equal(expected.asd_array, psd.asd_array))

    @mock.patch.object(bilby.gw.detector.psd.logger, "warning")
    def test_check_file_psd_file_set_to_asd_file(self, mock_warning):
        _ = bilby.gw.detector.PowerSpectralDensity(psd_file=self.asd_file)
        self.assertEqual(4, mock_warning.call_count)

    @mock.patch.object(bilby.gw.detector.psd.logger, "warning")
    def test_check_file_not_called_psd_file_set_to_psd_file(self, mock_warning):
        _ = bilby.gw.detector.PowerSpectralDensity(psd_file=self.psd_file)
        self.assertEqual(0, mock_warning.call_count)

    @mock.patch.object(bilby.gw.detector.psd.logger, "warning")
    def test_check_file_asd_file_set_to_psd_file(self, mock_warning):
        _ = bilby.gw.detector.PowerSpectralDensity(asd_file=self.psd_file)
        self.assertEqual(4, mock_warning.call_count)

    @mock.patch.object(bilby.gw.detector.psd.logger, "warning")
    def test_check_file_not_called_asd_file_set_to_asd_file(self, mock_warning):
        _ = bilby.gw.detector.PowerSpectralDensity(asd_file=self.asd_file)
        self.assertEqual(0, mock_warning.call_count)

    def test_from_frame_file(self):
        expected_frequency_array = np.array([1.0, 2.0, 3.0])
        expected_psd_array = np.array([16.0, 25.0, 36.0])
        with mock.patch(
            "bilby.gw.detector.InterferometerStrainData.set_from_frame_file"
        ) as _:
            with mock.patch(
                "bilby.gw.detector.InterferometerStrainData.create_power_spectral_density"
            ) as n:
                n.return_value = expected_frequency_array, expected_psd_array
                psd = bilby.gw.detector.PowerSpectralDensity.from_frame_file(
                    frame_file=self.asd_file, psd_start_time=0, psd_duration=4
                )
                self.assertTrue(
                    np.array_equal(expected_frequency_array, psd.frequency_array)
                )
                self.assertTrue(np.array_equal(expected_psd_array, psd.psd_array))

    def test_repr(self):
        psd = bilby.gw.detector.PowerSpectralDensity(psd_file=self.psd_file)
        expected = "PowerSpectralDensity(psd_file='{}', asd_file='{}')".format(
            self.psd_file, None
        )
        self.assertEqual(expected, repr(psd))


class TestPowerSpectralDensityEquals(unittest.TestCase):
    def setUp(self):
        self.psd_from_file_1 = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        self.psd_from_file_2 = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        self.frequency_array = np.linspace(1, 100)
        self.psd_array = np.linspace(1, 100)
        self.psd_from_array_1 = (
            bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_array(
                frequency_array=self.frequency_array, psd_array=self.psd_array
            )
        )
        self.psd_from_array_2 = (
            bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_array(
                frequency_array=self.frequency_array, psd_array=self.psd_array
            )
        )

    def tearDown(self):
        del self.psd_from_file_1
        del self.psd_from_file_2
        del self.frequency_array
        del self.psd_array
        del self.psd_from_array_1
        del self.psd_from_array_2

    def test_eq_true_from_array(self):
        self.assertEqual(self.psd_from_array_1, self.psd_from_array_2)

    def test_eq_true_from_file(self):
        self.assertEqual(self.psd_from_file_1, self.psd_from_file_2)

    def test_eq_false_different_psd_file_name(self):
        self.psd_from_file_1._psd_file = "some_other_name"
        self.assertNotEqual(self.psd_from_file_1, self.psd_from_file_2)

    def test_eq_false_different_asd_file_name(self):
        self.psd_from_file_1._psd_file = None
        self.psd_from_file_2._psd_file = None
        self.psd_from_file_1._asd_file = "some_name"
        self.psd_from_file_2._asd_file = "some_other_name"
        self.assertNotEqual(self.psd_from_file_1, self.psd_from_file_2)

    def test_eq_false_different_frequency_array(self):
        self.psd_from_file_1.frequency_array[0] = 0.5
        self.psd_from_array_1.frequency_array[0] = 0.5
        self.assertNotEqual(self.psd_from_file_1, self.psd_from_file_2)
        self.assertNotEqual(self.psd_from_array_1, self.psd_from_array_2)

    def test_eq_false_different_psd(self):
        self.psd_from_file_1.psd_array[0] = 0.53544321
        self.psd_from_array_1.psd_array[0] = 0.53544321
        self.assertNotEqual(self.psd_from_file_1, self.psd_from_file_2)
        self.assertNotEqual(self.psd_from_array_1, self.psd_from_array_2)

    def test_eq_false_different_asd(self):
        self.psd_from_file_1.asd_array[0] = 0.53544321
        self.psd_from_array_1.asd_array[0] = 0.53544321
        self.assertNotEqual(self.psd_from_file_1, self.psd_from_file_2)
        self.assertNotEqual(self.psd_from_array_1, self.psd_from_array_2)


class TestPowerSpectralDensityAutocovariance(unittest.TestCase):
    def setUp(self):
        self.frequency_array = np.arange(0, 512.25, 0.25)
        self.aligo = bilby.gw.detector.PowerSpectralDensity.from_aligo()

    def tearDown(self):
        del self.frequency_array
        del self.aligo

    def test_white_psd_gives_a_diagonal_autocovariance(self):
        """A white one-sided PSD S0 up to Nyquist gives the ACF [S0 fs / 2, 0, 0, ...]"""
        psd = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.frequency_array,
            psd_array=2 * np.ones_like(self.frequency_array),
        )
        acf, fill_value, fill_multiplier = psd.get_autocovariance_array(
            256, 1024.0, 0.0, 512.0
        )
        np.testing.assert_allclose(acf[0], 2 * 1024 / 2, rtol=1e-12)
        np.testing.assert_allclose(acf[1:], 0, atol=1e-12 * acf[0])
        self.assertEqual(fill_multiplier, 100)

    def test_fill_below_and_above_the_band(self):
        """The PSD below (above) the band is replaced by 100 x its maximum there"""
        psd_array = np.where(
            self.frequency_array < 20,
            1e6,
            np.where(self.frequency_array > 256, 1e-3, 1.0),
        )
        acf, fill_value, fill_multiplier = power_spectral_density_to_autocovariance(
            self.frequency_array, psd_array, 20.0, 256.0
        )
        self.assertEqual(fill_value, (1e8, 0.1))
        self.assertEqual(fill_multiplier, 100)
        filled = np.where(
            self.frequency_array < 20,
            1e8,
            np.where(self.frequency_array > 256, 0.1, 1.0),
        )
        np.testing.assert_allclose(acf, 0.5 * np.fft.irfft(filled) * 1024)

    def test_fill_multiplier(self):
        psd_array = 1 + self.frequency_array
        _, fill_value, _ = power_spectral_density_to_autocovariance(
            self.frequency_array, psd_array, 20.0, 100.0, fill_multiplier=10
        )
        self.assertEqual(
            fill_value,
            (10 * psd_array[self.frequency_array < 20].max(), 10 * psd_array[-1]),
        )

    def test_explicit_fill_value(self):
        psd_array = np.ones_like(self.frequency_array)
        for given, expected in [(1e4, (1e4, 1e4)), ((1e4, 1e5), (1e4, 1e5))]:
            _, fill_value, fill_multiplier = power_spectral_density_to_autocovariance(
                self.frequency_array, psd_array, 20.0, 100.0, fill_value=given
            )
            self.assertEqual((fill_value, fill_multiplier), (expected, None))

    def test_fill_value_below_the_psd_outside_the_band_warns(self):
        psd_array = np.where(self.frequency_array < 20, 1e6, 1.0)
        with self.assertLogs("bilby", level="WARNING") as logs:
            power_spectral_density_to_autocovariance(
                self.frequency_array, psd_array, 20.0, 512.0, fill_value=1e4
            )
        self.assertIn(
            "The fill value below the band (1.00e+04) is below the PSD there",
            "".join(logs.output),
        )

    def test_psd_not_defined_outside_the_band_warns(self):
        psd_array = np.where(self.frequency_array < 20, np.inf, 1.0)
        with self.assertLogs("bilby", level="WARNING") as logs:
            _, fill_value, _ = power_spectral_density_to_autocovariance(
                self.frequency_array, psd_array, 20.0, 512.0
            )
        self.assertEqual(fill_value[0], 100)
        self.assertIn("not defined below the band", "".join(logs.output))

    def test_psd_not_defined_in_part_of_the_band_warns(self):
        """The aLIGO PSD is tabulated from 9 Hz: a band from 5 Hz is the band from 9 Hz"""
        with self.assertLogs("bilby", level="WARNING") as logs:
            acf, _, _ = self.aligo.get_autocovariance_array(1024, 1024.0, 5.0, 512.0)
        self.assertIn("not defined at", "".join(logs.output))
        expected, _, _ = self.aligo.get_autocovariance_array(1024, 1024.0, 9.0, 512.0)
        np.testing.assert_array_equal(acf, expected)

    def test_psd_not_defined_in_the_band_raises(self):
        with self.assertRaises(ValueError):
            self.aligo.get_autocovariance_array(1024, 1024.0, 1.0, 5.0)

    def test_frequencies_not_starting_at_zero(self):
        """Missing frequencies below the first one are treated as zeros"""
        psd_array = np.ones_like(self.frequency_array)
        start = np.searchsorted(self.frequency_array, 10)
        expected = power_spectral_density_to_autocovariance(
            self.frequency_array,
            np.where(self.frequency_array < 10, 0, psd_array),
            20,
            512,
        )[0]
        np.testing.assert_array_equal(
            power_spectral_density_to_autocovariance(
                self.frequency_array[start:], psd_array[start:], 20, 512
            )[0],
            expected,
        )

    def test_sampling_frequency_different_from_the_psd_warns(self):
        with self.assertLogs("bilby", level="WARNING") as logs:
            power_spectral_density_to_autocovariance(
                self.frequency_array,
                np.ones_like(self.frequency_array),
                20,
                512,
                sampling_frequency=2048,
            )
        self.assertIn(
            "is not equal to the one computed from the PSD", "".join(logs.output)
        )

    def test_window_longer_than_the_autocovariance_raises(self):
        with self.assertRaises(ValueError):
            self.aligo.get_autocovariance_array(
                1024, 1024.0, 20.0, 512.0, duration_factor=0.5
            )

    def test_from_autocovariance_array(self):
        acf, _, _ = self.aligo.get_autocovariance_array(1024, 1024.0, 20.0, 512.0)
        precomputed = bilby.gw.detector.PowerSpectralDensity.from_autocovariance_array(
            acf, 1024
        )
        lags, fill_value, fill_multiplier = precomputed.get_autocovariance_array(
            512, 1024.0, 20.0, 512.0
        )
        np.testing.assert_array_equal(lags, acf[:512])
        self.assertEqual((fill_value, fill_multiplier), (None, None))

    def test_from_autocovariance_array_wrong_sampling_frequency_raises(self):
        psd = bilby.gw.detector.PowerSpectralDensity.from_autocovariance_array(
            np.ones(100), 1024
        )
        with self.assertRaises(ValueError):
            psd.get_autocovariance_array(20, 2048.0, 20.0, 512.0)

    def test_from_autocovariance_array_too_short_raises(self):
        psd = bilby.gw.detector.PowerSpectralDensity.from_autocovariance_array(
            np.ones(10), 1024
        )
        with self.assertRaises(ValueError):
            psd.get_autocovariance_array(20, 1024.0, 20.0, 512.0)


if __name__ == "__main__":
    unittest.main()
