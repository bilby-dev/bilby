import os
import unittest
from unittest import mock

import numpy as np

import bilby


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
        actual = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_array(
            frequency_array=self.frequency_array, psd_array=self.psd_array
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
        psd = bilby.gw.detector.PowerSpectralDensity.from_amplitude_spectral_density_file(
            asd_file=self.asd_file
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
        self.psd_from_array_1 = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_array(
            frequency_array=self.frequency_array, psd_array=self.psd_array
        )
        self.psd_from_array_2 = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_array(
            frequency_array=self.frequency_array, psd_array=self.psd_array
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


if __name__ == "__main__":
    unittest.main()
