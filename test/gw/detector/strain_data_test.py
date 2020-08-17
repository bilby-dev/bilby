import unittest

import mock
import numpy as np
import scipy.signal
from mock import patch

import bilby


class TestInterferometerStrainData(unittest.TestCase):
    def setUp(self):
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        self.ifosd = bilby.gw.detector.InterferometerStrainData(
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
        )

    def tearDown(self):
        del self.minimum_frequency
        del self.maximum_frequency
        del self.ifosd

    def test_frequency_mask(self):
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        with mock.patch("bilby.core.utils.create_frequency_series") as m:
            m.return_value = np.array([5, 15, 25])
            self.ifosd.set_from_frequency_domain_strain(
                frequency_domain_strain=np.array([0, 1, 2]),
                frequency_array=np.array([5, 15, 25]),
            )
            self.assertTrue(
                np.array_equal(self.ifosd.frequency_mask, [False, True, False])
            )

    def test_set_data_fails(self):
        with mock.patch("bilby.core.utils.create_frequency_series") as m:
            m.return_value = [1, 2, 3]
            with self.assertRaises(ValueError):
                self.ifosd.set_from_frequency_domain_strain(
                    frequency_domain_strain=np.array([0, 1, 2])
                )

    def test_set_data_fails_too_much(self):
        with mock.patch("bilby.core.utils.create_frequency_series") as m:
            m.return_value = [1, 2, 3]
            with self.assertRaises(ValueError):
                self.ifosd.set_from_frequency_domain_strain(
                    frequency_domain_strain=np.array([0, 1, 2]),
                    frequency_array=np.array([1, 2, 3]),
                    duration=3,
                    sampling_frequency=1,
                )

    def test_start_time_init(self):
        with mock.patch("bilby.core.utils.create_frequency_series") as m:
            m.return_value = [1, 2, 3]
            duration = 3
            sampling_frequency = 1
            self.ifosd.set_from_frequency_domain_strain(
                frequency_domain_strain=np.array([0, 1, 2]),
                duration=duration,
                sampling_frequency=sampling_frequency,
            )
            self.assertTrue(self.ifosd.start_time == 0)

    def test_start_time_set(self):
        with mock.patch("bilby.core.utils.create_frequency_series") as m:
            m.return_value = [1, 2, 3]
            duration = 3
            sampling_frequency = 1
            self.ifosd.set_from_frequency_domain_strain(
                frequency_domain_strain=np.array([0, 1, 2]),
                duration=duration,
                sampling_frequency=sampling_frequency,
                start_time=10,
            )
            self.assertTrue(self.ifosd.start_time == 10)

    def test_time_array_frequency_array_consistency(self):
        duration = 1
        sampling_frequency = 10
        time_array = bilby.core.utils.create_time_series(
            sampling_frequency=sampling_frequency, duration=duration
        )
        time_domain_strain = np.random.normal(
            0, duration - 1 / sampling_frequency, len(time_array)
        )
        self.ifosd.roll_off = 0
        self.ifosd.set_from_time_domain_strain(
            time_domain_strain=time_domain_strain,
            duration=duration,
            sampling_frequency=sampling_frequency,
        )

        frequency_domain_strain, freqs = bilby.core.utils.nfft(
            time_domain_strain, sampling_frequency
        )

        self.assertTrue(
            np.all(
                self.ifosd.frequency_domain_strain
                == frequency_domain_strain * self.ifosd.frequency_mask
            )
        )

    def test_time_within_data_before(self):
        self.ifosd.start_time = 3
        self.ifosd.duration = 2
        self.assertFalse(self.ifosd.time_within_data(2))

    def test_time_within_data_during(self):
        self.ifosd.start_time = 3
        self.ifosd.duration = 2
        self.assertTrue(self.ifosd.time_within_data(3))
        self.assertTrue(self.ifosd.time_within_data(4))
        self.assertTrue(self.ifosd.time_within_data(5))

    def test_time_within_data_after(self):
        self.ifosd.start_time = 3
        self.ifosd.duration = 2
        self.assertFalse(self.ifosd.time_within_data(6))

    def test_time_domain_window_no_roll_off_no_alpha(self):
        self.ifosd._time_domain_strain = np.array([3])
        self.ifosd.duration = 5
        self.ifosd.roll_off = 2
        expected_window = scipy.signal.windows.tukey(
            len(self.ifosd._time_domain_strain), alpha=self.ifosd.alpha
        )
        self.assertEqual(expected_window, self.ifosd.time_domain_window())
        self.assertEqual(np.mean(expected_window ** 2), self.ifosd.window_factor)

    def test_time_domain_window_sets_roll_off_directly(self):
        self.ifosd._time_domain_strain = np.array([3])
        self.ifosd.duration = 5
        self.ifosd.roll_off = 2
        expected_roll_off = 6
        self.ifosd.time_domain_window(roll_off=expected_roll_off)
        self.assertEqual(expected_roll_off, self.ifosd.roll_off)

    def test_time_domain_window_sets_roll_off_indirectly(self):
        self.ifosd._time_domain_strain = np.array([3])
        self.ifosd.duration = 5
        self.ifosd.roll_off = 2
        alpha = 4
        expected_roll_off = alpha * self.ifosd.duration / 2
        self.ifosd.time_domain_window(alpha=alpha)
        self.assertEqual(expected_roll_off, self.ifosd.roll_off)

    def test_time_domain_strain_when_set(self):
        expected_strain = 5
        self.ifosd._time_domain_strain = expected_strain
        self.assertEqual(expected_strain, self.ifosd.time_domain_strain)

    @patch("bilby.core.utils.infft")
    def test_time_domain_strain_from_frequency_domain_strain(self, m):
        m.return_value = 5
        self.ifosd.sampling_frequency = 200
        self.ifosd.duration = 4
        self.ifosd.sampling_frequency = 123
        self.ifosd.frequency_domain_strain = self.ifosd.frequency_array
        self.assertEqual(m.return_value, self.ifosd.time_domain_strain)

    def test_time_domain_strain_not_set(self):
        self.ifosd._time_domain_strain = None
        self.ifosd._frequency_domain_strain = None
        with self.assertRaises(ValueError):
            _ = self.ifosd.time_domain_strain

    def test_frequency_domain_strain_when_set(self):
        self.ifosd.sampling_frequency = 200
        self.ifosd.duration = 4
        expected_strain = self.ifosd.frequency_array * self.ifosd.frequency_mask
        self.ifosd._frequency_domain_strain = expected_strain
        self.assertTrue(
            np.array_equal(expected_strain, self.ifosd.frequency_domain_strain)
        )

    @patch("bilby.core.utils.nfft")
    def test_frequency_domain_strain_from_frequency_domain_strain(self, m):
        self.ifosd.start_time = 0
        self.ifosd.duration = 4
        self.ifosd.sampling_frequency = 200
        m.return_value = self.ifosd.frequency_array, self.ifosd.frequency_array
        self.ifosd._time_domain_strain = self.ifosd.time_array
        self.assertTrue(
            np.array_equal(
                self.ifosd.frequency_array * self.ifosd.frequency_mask,
                self.ifosd.frequency_domain_strain,
            )
        )

    def test_frequency_domain_strain_not_set(self):
        self.ifosd._time_domain_strain = None
        self.ifosd._frequency_domain_strain = None
        with self.assertRaises(ValueError):
            _ = self.ifosd.frequency_domain_strain

    def test_set_frequency_domain_strain(self):
        self.ifosd.duration = 4
        self.ifosd.sampling_frequency = 200
        self.ifosd.frequency_domain_strain = np.ones(len(self.ifosd.frequency_array))
        self.assertTrue(
            np.array_equal(
                np.ones(len(self.ifosd.frequency_array)),
                self.ifosd._frequency_domain_strain,
            )
        )

    def test_set_frequency_domain_strain_wrong_length(self):
        self.ifosd.duration = 4
        self.ifosd.sampling_frequency = 200
        with self.assertRaises(ValueError):
            self.ifosd.frequency_domain_strain = np.array([1])


class TestInterferometerStrainDataEquals(unittest.TestCase):
    def setUp(self):
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        self.roll_off = 0.2
        self.sampling_frequency = 100
        self.duration = 2
        self.frequency_array = bilby.utils.create_frequency_series(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.strain = self.frequency_array
        self.ifosd_1 = bilby.gw.detector.InterferometerStrainData(
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            roll_off=self.roll_off,
        )
        self.ifosd_2 = bilby.gw.detector.InterferometerStrainData(
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            roll_off=self.roll_off,
        )
        self.ifosd_1.set_from_frequency_domain_strain(
            frequency_domain_strain=self.strain, frequency_array=self.frequency_array
        )
        self.ifosd_2.set_from_frequency_domain_strain(
            frequency_domain_strain=self.strain, frequency_array=self.frequency_array
        )

    def tearDown(self):
        del self.minimum_frequency
        del self.maximum_frequency
        del self.roll_off
        del self.sampling_frequency
        del self.duration
        del self.frequency_array
        del self.strain
        del self.ifosd_1
        del self.ifosd_2

    def test_eq_true(self):
        self.assertEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_minimum_frequency(self):
        self.ifosd_1.minimum_frequency -= 1
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_maximum_frequency(self):
        self.ifosd_1.maximum_frequency -= 1
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_roll_off(self):
        self.ifosd_1.roll_off -= 0.1
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_window_factor(self):
        self.ifosd_1.roll_off -= 0.1
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_sampling_frequency(self):
        self.ifosd_1.sampling_frequency *= 2
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_sampling_duration(self):
        self.ifosd_1.duration *= 2
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_start_time(self):
        self.ifosd_1.start_time -= 0.1
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_frequency_array(self):
        new_frequency_array = bilby.utils.create_frequency_series(
            sampling_frequency=self.sampling_frequency / 2, duration=self.duration * 2
        )
        self.ifosd_1.frequency_array = new_frequency_array
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_frequency_domain_strain(self):
        new_strain = bilby.utils.create_frequency_series(
            sampling_frequency=self.sampling_frequency / 2, duration=self.duration * 2
        )
        self.ifosd_1._frequency_domain_strain = new_strain
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_time_array(self):
        new_time_array = bilby.utils.create_time_series(
            sampling_frequency=self.sampling_frequency / 2, duration=self.duration * 2
        )
        self.ifosd_1.time_array = new_time_array
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)

    def test_eq_different_time_domain_strain(self):
        new_strain = bilby.utils.create_time_series(
            sampling_frequency=self.sampling_frequency / 2, duration=self.duration * 2
        )
        self.ifosd_1._time_domain_strain = new_strain
        self.assertNotEqual(self.ifosd_1, self.ifosd_2)


if __name__ == "__main__":
    unittest.main()
