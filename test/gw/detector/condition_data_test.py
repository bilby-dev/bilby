import unittest

import numpy as np
import pytest

import bilby
from bilby.gw.detector.condition_data import (
    butterworth_filter, condition_strain_data, filter_amplitude_response)


class TestButterworthFilter(unittest.TestCase):
    def test_no_frequencies_gives_no_filter(self):
        self.assertIsNone(butterworth_filter(4096))

    def test_filter_types(self):
        from scipy.signal import butter
        for kwargs, expected in [
                (dict(highpass_frequency=10), butter(4, 10 / 2048, btype="highpass")),
                (dict(lowpass_frequency=400), butter(4, 400 / 2048, btype="lowpass")),
                (dict(highpass_frequency=10, lowpass_frequency=400),
                 butter(4, (10 / 2048, 400 / 2048), btype="bandpass"))]:
            for coefficients, expected_coefficients in zip(butterworth_filter(4096, **kwargs), expected):
                np.testing.assert_array_equal(coefficients, expected_coefficients)

    def test_amplitude_response(self):
        frequencies = np.array([5.0, 100.0, 1500.0])
        np.testing.assert_array_equal(filter_amplitude_response(frequencies, 4096), 1)
        response = filter_amplitude_response(frequencies, 4096, highpass_frequency=10, lowpass_frequency=400)
        self.assertLess(response[0], 1e-2)
        self.assertAlmostEqual(response[1], 1, places=3)
        self.assertLess(response[2], 1e-2)


class TestConditionStrainData(unittest.TestCase):
    def setUp(self):
        self.gps_start = 1126259640.0
        self.sampling_frequency = 4096.0
        self.times = self.gps_start + np.arange(int(16 * self.sampling_frequency)) / self.sampling_frequency
        self.strain_data = bilby.gw.detector.InterferometerStrainData()
        self._set_data(self._cosine(100) + self._cosine(1500))

    def tearDown(self):
        del self.strain_data
        del self.times

    def _cosine(self, frequency, times=None):
        times = self.times if times is None else times
        return np.cos(2 * np.pi * frequency * (times - self.gps_start))

    def _set_data(self, strain, times=None):
        times = self.times if times is None else times
        self.strain_data.set_from_time_domain_strain(
            strain, sampling_frequency=self.sampling_frequency, duration=len(times) / self.sampling_frequency,
            start_time=self.gps_start)

    def test_without_downsampling_the_data_are_only_trimmed(self):
        condition_strain_data(self.strain_data, self.sampling_frequency, remove_mean=False)
        central = self.times[len(self.times) // 4:3 * len(self.times) // 4]
        np.testing.assert_array_equal(self.strain_data.time_domain_strain,
                                      self._cosine(100, central) + self._cosine(1500, central))
        self.assertEqual(self.strain_data.start_time, central[0])

    def test_downsampling_removes_frequencies_above_the_new_nyquist_frequency(self):
        condition_strain_data(self.strain_data, 512.0, preserve_time=self.gps_start + 8, remove_mean=False)
        self.assertEqual(self.strain_data.sampling_frequency, 512.0)
        np.testing.assert_allclose(self.strain_data.time_domain_strain,
                                   self._cosine(100, self.strain_data.time_array), atol=1e-3)

    def test_preserve_time_is_kept(self):
        preserve_time = self.gps_start + 6 + 3 / self.sampling_frequency
        condition_strain_data(self.strain_data, 512.0, preserve_time=preserve_time)
        self.assertLess(np.min(abs(self.strain_data.time_array - preserve_time)), 1e-6)

    def test_bandpass_and_lowpass_filters(self):
        for kwargs, strain in [
                (dict(highpass_frequency=10, lowpass_frequency=400),
                 self._cosine(100) + self._cosine(5) + self._cosine(1500)),
                (dict(lowpass_frequency=400), self._cosine(100) + self._cosine(1500))]:
            self._set_data(strain)
            condition_strain_data(self.strain_data, self.sampling_frequency, remove_mean=False, **kwargs)
            np.testing.assert_allclose(self.strain_data.time_domain_strain,
                                       self._cosine(100, self.strain_data.time_array), atol=1e-2)

    def test_remove_mean(self):
        self._set_data(self._cosine(100) + 3)
        condition_strain_data(self.strain_data, 512.0)
        self.assertAlmostEqual(np.mean(self.strain_data.time_domain_strain), 0, places=12)

    def test_rolled_samples_are_removed_without_trim(self):
        number_of_samples = len(self.times) + 5  # not a multiple of the downsampling factor
        times = self.gps_start + np.arange(number_of_samples) / self.sampling_frequency
        self._set_data(self._cosine(100, times), times)
        preserve_time = times[-6]  # rolled by 7 samples, one of which is kept by the downsampling
        condition_strain_data(self.strain_data, 512.0, preserve_time=preserve_time, trim=0, remove_mean=False)
        self.assertLess(np.min(abs(self.strain_data.time_array - preserve_time)), 1e-6)
        self.assertGreater(self.strain_data.start_time, self.gps_start)
        self.assertLessEqual(self.strain_data.time_array[-1], times[-1])

    def test_sampling_frequency_must_divide_the_original(self):
        with self.assertRaises(ValueError):
            condition_strain_data(self.strain_data, 3000.0)

    def test_preserve_time_outside_the_data_raises(self):
        with self.assertRaises(ValueError):
            condition_strain_data(self.strain_data, 512.0, preserve_time=self.gps_start - 1)

    def test_matches_ringdown(self):
        ringdown = pytest.importorskip("ringdown")
        strain = self._cosine(100) + self._cosine(5)
        preserve_time = self.gps_start + 6 + 3 / self.sampling_frequency
        expected = ringdown.Data(strain, index=self.times, ifo="H1").condition(
            t0=preserve_time, ds=8, f_min=10, trim=0.25)
        self._set_data(strain)
        condition_strain_data(self.strain_data, 512.0, preserve_time=preserve_time, highpass_frequency=10)
        np.testing.assert_allclose(self.strain_data.time_array, expected.index.values, atol=1e-6)
        np.testing.assert_allclose(self.strain_data.time_domain_strain, expected.values, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
