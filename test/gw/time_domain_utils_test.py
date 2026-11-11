import unittest

import numpy as np
import scipy.linalg as sl

from bilby.gw.detector import PowerSpectralDensity
from bilby.gw.time_domain_utils import (
    fractional_time_shift,
    gohberg_semencul_product,
    gohberg_semencul_vectors,
    place_time_domain_signal,
    times_from_merger,
)


class TestGohbergSemencul(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_vectors_are_first_column_of_inverse(self):
        k = np.arange(64)
        acf = np.exp(-((k / 5.0) ** 2)) + 1e-2 * (k == 0)
        x, y = gohberg_semencul_vectors(acf)
        unit_vector = np.zeros(64)
        unit_vector[0] = 1
        np.testing.assert_allclose(sl.toeplitz(acf) @ x, unit_vector, atol=1e-10)
        np.testing.assert_array_equal(y, x[::-1])

    def test_product_matches_dense_solve(self):
        k = np.arange(256)
        acf = np.exp(-((k / 20.0) ** 2)) * np.cos(0.3 * k) + 0.05 * np.exp(-k / 5.0)
        acf[0] += 1e-3
        vector = self.rng.normal(size=256)
        expected = sl.solve(sl.toeplitz(acf), vector, assume_a="pos")
        result = gohberg_semencul_product(*gohberg_semencul_vectors(acf), vector)
        np.testing.assert_allclose(
            result, expected, rtol=1e-9, atol=1e-9 * np.max(abs(expected))
        )

    def test_product_matches_cholesky_for_aligo_autocovariance(self):
        """The aLIGO ACF with the default fill is ill-conditioned (cond ~ 1e5)"""
        psd = PowerSpectralDensity.from_aligo()
        acf, _, _ = psd.get_autocovariance_array(1024, 1024.0, 20.0, 512.0)
        vector = self.rng.normal(size=1024) * 1e-21
        expected = sl.cho_solve(sl.cho_factor(sl.toeplitz(acf)), vector)
        result = gohberg_semencul_product(*gohberg_semencul_vectors(acf), vector)
        self.assertLess(
            abs(vector @ result - vector @ expected) / (vector @ expected), 1e-8
        )


class TestFractionalTimeShift(unittest.TestCase):
    def test_zero_shift_pads_the_signal(self):
        signal = np.arange(10.0)
        shifted = fractional_time_shift(signal, 0.0, pad=4)
        np.testing.assert_allclose(shifted[4:14], signal, atol=1e-12)
        np.testing.assert_allclose(shifted[:4], 0, atol=1e-12)
        self.assertEqual(len(shifted), 18)

    def test_band_limited_signal_is_delayed(self):
        n = np.arange(400)
        burst = lambda m: np.exp(-(((m - 200) / 30) ** 2)) * np.cos(0.2 * m)  # noqa: E731
        shifted = fractional_time_shift(burst(n), 0.3, pad=32)
        np.testing.assert_allclose(shifted[32:432], burst(n - 0.3), atol=1e-10)


class TestPlaceTimeDomainSignal(unittest.TestCase):
    def setUp(self):
        self.signal = np.arange(1.0, 11.0)

    def test_integer_shift_is_exact(self):
        placed = place_time_domain_signal(
            self.signal,
            merger_index=4,
            arrival_index=10,
            number_of_samples=20,
            placement="nearest",
        )
        expected = np.zeros(20)
        expected[6:16] = self.signal
        np.testing.assert_array_equal(placed, expected)

    def test_nearest_rounds_to_the_nearest_sample(self):
        for fraction, offset in [(0.3, 6), (-0.3, 6), (0.7, 7)]:
            placed = place_time_domain_signal(
                self.signal,
                merger_index=4,
                arrival_index=10 + fraction,
                number_of_samples=20,
                placement="nearest",
            )
            expected = np.zeros(20)
            expected[offset : offset + 10] = self.signal
            np.testing.assert_array_equal(placed, expected)

    def test_signal_partly_outside_the_data_is_dropped(self):
        placed = place_time_domain_signal(
            self.signal,
            merger_index=4,
            arrival_index=1,
            number_of_samples=5,
            placement="nearest",
        )
        np.testing.assert_array_equal(placed, self.signal[3:8])

    def test_signal_entirely_outside_the_data_gives_zeros(self):
        placed = place_time_domain_signal(
            self.signal,
            merger_index=0,
            arrival_index=100,
            number_of_samples=5,
            placement="nearest",
        )
        np.testing.assert_array_equal(placed, np.zeros(5))

    def test_fd_shift_matches_analytic_delay(self):
        n = np.arange(400)
        burst = lambda m: np.exp(-(((m - 200) / 30) ** 2)) * np.cos(0.2 * m)  # noqa: E731
        placed = place_time_domain_signal(
            burst(n),
            merger_index=200,
            arrival_index=300.3,
            number_of_samples=600,
            placement="fd_shift",
        )
        np.testing.assert_allclose(placed, burst(np.arange(600) - 100.3), atol=1e-10)

    def test_unknown_placement_raises(self):
        with self.assertRaises(ValueError):
            place_time_domain_signal(self.signal, 0, 0, 5, placement="model_time")


class TestTimesFromMerger(unittest.TestCase):
    def test_values(self):
        times = times_from_merger(
            start_time=1e9,
            number_of_samples=4,
            sampling_frequency=4.0,
            geocent_time=1e9 + 0.5,
            time_delay=0.01,
        )
        np.testing.assert_allclose(times, [-0.51, -0.26, -0.01, 0.24], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
