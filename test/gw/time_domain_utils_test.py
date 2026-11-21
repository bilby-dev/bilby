import unittest

import numpy as np
import scipy.linalg as sl

from bilby.gw.detector import PowerSpectralDensity
from bilby.gw.time_domain_utils import (
    gohberg_semencul_product,
    gohberg_semencul_vectors,
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


if __name__ == "__main__":
    unittest.main()
