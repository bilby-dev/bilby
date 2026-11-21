"""
Utilities for time-domain gravitational-wave inference: products with the
inverse of a Toeplitz noise covariance matrix.
"""

import numpy as np
from scipy.fft import next_fast_len
from scipy.linalg import solve_toeplitz


def gohberg_semencul_vectors(acf):
    """The Gohberg-Semencul vectors of the inverse of the symmetric Toeplitz matrix
    with first row `acf` (Gohberg & Semencul 1972), computed with the Levinson
    recursion

    Parameters
    ==========
    acf: array_like
        The first row of the covariance matrix

    Returns
    =======
    x, y: array_like
        The first column of the inverse matrix, and the same reversed
    """
    acf = np.asarray(acf, dtype=float)
    unit_vector = np.zeros(len(acf))
    unit_vector[0] = 1.0
    x = solve_toeplitz((acf, acf), unit_vector)
    return x, x[::-1]


def gohberg_semencul_product(x, y, vector):
    """The product of the inverse of a symmetric Toeplitz matrix with a vector,
    computed from its Gohberg-Semencul vectors with FFTs, without forming the
    inverse matrix (Gohberg & Semencul 1972)

    Parameters
    ==========
    x, y: array_like
        The Gohberg-Semencul vectors, from :code:`gohberg_semencul_vectors`
    vector: array_like
        The vector, of the same length as `x`

    Returns
    =======
    array_like: The product of the inverse matrix with `vector`
    """
    x = np.asarray(x, dtype=float)
    number_of_samples = len(x)
    fft_length = next_fast_len(2 * number_of_samples - 1, real=True)
    x_fft = np.fft.rfft(x, n=fft_length)
    zy_fft = np.fft.rfft(
        np.concatenate(([0.0], np.asarray(y, dtype=float)[:-1])), n=fft_length
    )

    def lower(w_fft, vector_fft):
        """The product of the lower-triangular Toeplitz matrix with first column w and v, from their FFTs"""
        return np.fft.irfft(w_fft * vector_fft, n=fft_length)[:number_of_samples]

    # L(w)^T v = reverse(L(w) reverse(v))
    reversed_fft = np.fft.rfft(np.asarray(vector, dtype=float)[::-1], n=fft_length)
    first = lower(x_fft, np.fft.rfft(lower(x_fft, reversed_fft)[::-1], n=fft_length))
    second = lower(zy_fft, np.fft.rfft(lower(zy_fft, reversed_fft)[::-1], n=fft_length))
    return (first - second) / x[0]
