"""
Utilities for time-domain gravitational-wave inference.

* Gohberg-Semencul representation of the inverse of a symmetric Toeplitz
  noise covariance matrix.
* Placement of a time-domain waveform, generated on its own uniform grid, into
  a segment of detector data.
"""
import numpy as np
from scipy.fft import next_fast_len
from scipy.linalg import solve_toeplitz


def gohberg_semencul_vectors(acf):
    """ Gohberg-Semencul generators of the inverse of the Toeplitz matrix
    built from :code:`acf`.

    Parameters
    ==========
    acf: array_like
        Autocovariance function at lags 0, 1, ..., N - 1 (first row of the
        N x N covariance matrix).

    Returns
    =======
    x, y: array_like
        x is the first column of C^-1 and y is x reversed.
    """
    acf = np.asarray(acf, dtype=float)
    unit_vector = np.zeros(len(acf))
    unit_vector[0] = 1.0
    x = solve_toeplitz((acf, acf), unit_vector)
    return x, x[::-1]


def gohberg_semencul_product(x, y, vector):
    """ Product :math:`C^{-1} v` of the inverse of a symmetric Toeplitz matrix
    with a vector, from its Gohberg-Semencul vectors, without forming
    :math:`C^{-1}`:

    .. math::

        C^{-1} v = \\frac{1}{x_0} \\left[ L(x) L(x)^T v - L(Z y) L(Z y)^T v \\right],

    where :math:`L(w)` is the lower-triangular Toeplitz matrix with first
    column :math:`w` and :math:`Z` shifts down by one sample. Each product
    with a triangular Toeplitz matrix is computed with FFTs of length
    :code:`next_fast_len(2 N - 1)`.

    Parameters
    ==========
    x, y: array_like
        Gohberg-Semencul vectors from :code:`gohberg_semencul_vectors`.
    vector: array_like
        The vector v, of the same length as x.

    Returns
    =======
    array_like: :math:`C^{-1} v`
    """
    x = np.asarray(x, dtype=float)
    number_of_samples = len(x)
    fft_length = next_fast_len(2 * number_of_samples - 1, real=True)
    x_fft = np.fft.rfft(x, n=fft_length)
    zy_fft = np.fft.rfft(np.concatenate(([0.0], np.asarray(y, dtype=float)[:-1])), n=fft_length)

    def lower(w_fft, vector_fft):
        """ L(w) v from the transforms of w and v. """
        return np.fft.irfft(w_fft * vector_fft, n=fft_length)[:number_of_samples]

    # L(w)^T v = reverse(L(w) reverse(v))
    reversed_fft = np.fft.rfft(np.asarray(vector, dtype=float)[::-1], n=fft_length)
    first = lower(x_fft, np.fft.rfft(lower(x_fft, reversed_fft)[::-1], n=fft_length))
    second = lower(zy_fft, np.fft.rfft(lower(zy_fft, reversed_fft)[::-1], n=fft_length))
    return (first - second) / x[0]


def fractional_time_shift(signal, fraction, pad=32):
    """ Delay a time series by a fraction of a sample with a Fourier phase
    shift, on a zero-padded copy so that nothing wraps around.

    Parameters
    ==========
    signal: array_like
        The time series.
    fraction: float
        Delay in samples (positive values move the signal later).
    pad: int
        Number of zero samples added before and after the signal.

    Returns
    =======
    shifted: array_like
        The delayed series, of length len(signal) + 2 * pad (sample k
        corresponds to sample k - pad of the input).
    """
    signal = np.asarray(signal, dtype=float)
    length = next_fast_len(len(signal) + 2 * pad)
    buffer = np.zeros(length)
    buffer[pad:pad + len(signal)] = signal
    frequencies = np.fft.rfftfreq(length)
    shifted = np.fft.irfft(
        np.fft.rfft(buffer) * np.exp(-2j * np.pi * frequencies * fraction), n=length)
    return shifted[:len(signal) + 2 * pad]


def _copy_into_segment(signal, offset, number_of_samples):
    """ Put signal sample k at data index offset + k and return data indices
    [0, number_of_samples); zero elsewhere. """
    segment = np.zeros(number_of_samples)
    first = max(0, -offset)
    last = min(len(signal), number_of_samples - offset)
    if last > first:
        segment[offset + first:offset + last] = signal[first:last]
    return segment


def times_from_merger(start_time, number_of_samples, sampling_frequency, geocent_time, time_delay):
    """ Times of data samples relative to the arrival of the model's t=0 in a
    detector, t_k - (geocent_time + time_delay), as for ringdown analyses.

    Parameters
    ==========
    start_time: float
        GPS time of the first data sample.
    number_of_samples: int
    sampling_frequency: float
    geocent_time: float
        GPS time at which the model's t=0 arrives at the geocenter.
    time_delay: float
        Time delay of the detector from the geocenter (s).

    Returns
    =======
    array_like: times (s)
    """
    # subtract the two GPS times (~1e9 s) first, then add the small terms
    return (start_time - geocent_time) - time_delay + np.arange(number_of_samples) / sampling_frequency


def place_time_domain_signal(signal, merger_index, arrival_index, number_of_samples,
                             placement="nearest"):
    """ Place a time-domain signal, generated on its own grid with the data's
    sample spacing, on the data samples.

    Waveform sample k has time (k - merger_index) / f_s relative to the
    model's t=0, which arrives at data index `arrival_index`. Waveform samples
    outside the data are dropped and data samples the waveform does not reach
    are zero.

    Parameters
    ==========
    signal: array_like
        Detector signal on its own grid.
    merger_index: float
        Index of the model's t=0 in `signal`.
    arrival_index: float
        Index in the data at which the model's t=0 arrives.
    number_of_samples: int
        Number of data samples.
    placement: str
        - "nearest" (default): `merger_index` and `arrival_index` are each
          rounded to the nearest sample.
        - "fd_shift": whole samples are shifted by index and the remaining
          fraction with a Fourier phase shift of a zero-padded copy.

    Returns
    =======
    array_like: The signal on the data samples.
    """
    if placement == "nearest":
        offset = round(arrival_index) - round(merger_index)
        return _copy_into_segment(signal, offset, number_of_samples)
    elif placement == "fd_shift":
        position = arrival_index - merger_index
        offset = int(np.floor(position + 0.5))
        fraction = position - offset
        pad = 32
        shifted = fractional_time_shift(signal, fraction, pad=pad)
        return _copy_into_segment(shifted, offset - pad, number_of_samples)
    else:
        raise ValueError(f"placement must be 'nearest' or 'fd_shift', not {placement!r}")
