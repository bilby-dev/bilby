"""
Utilities for time-domain gravitational-wave inference: products with the
inverse of a Toeplitz noise covariance matrix, and placement of a time-domain
waveform on the data samples.
"""
import numpy as np
from scipy.fft import next_fast_len
from scipy.linalg import solve_toeplitz

from .utils import safe_cast_mode_to_int

# largest l of the default modes of time-domain approximants with higher modes
HIGHER_MODE_L_MAX = dict(IMRPhenomTHM=5, IMRPhenomTPHM=5, SEOBNRv4HM=5, SEOBNRv4PHM=5,
                         NRSur7dq4=4, NRHybSur3dq8=5)


def time_domain_start_frequency(minimum_frequency, waveform_approximant, mode_array=None):
    """ The frequency of the (2, 2) mode at which a time-domain waveform starts
    so that its modes with l <= 4 are present from `minimum_frequency`

    In the inspiral, the frequency of the (l, m) mode is m / 2 times that of the
    (2, 2) mode, so the waveform starts at 2 / min(l_max, 4) x `minimum_frequency`,
    where l_max is the largest l of the modes. See Ursell et al. arXiv:2510.15048.

    Parameters
    ==========
    minimum_frequency: float
        The frequency (in Hz) from which all modes should be present
    waveform_approximant: str
        The waveform approximant; approximants not in :code:`HIGHER_MODE_L_MAX`
        are taken to contain only l = 2 modes
    mode_array: list, optional
        The modes used, as in :code:`bilby.gw.source.lal_binary_black_hole`;
        default: all modes of the approximant

    Returns
    =======
    float: The start frequency (in Hz)
    """
    if mode_array is None:
        l_max = HIGHER_MODE_L_MAX.get(waveform_approximant, 2)
    else:
        l_max = max(safe_cast_mode_to_int(mode[0]) for mode in mode_array)
    return minimum_frequency * 2 / min(max(l_max, 2), 4)


def gohberg_semencul_vectors(acf):
    """ The Gohberg-Semencul vectors of the inverse of the symmetric Toeplitz matrix
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
    """ The product of the inverse of a symmetric Toeplitz matrix with a vector,
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
    zy_fft = np.fft.rfft(np.concatenate(([0.0], np.asarray(y, dtype=float)[:-1])), n=fft_length)

    def lower(w_fft, vector_fft):
        """ The product of the lower-triangular Toeplitz matrix with first column w and v, from their FFTs """
        return np.fft.irfft(w_fft * vector_fft, n=fft_length)[:number_of_samples]

    # L(w)^T v = reverse(L(w) reverse(v))
    reversed_fft = np.fft.rfft(np.asarray(vector, dtype=float)[::-1], n=fft_length)
    first = lower(x_fft, np.fft.rfft(lower(x_fft, reversed_fft)[::-1], n=fft_length))
    second = lower(zy_fft, np.fft.rfft(lower(zy_fft, reversed_fft)[::-1], n=fft_length))
    return (first - second) / x[0]


def fractional_time_shift(signal, fraction, pad=32):
    """ Delay a time series by a fraction of a sample with a Fourier phase shift

    Parameters
    ==========
    signal: array_like
        The time series
    fraction: float
        The delay in samples (positive values delay the signal)
    pad: int
        The number of zeros added before and after the signal, so that nothing wraps around

    Returns
    =======
    array_like: The delayed time series, of length len(signal) + 2 * pad
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
    """ Put `signal` at data indices `offset`, `offset` + 1, ... of a segment of `number_of_samples` zeros """
    segment = np.zeros(number_of_samples)
    first = max(0, -offset)
    last = min(len(signal), number_of_samples - offset)
    if last > first:
        segment[offset + first:offset + last] = signal[first:last]
    return segment


def times_from_merger(start_time, number_of_samples, sampling_frequency, geocent_time, time_delay):
    """ The times of data samples relative to the arrival of the merger at a detector

    Parameters
    ==========
    start_time: float
        The GPS time of the first data sample
    number_of_samples: int
        The number of data samples
    sampling_frequency: float
        The sampling frequency (in Hz)
    geocent_time: float
        The GPS time of the merger at the geocenter
    time_delay: float
        The time delay of the detector from the geocenter (in s)

    Returns
    =======
    array_like: The times (in s)
    """
    # subtract the two GPS times (~1e9 s) first, then add the small terms
    return (start_time - geocent_time) - time_delay + np.arange(number_of_samples) / sampling_frequency


def place_time_domain_signal(signal, merger_index, arrival_index, number_of_samples,
                             placement="nearest"):
    """ Place a time-domain signal, sampled at the sampling frequency of the data,
    on the data samples

    Signal samples outside the data are dropped and data samples not covered by
    the signal are zero.

    Parameters
    ==========
    signal: array_like
        The signal on its own time grid
    merger_index: float
        The index of the merger in `signal`
    arrival_index: float
        The index in the data at which the merger arrives
    number_of_samples: int
        The number of data samples
    placement: str
        "nearest": shift by the nearest whole number of samples;
        "fd_shift": shift by whole samples and the remaining fraction of a
        sample with :code:`fractional_time_shift`

    Returns
    =======
    array_like: The signal on the data samples
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
