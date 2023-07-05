import numpy as np

_TOL = 14


def get_sampling_frequency(time_array):
    """
    Calculate sampling frequency from a time series

    Attributes
    ==========
    time_array: array_like
        Time array to get sampling_frequency from

    Returns
    =======
    Sampling frequency of the time series: float

    Raises
    ======
    ValueError: If the time series is not evenly sampled.

    """
    tol = 1e-10
    if np.ptp(np.diff(time_array)) > tol:
        raise ValueError("Your time series was not evenly sampled")
    else:
        return np.round(1. / (time_array[1] - time_array[0]), decimals=_TOL)


def get_sampling_frequency_and_duration_from_time_array(time_array):
    """
    Calculate sampling frequency and duration from a time array

    Attributes
    ==========
    time_array: array_like
        Time array to get sampling_frequency/duration from: array_like

    Returns
    =======
    sampling_frequency, duration: float, float

    Raises
    ======
    ValueError: If the time_array is not evenly sampled.

    """

    sampling_frequency = get_sampling_frequency(time_array)
    duration = len(time_array) / sampling_frequency
    return sampling_frequency, duration


def get_sampling_frequency_and_duration_from_frequency_array(frequency_array):
    """
    Calculate sampling frequency and duration from a frequency array

    Attributes
    ==========
    frequency_array: array_like
        Frequency array to get sampling_frequency/duration from: array_like

    Returns
    =======
    sampling_frequency, duration: float, float

    Raises
    ======
    ValueError: If the frequency_array is not evenly sampled.

    """

    tol = 1e-10
    if np.ptp(np.diff(frequency_array)) > tol:
        raise ValueError("Your frequency series was not evenly sampled")

    number_of_frequencies = len(frequency_array)
    delta_freq = frequency_array[1] - frequency_array[0]
    duration = np.round(1 / delta_freq, decimals=_TOL)

    sampling_frequency = np.round(2 * (number_of_frequencies - 1) / duration, decimals=14)
    return sampling_frequency, duration


def create_time_series(sampling_frequency, duration, starting_time=0.):
    """

    Parameters
    ==========
    sampling_frequency: float
    duration: float
    starting_time: float, optional

    Returns
    =======
    float: An equidistant time series given the parameters

    """
    _check_legal_sampling_frequency_and_duration(sampling_frequency, duration)
    number_of_samples = int(duration * sampling_frequency)
    return np.linspace(start=starting_time,
                       stop=duration + starting_time - 1 / sampling_frequency,
                       num=number_of_samples)


def create_frequency_series(sampling_frequency, duration):
    """ Create a frequency series with the correct length and spacing.

    Parameters
    ==========
    sampling_frequency: float
    duration: float

    Returns
    =======
    array_like: frequency series

    """
    _check_legal_sampling_frequency_and_duration(sampling_frequency, duration)
    number_of_samples = int(np.round(duration * sampling_frequency))
    number_of_frequencies = int(np.round(number_of_samples / 2) + 1)

    return np.linspace(start=0,
                       stop=sampling_frequency / 2,
                       num=number_of_frequencies)


def _check_legal_sampling_frequency_and_duration(sampling_frequency, duration):
    """ By convention, sampling_frequency and duration have to multiply to an integer

    This will check if the product of both parameters multiplies reasonably close
    to an integer.

    Parameters
    ==========
    sampling_frequency: float
    duration: float

    """
    num = sampling_frequency * duration
    if np.abs(num - np.round(num)) > 10**(-_TOL):
        raise IllegalDurationAndSamplingFrequencyException(
            '\nYour sampling frequency and duration must multiply to a number'
            'up to (tol = {}) decimals close to an integer number. '
            '\nBut sampling_frequency={} and  duration={} multiply to {}'.format(
                _TOL, sampling_frequency, duration,
                sampling_frequency * duration
            )
        )


def create_white_noise(sampling_frequency, duration):
    """ Create white_noise which is then coloured by a given PSD

    Parameters
    ==========
    sampling_frequency: float
    duration: float
        duration of the data

    Returns
    =======
    array_like: white noise
    array_like: frequency array
    """
    from .random import rng

    number_of_samples = duration * sampling_frequency
    number_of_samples = int(np.round(number_of_samples))

    frequencies = create_frequency_series(sampling_frequency, duration)

    norm1 = 0.5 * duration**0.5
    re1, im1 = rng.normal(0, norm1, (2, len(frequencies)))
    white_noise = re1 + 1j * im1

    # set DC and Nyquist = 0
    white_noise[0] = 0
    # no Nyquist frequency when N=odd
    if np.mod(number_of_samples, 2) == 0:
        white_noise[-1] = 0

    # python: transpose for use with infft
    white_noise = np.transpose(white_noise)
    frequencies = np.transpose(frequencies)

    return white_noise, frequencies


def nfft(time_domain_strain, sampling_frequency):
    """ Perform an FFT while keeping track of the frequency bins. Assumes input
        time series is real (positive frequencies only).

    Parameters
    ==========
    time_domain_strain: array_like
        Time series of strain data.
    sampling_frequency: float
        Sampling frequency of the data.

    Returns
    =======
    frequency_domain_strain, frequency_array: (array_like, array_like)
        Single-sided FFT of time domain strain normalised to units of
        strain / Hz, and the associated frequency_array.

    """
    frequency_domain_strain = np.fft.rfft(time_domain_strain)
    frequency_domain_strain /= sampling_frequency

    frequency_array = np.linspace(
        0, sampling_frequency / 2, len(frequency_domain_strain))

    return frequency_domain_strain, frequency_array


def infft(frequency_domain_strain, sampling_frequency):
    """ Inverse FFT for use in conjunction with nfft.

    Parameters
    ==========
    frequency_domain_strain: array_like
        Single-sided, normalised FFT of the time-domain strain data (in units
        of strain / Hz).
    sampling_frequency: int, float
        Sampling frequency of the data.

    Returns
    =======
    time_domain_strain: array_like
        An array of the time domain strain
    """
    time_domain_strain_norm = np.fft.irfft(frequency_domain_strain)
    time_domain_strain = time_domain_strain_norm * sampling_frequency
    return time_domain_strain


class IllegalDurationAndSamplingFrequencyException(Exception):
    pass
