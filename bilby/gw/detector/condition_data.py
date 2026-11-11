import numpy as np

__all__ = ["condition_strain_data"]


def butterworth_filter(
    sampling_frequency, highpass_frequency=None, lowpass_frequency=None
):
    """The Butterworth filter of :code:`condition_strain_data` (order 4)

    Parameters
    ==========
    sampling_frequency: float
        The sampling frequency (in Hz) of the data to be filtered
    highpass_frequency, lowpass_frequency: float, optional
        The corner frequencies (in Hz); a band-pass filter if both are given

    Returns
    =======
    tuple: The filter coefficients (b, a), or None if neither frequency is given
    """
    from scipy.signal import butter

    order = 4
    nyquist_frequency = sampling_frequency / 2
    if highpass_frequency and lowpass_frequency:
        return butter(
            order,
            (
                highpass_frequency / nyquist_frequency,
                lowpass_frequency / nyquist_frequency,
            ),
            btype="bandpass",
        )
    elif highpass_frequency:
        return butter(order, highpass_frequency / nyquist_frequency, btype="highpass")
    elif lowpass_frequency:
        return butter(order, lowpass_frequency / nyquist_frequency, btype="lowpass")
    return None


def filter_amplitude_response(
    frequency_array, sampling_frequency, highpass_frequency=None, lowpass_frequency=None
):
    """The factor by which :code:`condition_strain_data` multiplies the amplitude of the
    data at each frequency: |H(f)|^2, as the filter is applied forwards and backwards

    Parameters
    ==========
    frequency_array: array_like
        The frequencies (in Hz)
    sampling_frequency: float
        The sampling frequency (in Hz) of the data when filtered
    highpass_frequency, lowpass_frequency: float, optional
        The corner frequencies (in Hz)

    Returns
    =======
    array_like: The amplitude response (1 if no filter)
    """
    from scipy.signal import freqz

    frequency_array = np.asarray(frequency_array, dtype=float)
    coefficients = butterworth_filter(
        sampling_frequency, highpass_frequency, lowpass_frequency
    )
    if coefficients is None:
        return np.ones(len(frequency_array))
    _, response = freqz(*coefficients, worN=frequency_array, fs=sampling_frequency)
    return abs(response) ** 2


def condition_strain_data(
    strain_data,
    sampling_frequency,
    preserve_time=None,
    highpass_frequency=None,
    lowpass_frequency=None,
    trim=0.25,
    remove_mean=True,
):
    """Filter, downsample and trim time-domain strain data

    Adapted from :code:`ringdown.data.Data.condition` (ringdown v1.1.0, Isi & Farr
    arXiv:2107.05609, MIT licence). The data are rolled so that the sample nearest
    `preserve_time` is kept, optionally filtered with an order-4 Butterworth filter
    (:code:`scipy.signal.filtfilt`), tapered, low-passed at the new Nyquist
    frequency, downsampled and trimmed by `trim` at each end. The result replaces
    the data in `strain_data`.

    The filter is applied to the data only, not to the waveform or the noise model:
    use a `highpass_frequency` of at most half the minimum frequency of the analysis,
    and no `lowpass_frequency` inside the band.

    Parameters
    ==========
    strain_data: bilby.gw.detector.InterferometerStrainData
        The strain data, modified in place
    sampling_frequency: float
        The new sampling frequency (in Hz)
    preserve_time: float, optional
        GPS time whose nearest sample is kept
    highpass_frequency: float, optional
        The corner frequency (in Hz) of the high-pass filter (default: none)
    lowpass_frequency: float, optional
        The corner frequency (in Hz) of the low-pass filter (default: none)
    trim: float
        The fraction of the data removed at each end
    remove_mean: bool
        If True, subtract the mean of the result
    """
    from scipy.signal import filtfilt
    from scipy.signal.windows import tukey

    original_sampling_frequency = strain_data.sampling_frequency
    factor = int(round(original_sampling_frequency / sampling_frequency))
    if factor < 1 or not np.isclose(
        factor * sampling_frequency, original_sampling_frequency, rtol=1e-9, atol=0
    ):
        raise ValueError(
            f"The new sampling frequency ({sampling_frequency:g} Hz) must divide the current one "
            f"({original_sampling_frequency:g} Hz)"
        )
    strain = np.array(strain_data.time_domain_strain, dtype=float)
    times = np.array(strain_data.time_array, dtype=float)

    # 1. roll, so that the sample nearest preserve_time is kept by the downsampling
    if preserve_time is not None:
        if not times[0] <= preserve_time <= times[-1]:
            raise ValueError(
                f"preserve_time ({preserve_time:.6f}) must be within the data ({times[0]:.6f} - {times[-1]:.6f})"
            )
        shift = int(np.argmin(abs(times - preserve_time))) % factor
        strain, times = np.roll(strain, -shift), np.roll(times, -shift)

    # 2. Butterworth filter, forwards and backwards
    coefficients = butterworth_filter(
        original_sampling_frequency, highpass_frequency, lowpass_frequency
    )
    if coefficients is not None:
        strain = filtfilt(*coefficients, strain)

    # 3. downsample
    if factor > 1:
        strain_fd = np.fft.rfft(strain * tukey(len(strain), trim))
        strain_fd[
            np.fft.rfftfreq(len(strain), 1 / original_sampling_frequency)
            > sampling_frequency / 2
        ] = 0
        strain = np.fft.irfft(strain_fd, n=len(strain))[::factor]
        times = times[::factor]

    # 4. trim
    number_of_samples = len(strain)
    keep = slice(
        int(round(trim * number_of_samples)), int(round((1 - trim) * number_of_samples))
    )
    strain, times = strain[keep], times[keep]
    out_of_order = np.flatnonzero(np.diff(times) <= 0)
    if len(out_of_order) > 0:  # rolled samples left at the end
        strain, times = strain[: out_of_order[0] + 1], times[: out_of_order[0] + 1]

    # 5. remove the mean
    if remove_mean:
        strain = strain - np.mean(strain)
    strain_data.set_from_time_domain_strain(
        strain,
        sampling_frequency=sampling_frequency,
        duration=len(strain) / sampling_frequency,
        start_time=times[0],
    )
