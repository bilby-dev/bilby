import numpy as np


def condition_strain_data(strain_data, sampling_frequency, start_time, duration,
                          preserve_time=None, trim=0.25, taper_and_trim=True,
                          remove_mean=True):
    """ Downsample time-domain strain data around an analysis window

    By default, the data in the window plus a buffer of
    `trim` / (1 - 2 `trim`) x `duration` on each side are tapered with a Tukey
    window (alpha = `trim`), low-passed below the new Nyquist frequency,
    downsampled, and the buffer is removed. The samples are aligned so that the
    sample nearest `preserve_time` is kept. With `taper_and_trim` = False,
    :code:`scipy.signal.decimate` is used instead. The result replaces the
    data in `strain_data`.

    Parameters
    ==========
    strain_data: bilby.gw.detector.InterferometerStrainData
        The strain data, modified in place
    sampling_frequency: float
        The new sampling frequency (in Hz); must divide the current one
    start_time: float
        The GPS start time of the analysis window
    duration: float
        The duration of the analysis window (in s)
    preserve_time: float, optional
        GPS time whose nearest sample is kept (default: `start_time`)
    trim: float
        The fraction of the data removed at each end
    taper_and_trim: bool
        If True, taper, low-pass and trim; if False, use :code:`scipy.signal.decimate`
    remove_mean: bool
        If True, subtract the mean of the result

    """
    from scipy.signal import decimate
    from scipy.signal.windows import tukey

    factor = int(round(strain_data.sampling_frequency / sampling_frequency))
    strain = np.asarray(strain_data.time_domain_strain, dtype=float)
    times = np.asarray(strain_data.time_array, dtype=float)
    if taper_and_trim:
        # step 1: the window and a buffer on each side
        buffer = duration * trim / (1 - 2 * trim)
        first, last = start_time - buffer, start_time + duration + buffer
        data_end = strain_data.start_time + strain_data.duration
        if (strain_data.start_time > first + 1 / strain_data.sampling_frequency
                or data_end < last - 1 / strain_data.sampling_frequency):
            raise ValueError(
                f"The data ({strain_data.start_time:.6f} - {data_end:.6f}) must cover {first:.6f} - {last:.6f}: "
                f"the analysis window plus {buffer:g} s on each side")
        keep = slice(max(int(round((first - strain_data.start_time) * strain_data.sampling_frequency)), 0),
                     int(round((last - strain_data.start_time) * strain_data.sampling_frequency)))
        strain, times = strain[keep], times[keep]

    # step 2: keep the sample nearest preserve_time
    shift = int(np.argmin(abs(times - (start_time if preserve_time is None else preserve_time)))) % factor
    if taper_and_trim and trim > 0:  # the wrapped samples are at the end and are removed in step 4
        strain, times = np.roll(strain, -shift), np.roll(times, -shift)
    else:
        strain, times = strain[shift:], times[shift:]

    if taper_and_trim:
        # step 3: taper, low-pass and downsample
        strain_fd = np.fft.rfft(strain * tukey(len(strain), trim))
        strain_fd[np.fft.rfftfreq(len(strain), 1 / strain_data.sampling_frequency) > sampling_frequency / 2] = 0
        strain, times = np.fft.irfft(strain_fd, n=len(strain))[::factor], times[::factor]
        # step 4: remove the fraction trim at each end, but always keep the analysis samples
        window = int(np.argmin(abs(times - start_time)))  # first sample of the analysis window
        keep = slice(min(int(round(trim * len(strain))), window),
                     max(int(round((1 - trim) * len(strain))), window + int(round(duration * sampling_frequency))))
        strain, times = strain[keep], times[keep]
    else:
        strain, times = decimate(strain, factor, zero_phase=True), times[::factor]

    if remove_mean:
        strain = strain - np.mean(strain)
    strain_data.set_from_time_domain_strain(
        strain, sampling_frequency=sampling_frequency, duration=len(strain) / sampling_frequency,
        start_time=times[0])


__all__ = ["condition_strain_data"]
