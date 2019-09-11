from ..conversion import convert_to_lal_binary_black_hole_parameters
from .calibration import *
from .interferometer import *
from .networks import *
from .psd import *
from .strain_data import *

try:
    import lal
    import lalsimulation as lalsim
except ImportError:
    logger.warning("You do not have lalsuite installed currently. You will"
                   " not be able to use some of the prebuilt functions.")


def get_safe_signal_duration(mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, flow=10):
    """ Calculate the safe signal duration, given the parameters

    Parameters
    ----------
    mass_1, mass_2, a_1, a_2, tilt_1, tilt_2: float
        The signal parameters
    flow: float
        The lower frequency cutoff from which to calculate the signal duration

     Returns
     -------
     safe_signal_duration: float
        The shortest safe signal duration (i.e., the signal duration rounded up
        to the nearest power of 2)

    """
    chirp_time = lalsim.SimInspiralChirpTimeBound(
        flow, mass_1 * lal.MSUN_SI, mass_2 * lal.MSUN_SI,
        a_1 * np.cos(tilt_1), a_2 * np.cos(tilt_2))
    return max(2**(int(np.log2(chirp_time)) + 1), 4)


def inject_signal_into_gwpy_timeseries(
        data, waveform_generator, parameters, det, outdir=None, label=None):
    """ Inject a signal into a gwpy timeseries

    Parameters
    ----------
    data: gwpy.timeseries.TimeSeries
        The time-series data into which we want to inject the signal
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
        An initialised waveform_generator
    parameters: dict
        A dictionary of the signal-parameters to inject
    ifo: bilby.gw.detector.Interferometer
        The interferometer for which the data refers too
    outdir, label: str
        If given, the outdir and label used to generate a plot

    Returns
    -------
    data_and_signal: gwpy.timeseries.TimeSeries
        The data with the time-domain signal added
    meta_data: dict
        A dictionary of meta data about the injection

    """
    ifo = get_empty_interferometer(det)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    parameters_check, _ = convert_to_lal_binary_black_hole_parameters(parameters)
    parameters_check = {key: parameters_check[key] for key in
                        ['mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2']}
    safe_time = get_safe_signal_duration(**parameters_check)
    if data.duration.value < safe_time:
        ValueError(
            "Injecting a signal with safe-duration {} longer than the data {}"
            .format(safe_time, data.duration.value))

    waveform_polarizations = waveform_generator.time_domain_strain(parameters)

    signal = np.zeros(len(data))

    for mode in waveform_polarizations.keys():
        det_response = ifo.antenna_response(
            parameters['ra'], parameters['dec'], parameters['geocent_time'],
            parameters['psi'], mode)
        signal += waveform_polarizations[mode] * det_response
    time_shift = ifo.time_delay_from_geocenter(
        parameters['ra'], parameters['dec'], parameters['geocent_time'])

    dt = parameters['geocent_time'] + time_shift - data.times[0].value
    n_roll = dt * data.sample_rate.value
    n_roll = int(np.round(n_roll))
    signal_shifted = gwpy.timeseries.TimeSeries(
        data=np.roll(signal, n_roll), times=data.times, unit=data.unit)

    signal_and_data = data.inject(signal_shifted)

    if outdir is not None and label is not None:
        fig = gwpy.plot.Plot(signal_shifted)
        fig.savefig('{}/{}_{}_time_domain_injected_signal'.format(
            outdir, ifo.name, label))

    meta_data = dict()
    frequency_domain_signal, _ = utils.nfft(signal, waveform_generator.sampling_frequency)
    meta_data['optimal_SNR'] = (
        np.sqrt(ifo.optimal_snr_squared(signal=frequency_domain_signal)).real)
    meta_data['matched_filter_SNR'] = (
        ifo.matched_filter_snr(signal=frequency_domain_signal))
    meta_data['parameters'] = parameters

    logger.info("Injected signal in {}:".format(ifo.name))
    logger.info("  optimal SNR = {:.2f}".format(meta_data['optimal_SNR']))
    logger.info("  matched filter SNR = {:.2f}".format(meta_data['matched_filter_SNR']))
    for key in parameters:
        logger.info('  {} = {}'.format(key, parameters[key]))

    return signal_and_data, meta_data


def get_interferometer_with_fake_noise_and_injection(
        name, injection_parameters, injection_polarizations=None,
        waveform_generator=None, sampling_frequency=4096, duration=4,
        start_time=None, outdir='outdir', label=None, plot=True, save=True,
        zero_noise=False):
    """
    Helper function to obtain an Interferometer instance with appropriate
    power spectral density and data, given an center_time.

    Note: by default this generates an Interferometer with a power spectral
    density based on advanced LIGO.

    Parameters
    ----------
    name: str
        Detector name, e.g., 'H1'.
    injection_parameters: dict
        injection parameters, needed for sky position and timing
    injection_polarizations: dict
       Polarizations of waveform to inject, output of
       `waveform_generator.frequency_domain_strain()`. If
       `waveform_generator` is also given, the injection_polarizations will
       be calculated directly and this argument can be ignored.
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
        A WaveformGenerator instance using the source model to inject. If
        `injection_polarizations` is given, this will be ignored.
    sampling_frequency: float
        sampling frequency for data, should match injection signal
    duration: float
        length of data, should be the same as used for signal generation
    start_time: float
        Beginning of data segment, if None, injection is placed 2s before
        end of segment.
    outdir: str
        directory in which to store output
    label: str
        If given, an identifying label used in generating file names.
    plot: bool
        If true, create an ASD + strain plot
    save: bool
        If true, save frequency domain data and PSD to file
    zero_noise: bool
        If true, set noise to zero.

    Returns
    -------
    bilby.gw.detector.Interferometer: An Interferometer instance with a PSD and frequency-domain strain data.

    """

    utils.check_directory_exists_and_if_not_mkdir(outdir)

    if start_time is None:
        start_time = injection_parameters['geocent_time'] + 2 - duration

    interferometer = get_empty_interferometer(name)
    interferometer.power_spectral_density = PowerSpectralDensity.from_aligo()
    if zero_noise:
        interferometer.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=start_time)
    else:
        interferometer.set_strain_data_from_power_spectral_density(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=start_time)

    injection_polarizations = interferometer.inject_signal(
        parameters=injection_parameters,
        injection_polarizations=injection_polarizations,
        waveform_generator=waveform_generator)

    signal = interferometer.get_detector_response(
        injection_polarizations, injection_parameters)

    if plot:
        interferometer.plot_data(signal=signal, outdir=outdir, label=label)

    if save:
        interferometer.save_data(outdir, label=label)

    return interferometer


def load_data_from_cache_file(
        cache_file, start_time, segment_duration, psd_duration, psd_start_time,
        channel_name=None, sampling_frequency=4096, roll_off=0.2,
        overlap=0, outdir=None):
    """ Helper routine to generate an interferometer from a cache file

    Parameters
    ----------
    cache_file: str
        Path to the location of the cache file
    start_time, psd_start_time: float
        GPS start time of the segment and data stretch used for the PSD
    segment_duration, psd_duration: float
        Segment duration and duration of data to use to generate the PSD (in
        seconds).
    roll_off: float, optional
        Rise time in seconds of tukey window.
    overlap: float,
        Number of seconds of overlap between FFTs.
    channel_name: str
        Channel name
    sampling_frequency: int
        Sampling frequency
    outdir: str, optional
        The output directory in which the data is saved

    Returns
    -------
    ifo: bilby.gw.detector.Interferometer
        An initialised interferometer object with strain data set to the
        appropriate data in the cache file and a PSD.
    """

    data_set = False
    psd_set = False

    with open(cache_file, 'r') as ff:
        lines = ff.readlines()
        if len(lines)>1:
            raise ValueError('This method cannot handle cache files with'
                             ' multiple frames. Use `load_data_by_channel_name'
                             ' instead.')
        else:
            line = lines[0]
            cache = lal.utils.cache.CacheEntry(line)
            data_in_cache = (
                (cache.segment[0].gpsSeconds < start_time) &
                (cache.segment[1].gpsSeconds > start_time + segment_duration))
            psd_in_cache = (
                (cache.segment[0].gpsSeconds < psd_start_time) &
                (cache.segment[1].gpsSeconds > psd_start_time + psd_duration))
            ifo = get_empty_interferometer(
                "{}1".format(cache.observatory))
            if not data_in_cache:
                raise ValueError('The specified data segment does not exist in' 
                                 ' this frame.')
            if not psd_in_cache:
                raise ValueError('The specified PSD data segment does not exist' 
                                 ' in this frame.')
            if (not data_set) & data_in_cache:
                ifo.set_strain_data_from_frame_file(
                    frame_file=cache.path,
                    sampling_frequency=sampling_frequency,
                    duration=segment_duration,
                    start_time=start_time,
                    channel=channel_name, buffer_time=0)
                data_set = True
            if (not psd_set) & psd_in_cache:
                ifo.power_spectral_density = \
                    PowerSpectralDensity.from_frame_file(
                        cache.path,
                        psd_start_time=psd_start_time,
                        psd_duration=psd_duration,
                        fft_length=segment_duration,
                        sampling_frequency=sampling_frequency,
                        roll_off=roll_off,
                        overlap=overlap,
                        channel=channel_name,
                        name=cache.observatory,
                        outdir=outdir,
                        analysis_segment_start_time=start_time)
                psd_set = True
    if data_set and psd_set:
        return ifo
    elif not data_set:
        raise ValueError('Data not loaded for {}'.format(ifo.name))
    elif not psd_set:
        raise ValueError('PSD not created for {}'.format(ifo.name))


def load_data_by_channel_name(
        channel_name, start_time, segment_duration, psd_duration, psd_start_time,
        sampling_frequency=4096, roll_off=0.2,
        overlap=0, outdir=None):
    """ Helper routine to generate an interferometer from a channel name
    This function creates an empty interferometer specified in the name 
    of the channel. It calls `ifo.set_strain_data_from_channel_name` to 
    set the data and PSD in the interferometer using data retrieved from 
    the specified channel using gwpy.TimeSeries.get()

    Parameters
    ----------
    channel_name: str
        Channel name with the format `IFO:Channel`
    start_time, psd_start_time: float
        GPS start time of the segment and data stretch used for the PSD
    segment_duration, psd_duration: float
        Segment duration and duration of data to use to generate the PSD (in
        seconds).
    roll_off: float, optional
        Rise time in seconds of tukey window.
    overlap: float,
        Number of seconds of overlap between FFTs.
    sampling_frequency: int
        Sampling frequency
    outdir: str, optional
        The output directory in which the data is saved

    Returns
    -------
    ifo: bilby.gw.detector.Interferometer
        An initialised interferometer object with strain data set to the
        appropriate data fetched from the specified channel and a PSD.
    """
    try:
        det = channel_name.split(':')[-2]
    except IndexError:
        raise IndexError("Channel name must be of the format `IFO:Channel`")
    ifo = get_empty_interferometer(det)

    ifo.set_strain_data_from_channel_name(
        channel = channel_name,
        sampling_frequency=sampling_frequency,
        duration=segment_duration,
        start_time=start_time)

    ifo.power_spectral_density = \
        PowerSpectralDensity.from_channel_name(
            channel=channel_name,
            psd_start_time=psd_start_time,
            psd_duration=psd_duration,
            fft_length=segment_duration,
            sampling_frequency=sampling_frequency,
            roll_off=roll_off,
            overlap=overlap,
            name=det,
            outdir=outdir,
            analysis_segment_start_time=start_time)
    return ifo
