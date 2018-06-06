from __future__ import division
import logging
import os
import numpy as np
from math import fmod
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
from scipy import signal
import argparse

# Constants
speed_of_light = 299792458.0  # speed of light in m/s
parsec = 3.085677581 * 1e16
solar_mass = 1.98855 * 1e30


def get_sampling_frequency(time_series):
    """
    Calculate sampling frequency from a time series
    """
    tol = 1e-10
    if np.ptp(np.diff(time_series)) > tol:
        raise ValueError("Your time series was not evenly sampled")
    else:
        return 1. / (time_series[1] - time_series[0])


def create_time_series(sampling_frequency, duration, starting_time=0.):
    return np.arange(starting_time, duration, 1./sampling_frequency)


def ra_dec_to_theta_phi(ra, dec, gmst):
    """
    Convert from RA and DEC to polar coordinates on celestial sphere
    Input:
    ra - right ascension in radians
    dec - declination in radians
    gmst - Greenwich mean sidereal time of arrival of the signal in radians
    Output:
    theta - zenith angle in radians
    phi - azimuthal angle in radians
    """
    phi = ra - gmst
    theta = np.pi / 2 - dec
    return theta, phi


def gps_time_to_gmst(gps_time):
    """
    Convert gps time to Greenwich mean sidereal time in radians

    This method assumes a constant rotation rate of earth since 00:00:00, 1 Jan. 2000
    A correction has been applied to give the exact correct value for 00:00:00, 1 Jan. 2018
    Error accumulates at a rate of ~0.0001 radians/decade.

    Input:
    time - gps time
    Output:
    gmst - Greenwich mean sidereal time in radians
    """
    omega_earth = 2 * np.pi * (1 / 365.2425 + 1) / 86400.
    gps_2000 = 630720013.
    gmst_2000 = (6 + 39. / 60 + 51.251406103947375 / 3600) * np.pi / 12
    correction_2018 = -0.00017782487379358614
    sidereal_time = omega_earth * (gps_time - gps_2000) + gmst_2000 + correction_2018
    gmst = fmod(sidereal_time, 2 * np.pi)
    return gmst


def create_frequency_series(sampling_frequency, duration):
    """
    Create a frequency series with the correct length and spacing.

    :param sampling_frequency: sampling frequency
    :param duration: duration of data
    :return: frequencies, frequency series
    """
    number_of_samples = duration * sampling_frequency
    number_of_samples = int(np.round(number_of_samples))

    # prepare for FFT
    number_of_frequencies = (number_of_samples-1)//2
    delta_freq = 1./duration

    frequencies = delta_freq * np.linspace(1, number_of_frequencies, number_of_frequencies)

    if len(frequencies) % 2 == 1:
        frequencies = np.concatenate(([0], frequencies, [sampling_frequency / 2.]))
    else:
        # no Nyquist frequency when N=odd
        frequencies = np.concatenate(([0], frequencies))

    return frequencies


def create_white_noise(sampling_frequency, duration):
    """
    Create white_noise which is then coloured by a given PSD


    :param sampling_frequency: sampling frequency
    :param duration: duration of data
    """

    number_of_samples = duration * sampling_frequency
    number_of_samples = int(np.round(number_of_samples))

    delta_freq = 1./duration

    frequencies = create_frequency_series(sampling_frequency, duration)

    norm1 = 0.5*(1./delta_freq)**0.5
    re1 = np.random.normal(0, norm1, len(frequencies))
    im1 = np.random.normal(0, norm1, len(frequencies))
    htilde1 = re1 + 1j*im1

    # convolve data with instrument transfer function
    otilde1 = htilde1 * 1.
    # set DC and Nyquist = 0
    otilde1[0] = 0
    # no Nyquist frequency when N=odd
    if np.mod(number_of_samples, 2) == 0:
        otilde1[-1] = 0

    # normalise for positive frequencies and units of strain/rHz
    white_noise = otilde1
    # python: transpose for use with infft
    white_noise = np.transpose(white_noise)
    frequencies = np.transpose(frequencies)

    return white_noise, frequencies


def nfft(ht, Fs):
    """
    performs an FFT while keeping track of the frequency bins
    assumes input time series is real (positive frequencies only)

    ht = time series
    Fs = sampling frequency

    returns
    hf = single-sided FFT of ft normalised to units of strain / sqrt(Hz)
    f = frequencies associated with hf
    """
    # add one zero padding if time series does not have even number of sampling times
    if np.mod(len(ht), 2) == 1:
        ht = np.append(ht, 0)
    LL = len(ht)
    # frequency range
    ff = Fs / 2 * np.linspace(0, 1, int(LL/2+1))

    # calculate FFT
    # rfft computes the fft for real inputs
    hf = np.fft.rfft(ht)

    # normalise to units of strain / sqrt(Hz)
    hf = hf / Fs

    return hf, ff


def infft(hf, Fs):
    """
    inverse FFT for use in conjunction with nfft
    eric.thrane@ligo.org
    input:
    hf = single-side FFT calculated by fft_eht
    Fs = sampling frequency
    output:
    h = time series
    """
    # use irfft to work with positive frequencies only
    h = np.fft.irfft(hf)
    # undo LAL/Lasky normalisation
    h = h*Fs

    return h


def asd_from_freq_series(freq_data, df):
    """
    Calculate the ASD from the frequency domain output of gaussian_noise()    
    Input:
    freq_data - array of complex frequency domain data
    df - spacing of freq_data, 1/(segment length) used to generate the gaussian noise
    Output:
    asd = array of real-valued normalized frequency domain ASD data
    """
    asd = np.absolute(freq_data) * (2 * df)**0.5
    return asd


def psd_from_freq_series(freq_data, df):
    """
    Calculate the PSD from the frequency domain output of gaussian_noise()
    Calls asd_from_freq_series() and squares the output
    Input:
    freq_data - array of complex frequency domain data
    df - spacing of freq_data, 1/(segment length) used to generate the gaussian noise
    Output:
    psd - array of real-valued normalized frequency domain PSD data
    """
    psd = np.power(asd_from_freq_series(freq_data, df), 2)
    return psd


def time_delay_geocentric(detector1, detector2, ra, dec, time):
    """
    Calculate time delay between two detectors in geocentric coordinates based on XLALArrivaTimeDiff in TimeDelay.c
    Input:
    detector1 - cartesian coordinate vector for the first detector in the geocentric frame
                generated by the Interferometer class as self.vertex
    detector2 - cartesian coordinate vector for the second detector in the geocentric frame
    To get time delay from Earth center, use detector2 = np.array([0,0,0])
    ra - right ascension of the source in radians
    dec - declination of the source in radians
    time - GPS time in the geocentric frame
    Output:
    delta_t - time delay between the two detectors in the geocentric frame
    """
    gmst = gps_time_to_gmst(time)
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    omega = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    delta_d = detector2 - detector1
    delta_t = np.dot(omega, delta_d) / speed_of_light
    return delta_t


def get_polarization_tensor(ra, dec, time, psi, mode):
    """
    Calculate the polarization tensor for a given sky location and time

    See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
    [u, v, w] represent the Earth-frame
    [m, n, omega] represent the wave-frame
    Note: there is a typo in the definition of the wave-frame in Nishizawa et al.

    :param ra: right ascension in radians
    :param dec: declination in radians
    :param time: geocentric GPS time
    :param psi: binary polarisation angle counter-clockwise about the direction of propagation
    :param mode: polarisation mode
    :return: polarization_tensor(ra, dec, time, psi, mode): polarization tensor for the specified mode.
    """
    greenwich_mean_sidereal_time = gps_time_to_gmst(time)
    theta, phi = ra_dec_to_theta_phi(ra, dec, greenwich_mean_sidereal_time)
    u = np.array([np.cos(phi) * np.cos(theta), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    v = np.array([-np.sin(phi), np.cos(phi), 0])
    m = -u * np.sin(psi) - v * np.cos(psi)
    n = -u * np.cos(psi) + v * np.sin(psi)

    if mode.lower() == 'plus':
        return np.einsum('i,j->ij', m, m) - np.einsum('i,j->ij', n, n)
    elif mode.lower() == 'cross':
        return np.einsum('i,j->ij', m, n) + np.einsum('i,j->ij', n, m)
    elif mode.lower() == 'breathing':
        return np.einsum('i,j->ij', m, m) + np.einsum('i,j->ij', n, n)

    omega = np.cross(m, n)
    if mode.lower() == 'longitudinal':
        return np.sqrt(2) * np.einsum('i,j->ij', omega, omega)
    elif mode.lower() == 'x':
        return np.einsum('i,j->ij', m, omega) + np.einsum('i,j->ij', omega, m)
    elif mode.lower() == 'y':
        return np.einsum('i,j->ij', n, omega) + np.einsum('i,j->ij', omega, n)
    else:
        logging.warning("{} not a polarization mode!".format(mode))
        return None


def get_vertex_position_geocentric(latitude, longitude, elevation):
    """
    Calculate the position of the IFO vertex in geocentric coordiantes in meters.

    Based on arXiv:gr-qc/0008066 Eqs. B11-B13 except for the typo in the definition of the local radius.
    See Section 2.1 of LIGO-T980044-10 for the correct expression
    """
    semi_major_axis = 6378137  # for ellipsoid model of Earth, in m
    semi_minor_axis = 6356752.314  # in m
    radius = semi_major_axis**2 * (semi_major_axis**2 * np.cos(latitude)**2
                                   + semi_minor_axis**2 * np.sin(latitude)**2)**(-0.5)
    x_comp = (radius + elevation) * np.cos(latitude) * np.cos(longitude)
    y_comp = (radius + elevation) * np.cos(latitude) * np.sin(longitude)
    z_comp = ((semi_minor_axis / semi_major_axis)**2 * radius + elevation) * np.sin(latitude)
    return np.array([x_comp, y_comp, z_comp])


def setup_logger(outdir=None, label=None, log_level=None):
    """ Setup logging output: call at the start of the script to use

    Parameters
    ----------
    outdir, label: str
        If supplied, write the logging output to outdir/label.log
    log_level = ['debug', 'info', 'warning']
        Either a string from the list above, or an interger as specified
        in https://docs.python.org/2/library/logging.html#logging-levels
    """

    if type(log_level) is str:
        try:
            LEVEL = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError('log_level {} not understood'.format(log_level))
    elif log_level is None:
        LEVEL = command_line_args.log_level
    else:
        LEVEL = int(log_level)

    logger = logging.getLogger()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))
    logger.setLevel(LEVEL)
    stream_handler.setLevel(LEVEL)
    logger.addHandler(stream_handler)

    if label:
        if outdir:
            check_directory_exists_and_if_not_mkdir(outdir)
        else:
            outdir = '.'
        log_file = '{}/{}.log'.format(outdir, label)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))

        file_handler.setLevel(LEVEL)
        logger.addHandler(file_handler)

    version_file = os.path.join(os.path.dirname(__file__), '.version')
    with open(version_file, 'r') as f:
        version = f.readline()
    logging.info('Running tupak version: {}'.format(version))


def get_progress_bar(module='tqdm'):
    if module in ['tqdm']:
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, *args, **kwargs):
                return x
        return tqdm
    elif module in ['tqdm_notebook']:
        try:
            from tqdm import tqdm_notebook as tqdm
        except ImportError:
            def tqdm(x, *args, **kwargs):
                return x
        return tqdm


def spherical_to_cartesian(radius, theta, phi):
    """
    Convert from spherical coordinates to cartesian.

    :param radius: radial coordinate
    :param theta: axial coordinate
    :param phi: azimuthal coordinate
    :return cartesian: cartesian vector
    """
    cartesian = [radius * np.sin(theta) * np.cos(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(theta)]
    return cartesian


def check_directory_exists_and_if_not_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.debug('Making directory {}'.format(directory))
    else:
        logging.debug('Directory {} exists'.format(directory))


def inner_product(aa, bb, frequency, PSD):
    """
    Calculate the inner product defined in the matched filter statistic

    arguments:
    aai, bb: single-sided Fourier transform, created, e.g., by the nfft function above
    frequency: an array of frequencies associated with aa, bb, also returned by nfft
    PSD: PSD object

    Returns:
    The matched filter inner product for aa and bb
    """
    PSD_interp = PSD.power_spectral_density_interpolated(frequency)

    # calculate the inner product
    integrand = np.conj(aa) * bb / PSD_interp

    df = frequency[1] - frequency[0]
    integral = np.sum(integrand) * df

    product = 4. * np.real(integral)

    return product


def noise_weighted_inner_product(aa, bb, power_spectral_density, time_duration):
    """
    Calculate the noise weighted inner product between two arrays.

    Parameters
    ----------
    aa: array
        Array to be complex conjugated
    bb: array
        Array not to be complex conjugated
    power_spectral_density: array
        Power spectral density
    time_duration: float
        time_duration of the data

    Return
    ------
    Noise-weighted inner product.
    """

    # caluclate the inner product
    integrand = np.conj(aa) * bb / power_spectral_density
    product = 4 / time_duration * np.sum(integrand)
    return product


def matched_filter_snr_squared(signal, interferometer, time_duration):
    return noise_weighted_inner_product(signal, interferometer.data, interferometer.power_spectral_density_array,
                                        time_duration)


def optimal_snr_squared(signal, interferometer, time_duration):
    return noise_weighted_inner_product(signal, signal, interferometer.power_spectral_density_array, time_duration)


def get_event_time(event):
    """
    Get the merger time for known GW events.

    We currently know about:
        GW150914
        LVT151012
        GW151226
        GW170104
        GW170608
        GW170814
        GW170817

    Parameters
    ----------
    event: str
        Event descriptor, this can deal with some prefixes, e.g., '150914', 'GW150914', 'LVT151012'

    Return
    ------
    event_time: float
        Merger time
    """
    event_times = {'150914': 1126259462.422, '151012': 1128678900.4443,  '151226': 1135136350.65,
                   '170104': 1167559936.5991, '170608': 1180922494.4902, '170814': 1186741861.5268,
                   '170817': 1187008882.4457}
    if 'GW' or 'LVT' in event:
        event = event[-6:]

    try:
        event_time = event_times[event[-6:]]
        return event_time
    except KeyError:
        print('Unknown event {}.'.format(event))
        return None


def get_open_strain_data(
        name, t1, t2, outdir, cache=False, raw_data_file=None, **kwargs):
    """ A function which accesses the open strain data

    This uses `gwpy` to download the open data and then saves a cached copy for
    later use

    Parameters
    ----------
    name: str
        The name of the detector to get data for
    t1, t2: float
        The GPS time of the start and end of the data
    outdir: str
        The output directory to place data in
    cache: bool
        If true, cache the data
    **kwargs:
        Passed to `gwpy.timeseries.TimeSeries.fetch_open_data`
    raw_data_file

    Returns
    -----------
    strain: gwpy.timeseries.TimeSeries

    """
    filename = '{}/{}_{}_{}.txt'.format(outdir, name, t1, t2)
    if raw_data_file:
        logging.info('Using raw_data_file {}'.format(raw_data_file))
        strain = TimeSeries.read(raw_data_file)
        if (t1 > strain.times[0].value) and (t2 < strain.times[-1].value):
            logging.info('Using supplied raw data file')
            strain = strain.crop(t1, t2)
        else:
            raise ValueError('Supplied file does not contain requested data')
    elif os.path.isfile(filename) and cache:
        logging.info('Using cached data from {}'.format(filename))
        strain = TimeSeries.read(filename)
    else:
        logging.info('Fetching open data ...')
        strain = TimeSeries.fetch_open_data(name, t1, t2, **kwargs)
        logging.info('Saving data to {}'.format(filename))
        strain.write(filename)
    return strain


def read_frame_file(file_name, t1, t2, channel=None, **kwargs):
    """ A function which accesses the open strain data

    This uses `gwpy` to download the open data and then saves a cached copy for
    later use

    Parameters
    ----------
    file_name: str
        The name of the frame to read
    t1, t2: float
        The GPS time of the start and end of the data
    channel: str
        The name of the channel being searched for, some standard channel names are attempted
        if channel is not specified or if specified channel is not found.
    **kwargs:
        Passed to `gwpy.timeseries.TimeSeries.fetch_open_data`

    Returns
    -----------
    strain: gwpy.timeseries.TimeSeries

    """
    loaded = False
    if channel is not None:
        try:
            strain = TimeSeries.read(source=file_name, channel=channel, start=t1, end=t2, **kwargs)
            loaded = True
            logging.info('Successfully loaded {}.'.format(channel))
        except RuntimeError:
            logging.warning('Channel {} not found. Trying preset channel names'.format(channel))
    for channel_type in ['GDS-CALIB_STRAIN', 'DCS-CALIB_STRAIN_C01', 'DCS-CALIB_STRAIN_C02']:
        for ifo_name in ['H1', 'L1']:
            channel = '{}:{}'.format(ifo_name, channel_type)
            if loaded:
                continue
            try:
                strain = TimeSeries.read(source=file_name, channel=channel, start=t1, end=t2, **kwargs)
                loaded = True
                logging.info('Successfully loaded {}.'.format(channel))
            except RuntimeError:
                None

    if loaded:
        return strain
    else:
        logging.warning('No data loaded.')
        return None


def process_strain_data(
        strain, alpha=0.25, filter_freq=1024, **kwargs):
    """
    Helper function to obtain an Interferometer instance with appropriate
    PSD and data, given an center_time.

    Parameters
    ----------
    name: str
        Detector name, e.g., 'H1'.
    center_time: float
        GPS time of the center_time about which to perform the analysis.
        Note: the analysis data is from `center_time-T/2` to `center_time+T/2`.
    T: float
        The total time (in seconds) to analyse. Defaults to 4s.
    alpha: float
        The tukey window shape parameter passed to `scipy.signal.tukey`.
    psd_offset, psd_duration: float
        The power spectral density (psd) is estimated using data from
        `center_time+psd_offset` to `center_time+psd_offset + psd_duration`.
    outdir: str
        Directory where the psd files are saved
    plot: bool
        If true, create an ASD + strain plot
    filter_freq: float
        Low pass filter frequency
    **kwargs:
        All keyword arguments are passed to
        `gwpy.timeseries.TimeSeries.fetch_open_data()`.

    Returns
    -------
    interferometer: `tupak.detector.Interferometer`
        An Interferometer instance with a PSD and frequency-domain strain data.

    """

    sampling_frequency = int(strain.sample_rate.value)

    # Low pass filter
    bp = filter_design.lowpass(filter_freq, strain.sample_rate)
    strain = strain.filter(bp, filtfilt=True)
    strain = strain.crop(*strain.span.contract(1))

    time_series = strain.times.value
    time_duration = time_series[-1] - time_series[0]

    # Apply Tukey window
    N = len(time_series)
    strain = strain * signal.windows.tukey(N, alpha=alpha)

    frequency_domain_strain, frequencies = nfft(strain.value, sampling_frequency)

    return frequency_domain_strain, frequencies


def set_up_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="Command line interface for tupak scripts")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help=("Increase output verbosity [logging.DEBUG]." +
                              " Overridden by script level settings"))
    parser.add_argument("-q", "--quite", action="store_true",
                        help=("Decrease output verbosity [logging.WARNING]." +
                              " Overridden by script level settings"))
    parser.add_argument("-c", "--clean", action="store_true",
                        help="Force clean data, never use cached data")
    parser.add_argument("-u", "--use-cached", action="store_true",
                        help="Force cached data and do not check its validity")
    parser.add_argument("-d", "--detectors",  nargs='+',
                        default=['H1', 'L1', 'V1'],
                        help=("List of detectors to use in open data calls, "
                              "e.g. -d H1 L1 for H1 and L1"))
    parser.add_argument("-t", "--test", action="store_true",
                        help=("Used for testing only: don't run full PE, but"
                              " just check nothing breaks"))
    args, _ = parser.parse_known_args()

    if args.quite:
        args.log_level = logging.WARNING
    elif args.verbose:
        args.log_level = logging.DEBUG
    else:
        args.log_level = logging.INFO

    return args


command_line_args = set_up_command_line_arguments()

if 'DISPLAY' in os.environ:
    pass
else:
    logging.info('No $DISPLAY environment variable found, so importing \
                  matplotlib.pyplot with non-interactive "Agg" backend.')
    import matplotlib
    matplotlib.use('Agg')




