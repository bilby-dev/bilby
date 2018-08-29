from __future__ import division
import logging
import os
import numpy as np
from math import fmod
import argparse
import traceback
import inspect

logger = logging.getLogger('tupak')

# Constants

speed_of_light = 299792458.0  # speed of light in m/s
parsec = 3.085677581 * 1e16
solar_mass = 1.98855 * 1e30
radius_of_earth = 6371 * 1e3  # metres


def infer_parameters_from_function(func):
    """ Infers the arguments of function (except the first arg which is
        assumed to be the dep. variable)
    """
    parameters = inspect.getargspec(func).args
    parameters.pop(0)
    return parameters


def get_sampling_frequency(time_series):
    """
    Calculate sampling frequency from a time series

    Returns
    -------
    float: Sampling frequency of the time series

    Raises
    -------
    ValueError: If the time series is not evenly sampled.

    """
    tol = 1e-10
    if np.ptp(np.diff(time_series)) > tol:
        raise ValueError("Your time series was not evenly sampled")
    else:
        return 1. / (time_series[1] - time_series[0])


def get_sampling_frequency_and_duration_from_time_array(time_array):
    """
    Calculate sampling frequency and duration from a time array

    Returns
    -------
    sampling_frequency, duration:

    Raises
    -------
    ValueError: If the time_array is not evenly sampled.

    """

    sampling_frequency = get_sampling_frequency(time_array)
    duration = time_array[-1] - time_array[0]
    return sampling_frequency, duration


def get_sampling_frequency_and_duration_from_frequency_array(frequency_array):
    """
    Calculate sampling frequency and duration from a frequency array

    Returns
    -------
    sampling_frequency, duration:

    Raises
    -------
    ValueError: If the frequency_array is not evenly sampled.

    """

    tol = 1e-10
    if np.ptp(np.diff(frequency_array)) > tol:
        raise ValueError("Your frequency series was not evenly sampled")

    number_of_frequencies = len(frequency_array)
    delta_freq = frequency_array[1] - frequency_array[0]
    duration = 1 / delta_freq
    sampling_frequency = 2 * number_of_frequencies / duration
    return sampling_frequency, duration


def create_time_series(sampling_frequency, duration, starting_time=0.):
    """

    Parameters
    ----------
    sampling_frequency: float
    duration: float
    starting_time: float, optional

    Returns
    -------
    float: An equidistant time series given the parameters

    """
    return np.arange(starting_time, starting_time+duration, 1./sampling_frequency)


def ra_dec_to_theta_phi(ra, dec, gmst):
    """ Convert from RA and DEC to polar coordinates on celestial sphere

    Parameters
    -------
    ra: float
        right ascension in radians
    dec: float
        declination in radians
    gmst: float
        Greenwich mean sidereal time of arrival of the signal in radians

    Returns
    -------
    float: zenith angle in radians
    float: azimuthal angle in radians

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

    Parameters
    -------
    gps_time: float
        gps time

    Returns
    -------
    float: Greenwich mean sidereal time in radians

    """
    omega_earth = 2 * np.pi * (1 / 365.2425 + 1) / 86400.
    gps_2000 = 630720013.
    gmst_2000 = (6 + 39. / 60 + 51.251406103947375 / 3600) * np.pi / 12
    correction_2018 = -0.00017782487379358614
    sidereal_time = omega_earth * (gps_time - gps_2000) + gmst_2000 + correction_2018
    gmst = fmod(sidereal_time, 2 * np.pi)
    return gmst


def create_frequency_series(sampling_frequency, duration):
    """ Create a frequency series with the correct length and spacing.

    Parameters
    -------
    sampling_frequency: float
    duration: float
        duration of data

    Returns
    -------
    array_like: frequency series

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
    """ Create white_noise which is then coloured by a given PSD

    Parameters
    -------
    sampling_frequency: float
    duration: float
        duration of the data

    Returns
    -------
    array_like: white noise
    array_like: frequency array
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


def nfft(time_domain_strain, sampling_frequency):
    """ Perform an FFT while keeping track of the frequency bins. Assumes input
        time series is real (positive frequencies only).

    Parameters
    ----------
    time_domain_strain: array_like
        Time series of strain data.
    sampling_frequency: float
        Sampling frequency of the data.

    Returns
    -------
    frequency_domain_strain, frequency_array: (array, array)
        Single-sided FFT of time domain strain normalised to units of
        strain / Hz, and the associated frequency_array.

    """

    if np.ndim(sampling_frequency) != 0:
        raise ValueError("Sampling frequency must be interger or float")

    # add one zero padding if time series doesn't have even number of samples
    if np.mod(len(time_domain_strain), 2) == 1:
        time_domain_strain = np.append(time_domain_strain, 0)
    LL = len(time_domain_strain)
    # frequency range
    frequency_array = sampling_frequency / 2 * np.linspace(0, 1, int(LL/2+1))

    # calculate FFT
    # rfft computes the fft for real inputs
    frequency_domain_strain = np.fft.rfft(time_domain_strain)

    # normalise to units of strain / Hz
    norm_frequency_domain_strain = frequency_domain_strain / sampling_frequency

    return norm_frequency_domain_strain, frequency_array


def infft(frequency_domain_strain, sampling_frequency):
    """ Inverse FFT for use in conjunction with nfft.

    Parameters
    ----------
    frequency_domain_strain: array_like
        Single-sided, normalised FFT of the time-domain strain data (in units
        of strain / Hz).
    sampling_frequency: float
        Sampling frequency of the data.

    Returns
    -------
    time_domain_strain: array
        An array of the time domain strain
    """

    if np.ndim(sampling_frequency) != 0:
        raise ValueError("Sampling frequency must be integer or float")

    time_domain_strain_norm = np.fft.irfft(frequency_domain_strain)
    time_domain_strain = time_domain_strain_norm * sampling_frequency
    return time_domain_strain


def setup_logger(outdir=None, label=None, log_level=None, print_version=False):
    """ Setup logging output: call at the start of the script to use

    Parameters
    ----------
    outdir, label: str
        If supplied, write the logging output to outdir/label.log
    log_level: str, optional
        ['debug', 'info', 'warning']
        Either a string from the list above, or an integer as specified
        in https://docs.python.org/2/library/logging.html#logging-levels
    print_version: bool
        If true, print version information
    """

    if type(log_level) is str:
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError('log_level {} not understood'.format(log_level))
    elif log_level is None:
        level = command_line_args.log_level
    else:
        level = int(log_level)

    logger = logging.getLogger('tupak')
    logger.propagate = False
    logger.setLevel(level)

    if any([type(h) == logging.StreamHandler for h in logger.handlers]) is False:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(name)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if any([type(h) == logging.FileHandler for h in logger.handlers]) is False:
        if label:
            if outdir:
                check_directory_exists_and_if_not_mkdir(outdir)
            else:
                outdir = '.'
            log_file = '{}/{}.log'.format(outdir, label)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))

            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    version_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), '.version')
    with open(version_file, 'r') as f:
        version = f.readline().rstrip()

    if print_version:
        logger.info('Running tupak version: {}'.format(version))


def get_progress_bar(module='tqdm'):
    """
    TODO: Write proper docstring
    """
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
    """ Convert from spherical coordinates to cartesian.

    Parameters
    -------
    radius: float
        radial coordinate
    theta: float
        axial coordinate
    phi: float
        azimuthal coordinate

    Returns
    -------
    list: cartesian vector
    """
    cartesian = [radius * np.sin(theta) * np.cos(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(theta)]
    return cartesian


def check_directory_exists_and_if_not_mkdir(directory):
    """ Checks if the given directory exists and creates it if it does not exist

    Parameters
    ----------
    directory: str
        Name of the directory

    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.debug('Making directory {}'.format(directory))
    else:
        logger.debug('Directory {} exists'.format(directory))


def set_up_command_line_arguments():
    """ Sets up some command line arguments that can be used to modify how scripts are run.

    Returns
    -------
    list: A list of the used arguments

    """
    parser = argparse.ArgumentParser(
        description="Command line interface for tupak scripts")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help=("Increase output verbosity [logging.DEBUG]." +
                              " Overridden by script level settings"))
    parser.add_argument("-q", "--quiet", action="store_true",
                        help=("Decrease output verbosity [logging.WARNING]." +
                              " Overridden by script level settings"))
    parser.add_argument("-c", "--clean", action="store_true",
                        help="Force clean data, never use cached data")
    parser.add_argument("-u", "--use-cached", action="store_true",
                        help="Force cached data and do not check its validity")
    parser.add_argument("-t", "--test", action="store_true",
                        help=("Used for testing only: don't run full PE, but"
                              " just check nothing breaks"))
    args, _ = parser.parse_known_args()

    if args.quiet:
        args.log_level = logging.WARNING
    elif args.verbose:
        args.log_level = logging.DEBUG
    else:
        args.log_level = logging.INFO

    return args


command_line_args = set_up_command_line_arguments()
setup_logger(print_version=True)

if 'DISPLAY' in os.environ:
    logger.debug("DISPLAY={} environment found".format(os.environ['DISPLAY']))
    pass
else:
    logger.debug('No $DISPLAY environment variable found, so importing \
                   matplotlib.pyplot with non-interactive "Agg" backend.')
    import matplotlib
    import matplotlib.pyplot as plt

    non_gui_backends = matplotlib.rcsetup.non_interactive_bk
    for backend in non_gui_backends:
        try:
            logger.debug("Trying backend {}".format(backend))
            matplotlib.use(backend, warn=False)
            plt.switch_backend(backend)
            break
        except Exception as e:
            print(traceback.format_exc())
