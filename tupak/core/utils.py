from __future__ import division
import logging
import os
import numpy as np
from math import fmod
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




