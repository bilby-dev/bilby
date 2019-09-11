from __future__ import division

import logging
import os
from math import fmod
import argparse
import traceback
import inspect
import subprocess
import json
import multiprocessing

import numpy as np
from scipy.interpolate import interp2d
from scipy.special import logsumexp
import pandas as pd

logger = logging.getLogger('bilby')


# Constants: values taken from astropy v3.0.4
speed_of_light = 299792458.0  # m/s
parsec = 3.0856775814671916e+16  # m
solar_mass = 1.9884754153381438e+30  # Kg
radius_of_earth = 6378100.0  # m

_TOL = 14


def infer_parameters_from_function(func):
    """ Infers the arguments of a function
        (except the first arg which is assumed to be the dep. variable).

        Throws out *args and **kwargs type arguments

        Can deal with type hinting!

        Returns
        ---------
        list: A list of strings with the parameters
    """
    return _infer_args_from_function_except_for_first_arg(func=func)


def infer_args_from_method(method):
    """ Infers all arguments of a method except for 'self'

    Throws out *args and **kwargs type arguments.

    Can deal with type hinting!

    Returns
    ---------
    list: A list of strings with the parameters
    """
    return _infer_args_from_function_except_for_first_arg(func=method)


def _infer_args_from_function_except_for_first_arg(func):
    try:
        parameters = inspect.getfullargspec(func).args
    except AttributeError:
        parameters = inspect.getargspec(func).args
    parameters.pop(0)
    return parameters


def get_sampling_frequency(time_array):
    """
    Calculate sampling frequency from a time series

    Attributes:
    -------
    time_array: array_like
        Time array to get sampling_frequency from

    Returns
    -------
    Sampling frequency of the time series: float

    Raises
    -------
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

    Attributes:
    -------
    time_array: array_like
        Time array to get sampling_frequency/duration from: array_like

    Returns
    -------
    sampling_frequency, duration: float, float

    Raises
    -------
    ValueError: If the time_array is not evenly sampled.

    """

    sampling_frequency = get_sampling_frequency(time_array)
    duration = len(time_array) / sampling_frequency
    return sampling_frequency, duration


def get_sampling_frequency_and_duration_from_frequency_array(frequency_array):
    """
    Calculate sampling frequency and duration from a frequency array

    Attributes:
    -------
    frequency_array: array_like
        Frequency array to get sampling_frequency/duration from: array_like

    Returns
    -------
    sampling_frequency, duration: float, float

    Raises
    -------
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
    ----------
    sampling_frequency: float
    duration: float
    starting_time: float, optional

    Returns
    -------
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
    -------
    sampling_frequency: float
    duration: float

    Returns
    -------
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
    -------
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

    delta_freq = 1. / duration

    frequencies = create_frequency_series(sampling_frequency, duration)

    norm1 = 0.5 * (1. / delta_freq)**0.5
    re1 = np.random.normal(0, norm1, len(frequencies))
    im1 = np.random.normal(0, norm1, len(frequencies))
    htilde1 = re1 + 1j * im1

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
    ----------
    frequency_domain_strain: array_like
        Single-sided, normalised FFT of the time-domain strain data (in units
        of strain / Hz).
    sampling_frequency: int, float
        Sampling frequency of the data.

    Returns
    -------
    time_domain_strain: array_like
        An array of the time domain strain
    """
    time_domain_strain_norm = np.fft.irfft(frequency_domain_strain)
    time_domain_strain = time_domain_strain_norm * sampling_frequency
    return time_domain_strain


def setup_logger(outdir=None, label=None, log_level='INFO', print_version=False):
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
    else:
        level = int(log_level)

    logger = logging.getLogger('bilby')
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

    if print_version:
        version = get_version_information()
        logger.info('Running bilby version: {}'.format(version))


def get_version_information():
    version_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), '.version')
    try:
        with open(version_file, 'r') as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")


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
    """ Sets up command line arguments that can be used to modify how scripts are run.

    Returns
    -------
    command_line_args, command_line_parser: tuple
        The command_line_args is a Namespace of the command line arguments while
        the command_line_parser can be given to a new `argparse.ArgumentParser`
        as a parent object from which to inherit.

    Notes
    -----
        The command line arguments are passed initially at runtime, but this parser
        does not have a `--help` option (i.e., the command line options are
        available for any script which includes `import bilby`, but no help command
        is available. This is done to avoid conflicts with child argparse routines
        (see the example below).

    Example
    -------
    In the following example we demonstrate how to setup a custom command line for a
    project which uses bilby.

        # Here we import bilby, which initialses and parses the default command-line args
        >>> import bilby
        # The command line arguments can then be accessed via
        >>> bilby.core.utils.command_line_args
        Namespace(clean=False, log_level=20, quite=False)
        # Next, we import argparse and define a new argparse object
        >>> import argparse
        >>> parser = argparse.ArgumentParser(parents=[bilby.core.utils.command_line_parser])
        >>> parser.add_argument('--argument', type=int, default=1)
        >>> args = parser.parse_args()
        Namespace(clean=False, log_level=20, quite=False, argument=1)

    Placing these lines into a script, you'll be able to pass in the usual bilby default
    arguments, in addition to `--argument`. To see a list of all options, call the script
    with `--help`.

    """
    try:
        parser = argparse.ArgumentParser(
            description="Command line interface for bilby scripts",
            add_help=False, allow_abbrev=False)
    except TypeError:
        parser = argparse.ArgumentParser(
            description="Command line interface for bilby scripts",
            add_help=False)
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
    parser.add_argument("--sampler-help", nargs='?', default=False,
                        const='None', help="Print help for given sampler")
    parser.add_argument("--bilby-test-mode", action="store_true",
                        help=("Used for testing only: don't run full PE, but"
                              " just check nothing breaks"))
    parser.add_argument("--bilby-zero-likelihood-mode", action="store_true",
                        help=("Used for testing only: don't run full PE, but"
                              " just check nothing breaks"))
    args, unknown_args = parser.parse_known_args()
    if args.quiet:
        args.log_level = logging.WARNING
    elif args.verbose:
        args.log_level = logging.DEBUG
    else:
        args.log_level = logging.INFO

    return args, parser


def derivatives(vals, func, releps=1e-3, abseps=None, mineps=1e-9, reltol=1e-3,
                epsscale=0.5, nonfixedidx=None):
    """
    Calculate the partial derivatives of a function at a set of values. The
    derivatives are calculated using the central difference, using an iterative
    method to check that the values converge as step size decreases.

    Parameters
    ----------
    vals: array_like
        A set of values, that are passed to a function, at which to calculate
        the gradient of that function
    func:
        A function that takes in an array of values.
    releps: float, array_like, 1e-3
        The initial relative step size for calculating the derivative.
    abseps: float, array_like, None
        The initial absolute step size for calculating the derivative.
        This overrides `releps` if set.
        `releps` is set then that is used.
    mineps: float, 1e-9
        The minimum relative step size at which to stop iterations if no
        convergence is achieved.
    epsscale: float, 0.5
        The factor by which releps if scaled in each iteration.
    nonfixedidx: array_like, None
        An array of indices in `vals` that are _not_ fixed values and therefore
        can have derivatives taken. If `None` then derivatives of all values
        are calculated.

    Returns
    -------
    grads: array_like
        An array of gradients for each non-fixed value.
    """

    if nonfixedidx is None:
        nonfixedidx = range(len(vals))

    if len(nonfixedidx) > len(vals):
        raise ValueError("To many non-fixed values")

    if max(nonfixedidx) >= len(vals) or min(nonfixedidx) < 0:
        raise ValueError("Non-fixed indexes contain non-existant indices")

    grads = np.zeros(len(nonfixedidx))

    # maximum number of times the gradient can change sign
    flipflopmax = 10.

    # set steps
    if abseps is None:
        if isinstance(releps, float):
            eps = np.abs(vals) * releps
            eps[eps == 0.] = releps  # if any values are zero set eps to releps
            teps = releps * np.ones(len(vals))
        elif isinstance(releps, (list, np.ndarray)):
            if len(releps) != len(vals):
                raise ValueError("Problem with input relative step sizes")
            eps = np.multiply(np.abs(vals), releps)
            eps[eps == 0.] = np.array(releps)[eps == 0.]
            teps = releps
        else:
            raise RuntimeError("Relative step sizes are not a recognised type!")
    else:
        if isinstance(abseps, float):
            eps = abseps * np.ones(len(vals))
        elif isinstance(abseps, (list, np.ndarray)):
            if len(abseps) != len(vals):
                raise ValueError("Problem with input absolute step sizes")
            eps = np.array(abseps)
        else:
            raise RuntimeError("Absolute step sizes are not a recognised type!")
        teps = eps

    # for each value in vals calculate the gradient
    count = 0
    for i in nonfixedidx:
        # initial parameter diffs
        leps = eps[i]
        cureps = teps[i]

        flipflop = 0

        # get central finite difference
        fvals = np.copy(vals)
        bvals = np.copy(vals)

        # central difference
        fvals[i] += 0.5 * leps  # change forwards distance to half eps
        bvals[i] -= 0.5 * leps  # change backwards distance to half eps
        cdiff = (func(fvals) - func(bvals)) / leps

        while 1:
            fvals[i] -= 0.5 * leps  # remove old step
            bvals[i] += 0.5 * leps

            # change the difference by a factor of two
            cureps *= epsscale
            if cureps < mineps or flipflop > flipflopmax:
                # if no convergence set flat derivative (TODO: check if there is a better thing to do instead)
                logger.warning("Derivative calculation did not converge: setting flat derivative.")
                grads[count] = 0.
                break
            leps *= epsscale

            # central difference
            fvals[i] += 0.5 * leps  # change forwards distance to half eps
            bvals[i] -= 0.5 * leps  # change backwards distance to half eps
            cdiffnew = (func(fvals) - func(bvals)) / leps

            if cdiffnew == cdiff:
                grads[count] = cdiff
                break

            # check whether previous diff and current diff are the same within reltol
            rat = (cdiff / cdiffnew)
            if np.isfinite(rat) and rat > 0.:
                # gradient has not changed sign
                if np.abs(1. - rat) < reltol:
                    grads[count] = cdiffnew
                    break
                else:
                    cdiff = cdiffnew
                    continue
            else:
                cdiff = cdiffnew
                flipflop += 1
                continue

        count += 1

    return grads


def logtrapzexp(lnf, dx):
    """
    Perform trapezium rule integration for the logarithm of a function on a regular grid.

    Parameters
    ----------
    lnf: array_like
        A :class:`numpy.ndarray` of values that are the natural logarithm of a function
    dx: Union[array_like, float]
        A :class:`numpy.ndarray` of steps sizes between values in the function, or a
        single step size value.

    Returns
    -------
    The natural logarithm of the area under the function.
    """
    return np.log(dx / 2.) + logsumexp([logsumexp(lnf[:-1]), logsumexp(lnf[1:])])


class SamplesSummary(object):
    """ Object to store a set of samples and calculate summary statistics

    Parameters
    ----------
    samples: array_like
        Array of samples
    average: str {'median', 'mean'}
        Use either a median average or mean average when calculating relative
        uncertainties
    level: float
        The default confidence interval level, defaults t0 0.9

    """
    def __init__(self, samples, average='median', confidence_level=.9):
        self.samples = samples
        self.average = average
        self.confidence_level = confidence_level

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        self._samples = samples

    @property
    def confidence_level(self):
        return self._confidence_level

    @confidence_level.setter
    def confidence_level(self, confidence_level):
        if 0 < confidence_level and confidence_level < 1:
            self._confidence_level = confidence_level
        else:
            raise ValueError("Confidence level must be between 0 and 1")

    @property
    def average(self):
        if self._average == 'mean':
            return self.mean
        elif self._average == 'median':
            return self.median

    @average.setter
    def average(self, average):
        allowed_averages = ['mean', 'median']
        if average in allowed_averages:
            self._average = average
        else:
            raise ValueError("Average {} not in allowed averages".format(average))

    @property
    def median(self):
        return np.median(self.samples, axis=0)

    @property
    def mean(self):
        return np.mean(self.samples, axis=0)

    @property
    def _lower_level(self):
        """ The credible interval lower quantile value """
        return (1 - self.confidence_level) / 2.

    @property
    def _upper_level(self):
        """ The credible interval upper quantile value """
        return (1 + self.confidence_level) / 2.

    @property
    def lower_absolute_credible_interval(self):
        """ Absolute lower value of the credible interval """
        return np.quantile(self.samples, self._lower_level, axis=0)

    @property
    def upper_absolute_credible_interval(self):
        """ Absolute upper value of the credible interval """
        return np.quantile(self.samples, self._upper_level, axis=0)

    @property
    def lower_relative_credible_interval(self):
        """ Relative (to average) lower value of the credible interval """
        return self.lower_absolute_credible_interval - self.average

    @property
    def upper_relative_credible_interval(self):
        """ Relative (to average) upper value of the credible interval """
        return self.upper_absolute_credible_interval - self.average


def run_commandline(cl, log_level=20, raise_error=True, return_output=True):
    """Run a string cmd as a subprocess, check for errors and return output.

    Parameters
    ----------
    cl: str
        Command to run
    log_level: int
        See https://docs.python.org/2/library/logging.html#logging-levels,
        default is '20' (INFO)

    """

    logger.log(log_level, 'Now executing: ' + cl)
    if return_output:
        try:
            out = subprocess.check_output(
                cl, stderr=subprocess.STDOUT, shell=True,
                universal_newlines=True)
        except subprocess.CalledProcessError as e:
            logger.log(log_level, 'Execution failed: {}'.format(e.output))
            if raise_error:
                raise
            else:
                out = 0
        os.system('\n')
        return(out)
    else:
        process = subprocess.Popen(cl, shell=True)
        process.communicate()


class Counter(object):
    """
    General class to count number of times a function is Called, returns total
    number of function calls
    Parameters
    ----------
    initalval : int, 0
    number to start counting from
    """
    def __init__(self, initval=0):
        self.val = multiprocessing.RawValue('i', initval)
        self.lock = multiprocessing.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    @property
    def value(self):
        return self.val.value


class UnsortedInterp2d(interp2d):

    def __call__(self, x, y, dx=0, dy=0, assume_sorted=False):
        """  Wrapper to scipy.interpolate.interp2d which preserves the input ordering.

        See https://stackoverflow.com/questions/44941271/scipy-interp2d-returned-function-sorts-input-argument-automatically-and-undesira
        for the implementation details.


        Parameters
        ----------
        x: See superclass
        y: See superclass
        dx: See superclass
        dy: See superclass
        assume_sorted: bool, optional
            This is just a place holder to prevent a warning.
            Overwriting this will not do anything

        Returns
        ----------
        array_like: See superclass

        """
        unsorted_idxs = np.argsort(np.argsort(x))
        return interp2d.__call__(self, x, y, dx=dx, dy=dy, assume_sorted=False)[unsorted_idxs]


#  Instantiate the default argument parser at runtime
command_line_args, command_line_parser = set_up_command_line_arguments()
#  Instantiate the default logging
setup_logger(print_version=True, log_level=command_line_args.log_level)

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
        except Exception:
            print(traceback.format_exc())


class BilbyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {'__array__': True, 'content': obj.tolist()}
        if isinstance(obj, complex):
            return {'__complex__': True, 'real': obj.real, 'imag': obj.imag}
        if isinstance(obj, pd.DataFrame):
            return {'__dataframe__': True, 'content': obj.to_dict(orient='list')}
        return json.JSONEncoder.default(self, obj)


def decode_bilby_json(dct):
    if dct.get("__array__", False):
        return np.asarray(dct["content"])
    if dct.get("__complex__", False):
        return complex(dct["real"], dct["imag"])
    if dct.get("__dataframe__", False):
        return pd.DataFrame(dct['content'])
    return dct


class IllegalDurationAndSamplingFrequencyException(Exception):
    pass
