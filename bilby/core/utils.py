
from distutils.spawn import find_executable
import logging
import os
import shutil
import sys
from math import fmod
import argparse
import inspect
import functools
import types
import subprocess
import multiprocessing
from importlib import import_module
from numbers import Number
import json
import warnings

import numpy as np
from scipy.interpolate import interp2d
from scipy.special import logsumexp
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger('bilby')

# Constants: values taken from LAL 505df9dd2e69b4812f1e8eee3a6d468ba7f80674
speed_of_light = 299792458.0  # m/s
parsec = 3.085677581491367e+16  # m
solar_mass = 1.9884099021470415e+30  # Kg
radius_of_earth = 6378136.6  # m
gravitational_constant = 6.6743e-11  # m^3 kg^-1 s^-2

_TOL = 14


def infer_parameters_from_function(func):
    """ Infers the arguments of a function
    (except the first arg which is assumed to be the dep. variable).

    Throws out `*args` and `**kwargs` type arguments

    Can deal with type hinting!

    Parameters
    ==========
    func: function or method
       The function or method for which the parameters should be inferred.

    Returns
    =======
    list: A list of strings with the parameters

    Raises
    ======
    ValueError
       If the object passed to the function is neither a function nor a method.

    Notes
    =====
    In order to handle methods the `type` of the function is checked, and
    if a method has been passed the first *two* arguments are removed rather than just the first one.
    This allows the reference to the instance (conventionally named `self`)
    to be removed.
    """
    if isinstance(func, types.MethodType):
        return infer_args_from_function_except_n_args(func=func, n=2)
    elif isinstance(func, types.FunctionType):
        return _infer_args_from_function_except_for_first_arg(func=func)
    else:
        raise ValueError("This doesn't look like a function.")


def infer_args_from_method(method):
    """ Infers all arguments of a method except for `self`

    Throws out `*args` and `**kwargs` type arguments.

    Can deal with type hinting!

    Returns
    =======
    list: A list of strings with the parameters
    """
    return infer_args_from_function_except_n_args(func=method, n=1)


def infer_args_from_function_except_n_args(func, n=1):
    """ Inspects a function to find its arguments, and ignoring the
    first n of these, returns a list of arguments from the function's
    signature.

    Parameters
    ==========
    func : function or method
       The function from which the arguments should be inferred.
    n : int
       The number of arguments which should be ignored, staring at the beginning.

    Returns
    =======
    parameters: list
       A list of parameters of the function, omitting the first `n`.

    Extended Summary
    ================
    This function is intended to allow the handling of named arguments
    in both functions and methods; this is important, since the first
    argument of an instance method will be the instance.

    See Also
    ========
    infer_args_from_method: Provides the arguments for a method
    infer_args_from_function: Provides the arguments for a function
    infer_args_from_function_except_first_arg: Provides all but first argument of a function or method.

    Examples
    ========

    .. code-block:: python

        >>> def hello(a, b, c, d):
        >>>     pass
        >>>
        >>> infer_args_from_function_except_n_args(hello, 2)
        ['c', 'd']

    """
    try:
        parameters = inspect.getfullargspec(func).args
    except AttributeError:
        parameters = inspect.getargspec(func).args
    del(parameters[:n])
    return parameters


def _infer_args_from_function_except_for_first_arg(func):
    return infer_args_from_function_except_n_args(func=func, n=1)


def get_dict_with_properties(obj):
    property_names = [p for p in dir(obj.__class__)
                      if isinstance(getattr(obj.__class__, p), property)]
    dict_with_properties = obj.__dict__.copy()
    for key in property_names:
        dict_with_properties[key] = getattr(obj, key)
    return dict_with_properties


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


def ra_dec_to_theta_phi(ra, dec, gmst):
    """ Convert from RA and DEC to polar coordinates on celestial sphere

    Parameters
    ==========
    ra: float
        right ascension in radians
    dec: float
        declination in radians
    gmst: float
        Greenwich mean sidereal time of arrival of the signal in radians

    Returns
    =======
    float: zenith angle in radians
    float: azimuthal angle in radians

    """
    phi = ra - gmst
    theta = np.pi / 2 - dec
    return theta, phi


def theta_phi_to_ra_dec(theta, phi, gmst):
    ra = phi + gmst
    dec = np.pi / 2 - theta
    return ra, dec


def gps_time_to_gmst(gps_time):
    """
    Convert gps time to Greenwich mean sidereal time in radians

    This method assumes a constant rotation rate of earth since 00:00:00, 1 Jan. 2000
    A correction has been applied to give the exact correct value for 00:00:00, 1 Jan. 2018
    Error accumulates at a rate of ~0.0001 radians/decade.

    Parameters
    ==========
    gps_time: float
        gps time

    Returns
    =======
    float: Greenwich mean sidereal time in radians

    """
    warnings.warn(
        "Function gps_time_to_gmst deprecated, use "
        "lal.GreenwichMeanSiderealTime(time) instead",
        DeprecationWarning)
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
    ==========
    sampling_frequency: float
    duration: float
        duration of the data

    Returns
    =======
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


def setup_logger(outdir=None, label=None, log_level='INFO', print_version=False):
    """ Setup logging output: call at the start of the script to use

    Parameters
    ==========
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


def spherical_to_cartesian(radius, theta, phi):
    """ Convert from spherical coordinates to cartesian.

    Parameters
    ==========
    radius: float
        radial coordinate
    theta: float
        axial coordinate
    phi: float
        azimuthal coordinate

    Returns
    =======
    list: cartesian vector
    """
    cartesian = [radius * np.sin(theta) * np.cos(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(theta)]
    return cartesian


def check_directory_exists_and_if_not_mkdir(directory):
    """ Checks if the given directory exists and creates it if it does not exist

    Parameters
    ==========
    directory: str
        Name of the directory

    """
    if directory == "":
        return
    elif not os.path.exists(directory):
        os.makedirs(directory)
        logger.debug('Making directory {}'.format(directory))
    else:
        logger.debug('Directory {} exists'.format(directory))


def set_up_command_line_arguments():
    """ Sets up command line arguments that can be used to modify how scripts are run.

    Returns
    =======
    command_line_args, command_line_parser: tuple
        The command_line_args is a Namespace of the command line arguments while
        the command_line_parser can be given to a new `argparse.ArgumentParser`
        as a parent object from which to inherit.

    Notes
    =====
        The command line arguments are passed initially at runtime, but this parser
        does not have a `--help` option (i.e., the command line options are
        available for any script which includes `import bilby`, but no help command
        is available. This is done to avoid conflicts with child argparse routines
        (see the example below).

    Examples
    ========
    In the following example we demonstrate how to setup a custom command line for a
    project which uses bilby.

    .. code-block:: python

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
    ==========
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
    =======
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
    ==========
    lnf: array_like
        A :class:`numpy.ndarray` of values that are the natural logarithm of a function
    dx: Union[array_like, float]
        A :class:`numpy.ndarray` of steps sizes between values in the function, or a
        single step size value.

    Returns
    =======
    The natural logarithm of the area under the function.
    """
    return np.log(dx / 2.) + logsumexp([logsumexp(lnf[:-1]), logsumexp(lnf[1:])])


class SamplesSummary(object):
    """ Object to store a set of samples and calculate summary statistics

    Parameters
    ==========
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
    ==========
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
    ==========
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
        """  Modified version of the interp2d call method.

        This avoids the outer product that is done when two numpy
        arrays are passed.

        Parameters
        ==========
        x: See superclass
        y: See superclass
        dx: See superclass
        dy: See superclass
        assume_sorted: bool, optional
            This is just a place holder to prevent a warning.
            Overwriting this will not do anything

        Returns
        =======
        array_like: See superclass

        """
        from scipy.interpolate.dfitpack import bispeu
        x, y = self._sanitize_inputs(x, y)
        out_of_bounds_x = (x < self.x_min) | (x > self.x_max)
        out_of_bounds_y = (y < self.y_min) | (y > self.y_max)
        bad = out_of_bounds_x | out_of_bounds_y
        if isinstance(x, Number) and isinstance(y, Number):
            if bad:
                output = self.fill_value
                ier = 0
            else:
                output, ier = bispeu(*self.tck, x, y)
                output = float(output)
        else:
            output = np.empty_like(x)
            output[bad] = self.fill_value
            output[~bad], ier = bispeu(*self.tck, x[~bad], y[~bad])
        if ier == 10:
            raise ValueError("Invalid input data")
        elif ier:
            raise TypeError("An error occurred")
        return output

    @staticmethod
    def _sanitize_inputs(x, y):
        if isinstance(x, np.ndarray) and x.size == 1:
            x = float(x)
        if isinstance(y, np.ndarray) and y.size == 1:
            y = float(y)
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if x.shape != y.shape:
                raise ValueError(
                    "UnsortedInterp2d received unequally shaped arrays"
                )
        elif isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
            y = y * np.ones_like(x)
        elif not isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            x = x * np.ones_like(y)
        return x, y


#  Instantiate the default argument parser at runtime
command_line_args, command_line_parser = set_up_command_line_arguments()
#  Instantiate the default logging
setup_logger(print_version=False, log_level=command_line_args.log_level)


class BilbyJsonEncoder(json.JSONEncoder):

    def default(self, obj):
        from .prior import MultivariateGaussianDist, Prior, PriorDict
        from ..gw.prior import HealPixMapPriorDist
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, PriorDict):
            return {'__prior_dict__': True, 'content': obj._get_json_dict()}
        if isinstance(obj, (MultivariateGaussianDist, HealPixMapPriorDist, Prior)):
            return {'__prior__': True, '__module__': obj.__module__,
                    '__name__': obj.__class__.__name__,
                    'kwargs': dict(obj.get_instantiation_dict())}
        try:
            from astropy import cosmology as cosmo, units
            if isinstance(obj, cosmo.FLRW):
                return encode_astropy_cosmology(obj)
            if isinstance(obj, units.Quantity):
                return encode_astropy_quantity(obj)
            if isinstance(obj, units.PrefixUnit):
                return str(obj)
        except ImportError:
            logger.debug("Cannot import astropy, cannot write cosmological priors")
        if isinstance(obj, np.ndarray):
            return {'__array__': True, 'content': obj.tolist()}
        if isinstance(obj, complex):
            return {'__complex__': True, 'real': obj.real, 'imag': obj.imag}
        if isinstance(obj, pd.DataFrame):
            return {'__dataframe__': True, 'content': obj.to_dict(orient='list')}
        if isinstance(obj, pd.Series):
            return {'__series__': True, 'content': obj.to_dict()}
        if inspect.isfunction(obj):
            return {"__function__": True, "__module__": obj.__module__, "__name__": obj.__name__}
        if inspect.isclass(obj):
            return {"__class__": True, "__module__": obj.__module__, "__name__": obj.__name__}
        return json.JSONEncoder.default(self, obj)


def encode_astropy_cosmology(obj):
    cls_name = obj.__class__.__name__
    dct = {key: getattr(obj, key) for
           key in infer_args_from_method(obj.__init__)}
    dct['__cosmology__'] = True
    dct['__name__'] = cls_name
    return dct


def encode_astropy_quantity(dct):
    dct = dict(__astropy_quantity__=True, value=dct.value, unit=str(dct.unit))
    if isinstance(dct['value'], np.ndarray):
        dct['value'] = list(dct['value'])
    return dct


def move_old_file(filename, overwrite=False):
    """ Moves or removes an old file.

    Parameters
    ==========
    filename: str
        Name of the file to be move
    overwrite: bool, optional
        Whether or not to remove the file or to change the name
        to filename + '.old'
    """
    if os.path.isfile(filename):
        if overwrite:
            logger.debug('Removing existing file {}'.format(filename))
            os.remove(filename)
        else:
            logger.debug(
                'Renaming existing file {} to {}.old'.format(filename,
                                                             filename))
            shutil.move(filename, filename + '.old')
    logger.debug("Saving result to {}".format(filename))


def load_json(filename, gzip):
    if gzip or os.path.splitext(filename)[1].lstrip('.') == 'gz':
        import gzip
        with gzip.GzipFile(filename, 'r') as file:
            json_str = file.read().decode('utf-8')
        dictionary = json.loads(json_str, object_hook=decode_bilby_json)
    else:
        with open(filename, 'r') as file:
            dictionary = json.load(file, object_hook=decode_bilby_json)
    return dictionary


def decode_bilby_json(dct):
    if dct.get("__prior_dict__", False):
        cls = getattr(import_module(dct['__module__']), dct['__name__'])
        obj = cls._get_from_json_dict(dct)
        return obj
    if dct.get("__prior__", False):
        cls = getattr(import_module(dct['__module__']), dct['__name__'])
        obj = cls(**dct['kwargs'])
        return obj
    if dct.get("__cosmology__", False):
        return decode_astropy_cosmology(dct)
    if dct.get("__astropy_quantity__", False):
        return decode_astropy_quantity(dct)
    if dct.get("__array__", False):
        return np.asarray(dct["content"])
    if dct.get("__complex__", False):
        return complex(dct["real"], dct["imag"])
    if dct.get("__dataframe__", False):
        return pd.DataFrame(dct['content'])
    if dct.get("__series__", False):
        return pd.Series(dct['content'])
    if dct.get("__function__", False) or dct.get("__class__", False):
        default = ".".join([dct["__module__"], dct["__name__"]])
        return getattr(import_module(dct["__module__"]), dct["__name__"], default)
    return dct


def decode_astropy_cosmology(dct):
    try:
        from astropy import cosmology as cosmo
        cosmo_cls = getattr(cosmo, dct['__name__'])
        del dct['__cosmology__'], dct['__name__']
        return cosmo_cls(**dct)
    except ImportError:
        logger.debug("Cannot import astropy, cosmological priors may not be "
                     "properly loaded.")
        return dct


def decode_astropy_quantity(dct):
    try:
        from astropy import units
        if dct['value'] is None:
            return None
        else:
            del dct['__astropy_quantity__']
            return units.Quantity(**dct)
    except ImportError:
        logger.debug("Cannot import astropy, cosmological priors may not be "
                     "properly loaded.")
        return dct


def reflect(u):
    """
    Iteratively reflect a number until it is contained in [0, 1].

    This is for priors with a reflective boundary condition, all numbers in the
    set `u = 2n +/- x` should be mapped to x.

    For the `+` case we just take `u % 1`.
    For the `-` case we take `1 - (u % 1)`.

    E.g., -0.9, 1.1, and 2.9 should all map to 0.9.

    Parameters
    ==========
    u: array-like
        The array of points to map to the unit cube

    Returns
    =======
    u: array-like
       The input array, modified in place.
    """
    idxs_even = np.mod(u, 2) < 1
    u[idxs_even] = np.mod(u[idxs_even], 1)
    u[~idxs_even] = 1 - np.mod(u[~idxs_even], 1)
    return u


def safe_file_dump(data, filename, module):
    """ Safely dump data to a .pickle file

    Parameters
    ==========
    data:
        data to dump
    filename: str
        The file to dump to
    module: pickle, dill
        The python module to use
    """

    temp_filename = filename + ".temp"
    with open(temp_filename, "wb") as file:
        module.dump(data, file)
    shutil.move(temp_filename, filename)


def latex_plot_format(func):
    """
    Wrap the plotting function to set rcParams dependent on environment variables

    The rcparams can be set directly from the env. variable `BILBY_STYLE` to
    point to a matplotlib style file. Or, if `BILBY_STYLE=default` (any case) a
    default setup is used, this is enabled by default. To not use any rcParams,
    set `BILBY_STYLE=none`. Occasionally, issues arrise with the latex
    `mathdefault` command. A fix is to define this command in the rcParams. An
    env. variable `BILBY_MATHDEFAULT` can be used to turn this fix on/off.
    Setting `BILBY_MATHDEFAULT=1` will enable the fix, all other choices
    (including undefined) will disable it. Additionally, the BILBY_STYLE and
    BILBY_MATHDEFAULT arguments can be passed into any
    latex_plot_format-wrapped plotting function and will be set directly.

    """
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        from matplotlib import rcParams

        if "BILBY_STYLE" in kwargs:
            bilby_style = kwargs.pop("BILBY_STYLE")
        else:
            bilby_style = os.environ.get("BILBY_STYLE", "default")

        if "BILBY_MATHDEFAULT" in kwargs:
            bilby_mathdefault = kwargs.pop("BILBY_MATHDEFAULT")
        else:
            bilby_mathdefault = int(os.environ.get("BILBY_MATHDEFAULT", "0"))

        if bilby_mathdefault == 1:
            logger.debug("Setting mathdefault in the rcParams")
            rcParams['text.latex.preamble'] = r'\providecommand{\mathdefault}[1][]{}'

        logger.debug("Using BILBY_STYLE={}".format(bilby_style))
        if bilby_style.lower() == "none":
            return func(*args, **kwargs)
        elif os.path.isfile(bilby_style):
            plt.style.use(bilby_style)
            return func(*args, **kwargs)
        elif bilby_style in plt.style.available:
            plt.style.use(bilby_style)
            return func(*args, **kwargs)
        elif bilby_style.lower() == "default":
            _old_tex = rcParams["text.usetex"]
            _old_serif = rcParams["font.serif"]
            _old_family = rcParams["font.family"]
            if find_executable("latex"):
                rcParams["text.usetex"] = True
            else:
                rcParams["text.usetex"] = False
            rcParams["font.serif"] = "Computer Modern Roman"
            rcParams["font.family"] = "serif"
            rcParams["text.usetex"] = _old_tex
            rcParams["font.serif"] = _old_serif
            rcParams["font.family"] = _old_family
            return func(*args, **kwargs)
        else:
            logger.debug(
                "Environment variable BILBY_STYLE={} not used"
                .format(bilby_style)
            )
            return func(*args, **kwargs)
    return wrapper_decorator


def safe_save_figure(fig, filename, **kwargs):
    check_directory_exists_and_if_not_mkdir(os.path.dirname(filename))
    from matplotlib import rcParams
    try:
        fig.savefig(fname=filename, **kwargs)
    except RuntimeError:
        logger.debug(
            "Failed to save plot with tex labels turning off tex."
        )
        rcParams["text.usetex"] = False
        fig.savefig(fname=filename, **kwargs)


def kish_log_effective_sample_size(ln_weights):
    """ Calculate the Kish effective sample size from the natural-log weights

    See https://en.wikipedia.org/wiki/Effective_sample_size for details

    Parameters
    ==========
    ln_weights: array
        An array of the ln-weights

    Returns
    =======
    ln_n_eff:
        The natural-log of the effective sample size

    """
    log_n_eff = 2 * logsumexp(ln_weights) - logsumexp(2 * ln_weights)
    return log_n_eff


def get_function_path(func):
    if hasattr(func, "__module__") and hasattr(func, "__name__"):
        return "{}.{}".format(func.__module__, func.__name__)
    else:
        return func


def loaded_modules_dict():
    module_names = list(sys.modules.keys())
    vdict = {}
    for key in module_names:
        if "." not in key:
            vdict[key] = str(getattr(sys.modules[key], "__version__", "N/A"))
    return vdict


class IllegalDurationAndSamplingFrequencyException(Exception):
    pass


class tcolors:
    KEY = '\033[93m'
    VALUE = '\033[91m'
    HIGHLIGHT = '\033[95m'
    END = '\033[0m'


def decode_from_hdf5(item):
    """
    Decode an item from HDF5 format to python type.

    This currently just converts __none__ to None and some arrays to lists

    .. versionadded:: 1.0.0

    Parameters
    ----------
    item: object
        Item to be decoded

    Returns
    -------
    output: object
        Converted input item
    """
    if isinstance(item, str) and item == "__none__":
        output = None
    elif isinstance(item, bytes) and item == b"__none__":
        output = None
    elif isinstance(item, (bytes, bytearray)):
        output = item.decode()
    elif isinstance(item, np.ndarray):
        if "|S" in str(item.dtype) or isinstance(item[0], bytes):
            output = [it.decode() for it in item]
        else:
            output = item
    else:
        output = item
    return output


def encode_for_hdf5(item):
    """
    Encode an item to a HDF5 savable format.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    item: object
        Object to be encoded, specific options are provided for Bilby types

    Returns
    -------
    output: object
        Input item converted into HDF5 savable format
    """
    from .prior.dict import PriorDict
    if isinstance(item, np.int_):
        item = int(item)
    elif isinstance(item, np.float_):
        item = float(item)
    elif isinstance(item, np.complex_):
        item = complex(item)
    if isinstance(item, (np.ndarray, int, float, complex, str, bytes)):
        output = item
    elif item is None:
        output = "__none__"
    elif isinstance(item, list):
        if len(item) == 0:
            output = item
        elif isinstance(item[0], (str, bytes)) or item[0] is None:
            output = list()
            for value in item:
                if isinstance(value, str):
                    output.append(value.encode("utf-8"))
                elif isinstance(value, bytes):
                    output.append(value)
                else:
                    output.append(b"__none__")
        elif isinstance(item[0], (int, float, complex)):
            output = np.array(item)
    elif isinstance(item, PriorDict):
        output = json.dumps(item._get_json_dict())
    elif isinstance(item, pd.DataFrame):
        output = item.to_dict(orient="list")
    elif isinstance(item, pd.Series):
        output = item.to_dict()
    elif inspect.isfunction(item) or inspect.isclass(item):
        output = dict(__module__=item.__module__, __name__=item.__name__)
    elif isinstance(item, dict):
        output = item.copy()
    else:
        raise ValueError(f'Cannot save {type(item)} type')
    return output


def recursively_load_dict_contents_from_group(h5file, path):
    """
    Recursively load a HDF5 file into a dictionary

    .. versionadded:: 1.1.0

    Parameters
    ----------
    h5file: h5py.File
        Open h5py file object
    path: str
        Path within the HDF5 file

    Returns
    -------
    output: dict
        The contents of the HDF5 file unpacked into the dictionary.
    """
    import h5py
    output = dict()
    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):
            output[key] = decode_from_hdf5(item[()])
        elif isinstance(item, h5py.Group):
            output[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return output


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Recursively save a dictionary to a HDF5 group

    .. versionadded:: 1.1.0

    Parameters
    ----------
    h5file: h5py.File
        Open HDF5 file
    path: str
        Path inside the HDF5 file
    dic: dict
        The dictionary containing the data
    """
    for key, item in dic.items():
        item = encode_for_hdf5(item)
        if isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            h5file[path + key] = item


def docstring(docstr, sep="\n"):
    """
    Decorator: Append to a function's docstring.

    This is required for e.g., :code:`classmethods` as the :code:`__doc__`
    can't be changed after.

    Parameters
    ==========
    docstr: str
        The docstring
    sep: str
        Separation character for appending the existing docstring.
    """
    def _decorator(func):
        if func.__doc__ is None:
            func.__doc__ = docstr
        else:
            func.__doc__ = sep.join([func.__doc__, docstr])
        return func
    return _decorator
