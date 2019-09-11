import os
import sys

import numpy as np

from ...core import utils
from ...core.utils import logger
from .. import utils as gwutils
from .interferometer import Interferometer
from .psd import PowerSpectralDensity
from .strain_data import InterferometerStrainData


class InterferometerList(list):
    """ A list of Interferometer objects """

    def __init__(self, interferometers):
        """ Instantiate a InterferometerList

        The InterferometerList is a list of Interferometer objects, each
        object has the data used in evaluating the likelihood

        Parameters
        ----------
        interferometers: iterable
            The list of interferometers
        """

        list.__init__(self)
        if type(interferometers) == str:
            raise TypeError("Input must not be a string")
        for ifo in interferometers:
            if type(ifo) == str:
                ifo = get_empty_interferometer(ifo)
            if type(ifo) not in [Interferometer, TriangularInterferometer]:
                raise TypeError("Input list of interferometers are not all Interferometer objects")
            else:
                self.append(ifo)
        self._check_interferometers()

    def _check_interferometers(self):
        """ Check certain aspects of the set are the same """
        consistent_attributes = ['duration', 'start_time', 'sampling_frequency']
        for attribute in consistent_attributes:
            x = [getattr(interferometer.strain_data, attribute)
                 for interferometer in self]
            if not all(y == x[0] for y in x):
                raise ValueError("The {} of all interferometers are not the same".format(attribute))

    def set_strain_data_from_power_spectral_densities(self, sampling_frequency, duration, start_time=0):
        """ Set the `Interferometer.strain_data` from the power spectral densities of the detectors

        This uses the `interferometer.power_spectral_density` object to set
        the `strain_data` to a noise realization. See
        `bilby.gw.detector.InterferometerStrainData` for further information.

        Parameters
        ----------
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data

        """
        for interferometer in self:
            interferometer.set_strain_data_from_power_spectral_density(sampling_frequency=sampling_frequency,
                                                                       duration=duration,
                                                                       start_time=start_time)

    def set_strain_data_from_zero_noise(self, sampling_frequency, duration, start_time=0):
        """ Set the `Interferometer.strain_data` from the power spectral densities of the detectors

        This uses the `interferometer.power_spectral_density` object to set
        the `strain_data` to zero noise. See
        `bilby.gw.detector.InterferometerStrainData` for further information.

        Parameters
        ----------
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data

        """
        for interferometer in self:
            interferometer.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency,
                                                           duration=duration,
                                                           start_time=start_time)

    def inject_signal(self, parameters=None, injection_polarizations=None, waveform_generator=None):
        """ Inject a signal into noise in each of the three detectors.

        Parameters
        ----------
        parameters: dict
            Parameters of the injection.
        injection_polarizations: dict
           Polarizations of waveform to inject, output of
           `waveform_generator.frequency_domain_strain()`. If
           `waveform_generator` is also given, the injection_polarizations will
           be calculated directly and this argument can be ignored.
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject. If
            `injection_polarizations` is given, this will be ignored.

        Note
        ----------
        if your signal takes a substantial amount of time to generate, or
        you experience buggy behaviour. It is preferable to provide the
        injection_polarizations directly.

        Returns
        -------
        injection_polarizations: dict

        """
        if injection_polarizations is None:
            if waveform_generator is not None:
                injection_polarizations = \
                    waveform_generator.frequency_domain_strain(parameters)
            else:
                raise ValueError(
                    "inject_signal needs one of waveform_generator or "
                    "injection_polarizations.")

        all_injection_polarizations = list()
        for interferometer in self:
            all_injection_polarizations.append(
                interferometer.inject_signal(parameters=parameters, injection_polarizations=injection_polarizations))

        return all_injection_polarizations

    def save_data(self, outdir, label=None):
        """ Creates a save file for the data in plain text format

        Parameters
        ----------
        outdir: str
            The output directory in which the data is supposed to be saved
        label: str
            The string labelling the data
        """
        for interferometer in self:
            interferometer.save_data(outdir=outdir, label=label)

    def plot_data(self, signal=None, outdir='.', label=None):
        if utils.command_line_args.bilby_test_mode:
            return

        for interferometer in self:
            interferometer.plot_data(signal=signal, outdir=outdir, label=label)

    @property
    def number_of_interferometers(self):
        return len(self)

    @property
    def duration(self):
        return self[0].strain_data.duration

    @property
    def start_time(self):
        return self[0].strain_data.start_time

    @property
    def sampling_frequency(self):
        return self[0].strain_data.sampling_frequency

    @property
    def frequency_array(self):
        return self[0].strain_data.frequency_array

    def append(self, interferometer):
        if isinstance(interferometer, InterferometerList):
            super(InterferometerList, self).extend(interferometer)
        else:
            super(InterferometerList, self).append(interferometer)
        self._check_interferometers()

    def extend(self, interferometers):
        super(InterferometerList, self).extend(interferometers)
        self._check_interferometers()

    def insert(self, index, interferometer):
        super(InterferometerList, self).insert(index, interferometer)
        self._check_interferometers()

    @property
    def meta_data(self):
        """ Dictionary of the per-interferometer meta_data """
        return {interferometer.name: interferometer.meta_data
                for interferometer in self}

    @staticmethod
    def _hdf5_filename_from_outdir_label(outdir, label):
        return os.path.join(outdir, label + '.h5')

    def to_hdf5(self, outdir='outdir', label='ifo_list'):
        """ Saves the object to a hdf5 file

        Parameters
        ----------
        outdir: str, optional
            Output directory name of the file
        label: str, optional
            Output file name, is 'ifo_list' if not given otherwise. A list of
            the included interferometers will be appended.
        """
        import deepdish
        if sys.version_info[0] < 3:
            raise NotImplementedError('Pickling of InterferometerList is not supported in Python 2.'
                                      'Use Python 3 instead.')
        label = label + '_' + ''.join(ifo.name for ifo in self)
        utils.check_directory_exists_and_if_not_mkdir(outdir)
        deepdish.io.save(self._hdf5_filename_from_outdir_label(outdir, label), self)

    @classmethod
    def from_hdf5(cls, filename=None):
        """ Loads in an InterferometerList object from an hdf5 file

        Parameters
        ----------
        filename: str
            If given, try to load from this filename

        """
        import deepdish
        if sys.version_info[0] < 3:
            raise NotImplementedError('Pickling of InterferometerList is not supported in Python 2.'
                                      'Use Python 3 instead.')
        res = deepdish.io.load(filename)
        if res.__class__ == list:
            res = cls(res)
        if res.__class__ != cls:
            raise TypeError('The loaded object is not a InterferometerList')
        return res


class TriangularInterferometer(InterferometerList):

    def __init__(self, name, power_spectral_density, minimum_frequency, maximum_frequency,
                 length, latitude, longitude, elevation, xarm_azimuth, yarm_azimuth,
                 xarm_tilt=0., yarm_tilt=0.):
        InterferometerList.__init__(self, [])
        self.name = name
        # for attr in ['power_spectral_density', 'minimum_frequency', 'maximum_frequency']:
        if isinstance(power_spectral_density, PowerSpectralDensity):
            power_spectral_density = [power_spectral_density] * 3
        if isinstance(minimum_frequency, float) or isinstance(minimum_frequency, int):
            minimum_frequency = [minimum_frequency] * 3
        if isinstance(maximum_frequency, float) or isinstance(maximum_frequency, int):
            maximum_frequency = [maximum_frequency] * 3

        for ii in range(3):
            self.append(Interferometer(
                '{}{}'.format(name, ii + 1), power_spectral_density[ii], minimum_frequency[ii], maximum_frequency[ii],
                length, latitude, longitude, elevation, xarm_azimuth, yarm_azimuth, xarm_tilt, yarm_tilt))

            xarm_azimuth += 240
            yarm_azimuth += 240

            latitude += np.arctan(length * np.sin(xarm_azimuth * np.pi / 180) * 1e3 / utils.radius_of_earth)
            longitude += np.arctan(length * np.cos(xarm_azimuth * np.pi / 180) * 1e3 / utils.radius_of_earth)


def get_event_data(
        event, interferometer_names=None, duration=4, roll_off=0.2,
        psd_offset=-1024, psd_duration=100, cache=True, outdir='outdir',
        label=None, plot=True, filter_freq=None, **kwargs):
    """
    Get open data for a specified event.

    Parameters
    ----------
    event: str
        Event descriptor, this can deal with some prefixes, e.g., '150914',
        'GW150914', 'LVT151012'
    interferometer_names: list, optional
        List of interferometer identifiers, e.g., 'H1'.
        If None will look for data in 'H1', 'V1', 'L1'
    duration: float
        Time duration to search for.
    roll_off: float
        The roll-off (in seconds) used in the Tukey window.
    psd_offset, psd_duration: float
        The power spectral density (psd) is estimated using data from
        `center_time+psd_offset` to `center_time+psd_offset + psd_duration`.
    cache: bool
        Whether or not to store the acquired data.
    outdir: str
        Directory where the psd files are saved
    label: str
        If given, an identifying label used in generating file names.
    plot: bool
        If true, create an ASD + strain plot
    filter_freq: float
        Low pass filter frequency
    **kwargs:
        All keyword arguments are passed to
        `gwpy.timeseries.TimeSeries.fetch_open_data()`.

    Return
    ------
    list: A list of bilby.gw.detector.Interferometer objects
    """
    event_time = gwutils.get_event_time(event)

    interferometers = []

    if interferometer_names is None:
        interferometer_names = ['H1', 'L1', 'V1']

    for name in interferometer_names:
        try:
            interferometers.append(get_interferometer_with_open_data(
                name, trigger_time=event_time, duration=duration, roll_off=roll_off,
                psd_offset=psd_offset, psd_duration=psd_duration, cache=cache,
                outdir=outdir, label=label, plot=plot, filter_freq=filter_freq,
                **kwargs))
        except ValueError as e:
            logger.debug("Error raised {}".format(e))
            logger.warning('No data found for {}.'.format(name))

    return InterferometerList(interferometers)


def get_empty_interferometer(name):
    """
    Get an interferometer with standard parameters for known detectors.

    These objects do not have any noise instantiated.

    The available instruments are:
        H1, L1, V1, GEO600, CE

    Detector positions taken from:
        L1/H1: LIGO-T980044-10
        V1/GEO600: arXiv:gr-qc/0008066 [45]
        CE: located at the site of H1

    Detector sensitivities:
        H1/L1/V1: https://dcc.ligo.org/LIGO-P1200087-v42/public
        GEO600: http://www.geo600.org/1032083/GEO600_Sensitivity_Curves
        CE: https://dcc.ligo.org/LIGO-P1600143/public


    Parameters
    ----------
    name: str
        Interferometer identifier.

    Returns
    -------
    interferometer: Interferometer
        Interferometer instance
    """
    filename = os.path.join(os.path.dirname(__file__), 'detectors', '{}.interferometer'.format(name))
    try:
        return load_interferometer(filename)
    except OSError:
        raise ValueError('Interferometer {} not implemented'.format(name))


def load_interferometer(filename):
    """Load an interferometer from a file."""
    parameters = dict()
    with open(filename, 'r') as parameter_file:
        lines = parameter_file.readlines()
        for line in lines:
            if line[0] == '#':
                continue
            split_line = line.split('=')
            key = split_line[0].strip()
            value = eval('='.join(split_line[1:]))
            parameters[key] = value
    if 'shape' not in parameters.keys():
        ifo = Interferometer(**parameters)
        logger.debug('Assuming L shape for {}'.format('name'))
    elif parameters['shape'].lower() in ['l', 'ligo']:
        parameters.pop('shape')
        ifo = Interferometer(**parameters)
    elif parameters['shape'].lower() in ['triangular', 'triangle']:
        parameters.pop('shape')
        ifo = TriangularInterferometer(**parameters)
    else:
        raise IOError("{} could not be loaded. Invalid parameter 'shape'.".format(filename))
    return ifo


def get_interferometer_with_open_data(
        name, trigger_time, duration=4, start_time=None, roll_off=0.2,
        psd_offset=-1024, psd_duration=100, cache=True, outdir='outdir',
        label=None, plot=True, filter_freq=None, **kwargs):
    """
    Helper function to obtain an Interferometer instance with appropriate
    PSD and data, given an center_time.

    Parameters
    ----------

    name: str
        Detector name, e.g., 'H1'.
    trigger_time: float
        Trigger GPS time.
    duration: float, optional
        The total time (in seconds) to analyse. Defaults to 4s.
    start_time: float, optional
        Beginning of the segment, if None, the trigger is placed 2s before the end
        of the segment.
    roll_off: float
        The roll-off (in seconds) used in the Tukey window.
    psd_offset, psd_duration: float
        The power spectral density (psd) is estimated using data from
        `center_time+psd_offset` to `center_time+psd_offset + psd_duration`.
    cache: bool, optional
        Whether or not to store the acquired data
    outdir: str
        Directory where the psd files are saved
    label: str
        If given, an identifying label used in generating file names.
    plot: bool
        If true, create an ASD + strain plot
    filter_freq: float
        Low pass filter frequency
    **kwargs:
        All keyword arguments are passed to
        `gwpy.timeseries.TimeSeries.fetch_open_data()`.

    Returns
    -------
    bilby.gw.detector.Interferometer: An Interferometer instance with a PSD and frequency-domain strain data.

    """

    logger.warning(
        "Parameter estimation for real interferometer data in bilby is in "
        "alpha testing at the moment: the routines for windowing and filtering"
        " have not been reviewed.")

    utils.check_directory_exists_and_if_not_mkdir(outdir)

    if start_time is None:
        start_time = trigger_time + 2 - duration

    strain = InterferometerStrainData(roll_off=roll_off)
    strain.set_from_open_data(
        name=name, start_time=start_time, duration=duration,
        outdir=outdir, cache=cache, **kwargs)
    strain.low_pass_filter(filter_freq)

    strain_psd = InterferometerStrainData(roll_off=roll_off)
    strain_psd.set_from_open_data(
        name=name, start_time=start_time + duration + psd_offset,
        duration=psd_duration, outdir=outdir, cache=cache, **kwargs)
    # Low pass filter
    strain_psd.low_pass_filter(filter_freq)
    # Create and save PSDs
    psd_frequencies, psd_array = strain_psd.create_power_spectral_density(
        name=name, outdir=outdir, fft_length=strain.duration)

    interferometer = get_empty_interferometer(name)
    interferometer.power_spectral_density = PowerSpectralDensity(
        psd_array=psd_array, frequency_array=psd_frequencies)
    interferometer.strain_data = strain

    if plot:
        interferometer.plot_data(outdir=outdir, label=label)

    return interferometer
