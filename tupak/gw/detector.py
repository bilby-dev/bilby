from __future__ import division, print_function, absolute_import

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d

from tupak.gw import utils as gwutils
from tupak.core import utils
from tupak.core.utils import logger
from .calibration import Recalibrate

try:
    import gwpy
except ImportError:
    logger.warning("You do not have gwpy installed currently. You will "
                   " not be able to use some of the prebuilt functions.")


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
            raise ValueError("Input must not be a string")
        for ifo in interferometers:
            if type(ifo) == str:
                ifo = get_empty_interferometer(ifo)
            if type(ifo) not in [Interferometer, TriangularInterferometer]:
                raise ValueError("Input list of interferometers are not all Interferometer objects")
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
        """ Set the `Interferometer.strain_data` from the power spectal densities of the detectors

        This uses the `interferometer.power_spectral_density` object to set
        the `strain_data` to a noise realization. See
        `tupak.gw.detector.InterferometerStrainData` for further information.

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
            interferometer.set_strain_data_from_power_spectral_density(sampling_frequency, duration, start_time)

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
        waveform_generator: tupak.gw.waveform_generator.WaveformGenerator
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
                waveform_generator.parameters = parameters
                injection_polarizations = waveform_generator.frequency_domain_strain()
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
            interferometer.save_data(outdir, label)

    def plot_data(self, signal=None, outdir='.', label=None):
        if utils.command_line_args.test:
            return

        fig = plt.figure()
        for ii, interferometer in enumerate(self):
            ax = fig.add_subplot(len(self) // 2, 2, ii + 1)
            ax.loglog(interferometer.frequency_array,
                      gwutils.asd_from_freq_series(freq_data=interferometer.frequency_domain_strain,
                                                   df=(interferometer.frequency_array[1] -
                                                       interferometer.frequency_array[0])),
                      color='C0', label=interferometer.name)
            ax.loglog(interferometer.frequency_array,
                      interferometer.amplitude_spectral_density_array,
                      color='C1', lw=0.5, label=interferometer.name + ' ASD')
            ax.grid('on')
            ax.set_ylabel(r'strain [strain/$\sqrt{\rm Hz}$]')
            ax.set_xlabel(r'frequency [Hz]')
            ax.set_xlim(20, 2000)
            ax.legend(loc='best')
        if signal is not None:
            ax.loglog(self.frequency_array,
                      gwutils.asd_from_freq_series(freq_data=signal,
                                                   df=(self.frequency_array[1] -
                                                       self.frequency_array[0])
                                                   ),
                      color='C2',
                      label='Signal')
        fig.tight_layout()
        if label is None:
            fig.savefig(
                '{}/frequency_domain_data.png'.format(outdir))
        else:
            fig.savefig(
                '{}/{}_frequency_domain_data.png'.format(
                    outdir, label))

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


class InterferometerStrainData(object):
    """ Strain data for an interferometer """

    def __init__(self, minimum_frequency=0, maximum_frequency=np.inf,
                 roll_off=0.2):
        """ Initiate an InterferometerStrainData object

        The initialised object contains no data, this should be added using one
        of the `set_from..` methods.

        Parameters
        ----------
        minimum_frequency: float
            Minimum frequency to analyse for detector. Default is 0.
        maximum_frequency: float
            Maximum frequency to analyse for detector. Default is infinity.
        roll_off: float
            The roll-off (in seconds) used in the Tukey window, default=0.2s.
            This corresponds to alpha * duration / 2 for scipy tukey window.

        """
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.roll_off = roll_off
        self.window_factor = 1

        self._set_time_and_frequency_array_parameters(None, None, None)

        self._frequency_domain_strain = None
        self._frequency_array = None
        self._time_domain_strain = None
        self._time_array = None

    @property
    def frequency_array(self):
        """ Frequencies of the data in Hz """
        if self._frequency_array is not None:
            return self._frequency_array
        else:
            self._calculate_frequency_array()
            return self._frequency_array

    @frequency_array.setter
    def frequency_array(self, frequency_array):
        self._frequency_array = frequency_array

    @property
    def time_array(self):
        """ Time of the data in seconds """
        if self._time_array is not None:
            return self._time_array
        else:
            self._calculate_time_array()
            return self._time_array

    @time_array.setter
    def time_array(self, time_array):
        self._time_array = time_array

    def _calculate_time_array(self):
        """ Calculate the time array """
        if (self.sampling_frequency is None) or (self.duration is None):
            raise ValueError(
                "You have not specified the sampling_frequency and duration")

        self.time_array = utils.create_time_series(
            sampling_frequency=self.sampling_frequency, duration=self.duration,
            starting_time=self.start_time)

    def _calculate_frequency_array(self):
        """ Calculate the frequency array """
        if (self.sampling_frequency is None) or (self.duration is None):
            raise ValueError(
                "You have not specified the sampling_frequency and duration")
        self.frequency_array = utils.create_frequency_series(
            sampling_frequency=self.sampling_frequency, duration=self.duration)

    def time_within_data(self, time):
        """ Check if time is within the data span

        Parameters
        ----------
        time: float
            The time to check

        Returns
        -------
        bool:
            A boolean stating whether the time is inside or outside the span

        """
        if time < self.start_time:
            logger.debug("Time is before the start_time")
            return False
        elif time > self.start_time + self.duration:
            logger.debug("Time is after the start_time + duration")
            return False
        else:
            return True

    @property
    def minimum_frequency(self):
        return self.__minimum_frequency

    @minimum_frequency.setter
    def minimum_frequency(self, minimum_frequency):
        self.__minimum_frequency = minimum_frequency

    @property
    def maximum_frequency(self):
        """ Force the maximum frequency be less than the Nyquist frequency """
        if self.sampling_frequency is not None:
            if 2 * self.__maximum_frequency > self.sampling_frequency:
                self.__maximum_frequency = self.sampling_frequency / 2.
        return self.__maximum_frequency

    @maximum_frequency.setter
    def maximum_frequency(self, maximum_frequency):
        self.__maximum_frequency = maximum_frequency

    @property
    def frequency_mask(self):
        """Masking array for limiting the frequency band.

        Returns
        -------
        array_like: An array of boolean values
        """
        return ((self.frequency_array > self.minimum_frequency) &
                (self.frequency_array < self.maximum_frequency))

    @property
    def alpha(self):
        return 2 * self.roll_off / self.duration

    def time_domain_window(self, roll_off=None, alpha=None):
        """
        Window function to apply to time domain data before FFTing.

        This defines self.window_factor as the power loss due to the windowing.
        See https://dcc.ligo.org/DocDB/0027/T040089/000/T040089-00.pdf

        Parameters
        ----------
        roll_off: float
            Rise time of window in seconds
        alpha: float
            Parameter to pass to tukey window, how much of segment falls
            into windowed part

        Return
        ------
        window: array
            Window function over time array
        """
        if roll_off is not None:
            self.roll_off = roll_off
        elif alpha is not None:
            self.roll_off = alpha * self.duration / 2
        window = tukey(len(self._time_domain_strain), alpha=self.alpha)
        self.window_factor = np.mean(window ** 2)
        return window

    @property
    def time_domain_strain(self):
        """ The time domain strain, in units of strain """
        if self._time_domain_strain is not None:
            return self._time_domain_strain
        elif self._frequency_domain_strain is not None:
            self._time_domain_strain = utils.infft(
                self.frequency_domain_strain, self.sampling_frequency)
            return self._time_domain_strain

        else:
            raise ValueError("time domain strain data not yet set")

    @property
    def frequency_domain_strain(self):
        """ Returns the frequency domain strain

        This is the frequency domain strain normalised to units of
        strain / Hz, obtained by a one-sided Fourier transform of the
        time domain data, divided by the sampling frequency.
        """
        if self._frequency_domain_strain is not None:
            return self._frequency_domain_strain * self.frequency_mask
        elif self._time_domain_strain is not None:
            logger.info("Generating frequency domain strain from given time "
                        "domain strain.")
            logger.info("Applying a tukey window with alpha={}, roll off={}".format(
                self.alpha, self.roll_off))
            # self.low_pass_filter()
            window = self.time_domain_window()
            frequency_domain_strain, self.frequency_array = utils.nfft(
                self._time_domain_strain * window, self.sampling_frequency)
            self._frequency_domain_strain = frequency_domain_strain
            return self._frequency_domain_strain * self.frequency_mask
        else:
            raise ValueError("frequency domain strain data not yet set")

    @frequency_domain_strain.setter
    def frequency_domain_strain(self, frequency_domain_strain):
        self._frequency_domain_strain = frequency_domain_strain

    def add_to_frequency_domain_strain(self, x):
        """Deprecated"""
        self._frequency_domain_strain += x

    def low_pass_filter(self, filter_freq=None):
        """ Low pass filter the data """

        if filter_freq is None:
            logger.debug(
                "Setting low pass filter_freq using given maximum frequency")
            filter_freq = self.maximum_frequency

        if 2 * filter_freq >= self.sampling_frequency:
            logger.info(
                "Low pass filter frequency of {}Hz requested, this is equal"
                " or greater than the Nyquist frequency so no filter applied"
                    .format(filter_freq))
            return

        logger.debug("Applying low pass filter with filter frequency {}".format(filter_freq))
        bp = gwpy.signal.filter_design.lowpass(
            filter_freq, self.sampling_frequency)
        strain = gwpy.timeseries.TimeSeries(
            self.time_domain_strain, sample_rate=self.sampling_frequency)
        strain = strain.filter(bp, filtfilt=True)
        self._time_domain_strain = strain.value

    def create_power_spectral_density(self, fft_length, name='unknown', outdir=None):
        """ Use the time domain strain to generate a power spectral density

        This create a Tukey-windowed power spectral density and writes it to a
        PSD file.

        Parameters
        ----------
        fft_length: float
            Duration of the analysis segment.
        name: str
            The name of the detector, used in storing the PSD. Defaults to
            "unknown".
        outdir: str
            The output directory to write the PSD file too. If not given,
            the PSD will not be written to file.

        Returns
        -------
        frequency_array, psd : array_like
            The frequencies and power spectral density array

        """
        strain = gwpy.timeseries.TimeSeries(self.time_domain_strain, sample_rate=self.sampling_frequency)
        psd_alpha = 2 * self.roll_off / fft_length
        logger.info("Creating PSD with non-overlapping tukey window, alpha={}, roll off={}".format(
            psd_alpha, self.roll_off))
        psd = strain.psd(fftlength=fft_length, overlap=0, window=('tukey', psd_alpha))

        if outdir:
            psd_file = '{}/{}_PSD_{}_{}.txt'.format(outdir, name, self.start_time, self.duration)
            with open('{}'.format(psd_file), 'w+') as opened_file:
                for f, p in zip(psd.frequencies.value, psd.value):
                    opened_file.write('{} {}\n'.format(f, p))

        return psd.frequencies.value, psd.value

    def _infer_time_domain_dependence(
            self, start_time, sampling_frequency, duration, time_array):
        """ Helper function to figure out if the time_array, or
            sampling_frequency and duration where given
        """
        self._infer_dependence(domain='time', array=time_array, duration=duration,
                               sampling_frequency=sampling_frequency, start_time=start_time)

    def _infer_frequency_domain_dependence(
            self, start_time, sampling_frequency, duration, frequency_array):
        """ Helper function to figure out if the frequency_array, or
            sampling_frequency and duration where given
        """

        self._infer_dependence(domain='frequency', array=frequency_array,
                               duration=duration, sampling_frequency=sampling_frequency, start_time=start_time)

    def _infer_dependence(self, domain, array, duration, sampling_frequency, start_time):
        if (sampling_frequency is not None) and (duration is not None):
            if array is not None:
                raise ValueError(
                    "You have given the sampling_frequency, duration, and "
                    "an array")
            pass
        elif array is not None:
            if domain == 'time':
                self.time_array = array
                sampling_frequency, duration = utils.get_sampling_frequency_and_duration_from_time_array(array)
            elif domain == 'frequency':
                self.frequency_array = array
                sampling_frequency, duration = utils.get_sampling_frequency_and_duration_from_frequency_array(array)
        elif sampling_frequency is None or duration is None:
            raise ValueError(
                "You must provide both sampling_frequency and duration")
        else:
            raise ValueError(
                "Insufficient information given to set arrays")
        self._set_time_and_frequency_array_parameters(duration=duration,
                                                      sampling_frequency=sampling_frequency,
                                                      start_time=start_time)

    def set_from_time_domain_strain(
            self, time_domain_strain, sampling_frequency=None, duration=None,
            start_time=0, time_array=None):
        """ Set the strain data from a time domain strain array

        This sets the time_domain_strain attribute, the frequency_domain_strain
        is automatically calculated after a low-pass filter and Tukey window
        is applied.

        Parameters
        ----------
        time_domain_strain: array_like
            An array of the time domain strain.
        sampling_frequency: float
            The sampling frequency (in Hz).
        duration: float
            The data duration (in s).
        start_time: float
            The GPS start-time of the data.
        time_array: array_like
            The array of times, if sampling_frequency and duration not
            given.

        """
        self._infer_time_domain_dependence(start_time=start_time,
                                           sampling_frequency=sampling_frequency,
                                           duration=duration,
                                           time_array=time_array)

        logger.debug('Setting data using provided time_domain_strain')
        if np.shape(time_domain_strain) == np.shape(self.time_array):
            self._time_domain_strain = time_domain_strain
            self._frequency_domain_strain = None
        else:
            raise ValueError("Data times do not match time array")

    def set_from_gwpy_timeseries(self, time_series):
        """ Set the strain data from a gwpy TimeSeries

        This sets the time_domain_strain attribute, the frequency_domain_strain
        is automatically calculated after a low-pass filter and Tukey window
        is applied.

        Parameters
        ----------
        time_series: gwpy.timeseries.timeseries.TimeSeries

        """
        logger.debug('Setting data using provided gwpy TimeSeries object')
        if type(time_series) != gwpy.timeseries.TimeSeries:
            raise ValueError("Input time_series is not a gwpy TimeSeries")
        self._set_time_and_frequency_array_parameters(duration=time_series.duration.value,
                                                      sampling_frequency=time_series.sample_rate.value,
                                                      start_time=time_series.epoch.value)
        self._time_domain_strain = time_series.value
        self._frequency_domain_strain = None

    def set_from_open_data(
            self, name, start_time, duration=4, outdir='outdir', cache=True,
            **kwargs):
        """ Set the strain data from open LOSC data

        This sets the time_domain_strain attribute, the frequency_domain_strain
        is automatically calculated after a low-pass filter and Tukey window
        is applied.

        Parameters
        ----------
        name: str
            Detector name, e.g., 'H1'.
        start_time: float
            Start GPS time of segment.
        duration: float, optional
            The total time (in seconds) to analyse. Defaults to 4s.
        outdir: str
            Directory where the psd files are saved
        cache: bool, optional
            Whether or not to store/use the acquired data.
        **kwargs:
            All keyword arguments are passed to
            `gwpy.timeseries.TimeSeries.fetch_open_data()`.

        """

        timeseries = gwutils.get_open_strain_data(
            name, start_time, start_time + duration, outdir=outdir, cache=cache,
            **kwargs)

        self.set_from_gwpy_timeseries(timeseries)

    def set_from_csv(self, filename):
        """ Set the strain data from a csv file

        Parameters
        ----------
        filename: str
            The path to the file to read in

        """
        timeseries = gwpy.timeseries.TimeSeries.read(filename, format='csv')
        self.set_from_gwpy_timeseries(timeseries)

    def set_from_frequency_domain_strain(
            self, frequency_domain_strain, sampling_frequency=None,
            duration=None, start_time=0, frequency_array=None):
        """ Set the `frequency_domain_strain` from a numpy array

        Parameters
        ----------
        frequency_domain_strain: array_like
            The data to set.
        sampling_frequency: float
            The sampling frequency (in Hz).
        duration: float
            The data duration (in s).
        start_time: float
            The GPS start-time of the data.
        frequency_array: array_like
            The array of frequencies, if sampling_frequency and duration not
            given.

        """

        self._infer_frequency_domain_dependence(start_time=start_time,
                                                sampling_frequency=sampling_frequency,
                                                duration=duration,
                                                frequency_array=frequency_array)

        logger.debug('Setting data using provided frequency_domain_strain')
        if np.shape(frequency_domain_strain) == np.shape(self.frequency_array):
            self._frequency_domain_strain = frequency_domain_strain
            self.window_factor = 1
        else:
            raise ValueError("Data frequencies do not match frequency_array")

    def set_from_power_spectral_density(
            self, power_spectral_density, sampling_frequency, duration,
            start_time=0):
        """ Set the `frequency_domain_strain` by generating a noise realisation

        Parameters
        ----------
        power_spectral_density: tupak.gw.detector.PowerSpectralDensity
            A PowerSpectralDensity object used to generate the data
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data

        """

        self._set_time_and_frequency_array_parameters(duration=duration,
                                                      sampling_frequency=sampling_frequency,
                                                      start_time=start_time)

        logger.debug(
            'Setting data using noise realization from provided'
            'power_spectal_density')
        frequency_domain_strain, frequency_array = \
            power_spectral_density.get_noise_realisation(
                self.sampling_frequency, self.duration)

        if np.array_equal(frequency_array, self.frequency_array):
            self._frequency_domain_strain = frequency_domain_strain
        else:
            raise ValueError("Data frequencies do not match frequency_array")

    def set_from_zero_noise(self, sampling_frequency, duration, start_time=0):
        """ Set the `frequency_domain_strain` to zero noise

        Parameters
        ----------
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data

        """

        self._set_time_and_frequency_array_parameters(duration=duration,
                                                      sampling_frequency=sampling_frequency,
                                                      start_time=start_time)

        logger.debug('Setting zero noise data')
        self._frequency_domain_strain = np.zeros_like(self.frequency_array,
                                                      dtype=np.complex)

    def set_from_frame_file(
            self, frame_file, sampling_frequency, duration, start_time=0,
            channel=None, buffer_time=1):
        """ Set the `frequency_domain_strain` from a frame fiile

        Parameters
        ----------
        frame_file: str
            File from which to load data.
        channel: str
            Channel to read from frame.
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data
        buffer_time: float
            Read in data with `start_time-buffer_time` and
            `start_time+duration+buffer_time`

        """

        self._set_time_and_frequency_array_parameters(duration=duration,
                                                      sampling_frequency=sampling_frequency,
                                                      start_time=start_time)

        logger.info('Reading data from frame')
        strain = gwutils.read_frame_file(
            frame_file, start_time=start_time, end_time=start_time + duration,
            buffer_time=buffer_time, channel=channel,
            resample=sampling_frequency)

        self.set_from_gwpy_timeseries(strain)

    def _set_time_and_frequency_array_parameters(self, duration, sampling_frequency, start_time):
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.start_time = start_time


class Interferometer(object):
    """Class for the Interferometer """

    def __init__(self, name, power_spectral_density, minimum_frequency, maximum_frequency,
                 length, latitude, longitude, elevation, xarm_azimuth, yarm_azimuth,
                 xarm_tilt=0., yarm_tilt=0., calibration_model=Recalibrate()):
        """
        Instantiate an Interferometer object.

        Parameters
        ----------
        name: str
            Interferometer name, e.g., H1.
        power_spectral_density: PowerSpectralDensity
            Power spectral density determining the sensitivity of the detector.
        minimum_frequency: float
            Minimum frequency to analyse for detector.
        maximum_frequency: float
            Maximum frequency to analyse for detector.
        length: float
            Length of the interferometer in km.
        latitude: float
            Latitude North in degrees (South is negative).
        longitude: float
            Longitude East in degrees (West is negative).
        elevation: float
            Height above surface in metres.
        xarm_azimuth: float
            Orientation of the x arm in degrees North of East.
        yarm_azimuth: float
            Orientation of the y arm in degrees North of East.
        xarm_tilt: float, optional
            Tilt of the x arm in radians above the horizontal defined by
            ellipsoid earth model in LIGO-T980044-08.
        yarm_tilt: float, optional
            Tilt of the y arm in radians above the horizontal.
        calibration_model: Recalibration
            Calibration model, this applies the calibration correction to the
            template, the default model applies no correction.
        """
        self.__x_updated = False
        self.__y_updated = False
        self.__vertex_updated = False
        self.__detector_tensor_updated = False

        self.name = name
        self.length = length
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.xarm_azimuth = xarm_azimuth
        self.yarm_azimuth = yarm_azimuth
        self.xarm_tilt = xarm_tilt
        self.yarm_tilt = yarm_tilt
        self.power_spectral_density = power_spectral_density
        self.calibration_model = calibration_model
        self._strain_data = InterferometerStrainData(
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency)

    @property
    def minimum_frequency(self):
        return self.strain_data.minimum_frequency

    @minimum_frequency.setter
    def minimum_frequency(self, minimum_frequency):
        self._strain_data.minimum_frequency = minimum_frequency

    @property
    def maximum_frequency(self):
        return self.strain_data.maximum_frequency

    @maximum_frequency.setter
    def maximum_frequency(self, maximum_frequency):
        self._strain_data.maximum_frequency = maximum_frequency

    @property
    def strain_data(self):
        """ A tupak.gw.detector.InterferometerStrainData instance """
        return self._strain_data

    @strain_data.setter
    def strain_data(self, strain_data):
        """ Set the strain_data

        This sets the Interferometer.strain_data equal to the provided
        strain_data. This will override the minimum_frequency and
        maximum_frequency of the provided strain_data object with those of
        the Interferometer object.
        """
        strain_data.minimum_frequency = self.minimum_frequency
        strain_data.maximum_frequency = self.maximum_frequency

        self._strain_data = strain_data

    def set_strain_data_from_frequency_domain_strain(
            self, frequency_domain_strain, sampling_frequency=None,
            duration=None, start_time=0, frequency_array=None):
        """ Set the `Interferometer.strain_data` from a numpy array

        Parameters
        ----------
        frequency_domain_strain: array_like
            The data to set.
        sampling_frequency: float
            The sampling frequency (in Hz).
        duration: float
            The data duration (in s).
        start_time: float
            The GPS start-time of the data.
        frequency_array: array_like
            The array of frequencies, if sampling_frequency and duration not
            given.

        """
        self.strain_data.set_from_frequency_domain_strain(
            frequency_domain_strain=frequency_domain_strain,
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=start_time, frequency_array=frequency_array)

    def set_strain_data_from_power_spectral_density(
            self, sampling_frequency, duration, start_time=0):
        """ Set the `Interferometer.strain_data` from a power spectal density

        This uses the `interferometer.power_spectral_density` object to set
        the `strain_data` to a noise realization. See
        `tupak.gw.detector.InterferometerStrainData` for further information.

        Parameters
        ----------
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data

        """
        self.strain_data.set_from_power_spectral_density(
            self.power_spectral_density, sampling_frequency=sampling_frequency,
            duration=duration, start_time=start_time)

    def set_strain_data_from_frame_file(
            self, frame_file, sampling_frequency, duration, start_time=0,
            channel=None, buffer_time=1):
        """ Set the `Interferometer.strain_data` from a frame file

        Parameters
        ----------
        frame_file: str
            File from which to load data.
        channel: str
            Channel to read from frame.
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data
        buffer_time: float
            Read in data with `start_time-buffer_time` and
            `start_time+duration+buffer_time`

        """
        self.strain_data.set_from_frame_file(
            frame_file=frame_file, sampling_frequency=sampling_frequency,
            duration=duration, start_time=start_time,
            channel=channel, buffer_time=buffer_time)

    def set_strain_data_from_csv(self, filename):
        """ Set the `Interferometer.strain_data` from a csv file

        Parameters
        ----------
        filename: str
            The path to the file to read in

        """
        self.strain_data.set_from_csv(filename)

    def set_strain_data_from_zero_noise(
            self, sampling_frequency, duration, start_time=0):
        """ Set the `Interferometer.strain_data` to zero noise

        Parameters
        ----------
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data

        """

        self.strain_data.set_from_zero_noise(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=start_time)

    @property
    def latitude(self):
        """ Saves latitude in rad internally. Updates related quantities if set to a different value.

        Returns
        -------
        float: The latitude position of the detector in degree
        """
        return self.__latitude * 180 / np.pi

    @latitude.setter
    def latitude(self, latitude):
        self.__latitude = latitude * np.pi / 180
        self.__x_updated = False
        self.__y_updated = False
        self.__vertex_updated = False

    @property
    def longitude(self):
        """ Saves longitude in rad internally. Updates related quantities if set to a different value.

        Returns
        -------
        float: The longitude position of the detector in degree
        """
        return self.__longitude * 180 / np.pi

    @longitude.setter
    def longitude(self, longitude):
        self.__longitude = longitude * np.pi / 180
        self.__x_updated = False
        self.__y_updated = False
        self.__vertex_updated = False

    @property
    def elevation(self):
        """ Updates related quantities if set to a different values.

        Returns
        -------
        float: The height about the surface in meters
        """
        return self.__elevation

    @elevation.setter
    def elevation(self, elevation):
        self.__elevation = elevation
        self.__vertex_updated = False

    @property
    def xarm_azimuth(self):
        """ Saves the x-arm azimuth in rad internally. Updates related quantities if set to a different values.

        Returns
        -------
        float: The x-arm azimuth in degrees.

        """
        return self.__xarm_azimuth * 180 / np.pi

    @xarm_azimuth.setter
    def xarm_azimuth(self, xarm_azimuth):
        self.__xarm_azimuth = xarm_azimuth * np.pi / 180
        self.__x_updated = False

    @property
    def yarm_azimuth(self):
        """ Saves the y-arm azimuth in rad internally. Updates related quantities if set to a different values.

        Returns
        -------
        float: The y-arm azimuth in degrees.

        """
        return self.__yarm_azimuth * 180 / np.pi

    @yarm_azimuth.setter
    def yarm_azimuth(self, yarm_azimuth):
        self.__yarm_azimuth = yarm_azimuth * np.pi / 180
        self.__y_updated = False

    @property
    def xarm_tilt(self):
        """ Updates related quantities if set to a different values.

        Returns
        -------
        float: The x-arm tilt in radians.

        """
        return self.__xarm_tilt

    @xarm_tilt.setter
    def xarm_tilt(self, xarm_tilt):
        self.__xarm_tilt = xarm_tilt
        self.__x_updated = False

    @property
    def yarm_tilt(self):
        """ Updates related quantities if set to a different values.

        Returns
        -------
        float: The y-arm tilt in radians.

        """
        return self.__yarm_tilt

    @yarm_tilt.setter
    def yarm_tilt(self, yarm_tilt):
        self.__yarm_tilt = yarm_tilt
        self.__y_updated = False

    @property
    def vertex(self):
        """ Position of the IFO vertex in geocentric coordinates in meters.

        Is automatically updated if related quantities are modified.

        Returns
        -------
        array_like: A 3D array representation of the vertex
        """
        if not self.__vertex_updated:
            self.__vertex = gwutils.get_vertex_position_geocentric(self.__latitude, self.__longitude,
                                                                   self.elevation)
            self.__vertex_updated = True
        return self.__vertex

    @property
    def x(self):
        """ A unit vector along the x-arm

        Is automatically updated if related quantities are modified.

        Returns
        -------
        array_like: A 3D array representation of a unit vector along the x-arm

        """
        if not self.__x_updated:
            self.__x = self.unit_vector_along_arm('x')
            self.__x_updated = True
            self.__detector_tensor_updated = False
        return self.__x

    @property
    def y(self):
        """ A unit vector along the y-arm

        Is automatically updated if related quantities are modified.

        Returns
        -------
        array_like: A 3D array representation of a unit vector along the y-arm

        """
        if not self.__y_updated:
            self.__y = self.unit_vector_along_arm('y')
            self.__y_updated = True
            self.__detector_tensor_updated = False
        return self.__y

    @property
    def detector_tensor(self):
        """
        Calculate the detector tensor from the unit vectors along each arm of the detector.

        See Eq. B6 of arXiv:gr-qc/0008066

        Is automatically updated if related quantities are modified.

        Returns
        -------
        array_like: A 3x3 array representation of the detector tensor

        """
        if not self.__detector_tensor_updated:
            self.__detector_tensor = 0.5 * (np.einsum('i,j->ij', self.x, self.x) - np.einsum('i,j->ij', self.y, self.y))
            self.__detector_tensor_updated = True
        return self.__detector_tensor

    def antenna_response(self, ra, dec, time, psi, mode):
        """
        Calculate the antenna response function for a given sky location

        See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
        [u, v, w] represent the Earth-frame
        [m, n, omega] represent the wave-frame
        Note: there is a typo in the definition of the wave-frame in Nishizawa et al.

        Parameters
        -------
        ra: float
            right ascension in radians
        dec: float
            declination in radians
        time: float
            geocentric GPS time
        psi: float
            binary polarisation angle counter-clockwise about the direction of propagation
        mode: str
            polarisation mode (e.g. 'plus', 'cross')

        Returns
        -------
        array_like: A 3x3 array representation of the antenna response for the specified mode

        """
        polarization_tensor = gwutils.get_polarization_tensor(ra, dec, time, psi, mode)
        return np.einsum('ij,ij->', self.detector_tensor, polarization_tensor)

    def get_detector_response(self, waveform_polarizations, parameters):
        """ Get the detector response for a particular waveform

        Parameters
        -------
        waveform_polarizations: dict
            polarizations of the waveform
        parameters: dict
            parameters describing position and time of arrival of the signal

        Returns
        -------
        array_like: A 3x3 array representation of the detector response (signal observed in the interferometer)
        """
        signal = {}
        for mode in waveform_polarizations.keys():
            det_response = self.antenna_response(
                parameters['ra'],
                parameters['dec'],
                parameters['geocent_time'],
                parameters['psi'], mode)

            signal[mode] = waveform_polarizations[mode] * det_response
        signal_ifo = sum(signal.values())

        signal_ifo *= self.strain_data.frequency_mask

        time_shift = self.time_delay_from_geocenter(
            parameters['ra'],
            parameters['dec'],
            self.strain_data.start_time)
        dt = parameters['geocent_time'] + time_shift - self.strain_data.start_time

        signal_ifo = signal_ifo * np.exp(
            -1j * 2 * np.pi * dt * self.frequency_array)

        signal_ifo *= self.calibration_model.get_calibration_factor(
            self.frequency_array, prefix='recalib_{}_'.format(self.name), **parameters)

        return signal_ifo

    def inject_signal(self, parameters=None, injection_polarizations=None,
                      waveform_generator=None):
        """ Inject a signal into noise

        Parameters
        ----------
        parameters: dict
            Parameters of the injection.
        injection_polarizations: dict
           Polarizations of waveform to inject, output of
           `waveform_generator.frequency_domain_strain()`. If
           `waveform_generator` is also given, the injection_polarizations will
           be calculated directly and this argument can be ignored.
        waveform_generator: tupak.gw.waveform_generator
            A WaveformGenerator instance using the source model to inject. If
            `injection_polarizations` is given, this will be ignored.

        Note
        -------
        if your signal takes a substantial amount of time to generate, or
        you experience buggy behaviour. It is preferable to provide the
        injection_polarizations directly.

        Returns
        -------
        injection_polarizations: dict

        """

        if injection_polarizations is None:
            if waveform_generator is not None:
                waveform_generator.parameters = parameters
                injection_polarizations = waveform_generator.frequency_domain_strain()
            else:
                raise ValueError(
                    "inject_signal needs one of waveform_generator or "
                    "injection_polarizations.")

            if injection_polarizations is None:
                raise ValueError(
                    'Trying to inject signal which is None. The most likely cause'
                    ' is that waveform_generator.frequency_domain_strain returned'
                    ' None. This can be caused if, e.g., mass_2 > mass_1.')

        if not self.strain_data.time_within_data(parameters['geocent_time']):
            logger.warning(
                'Injecting signal outside segment, start_time={}, merger time={}.'
                    .format(self.strain_data.start_time, parameters['geocent_time']))

        signal_ifo = self.get_detector_response(injection_polarizations, parameters)
        if np.shape(self.frequency_domain_strain).__eq__(np.shape(signal_ifo)):
            self.strain_data.frequency_domain_strain = \
                signal_ifo + self.strain_data.frequency_domain_strain
        else:
            logger.info('Injecting into zero noise.')
            self.set_strain_data_from_frequency_domain_strain(
                signal_ifo,
                sampling_frequency=self.strain_data.sampling_frequency,
                duration=self.strain_data.duration,
                start_time=self.strain_data.start_time)
        opt_snr = np.sqrt(self.optimal_snr_squared(signal=signal_ifo).real)
        mf_snr = np.sqrt(self.matched_filter_snr_squared(signal=signal_ifo).real)

        logger.info("Injected signal in {}:".format(self.name))
        logger.info("  optimal SNR = {:.2f}".format(opt_snr))
        logger.info("  matched filter SNR = {:.2f}".format(mf_snr))
        for key in parameters:
            logger.info('  {} = {}'.format(key, parameters[key]))

        return injection_polarizations

    def unit_vector_along_arm(self, arm):
        """
        Calculate the unit vector pointing along the specified arm in cartesian Earth-based coordinates.

        See Eqs. B14-B17 in arXiv:gr-qc/0008066

        Parameters
        -------
        arm: str
            'x' or 'y' (arm of the detector)

        Returns
        -------
        array_like: 3D unit vector along arm in cartesian Earth-based coordinates

        Raises
        -------
        ValueError: If arm is neither 'x' nor 'y'

        """
        if arm == 'x':
            return self.__calculate_arm(self.__xarm_tilt, self.__xarm_azimuth)
        elif arm == 'y':
            return self.__calculate_arm(self.__yarm_tilt, self.__yarm_azimuth)
        else:
            raise ValueError("Arm must either be 'x' or 'y'.")

    def __calculate_arm(self, arm_tilt, arm_azimuth):
        e_long = np.array([-np.sin(self.__longitude), np.cos(self.__longitude), 0])
        e_lat = np.array([-np.sin(self.__latitude) * np.cos(self.__longitude),
                          -np.sin(self.__latitude) * np.sin(self.__longitude), np.cos(self.__latitude)])
        e_h = np.array([np.cos(self.__latitude) * np.cos(self.__longitude),
                        np.cos(self.__latitude) * np.sin(self.__longitude), np.sin(self.__latitude)])

        return np.cos(arm_tilt) * np.cos(arm_azimuth) * e_long + \
               np.cos(arm_tilt) * np.sin(arm_azimuth) * e_lat + \
               np.sin(arm_tilt) * e_h

    @property
    def amplitude_spectral_density_array(self):
        """ Returns the amplitude spectral density (ASD) given we know a power spectral denstiy (PSD)

        Returns
        -------
        array_like: An array representation of the ASD

        """
        return self.power_spectral_density_array ** 0.5

    @property
    def power_spectral_density_array(self):
        """ Returns the power spectral density (PSD)

        This accounts for whether the data in the interferometer has been windowed.

        Returns
        -------
        array_like: An array representation of the PSD

        """
        return self.power_spectral_density.power_spectral_density_interpolated(self.frequency_array) \
               * self.strain_data.window_factor

    @property
    def frequency_array(self):
        return self.strain_data.frequency_array

    @property
    def frequency_mask(self):
        return self.strain_data.frequency_mask

    @property
    def frequency_domain_strain(self):
        """ The frequency domain strain in units of strain / Hz """
        return self.strain_data.frequency_domain_strain

    def time_delay_from_geocenter(self, ra, dec, time):
        """
        Calculate the time delay from the geocenter for the interferometer.

        Use the time delay function from utils.

        Parameters
        -------
        ra: float
            right ascension of source in radians
        dec: float
            declination of source in radians
        time: float
            GPS time

        Returns
        -------
        float: The time delay from geocenter in seconds
        """
        return gwutils.time_delay_geocentric(self.vertex, np.array([0, 0, 0]), ra, dec, time)

    def vertex_position_geocentric(self):
        """
        Calculate the position of the IFO vertex in geocentric coordinates in meters.

        Based on arXiv:gr-qc/0008066 Eqs. B11-B13 except for the typo in the definition of the local radius.
        See Section 2.1 of LIGO-T980044-10 for the correct expression

        Returns
        -------
        array_like: A 3D array representation of the vertex
        """
        return gwutils.get_vertex_position_geocentric(self.__latitude, self.__longitude, self.__elevation)

    def optimal_snr_squared(self, signal):
        """

        Parameters
        ----------
        signal: array_like
            Array containing the signal

        Returns
        -------
        float: The optimal signal to noise ratio possible squared
        """
        return gwutils.optimal_snr_squared(signal=signal,
                                           power_spectral_density=self.power_spectral_density_array,
                                           duration=self.strain_data.duration)

    def matched_filter_snr_squared(self, signal):
        """

        Parameters
        ----------
        signal: array_like
            Array containing the signal

        Returns
        -------
        float: The matched filter signal to noise ratio squared

        """
        return gwutils.matched_filter_snr_squared(signal=signal,
                                                  frequency_domain_strain=self.frequency_domain_strain,
                                                  power_spectral_density=self.power_spectral_density_array,
                                                  duration=self.strain_data.duration)

    @property
    def whitened_frequency_domain_strain(self):
        """ Calculates the whitened data by dividing data by the amplitude spectral density

        Returns
        -------
        array_like: The whitened data
        """
        return self.strain_data.frequency_domain_strain / self.amplitude_spectral_density_array

    def save_data(self, outdir, label=None):
        """ Creates a save file for the data in plain text format

        Parameters
        ----------
        outdir: str
            The output directory in which the data is supposed to be saved
        label: str
            The name of the output files
        """
        np.savetxt('{}/{}_frequency_domain_data.dat'.format(outdir, self.name),
                   np.array(
                       [self.frequency_array,
                        self.frequency_domain_strain.real,
                        self.frequency_domain_strain.imag]).T,
                   header='f real_h(f) imag_h(f)')
        if label is None:
            filename = '{}/{}_psd.dat'.format(outdir, self.name)
        else:
            filename = '{}/{}_{}_psd.dat'.format(outdir, self.name, label)
        np.savetxt(filename,
                   np.array(
                       [self.frequency_array,
                        self.amplitude_spectral_density_array]).T,
                   header='f h(f)')

    def plot_data(self, signal=None, outdir='.', label=None):
        if utils.command_line_args.test:
            return

        fig, ax = plt.subplots()
        ax.loglog(self.frequency_array,
                  gwutils.asd_from_freq_series(freq_data=self.frequency_domain_strain,
                                               df=(self.frequency_array[1] - self.frequency_array[0])),
                  color='C0', label=self.name)
        ax.loglog(self.frequency_array,
                  self.amplitude_spectral_density_array,
                  color='C1', lw=0.5, label=self.name + ' ASD')
        if signal is not None:
            ax.loglog(self.frequency_array,
                      gwutils.asd_from_freq_series(freq_data=signal,
                                                   df=(self.frequency_array[1] - self.frequency_array[0])),
                      color='C2',
                      label='Signal')
        ax.grid('on')
        ax.set_ylabel(r'strain [strain/$\sqrt{\rm Hz}$]')
        ax.set_xlabel(r'frequency [Hz]')
        ax.set_xlim(20, 2000)
        ax.legend(loc='best')
        if label is None:
            fig.savefig(
                '{}/{}_frequency_domain_data.png'.format(outdir, self.name))
        else:
            fig.savefig(
                '{}/{}_{}_frequency_domain_data.png'.format(
                    outdir, self.name, label))


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


class PowerSpectralDensity(object):

    def __init__(self, **kwargs):
        """
        Instantiate a new PowerSpectralDensity object.

        If called with no argument, `PowerSpectralDensity()` will return an
        empty instance which can be filled with one of the `set_from` methods.
        You can also initialise a new PowerSpectralDensity object giving the
        arguments of any `set_from` method and an attempt will be made to use
        this information to load/create the power spectral density.

        Example
        -------
        Using the `set_from` method directly (here `psd_file` is a string
        containing the path to the file to load):
        >>> power_spectral_density = PowerSpectralDensity()
        >>> power_spectral_density.set_from_power_spectral_density_file(psd_file)

        Alternatively (and equivalently) setting the psd_file directly:
        >>> power_spectral_density = PowerSpectralDensity(psd_file=psd_file)

        Note: for the "direct" method to work, you must provide the input
        as a keyword argument as above.

        Attributes
        ----------
        amplitude_spectral_density: array_like
            Array representation of the ASD
        amplitude_spectral_density_file: str
            Name of the ASD file
        frequency_array: array_like
            Array containing the frequencies of the ASD/PSD values
        power_spectral_density: array_like
            Array representation of the PSD
        power_spectral_density_file: str
            Name of the PSD file
        power_spectral_density_interpolated: scipy.interpolated.interp1d
            Interpolated function of the PSD

        """
        self.__power_spectral_density = None
        self.__amplitude_spectral_density = None

        self.frequency_array = []
        self.power_spectral_density_interpolated = None

        for key in kwargs:
            try:
                expanded_key = (key.replace('psd', 'power_spectral_density')
                                .replace('asd', 'amplitude_spectral_density'))
                m = getattr(self, 'set_from_{}'.format(expanded_key))
                m(**kwargs)
            except AttributeError:
                logger.info("Tried setting PSD from init kwarg {} and failed".format(key))

    def set_from_amplitude_spectral_density_file(self, asd_file):
        """ Set the amplitude spectral density from a given file

        Parameters
        ----------
        asd_file: str
            File containing amplitude spectral density, format 'f h_f'

        """

        self.amplitude_spectral_density_file = asd_file
        self.import_amplitude_spectral_density()
        if min(self.amplitude_spectral_density) < 1e-30:
            logger.warning("You specified an amplitude spectral density file.")
            logger.warning("{} WARNING {}".format("*" * 30, "*" * 30))
            logger.warning("The minimum of the provided curve is {:.2e}.".format(
                min(self.amplitude_spectral_density)))
            logger.warning(
                "You may have intended to provide this as a power spectral density.")

    def set_from_power_spectral_density_file(self, psd_file):
        """ Set the power spectral density from a given file

        Parameters
        ----------
        psd_file: str, optional
            File containing power spectral density, format 'f h_f'

        """

        self.power_spectral_density_file = psd_file
        self.import_power_spectral_density()
        if min(self.power_spectral_density) > 1e-30:
            logger.warning("You specified a power spectral density file.")
            logger.warning("{} WARNING {}".format("*" * 30, "*" * 30))
            logger.warning("The minimum of the provided curve is {:.2e}.".format(
                min(self.power_spectral_density)))
            logger.warning(
                "You may have intended to provide this as an amplitude spectral density.")

    def set_from_frame_file(self, frame_file, psd_start_time, psd_duration,
                            fft_length=4, sampling_frequency=4096, roll_off=0.2,
                            channel=None):
        """ Generate power spectral density from a frame file

        Parameters
        ----------
        frame_file: str, optional
            Frame file to read data from.
        psd_start_time: float
            Beginning of segment to analyse.
        psd_duration: float, optional
            Duration of data (in seconds) to generate PSD from.
        fft_length: float, optional
            Number of seconds in a single fft.
        sampling_frequency: float, optional
            Sampling frequency for time series.
            This is twice the maximum frequency.
        roll_off: float, optional
            Rise time in seconds of tukey window.
        channel: str, optional
            Name of channel to use to generate PSD.

        """

        strain = InterferometerStrainData(roll_off=roll_off)
        strain.set_from_frame_file(
            frame_file, start_time=psd_start_time, duration=psd_duration,
            channel=channel, sampling_frequency=sampling_frequency)

        f, psd = strain.create_power_spectral_density(fft_length=fft_length)
        self.frequency_array = f
        self.power_spectral_density = psd

    def set_from_amplitude_spectral_density_array(self, frequency_array,
                                                  asd_array):
        self.frequency_array = frequency_array
        self.amplitude_spectral_density = asd_array

    def set_from_power_spectral_density_array(self, frequency_array, psd_array):
        self.frequency_array = frequency_array
        self.power_spectral_density = psd_array

    def set_from_aLIGO(self):
        psd_file = 'aLIGO_ZERO_DET_high_P_psd.txt'
        logger.info("No power spectral density provided, using aLIGO,"
                    "zero detuning, high power.")
        self.set_from_power_spectral_density_file(psd_file)

    @property
    def power_spectral_density(self):
        if self.__power_spectral_density is not None:
            return self.__power_spectral_density
        else:
            self.set_to_aLIGO()
            return self.__power_spectral_density

    @power_spectral_density.setter
    def power_spectral_density(self, power_spectral_density):
        self._check_frequency_array_matches_density_array(power_spectral_density)
        self.__power_spectral_density = power_spectral_density
        self._interpolate_power_spectral_density()
        self.__amplitude_spectral_density = power_spectral_density ** 0.5

    @property
    def amplitude_spectral_density(self):
        return self.__amplitude_spectral_density

    @amplitude_spectral_density.setter
    def amplitude_spectral_density(self, amplitude_spectral_density):
        self._check_frequency_array_matches_density_array(amplitude_spectral_density)
        self.__amplitude_spectral_density = amplitude_spectral_density
        self.__power_spectral_density = amplitude_spectral_density ** 2
        self._interpolate_power_spectral_density()

    def import_amplitude_spectral_density(self):
        """
        Automagically load one of the amplitude spectral density curves
        contained in the noise_curves directory.

        Test if the file contains a path (i.e., contains '/').
        If not assume the file is in the default directory.
        """

        if '/' not in self.amplitude_spectral_density_file:
            self.amplitude_spectral_density_file = os.path.join(
                os.path.dirname(__file__), 'noise_curves',
                self.amplitude_spectral_density_file)

        self.frequency_array, self.amplitude_spectral_density = np.genfromtxt(
            self.amplitude_spectral_density_file).T

    def import_power_spectral_density(self):
        """
        Automagically load one of the power spectral density curves contained
        in the noise_curves directory.

        Test if the file contains a path (i.e., contains '/').
        If not assume the file is in the default directory.
        """
        if '/' not in self.power_spectral_density_file:
            self.power_spectral_density_file = os.path.join(
                os.path.dirname(__file__), 'noise_curves',
                self.power_spectral_density_file)
        self.frequency_array, self.power_spectral_density = np.genfromtxt(
            self.power_spectral_density_file).T

    def _check_frequency_array_matches_density_array(self, density_array):
        """Check the provided frequency and spectral density arrays match."""
        try:
            self.frequency_array - density_array
        except ValueError as e:
            raise (e, 'Provided spectral density does not match frequency array. Not updating.')

    def _interpolate_power_spectral_density(self):
        """Interpolate the loaded power spectral density so it can be resampled
           for arbitrary frequency arrays.
        """
        self.power_spectral_density_interpolated = interp1d(
            self.frequency_array, self.power_spectral_density, bounds_error=False,
            fill_value=np.inf)

    def get_noise_realisation(self, sampling_frequency, duration):
        """
        Generate frequency Gaussian noise scaled to the power spectral density.

        Parameters
        -------
        sampling_frequency: float
            sampling frequency of noise
        duration: float
            duration of noise

        Returns
        -------
        array_like: frequency domain strain of this noise realisation
        array_like: frequencies related to the frequency domain strain

        """
        white_noise, frequencies = utils.create_white_noise(sampling_frequency, duration)
        frequency_domain_strain = self.power_spectral_density_interpolated(frequencies) ** 0.5 * white_noise
        out_of_bounds = (frequencies < min(self.frequency_array)) | (frequencies > max(self.frequency_array))
        frequency_domain_strain[out_of_bounds] = 0 * (1 + 1j)
        return frequency_domain_strain, frequencies


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
        interferometer = load_interferometer(filename)
        return interferometer
    except FileNotFoundError:
        logger.warning('Interferometer {} not implemented'.format(name))


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
        interferometer = Interferometer(**parameters)
        logger.debug('Assuming L shape for {}'.format('name'))
    elif parameters['shape'].lower() in ['l', 'ligo']:
        parameters.pop('shape')
        interferometer = Interferometer(**parameters)
    elif parameters['shape'].lower() in ['triangular', 'triangle']:
        parameters.pop('shape')
        interferometer = TriangularInterferometer(**parameters)
    else:
        raise IOError("{} could not be loaded. Invalid parameter 'shape'.".format(filename))
    return interferometer


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
    tupak.gw.detector.Interferometer: An Interferometer instance with a PSD and frequency-domain strain data.

    """

    logger.warning(
        "Parameter estimation for real interferometer data in tupak is in "
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
    waveform_generator: tupak.gw.waveform_generator
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
    tupak.gw.detector.Interferometer: An Interferometer instance with a PSD and frequency-domain strain data.

    """

    utils.check_directory_exists_and_if_not_mkdir(outdir)

    if start_time is None:
        start_time = injection_parameters['geocent_time'] + 2 - duration

    interferometer = get_empty_interferometer(name)
    interferometer.power_spectral_density.set_from_aLIGO()
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
    list: A list of tupak.gw.detector.Interferometer objects
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
