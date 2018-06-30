from __future__ import division, print_function, absolute_import

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from gwpy.signal import filter_design
from scipy import signal
from scipy.interpolate import interp1d

import tupak.gw.utils
from tupak.core import utils


class InterferometerSet(list):
    """ A list of Interferometer objects """
    def __init__(self, interferometers):
        """ Instantiate a InterferometerSet

        The InterferometerSet contains a list of Interferometer objects, each
        object has the data used in evaluating the likelihood

        Parameters
        ----------
        interferometers: iterable
            The list of interferometers
        """

        if type(interferometers) == str:
            raise ValueError("Input must not be a string")
        for ifo in interferometers:
            if type(ifo) != Interferometer:
                raise ValueError("Input list of interferometers are not all Interferometer objects")
        list.__init__(self, interferometers)
        self._check_interferometers()

    def _check_interferometers(self):
        """ Check certain aspects of the set are the same """
        consistent_attributes = ['duration', 'start_time', 'sampling_frequency']
        for attribute in consistent_attributes:
            x = [getattr(interferometer.strain_data, attribute)
                 for interferometer in self]
            if not all(y == x[0] for y in x):
                raise ValueError("The {} of all interferometers are not the same".format(attribute))

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
        super(InterferometerSet, self).append(interferometer)
        self._check_interferometers()

    def extend(self, interferometers):
        super(InterferometerSet, self).extend(interferometers)
        self._check_interferometers()

    def insert(self, index, interferometer):
        super(InterferometerSet, self).insert(index, interferometer)
        self._check_interferometers()


class InterferometerStrainData(object):
    """ Strain data for an interferometer """
    def __init__(self, minimum_frequency=0, maximum_frequency=np.inf):
        """ Initiate an InterferometerStrainData object

        The initialised object contains no data, this should be added using one
        of the `set_from..` methods.

        Parameters
        ----------
        minimum_frequency: float
            Minimum frequency to analyse for detector. Default is 0.
        maximum_frequency: float
            Maximum frequency to analyse for detector. Default is infinity.

        """
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.sampling_frequency = None
        self.duration = None
        self.start_time = None

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
            logging.debug("Time is before the start_time")
            return False
        elif time > self.start_time + self.duration:
            logging.debug("Time is after the start_time + duration")
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
    def time_domain_strain(self):
        if self._time_domain_strain is not None:
            return self._time_domain_strain
        elif self._frequency_domain_strain is not None:
            time_domain_strain = utils.infft(
                self.frequency_domain_strain, self.frequency_array)
            self._time_domain_strain = time_domain_strain
            return self._time_domain_strain

        else:
            raise ValueError("time domain strain data not yet set")

    @property
    def frequency_domain_strain(self):
        if self._frequency_domain_strain is not None:
            return self._frequency_domain_strain * self.frequency_mask
        elif self._time_domain_strain is not None:
            frequency_domain_strain, _ = utils.nfft(
                self._time_domain_strain, self.sampling_frequency)
            self._frequency_domain_strain = frequency_domain_strain
            return self._frequency_domain_strain
        else:
            raise ValueError("frequency domain strain data not yet set")

    def add_to_frequency_domain_strain(self, x):
        self._frequency_domain_strain += x

    def _infer_time_domain_dependence(
            self, sampling_frequency, duration, time_array):
        """ Helper function to figure out if the time_array, or
            sampling_frequency and duration where given
        """

        if (sampling_frequency is not None) and (duration is not None):
            if time_array is not None:
                raise ValueError(
                    "You have given the sampling_frequency, duration, and "
                    "time_array")
            self.sampling_frequency = sampling_frequency
            self.duration = duration
        elif any(x is not None for x in [sampling_frequency, duration]):
            raise ValueError(
                "You must provide both sampling_frequency and duration")
        elif time_array is not None:
            self.sampling_frequency, self.duration = (
                utils.get_sampling_frequency_and_duration_from_time_array(
                    time_array))
            self.time_array = np.array(time_array)
        else:
            raise ValueError(
                "Insufficient information given to set time_array")

    def set_from_time_domain_strain(
            self, time_domain_strain, sampling_frequency=None, duration=None,
            start_time=0, time_array=None):

        self.start_time = start_time
        self._infer_time_domain_dependence(
            sampling_frequency=sampling_frequency, duration=duration,
            time_array=time_array)

        logging.debug('Setting data using provided time_domain_strain')
        if np.shape(time_domain_strain) == np.shape(self.time_array):
            self._time_domain_strain = time_domain_strain
        else:
            raise ValueError("Data times do not match time array")

    def _infer_frequency_domain_dependence(
            self, sampling_frequency, duration, frequency_array):
        """ Helper function to figure out if the frequency_array, or
            sampling_frequency and duration where given
        """

        if (sampling_frequency is not None) and (duration is not None):
            if frequency_array is not None:
                raise ValueError(
                    "You have given the sampling_frequency, duration, and "
                    "frequency_array")
            self.sampling_frequency = sampling_frequency
            self.duration = duration
        elif any(x is not None for x in [sampling_frequency, duration]):
            raise ValueError(
                "You must provide both sampling_frequency and duration")
        elif frequency_array is not None:
            self.sampling_frequency, self.duration = (
                utils.get_sampling_frequency_and_duration_from_frequency_array(
                    frequency_array))
            self.frequency_array = np.array(frequency_array)
        else:
            raise ValueError(
                "Insufficient information given to set frequency_array")

    def set_from_frequency_domain_strain(
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

        self.start_time = start_time
        self._infer_frequency_domain_dependence(
            sampling_frequency=sampling_frequency, duration=duration,
            frequency_array=frequency_array)

        logging.debug('Setting data using provided frequency_domain_strain')
        if np.shape(frequency_domain_strain) == np.shape(self.frequency_array):
            self._frequency_domain_strain = frequency_domain_strain
        else:
            raise ValueError("Data frequencies do not match frequency_array")

    def set_from_power_spectral_density(
            self, power_spectral_density, sampling_frequency, duration,
            start_time=0):
        """ Set the `Interferometer.strain_data` from a power spectral density

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

        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.start_time = start_time

        logging.debug(
            'Setting data using noise realization from provided'
            'power_spectal_density')
        frequency_domain_strain, frequencies = \
            power_spectral_density.get_noise_realisation(
                self.sampling_frequency, self.duration)

        if np.array_equal(frequencies, self.frequency_array):
            self._frequency_domain_strain = frequency_domain_strain
        else:
            raise ValueError("Data frequencies do not match frequency_array")

    def set_from_zero_noise(self, sampling_frequency, duration, start_time=0):
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

        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.start_time = start_time

        logging.debug('Setting zero noise data')
        self._frequency_domain_strain = np.zeros_like(self.frequency_array,
                                                      dtype=np.complex)

    def set_from_frame_file(
            self, frame_file, sampling_frequency, duration, start_time=0,
            channel_name=None, buffer_time=1, filter_freq=None, alpha=0.25):
        """ Set the `Interferometer.strain_data` from a frame file

        Parameters
        ----------
        frame_file: str
            File from which to load data.
        channel_name: str
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
        alpha: float
            The tukey window shape parameter passed to `scipy.signal.tukey`.
        filter_freq: float
            Low pass filter frequency. If None, defaults to the maximum
            frequency given to InterferometerStrainData.

        """

        if filter_freq is None:
            logging.debug(
                "Setting low pass filter_freq using given maximum frequency")
            filter_freq = self.maximum_frequency

        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.start_time = start_time

        logging.info('Reading data from frame')
        strain = tupak.gw.utils.read_frame_file(
            frame_file, t1=start_time, t2=start_time+duration,
            buffer_time=buffer_time, channel=channel_name,
            resample=sampling_frequency)

        frequency_domain_strain, freqs = tupak.gw.utils.process_strain_data(
            strain, filter_freq=filter_freq, alpha=alpha)
        if np.array_equal(freqs, self.frequency_array):
            self._frequency_domain_strain = frequency_domain_strain
        else:
            raise ValueError("Data frequencies do not match frequency_array")


class Interferometer(object):
    """Class for the Interferometer """

    def __init__(self, name, power_spectral_density, minimum_frequency, maximum_frequency,
                 length, latitude, longitude, elevation, xarm_azimuth, yarm_azimuth,
                 xarm_tilt=0., yarm_tilt=0.):
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
            Tilt of the x arm in radians above the horizontal defined by ellipsoid earth model in LIGO-T980044-08.
        yarm_tilt: float, optional
            Tilt of the y arm in radians above the horizontal.
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
        self.time_marginalization = False
        self._strain_data = InterferometerStrainData(
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency)

    @property
    def strain_data(self):
        """ A tupak.gw.detector.InterferometerStrainData instance """
        return self._strain_data

    @strain_data.setter
    def strain_data(self, strain_data):
        """ Set the strain_data """
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
            channel_name=None, buffer_time=1, alpha=0.25, filter_freq=None):
        """ Set the `Interferometer.strain_data` from a frame file

        Parameters
        ----------
        frame_file: str
            File from which to load data.
        channel_name: str
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
        alpha: float
            The tukey window shape parameter passed to `scipy.signal.tukey`.
        filter_freq: float
            Low pass filter frequency. If None, defaults to the maximum
            frequency given to InterferometerStrainData.

        """
        self.strain_data.set_from_frame_file(
            frame_file=frame_file, sampling_frequency=sampling_frequency,
            duration=duration, start_time=start_time,
            channel_name=channel_name, buffer_time=buffer_time,
            alpha=alpha, filter_freq=filter_freq)

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
            self.__vertex = tupak.gw.utils.get_vertex_position_geocentric(self.__latitude, self.__longitude,
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
        polarization_tensor = tupak.gw.utils.get_polarization_tensor(ra, dec, time, psi, mode)
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
            self.strain_data.start_time)  # parameters['geocent_time'])

        if self.time_marginalization:
            dt = time_shift  # when marginalizing over time we only care about relative time shifts between detectors and marginalized over
            # all candidate coalescence times
        else:
            dt = self.strain_data.start_time - (parameters['geocent_time'] - time_shift)

        signal_ifo = signal_ifo * np.exp(
            -1j * 2 * np.pi * dt * self.frequency_array)

        return signal_ifo

    def inject_signal(self,  parameters=None, injection_polarizations=None,
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

        Note: if your signal takes a substantial amount of time to generate, or
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

        if self.strain_data.time_within_data(parameters['geocent_time']):
            logging.warning(
                'Injecting signal outside segment, start_time={}, merger time={}.'
                .format(self.strain_data.start_time, parameters['geocent_time']))

        signal_ifo = self.get_detector_response(injection_polarizations, parameters)
        if np.shape(self.frequency_domain_strain).__eq__(np.shape(signal_ifo)):
            self.strain_data.add_to_frequency_domain_strain(signal_ifo)
        else:
            logging.info('Injecting into zero noise.')
            self.set_strain_data_from_frequency_domain_strain(
                signal_ifo,
                sampling_frequency=self.strain_data.sampling_frequency,
                duration=self.strain_data.duration,
                start_time=self.strain_data.start_time)
        opt_snr = np.sqrt(tupak.gw.utils.optimal_snr_squared(
            signal=signal_ifo, interferometer=self,
            time_duration=self.strain_data.duration).real)
        mf_snr = np.sqrt(tupak.gw.utils.matched_filter_snr_squared(
            signal=signal_ifo, interferometer=self,
            time_duration=self.strain_data.duration).real)

        logging.info("Injected signal in {}:".format(self.name))
        logging.info("  optimal SNR = {:.2f}".format(opt_snr))
        logging.info("  matched filter SNR = {:.2f}".format(mf_snr))
        for key in parameters:
            logging.info('  {} = {}'.format(key, parameters[key]))

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
        """ Calculates the amplitude spectral density (ASD) given we know a power spectral denstiy (PSD)

        Returns
        -------
        array_like: An array representation of the ASD

        """
        return self.power_spectral_density_array ** 0.5

    @property
    def power_spectral_density_array(self):
        """ Calculates the power spectral density (PSD)

        Returns
        -------
        array_like: An array representation of the PSD

        """
        return self.power_spectral_density.power_spectral_density_interpolated(self.frequency_array)

    @property
    def frequency_array(self):
        return self.strain_data.frequency_array

    @property
    def frequency_mask(self):
        return self.strain_data.frequency_mask

    @property
    def frequency_domain_strain(self):
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
        return tupak.gw.utils.time_delay_geocentric(self.vertex, np.array([0, 0, 0]), ra, dec, time)

    def vertex_position_geocentric(self):
        """
        Calculate the position of the IFO vertex in geocentric coordinates in meters.

        Based on arXiv:gr-qc/0008066 Eqs. B11-B13 except for the typo in the definition of the local radius.
        See Section 2.1 of LIGO-T980044-10 for the correct expression

        Returns
        -------
        array_like: A 3D array representation of the vertex
        """
        return tupak.gw.utils.get_vertex_position_geocentric(self.__latitude, self.__longitude, self.__elevation)

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
        fig, ax = plt.subplots()
        ax.loglog(self.frequency_array,
                  np.abs(self.frequency_domain_strain),
                  '-C0', label=self.name)
        ax.loglog(self.frequency_array,
                  self.amplitude_spectral_density_array,
                  '-C1', lw=0.5, label=self.name + ' ASD')
        if signal is not None:
            ax.loglog(self.frequency_array, abs(signal), '-C2',
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


class PowerSpectralDensity(object):

    def __init__(self, asd_file=None, psd_file='aLIGO_ZERO_DET_high_P_psd.txt', frame_file=None, asd_array=None,
                 psd_array=None, frequencies=None, start_time=0,
                 psd_duration=1024, psd_offset=16, channel_name=None, filter_freq=1024, alpha=0.25, fft_length=4):
        """
        Instantiate a new PowerSpectralDensity object.

        Only one of the asd_file or psd_file needs to be specified.
        If multiple are given, the first will be used.

        Parameters
        -------
        asd_file: str, optional
            File containing amplitude spectral density, format 'f h_f'
        psd_file: str, optional
            File containing power spectral density, format 'f h_f'
        frame_file: str, optional
            Frame file to read data from.
        asd_array: array_like, optional
            List of amplitude spectral density values corresponding to frequency_array,
            requires frequency_array to be specified.
        psd_array: array_like, optional
            List of power spectral density values corresponding to frequency_array,
            requires frequency_array to be specified.
        frequencies: array_like, optional
            List of frequency values corresponding to asd_array/psd_array,
        start_time: float, optional
            Beginning of segment to analyse.
        psd_duration: float, optional
            Duration of data to generate PSD from.
        psd_offset: float, optional
            Offset of data from beginning of analysed segment.
        channel_name: str, optional
            Name of channel to use to generate PSD.
        alpha: float, optional
            Parameter for Tukey window.
        fft_length: float, optional
            Number of seconds in a single fft.

        Attributes
        -------
        amplitude_spectral_density: array_like
            Array representation of the ASD
        amplitude_spectral_density_file: str
            Name of the ASD file
        frequencies: array_like
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

        self.frequencies = []
        self.power_spectral_density_interpolated = None

        if asd_file is not None:
            self.amplitude_spectral_density_file = asd_file
            self.import_amplitude_spectral_density()
            if min(self.amplitude_spectral_density) < 1e-30:
                logging.warning("You specified an amplitude spectral density file.")
                logging.warning("{} WARNING {}".format("*" * 30, "*" * 30))
                logging.warning("The minimum of the provided curve is {:.2e}.".format(
                    min(self.amplitude_spectral_density)))
                logging.warning("You may have intended to provide this as a power spectral density.")
        elif frame_file is not None:
            strain = tupak.gw.utils.read_frame_file(frame_file, t1=start_time - psd_duration - psd_offset,
                                                    t2=start_time - psd_offset, channel=channel_name)
            sampling_frequency = int(strain.sample_rate.value)

            # Low pass filter
            bp = filter_design.lowpass(filter_freq, strain.sample_rate)
            strain = strain.filter(bp, filtfilt=True)
            strain = strain.crop(*strain.span.contract(1))

            # Create and save PSDs
            nfft = int(sampling_frequency * fft_length)
            window = signal.windows.tukey(nfft, alpha=alpha)
            psd = strain.psd(fftlength=fft_length, window=window)
            self.frequencies = psd.frequencies
            self.power_spectral_density = psd.value
        elif frequencies is not None:
            self.frequencies = frequencies
            if asd_array is not None:
                self.amplitude_spectral_density = asd_array
            elif psd_array is not None:
                self.power_spectral_density = psd_array
        else:
            if psd_file is None:
                logging.info("No power spectral density provided, using aLIGO, zero detuning, high power.")
            self.power_spectral_density_file = psd_file
            self.import_power_spectral_density()
            if min(self.power_spectral_density) > 1e-30:
                logging.warning("You specified a power spectral density file.")
                logging.warning("{} WARNING {}".format("*" * 30, "*" * 30))
                logging.warning("The minimum of the provided curve is {:.2e}.".format(
                    min(self.power_spectral_density)))
                logging.warning("You may have intended to provide this as an amplitude spectral density.")

    @property
    def power_spectral_density(self):
        return self.__power_spectral_density

    @power_spectral_density.setter
    def power_spectral_density(self, power_spectral_density):
        self._check_frequency_array_matches_density_array(power_spectral_density)
        self.__power_spectral_density = power_spectral_density
        self._interpolate_power_spectral_density()
        self.__amplitude_spectral_density = power_spectral_density**0.5

    @property
    def amplitude_spectral_density(self):
        return self.__amplitude_spectral_density

    @amplitude_spectral_density.setter
    def amplitude_spectral_density(self, amplitude_spectral_density):
        self._check_frequency_array_matches_density_array(amplitude_spectral_density)
        self.__amplitude_spectral_density = amplitude_spectral_density
        self.__power_spectral_density = amplitude_spectral_density**2
        self._interpolate_power_spectral_density()

    def import_amplitude_spectral_density(self):
        """
        Automagically load one of the amplitude spectral density curves contained in the noise_curves directory.

        Test if the file contains a path (i.e., contains '/').
        If not assume the file is in the default directory.
        """
        if '/' not in self.amplitude_spectral_density_file:
            self.amplitude_spectral_density_file = os.path.join(os.path.dirname(__file__), 'noise_curves',
                                                                self.amplitude_spectral_density_file)
        self.frequencies, self.amplitude_spectral_density = np.genfromtxt(self.amplitude_spectral_density_file).T

    def import_power_spectral_density(self):
        """
        Automagically load one of the power spectral density curves contained in the noise_curves directory.

        Test if the file contains a path (i.e., contains '/').
        If not assume the file is in the default directory.
        """
        if '/' not in self.power_spectral_density_file:
            self.power_spectral_density_file = os.path.join(os.path.dirname(__file__), 'noise_curves',
                                                            self.power_spectral_density_file)
        self.frequencies, self.power_spectral_density = np.genfromtxt(self.power_spectral_density_file).T

    def _check_frequency_array_matches_density_array(self, density_array):
        """Check if the provided frequency and spectral density arrays match."""
        try:
            self.frequencies - density_array
        except ValueError as e:
            raise(e, 'Provided spectral density does not match frequency array. Not updating.')

    def _interpolate_power_spectral_density(self):
        """Interpolate the loaded PSD so it can be resampled for arbitrary frequency arrays."""
        self.power_spectral_density_interpolated = interp1d(self.frequencies, self.power_spectral_density,
                                                            bounds_error=False,
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
        interpolated_power_spectral_density = self.power_spectral_density_interpolated(frequencies)
        frequency_domain_strain = interpolated_power_spectral_density ** 0.5 * white_noise
        out_of_bounds = (frequencies < min(self.frequencies)) | (frequencies > max(self.frequencies))
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
        logging.warning('Interferometer {} not implemented'.format(name))


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
    interferometer = Interferometer(**parameters)
    return interferometer


def get_interferometer_with_open_data(
        name, trigger_time, time_duration=4, start_time=None, alpha=0.25, psd_offset=-1024,
        psd_duration=100, cache=True, outdir='outdir', label=None, plot=True, filter_freq=1024,
        raw_data_file=None, **kwargs):
    """
    Helper function to obtain an Interferometer instance with appropriate
    PSD and data, given an center_time.

    Parameters
    ----------

    name: str
        Detector name, e.g., 'H1'.
    trigger_time: float
        Trigger GPS time.
    time_duration: float, optional
        The total time (in seconds) to analyse. Defaults to 4s.
    start_time: float, optional
        Beginning of the segment, if None, the trigger is placed 2s before the end
        of the segment.
    alpha: float, optional
        The tukey window shape parameter passed to `scipy.signal.tukey`.
    psd_offset, psd_duration: float
        The power spectral density (psd) is estimated using data from
        `center_time+psd_offset` to `center_time+psd_offset + psd_duration`.
    cache: bool, optional
        Whether or not to store the acquired data
    raw_data_file: str
        Name of a raw data file if this supposed to be read from a local file
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

    logging.warning(
        "Parameter estimation for real interferometer data in tupak is in "
        "alpha testing at the moment: the routines for windowing and filtering"
        " have not been reviewed.")

    utils.check_directory_exists_and_if_not_mkdir(outdir)

    if start_time is None:
        start_time = trigger_time + 2 - time_duration

    strain = tupak.gw.utils.get_open_strain_data(
        name, start_time - 1, start_time + time_duration + 1, outdir=outdir, cache=cache,
        raw_data_file=raw_data_file, **kwargs)

    strain_psd = tupak.gw.utils.get_open_strain_data(
        name, start_time + time_duration + psd_offset,
        start_time + time_duration + psd_offset + psd_duration,
        raw_data_file=raw_data_file,
        outdir=outdir, cache=cache, **kwargs)

    sampling_frequency = int(strain.sample_rate.value)

    # Low pass filter
    bp = filter_design.lowpass(filter_freq, strain.sample_rate)
    strain = strain.filter(bp, filtfilt=True)
    strain = strain.crop(*strain.span.contract(1))
    strain_psd = strain_psd.filter(bp, filtfilt=True)
    strain_psd = strain_psd.crop(*strain_psd.span.contract(1))

    # Create and save PSDs
    NFFT = int(sampling_frequency * time_duration)
    window = signal.windows.tukey(NFFT, alpha=alpha)
    psd = strain_psd.psd(fftlength=time_duration, window=window)
    psd_file = '{}/{}_PSD_{}_{}.txt'.format(
        outdir, name, start_time + time_duration + psd_offset, psd_duration)
    with open('{}'.format(psd_file), 'w+') as file:
        for f, p in zip(psd.frequencies.value, psd.value):
            file.write('{} {}\n'.format(f, p))

    time_series = strain.times.value
    time_duration = time_series[-1] - time_series[0]

    # Apply Tukey window
    N = len(time_series)
    strain = strain * signal.windows.tukey(N, alpha=alpha)

    interferometer = get_empty_interferometer(name)
    interferometer.power_spectral_density = PowerSpectralDensity(
        psd_file=psd_file)
    interferometer.set_strain_data_from_frequency_domain_strain(
        frequency_domain_strain=utils.nfft(
            strain.value, sampling_frequency)[0],
        sampling_frequency=sampling_frequency, duration=time_duration,
        start_time=strain.epoch.value)

    if plot:
        interferometer.plot_data(outdir=outdir, label=label)

    return interferometer


def get_interferometer_with_fake_noise_and_injection(
        name, injection_parameters, injection_polarizations=None,
        waveform_generator=None, sampling_frequency=4096, time_duration=4,
        start_time=None, outdir='outdir', label=None, plot=True, save=True,
        zero_noise=False):
    """
    Helper function to obtain an Interferometer instance with appropriate
    power spectral density and data, given an center_time.

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
    time_duration: float
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
        start_time = injection_parameters['geocent_time'] + 2 - time_duration

    interferometer = get_empty_interferometer(name)
    if zero_noise:
        interferometer.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency, duration=time_duration,
            start_time=start_time)
    else:
        interferometer.set_strain_data_from_power_spectral_density(
            sampling_frequency=sampling_frequency, duration=time_duration,
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
        event, interferometer_names=None, time_duration=4, alpha=0.25,
        psd_offset=-1024, psd_duration=100, cache=True, outdir='outdir',
        label=None, plot=True, filter_freq=1024, raw_data_file=None, **kwargs):
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
    time_duration: float
        Time duration to search for.
    alpha: float
        The tukey window shape parameter passed to `scipy.signal.tukey`.
    psd_offset, psd_duration: float
        The power spectral density (psd) is estimated using data from
        `center_time+psd_offset` to `center_time+psd_offset + psd_duration`.
    cache: bool
        Whether or not to store the acquired data.
    raw_data_file:
        If we want to read the event data from a local file.
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
    event_time = tupak.gw.utils.get_event_time(event)

    interferometers = []

    if interferometer_names is None:
        if utils.command_line_args.detectors:
            interferometer_names = utils.command_line_args.detectors
        else:
            interferometer_names = ['H1', 'L1', 'V1']

    for name in interferometer_names:
        try:
            interferometers.append(get_interferometer_with_open_data(
                name, trigger_time=event_time, time_duration=time_duration, alpha=alpha,
                psd_offset=psd_offset, psd_duration=psd_duration, cache=cache,
                outdir=outdir, label=label, plot=plot, filter_freq=filter_freq,
                raw_data_file=raw_data_file, **kwargs))
        except ValueError:
            logging.warning('No data found for {}.'.format(name))

    return InterferometerSet(interferometers)
