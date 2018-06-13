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
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.length = length
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.xarm_azimuth = xarm_azimuth
        self.yarm_azimuth = yarm_azimuth
        self.xarm_tilt = xarm_tilt
        self.yarm_tilt = yarm_tilt
        self.power_spectral_density = power_spectral_density
        self.data = np.array([])
        self.frequency_array = np.array([])
        self.sampling_frequency = None
        self.duration = None
        self.time_marginalization = False
        self.epoch = 0

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
        return (self.frequency_array > self.minimum_frequency) & (self.frequency_array < self.maximum_frequency)

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
        if self.__vertex_updated is False:
            self.__vertex = tupak.gw.utils.get_vertex_position_geocentric(self.__latitude, self.__longitude,
                                                                          self.elevation)
            self.__vertex_updated = True
        return self.__vertex

    @propexrty
    def x(self):
        """ A unit vector along the x-arm

        Is automatically updated if related quantities are modified.

        Returns
        -------
        array_like: A 3D array representation of a unit vector along the x-arm

        """
        if self.__x_updated is False:
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
        if self.__y_updated is False:
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
        if self.__detector_tensor_updated is False:
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

        signal_ifo *= self.frequency_mask

        time_shift = self.time_delay_from_geocenter(
            parameters['ra'],
            parameters['dec'],
            self.epoch)  # parameters['geocent_time'])

        if self.time_marginalization:
            dt = time_shift  # when marginalizing over time we only care about relative time shifts between detectors and marginalized over
            # all candidate coalescence times
        else:
            dt = self.epoch - (parameters['geocent_time'] - time_shift)

        signal_ifo = signal_ifo * np.exp(
            -1j * 2 * np.pi * dt * self.frequency_array)

        return signal_ifo

    def inject_signal(self, waveform_polarizations, parameters):
        """ Inject a signal into noise and adds the requested signal to self.data

        Parameters
        -------
        waveform_polarizations: dict
            polarizations of the waveform
        parameters: dict
            parameters describing position and time of arrival of the signal
        """
        if waveform_polarizations is None:
            logging.warning('Trying to inject signal which is None.')
        else:
            signal_ifo = self.get_detector_response(waveform_polarizations, parameters)
            if np.shape(self.data).__eq__(np.shape(signal_ifo)):
                self.data += signal_ifo
            else:
                logging.info('Injecting into zero noise.')
                self.data = signal_ifo
            opt_snr = np.sqrt(tupak.gw.utils.optimal_snr_squared(signal=signal_ifo, interferometer=self,
                                                                 time_duration=1 / (self.frequency_array[1] -
                                                                                    self.frequency_array[0])).real)
            mf_snr = np.sqrt(tupak.gw.utils.matched_filter_snr_squared(signal=signal_ifo,
                                                                       interferometer=self,
                                                                       time_duration=1 / (self.frequency_array[1] -
                                                                                          self.frequency_array[0])).real)
            logging.info("Injection found with optimal SNR = {:.2f} and matched filter SNR = {:.2f} in {}".format(
                opt_snr, mf_snr, self.name))

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
            return self.__calculate_arm(self.xarm_tilt, self.xarm_azimuth)
        elif arm == 'y':
            return self.__calculate_arm(self.yarm_tilt, self.yarm_azimuth)
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

    def set_data(self, sampling_frequency, duration, epoch=0,
                 from_power_spectral_density=False, zero_noise=False,
                 frequency_domain_strain=None, frame_file=None,
                 channel_name=None, overwrite_psd=True, **kwargs):
        """
        Set the interferometer frequency-domain stain and accompanying PSD values.

        Parameters
        ----------
        sampling_frequency: float
            The sampling frequency of the data
        duration: float
            Duration of data
        epoch: float, optional
            The GPS time of the start of the data
        frequency_domain_strain: array_like, optional
            The frequency-domain strain
        from_power_spectral_density: bool, optional
            If frequency_domain_strain not given, use IFO's PSD object to
            generate noise
        zero_noise: bool, optional
            If true and frequency_domain_strain and from_power_spectral_density
            are false, set the data to be zero.
        frame_file: str, optional
            File from which to load data.
        channel_name: str, optional
            Channel to read from frame.
        overwrite_psd: bool, optional
            Whether to overwrite the psd in the interferometer with one calculated
            from the loaded data, default=True.
        kwargs: dict, optional
            Additional arguments for loading data.

        Raises
        -------
        ValueError: If no method to set data is provided
        """

        self.epoch = epoch

        if frequency_domain_strain is not None:
            logging.info(
                'Setting {} data using provided frequency_domain_strain'.format(self.name))
            frequencies = utils.create_frequency_series(sampling_frequency, duration)
        elif from_power_spectral_density:
            logging.info(
                'Setting {} data using noise realization from provided'
                'power_spectal_density'.format(self.name))
            frequency_domain_strain, frequencies = \
                self.power_spectral_density.get_noise_realisation(
                    sampling_frequency, duration)
        elif zero_noise:
            logging.info('Setting zero noise in {}'.format(self.name))
            frequencies = utils.create_frequency_series(sampling_frequency, duration)
            frequency_domain_strain = np.zeros_like(frequencies) * (1 + 1j)
        elif frame_file is not None:
            logging.info('Reading data from frame, {}.'.format(self.name))
            strain = tupak.gw.utils.read_frame_file(
                frame_file, t1=epoch, t2=epoch + duration, channel=channel_name, resample=sampling_frequency)
            frequency_domain_strain, frequencies = tupak.gw.utils.process_strain_data(strain, **kwargs)
            if overwrite_psd:
                self.power_spectral_density = PowerSpectralDensity(
                    frame_file=frame_file, channel_name=channel_name, epoch=epoch, **kwargs)
        else:
            raise ValueError("No method to set data provided.")

        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.frequency_array = frequencies
        self.data = frequency_domain_strain * self.frequency_mask

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
    def whitened_data(self):
        """ Calculates the whitened data by dividing data by the amplitude spectral density

        Returns
        -------
        array_like: The whitened data
        """
        return self.data / self.amplitude_spectral_density_array

    def save_data(self, outdir):
        """ Creates a save file for the data in plain text format

        Parameters
        ----------
        outdir: str
            The output directory in which the data is supposed to be saved
        """
        np.savetxt('{}/{}_frequency_domain_data.dat'.format(outdir, self.name),
                   [self.frequency_array, self.data.real, self.data.imag],
                   header='f real_h(f) imag_h(f)')
        np.savetxt('{}/{}_psd.dat'.format(outdir, self.name),
                   [self.frequency_array, self.amplitude_spectral_density_array],
                   header='f h(f)')


class PowerSpectralDensity:

    def __init__(self, asd_file=None, psd_file='aLIGO_ZERO_DET_high_P_psd.txt', frame_file=None, epoch=0,
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
        epoch: float, optional
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
        frequency_noise_realization: list
            TODO: This isn't doing anything right now
        interpolated_frequency: list
            TODO: This isn't doing anything right now
        power_spectral_density: array_like
            Array representation of the PSD
        power_spectral_density_file: str
            Name of the PSD file
        power_spectral_density_interpolated: scipy.interpolated.interp1d
            Interpolated function of the PSD
        """

        self.frequencies = []
        self.power_spectral_density = []
        self.amplitude_spectral_density = []
        self.frequency_noise_realization = []
        self.interpolated_frequency = []
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
            strain = tupak.gw.utils.read_frame_file(frame_file, t1=epoch - psd_duration - psd_offset,
                                                    t2=epoch - psd_duration, channel=channel_name)
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
            self.amplitude_spectral_density = self.power_spectral_density ** 0.5
            self.interpolate_power_spectral_density()
        else:
            self.power_spectral_density_file = psd_file
            self.import_power_spectral_density()
            if min(self.power_spectral_density) > 1e-30:
                logging.warning("You specified a power spectral density file.")
                logging.warning("{} WARNING {}".format("*" * 30, "*" * 30))
                logging.warning("The minimum of the provided curve is {:.2e}.".format(
                    min(self.power_spectral_density)))
                logging.warning("You may have intended to provide this as an amplitude spectral density.")

    def import_amplitude_spectral_density(self):
        """
        Automatically load one of the amplitude spectral density curves contained in the noise_curves directory.

        Test if the file contains a path (i.e., contains '/').
        If not assume the file is in the default directory.
        """
        if '/' not in self.amplitude_spectral_density_file:
            self.amplitude_spectral_density_file = os.path.join(os.path.dirname(__file__), 'noise_curves',
                                                                self.amplitude_spectral_density_file)
        spectral_density = np.genfromtxt(self.amplitude_spectral_density_file)
        self.frequencies = spectral_density[:, 0]
        self.amplitude_spectral_density = spectral_density[:, 1]
        self.power_spectral_density = self.amplitude_spectral_density ** 2
        self.interpolate_power_spectral_density()

    def import_power_spectral_density(self):
        """
        Automatically load one of the power spectral density curves contained in the noise_curves directory.

        Test if the file contains a path (i.e., contains '/').
        If not assume the file is in the default directory.
        """
        if '/' not in self.power_spectral_density_file:
            self.power_spectral_density_file = os.path.join(os.path.dirname(__file__), 'noise_curves',
                                                            self.power_spectral_density_file)
        spectral_density = np.genfromtxt(self.power_spectral_density_file)
        self.frequencies = spectral_density[:, 0]
        self.power_spectral_density = spectral_density[:, 1]
        self.amplitude_spectral_density = np.sqrt(self.power_spectral_density)
        self.interpolate_power_spectral_density()

    def interpolate_power_spectral_density(self):
        """Interpolate the loaded PSD so it can be resampled for arbitrary frequency arrays."""
        self.power_spectral_density_interpolated = interp1d(self.frequencies, self.power_spectral_density,
                                                            bounds_error=False,
                                                            fill_value=max(self.power_spectral_density))

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
        return frequency_domain_strain, frequencies


def get_empty_interferometer(name):
    """
    Get an interferometer with standard parameters for known detectors.

    These objects do not have any noise instantiated.

    The available instruments are:
        H1, L1, V1, GEO600

    Detector positions taken from LIGO-T980044-10 for L1/H1 and from
    arXiv:gr-qc/0008066 [45] for V1/GEO600

    Parameters
    ----------
    name: str
        Interferometer identifier.

    Returns
    -------
    interferometer: Interferometer
        Interferometer instance
    """
    known_interferometers = {
        'H1': Interferometer(name='H1', power_spectral_density=PowerSpectralDensity(),
                             minimum_frequency=20, maximum_frequency=2048,
                             length=4, latitude=46 + 27. / 60 + 18.528 / 3600,
                             longitude=-(119 + 24. / 60 + 27.5657 / 3600), elevation=142.554, xarm_azimuth=125.9994,
                             yarm_azimuth=215.994, xarm_tilt=-6.195e-4, yarm_tilt=1.25e-5),
        'L1': Interferometer(name='L1', power_spectral_density=PowerSpectralDensity(),
                             minimum_frequency=20, maximum_frequency=2048,
                             length=4, latitude=30 + 33. / 60 + 46.4196 / 3600,
                             longitude=-(90 + 46. / 60 + 27.2654 / 3600), elevation=-6.574, xarm_azimuth=197.7165,
                             yarm_azimuth=287.7165,
                             xarm_tilt=-3.121e-4, yarm_tilt=-6.107e-4),
        'V1': Interferometer(name='V1', power_spectral_density=PowerSpectralDensity(psd_file='AdV_psd.txt'), length=3,
                             minimum_frequency=20, maximum_frequency=2048,
                             latitude=43 + 37. / 60 + 53.0921 / 3600, longitude=10 + 30. / 60 + 16.1878 / 3600,
                             elevation=51.884, xarm_azimuth=70.5674, yarm_azimuth=160.5674),
        'GEO600': Interferometer(name='GEO600',
                                 power_spectral_density=PowerSpectralDensity(asd_file='GEO600_S6e_asd.txt'),
                                 minimum_frequency=40, maximum_frequency=2048, length=0.6,
                                 latitude=52 + 14. / 60 + 42.528 / 3600, longitude=9 + 48. / 60 + 25.894 / 3600,
                                 elevation=114.425, xarm_azimuth=115.9431, yarm_azimuth=21.6117),
        'CE': Interferometer(name='CE', power_spectral_density=PowerSpectralDensity(psd_file='CE_psd.txt'),
                             minimum_frequency=10, maximum_frequency=2048,
                             length=40, latitude=46 + 27. / 60 + 18.528 / 3600,
                             longitude=-(119 + 24. / 60 + 27.5657 / 3600), elevation=142.554, xarm_azimuth=125.9994,
                             yarm_azimuth=215.994, xarm_tilt=-6.195e-4, yarm_tilt=1.25e-5),
    }

    if name in known_interferometers.keys():
        interferometer = known_interferometers[name]
        return interferometer
    else:
        logging.warning('Interferometer {} not implemented'.format(name))


def get_interferometer_with_open_data(
        name, center_time, T=4, alpha=0.25, psd_offset=-1024, psd_duration=100,
        cache=True, outdir='outdir', plot=True, filter_freq=1024,
        raw_data_file=None, **kwargs):
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
    T: float, optional
        The total time (in seconds) to analyse. Defaults to 4s.
    alpha: float, optional
        The Tukey window shape parameter passed to `scipy.signal.tukey`.
    psd_offset, psd_duration: float, optional
        The power spectral density (psd) is estimated using data from
        `center_time+psd_offset` to `center_time+psd_offset + psd_duration`.
    cache: bool, optional
        Whether or not to store the acquired data
    raw_data_file: str
        Name of a raw data file if this supposed to be read from a local file
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
    tupak.gw.detector.Interferometer: An Interferometer instance with a PSD and frequency-domain strain data.

    """

    logging.warning(
        "Parameter estimation for real interferometer data in tupak is in "
        "alpha testing at the moment: the routines for windowing and filtering"
        " have not been reviewed.")

    utils.check_directory_exists_and_if_not_mkdir(outdir)

    strain = tupak.gw.utils.get_open_strain_data(
        name, center_time - T / 2, center_time + T / 2, outdir=outdir, cache=cache,
        raw_data_file=raw_data_file, **kwargs)

    strain_psd = tupak.gw.utils.get_open_strain_data(
        name, center_time + psd_offset, center_time + psd_offset + psd_duration,
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
    nfft = int(sampling_frequency * T)
    window = signal.windows.tukey(nfft, alpha=alpha)
    psd = strain_psd.psd(fftlength=T, window=window)
    psd_file = '{}/{}_PSD_{}_{}.txt'.format(
        outdir, name, center_time + psd_offset, psd_duration)
    with open('{}'.format(psd_file), 'w+') as psd_file:
        for f, p in zip(psd.frequencies.value, psd.value):
            psd_file.write('{} {}\n'.format(f, p))

    time_series = strain.times.value
    time_duration = time_series[-1] - time_series[0]

    # Apply Tukey window
    n = len(time_series)
    strain = strain * signal.windows.tukey(n, alpha=alpha)

    interferometer = get_empty_interferometer(name)
    interferometer.power_spectral_density = PowerSpectralDensity(
        psd_file=psd_file)
    interferometer.set_data(
        sampling_frequency, time_duration,
        frequency_domain_strain=utils.nfft(
            strain.value, sampling_frequency)[0],
        epoch=strain.epoch.value)

    if plot:
        fig, ax = plt.subplots()
        ax.loglog(interferometer.frequency_array, np.abs(interferometer.data),
                  '-C0', label=name)
        ax.loglog(interferometer.frequency_array,
                  interferometer.amplitude_spectral_density_array,
                  '-C1', lw=0.5, label=name + ' ASD')
        ax.grid('on')
        ax.set_ylabel(r'strain [strain/$\sqrt{\rm Hz}$]')
        ax.set_xlabel(r'frequency [Hz]')
        ax.set_xlim(20, 2000)
        ax.legend(loc='best')
        fig.savefig('{}/{}_frequency_domain_data.png'.format(outdir, name))

    return interferometer


def get_interferometer_with_fake_noise_and_injection(
        name, injection_polarizations, injection_parameters,
        sampling_frequency=4096, time_duration=4, outdir='outdir', plot=True,
        save=True, zero_noise=False):
    """
    Helper function to obtain an Interferometer instance with appropriate
    power spectral density and data, given an center_time.

    Parameters
    ----------
    name: str
        Detector name, e.g., 'H1'.
    injection_polarizations: dict
        polarizations of waveform to inject, output of
        `waveform_generator.get_frequency_domain_signal`
    injection_parameters: dict
        injection parameters, needed for sky position and timing
    sampling_frequency: float
        sampling frequency for data, should match injection signal
    time_duration: float
        length of data, should be the same as used for signal generation
    outdir: str
        directory in which to store output
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

    interferometer = get_empty_interferometer(name)
    if zero_noise:
        interferometer.set_data(
            sampling_frequency=sampling_frequency, duration=time_duration,
            zero_noise=True,
            epoch=(injection_parameters['geocent_time'] + 2) - time_duration)
    else:
        interferometer.set_data(
            sampling_frequency=sampling_frequency, duration=time_duration,
            from_power_spectral_density=True,
            epoch=(injection_parameters['geocent_time'] + 2) - time_duration)
    interferometer.inject_signal(
        waveform_polarizations=injection_polarizations,
        parameters=injection_parameters)

    interferometer_signal = interferometer.get_detector_response(
        injection_polarizations, injection_parameters)

    if plot:
        fig, ax = plt.subplots()
        ax.loglog(interferometer.frequency_array, np.abs(interferometer.data),
                  '-C0', label=name)
        ax.loglog(interferometer.frequency_array,
                  interferometer.amplitude_spectral_density_array,
                  '-C1', lw=0.5, label=name + ' ASD')
        ax.loglog(interferometer.frequency_array, abs(interferometer_signal),
                  '-C2', label='Signal')
        ax.grid('on')
        ax.set_ylabel(r'strain [strain/$\sqrt{\rm Hz}$]')
        ax.set_xlabel(r'frequency [Hz]')
        ax.set_xlim(20, 2000)
        ax.legend(loc='best')
        fig.savefig('{}/{}_frequency_domain_data.png'.format(outdir, name))

    if save:
        interferometer.save_data(outdir)

    return interferometer


def get_event_data(
        event, interferometer_names=None, time_duration=4, alpha=0.25,
        psd_offset=-1024, psd_duration=100, cache=True, outdir='outdir',
        plot=True, filter_freq=1024, raw_data_file=None, **kwargs):
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
                name, event_time, T=time_duration, alpha=alpha,
                psd_offset=psd_offset, psd_duration=psd_duration, cache=cache,
                outdir=outdir, plot=plot, filter_freq=filter_freq,
                raw_data_file=raw_data_file, **kwargs))
        except ValueError:
            logging.info('No data found for {}.'.format(name))

    return interferometers
