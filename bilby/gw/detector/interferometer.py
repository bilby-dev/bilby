import os
import sys

import numpy as np
from matplotlib import pyplot as plt

from bilby.core import utils
from bilby.core.utils import logger
from bilby.gw import utils as gwutils
from bilby.gw.calibration import Recalibrate
from .strain_data import InterferometerStrainData

try:
    import gwpy
    import gwpy.signal
except ImportError:
    logger.warning("You do not have gwpy installed currently. You will "
                   " not be able to use some of the prebuilt functions.")


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
        power_spectral_density: bilby.gw.detector.PowerSpectralDensity
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
        self.meta_data = dict()

    def __eq__(self, other):
        if self.name == other.name and \
                self.length == other.length and \
                self.latitude == other.latitude and \
                self.longitude == other.longitude and \
                self.elevation == other.elevation and \
                self.xarm_azimuth == other.xarm_azimuth and \
                self.xarm_tilt == other.xarm_tilt and \
                self.yarm_azimuth == other.yarm_azimuth and \
                self.yarm_tilt == other.yarm_tilt and \
                self.power_spectral_density.__eq__(other.power_spectral_density) and \
                self.calibration_model == other.calibration_model and \
                self.strain_data == other.strain_data:
            return True
        return False

    def __repr__(self):
        return self.__class__.__name__ + '(name=\'{}\', power_spectral_density={}, minimum_frequency={}, ' \
                                         'maximum_frequency={}, length={}, latitude={}, longitude={}, elevation={}, ' \
                                         'xarm_azimuth={}, yarm_azimuth={}, xarm_tilt={}, yarm_tilt={})' \
            .format(self.name, self.power_spectral_density, float(self.minimum_frequency),
                    float(self.maximum_frequency), float(self.length), float(self.latitude), float(self.longitude),
                    float(self.elevation), float(self.xarm_azimuth), float(self.yarm_azimuth), float(self.xarm_tilt),
                    float(self.yarm_tilt))

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
        """ A bilby.gw.detector.InterferometerStrainData instance """
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
        if not self.__x_updated or not self.__y_updated:
            _, _ = self.x, self.y  # noqa
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
            parameters['ra'], parameters['dec'], parameters['geocent_time'])
        dt = parameters['geocent_time'] + time_shift - self.strain_data.start_time

        signal_ifo[self.frequency_mask] = signal_ifo[self.frequency_mask] * np.exp(
            -1j * 2 * np.pi * dt * self.frequency_array[self.frequency_mask])

        signal_ifo[self.frequency_mask] *= self.calibration_model.get_calibration_factor(
            self.frequency_array[self.frequency_mask],
            prefix='recalib_{}_'.format(self.name), **parameters)

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
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
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
                injection_polarizations = \
                    waveform_generator.frequency_domain_strain(parameters)
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

        self.meta_data['optimal_SNR'] = (
            np.sqrt(self.optimal_snr_squared(signal=signal_ifo)).real)
        self.meta_data['matched_filter_SNR'] = (
            self.matched_filter_snr(signal=signal_ifo))
        self.meta_data['parameters'] = parameters

        logger.info("Injected signal in {}:".format(self.name))
        logger.info("  optimal SNR = {:.2f}".format(self.meta_data['optimal_SNR']))
        logger.info("  matched filter SNR = {:.2f}".format(self.meta_data['matched_filter_SNR']))
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

        return (np.cos(arm_tilt) * np.cos(arm_azimuth) * e_long +
                np.cos(arm_tilt) * np.sin(arm_azimuth) * e_lat +
                np.sin(arm_tilt) * e_h)

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
        return (self.power_spectral_density.power_spectral_density_interpolated(self.frequency_array) *
                self.strain_data.window_factor)

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

    @property
    def time_domain_strain(self):
        """ The time domain strain in units of s """
        return self.strain_data.time_domain_strain

    @property
    def time_array(self):
        return self.strain_data.time_array

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
        return gwutils.optimal_snr_squared(
            signal=signal[self.frequency_mask],
            power_spectral_density=self.power_spectral_density_array[self.frequency_mask],
            duration=self.strain_data.duration)

    def inner_product(self, signal):
        """

        Parameters
        ----------
        signal: array_like
            Array containing the signal

        Returns
        -------
        float: The optimal signal to noise ratio possible squared
        """
        return gwutils.noise_weighted_inner_product(
            aa=signal[self.frequency_mask],
            bb=self.frequency_domain_strain[self.frequency_mask],
            power_spectral_density=self.power_spectral_density_array[self.frequency_mask],
            duration=self.strain_data.duration)

    def matched_filter_snr(self, signal):
        """

        Parameters
        ----------
        signal: array_like
            Array containing the signal

        Returns
        -------
        float: The matched filter signal to noise ratio squared

        """
        return gwutils.matched_filter_snr(
            signal=signal[self.frequency_mask],
            frequency_domain_strain=self.frequency_domain_strain[self.frequency_mask],
            power_spectral_density=self.power_spectral_density_array[self.frequency_mask],
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

        if label is None:
            filename_psd = '{}/{}_psd.dat'.format(outdir, self.name)
            filename_data = '{}/{}_frequency_domain_data.dat'.format(outdir, self.name)
        else:
            filename_psd = '{}/{}_{}_psd.dat'.format(outdir, self.name, label)
            filename_data = '{}/{}_{}_frequency_domain_data.dat'.format(outdir, self.name, label)
        np.savetxt(filename_data,
                   np.array(
                       [self.frequency_array,
                        self.frequency_domain_strain.real,
                        self.frequency_domain_strain.imag]).T,
                   header='f real_h(f) imag_h(f)')
        np.savetxt(filename_psd,
                   np.array(
                       [self.frequency_array,
                        self.amplitude_spectral_density_array]).T,
                   header='f h(f)')

    def plot_data(self, signal=None, outdir='.', label=None):
        if utils.command_line_args.test:
            return

        fig, ax = plt.subplots()
        df = self.frequency_array[1] - self.frequency_array[0]
        asd = gwutils.asd_from_freq_series(
            freq_data=self.frequency_domain_strain, df=df)

        ax.loglog(self.frequency_array[self.frequency_mask],
                  asd[self.frequency_mask],
                  color='C0', label=self.name)
        ax.loglog(self.frequency_array[self.frequency_mask],
                  self.amplitude_spectral_density_array[self.frequency_mask],
                  color='C1', lw=1.0, label=self.name + ' ASD')
        if signal is not None:
            signal_asd = gwutils.asd_from_freq_series(
                freq_data=signal, df=df)

            ax.loglog(self.frequency_array[self.frequency_mask],
                      signal_asd[self.frequency_mask],
                      color='C2',
                      label='Signal')
        ax.grid(True)
        ax.set_ylabel(r'Strain [strain/$\sqrt{\rm Hz}$]')
        ax.set_xlabel(r'Frequency [Hz]')
        ax.legend(loc='best')
        fig.tight_layout()
        if label is None:
            fig.savefig(
                '{}/{}_frequency_domain_data.png'.format(outdir, self.name))
        else:
            fig.savefig(
                '{}/{}_{}_frequency_domain_data.png'.format(
                    outdir, self.name, label))
        plt.close(fig)

    def plot_time_domain_data(
            self, outdir='.', label=None, bandpass_frequencies=(50, 250),
            notches=None, start_end=None, t0=None):
        """ Plots the strain data in the time domain

        Parameters
        ----------
        outdir, label: str
            Used in setting the saved filename.
        bandpass: tuple, optional
            A tuple of the (low, high) frequencies to use when bandpassing the
            data, if None no bandpass is applied.
        notches: list, optional
            A list of frequencies specifying any lines to notch
        start_end: tuple
            A tuple of the (start, end) range of GPS times to plot
        t0: float
            If given, the reference time to subtract from the time series before
            plotting.

        """

        # We use the gwpy timeseries to perform bandpass and notching
        if notches is None:
            notches = list()
        timeseries = gwpy.timeseries.TimeSeries(
            data=self.time_domain_strain, times=self.time_array)
        zpks = []
        if bandpass_frequencies is not None:
            zpks.append(gwpy.signal.filter_design.bandpass(
                bandpass_frequencies[0], bandpass_frequencies[1],
                self.strain_data.sampling_frequency))
        if notches is not None:
            for line in notches:
                zpks.append(gwpy.signal.filter_design.notch(
                    line, self.strain_data.sampling_frequency))
        if len(zpks) > 0:
            zpk = gwpy.signal.filter_design.concatenate_zpks(*zpks)
            strain = timeseries.filter(zpk, filtfilt=True)
        else:
            strain = timeseries

        fig, ax = plt.subplots()

        if t0:
            x = self.time_array - t0
            xlabel = 'GPS time [s] - {}'.format(t0)
        else:
            x = self.time_array
            xlabel = 'GPS time [s]'

        ax.plot(x, strain)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Strain')

        if start_end is not None:
            ax.set_xlim(*start_end)

        fig.tight_layout()

        if label is None:
            fig.savefig(
                '{}/{}_time_domain_data.png'.format(outdir, self.name))
        else:
            fig.savefig(
                '{}/{}_{}_time_domain_data.png'.format(outdir, self.name, label))
        plt.close(fig)

    @staticmethod
    def _hdf5_filename_from_outdir_label(outdir, label):
        return os.path.join(outdir, label + '.h5')

    def to_hdf5(self, outdir='outdir', label=None):
        """ Save the object to a hdf5 file

        Attributes
        ----------
        outdir: str, optional
            Output directory name of the file, defaults to 'outdir'.
        label: str, optional
            Output file name, is self.name if not given otherwise.
        """
        import deepdish
        if sys.version_info[0] < 3:
            raise NotImplementedError('Pickling of Interferometer is not supported in Python 2.'
                                      'Use Python 3 instead.')
        if label is None:
            label = self.name
        utils.check_directory_exists_and_if_not_mkdir('outdir')
        filename = self._hdf5_filename_from_outdir_label(outdir, label)
        deepdish.io.save(filename, self)

    @classmethod
    def from_hdf5(cls, filename=None):
        """ Loads in an Interferometer object from an hdf5 file

        Parameters
        ----------
        filename: str
            If given, try to load from this filename

        """
        import deepdish
        if sys.version_info[0] < 3:
            raise NotImplementedError('Pickling of Interferometer is not supported in Python 2.'
                                      'Use Python 3 instead.')

        res = deepdish.io.load(filename)
        if res.__class__ != cls:
            raise TypeError('The loaded object is not an Interferometer')
        return res
