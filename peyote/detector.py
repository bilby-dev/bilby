from __future__ import division, print_function
import numpy as np
import logging
import os
from scipy.interpolate import interp1d
import peyote


class Interferometer:
    """Class for the Interferometer """

    def __init__(self, name, power_spectral_density, length, latitude, longitude, elevation, xarm_azimuth, yarm_azimuth,
                 xarm_tilt=0., yarm_tilt=0.):
        """
        Interferometer class

        :param name: interferometer name, e.g., H1
        :param power_spectral_density: PowerSpectralDensity object, default is aLIGO design sensitivity.
        :param length: length of the interferometer
        :param latitude: latitude North in degrees (South is negative)
        :param longitude: longitude East in degrees (West is negative)
        :param elevation: height above surface in meters
        :param xarm_azimuth: orientation of the x arm in degrees North of East
        :param yarm_azimuth: orientation of the y arm in degrees North of East
        :param xarm_tilt: tilt of the x arm in radians above the horizontal defined by ellipsoid earth model in
                          LIGO-T980044-08
        :param yarm_tilt: tilt of the y arm in radians above the horizontal
        """
        self.name = name
        self.length = length
        self.latitude = latitude * np.pi / 180  # convert to rads
        self.longitude = longitude * np.pi / 180
        self.elevation = elevation
        self.xarm_azimuth = xarm_azimuth * np.pi / 180
        self.yarm_azimuth = yarm_azimuth * np.pi / 180
        self.xarm_tilt = xarm_tilt
        self.yarm_tilt = yarm_tilt
        self.x = self.unit_vector_along_arm('x')
        self.y = self.unit_vector_along_arm('y')
        self.detector_tensor = self.detector_tensor()
        self.vertex = self.vertex_position_geocentric()
        self.power_spectral_density = power_spectral_density
        self.power_spectral_density_array = np.array([])
        self.amplitude_spectral_density_array = np.array([])
        self.data = np.array([])
        self.whitened_data = np.array([])
        self.frequency_array = []

    def antenna_response(self, ra, dec, time, psi, mode):
        """
        Calculate the antenna response function for a given sky location

        See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
        [u, v, w] represent the Earth-frame
        [m, n, omega] represent the wave-frame
        Note: there is a typo in the definition of the wave-frame in Nishizawa et al.

        :param ra: right ascension in radians
        :param dec: declination in radians
        :param time: geocentric GPS time
        :param psi: binary polarisation angle counter-clockwise about the direction of propagation
        :param mode: polarisation mode
        :return: detector_response(theta, phi, psi, mode): antenna response for the specified mode.
        """
        polarization_tensor = peyote.utils.get_polarization_tensor(ra, dec, time, psi, mode)
        detector_response = np.einsum('ij,ij->', self.detector_tensor, polarization_tensor)
        return detector_response

    def detector_tensor(self):
        """
        Calculate the detector tensor from the unit vectors along each arm of the detector.

        See Eq. B6 of arXiv:gr-qc/0008066
        """
        detector_tensor = 0.5 * (np.einsum('i,j->ij', self.x, self.x) - np.einsum('i,j->ij', self.y, self.y))
        return detector_tensor

    def inject_signal(self, source):
        """
        Inject a signal into noise.

        Adds the requested signal to self.data

        :param source: source type

        """
        signal = source.frequency_domain_strain()

        for mode in signal.keys():
            det_response = self.antenna_response(
                source.ra, source.dec, source.geocent_time,
                source.psi, mode)

            signal[mode] *= det_response
        signal_ifo = sum(signal.values())

        time_shift = self.time_delay_from_geocenter(
            source.ra, source.dec, source.geocent_time)
        signal_ifo *= np.exp(-1j*2*np.pi*time_shift*source.frequency_array)

        self.data += signal_ifo

    def unit_vector_along_arm(self, arm):
        """
        Calculate the unit vector pointing along the specified arm in cartesian Earth-based coordinates.

        See Eqs. B14-B17 in arXiv:gr-qc/0008066

        Input:
        arm - x or y arm of the detector
        Output:
        n - unit vector along arm in cartesian Earth-based coordinates
        """
        e_long = np.array([-np.sin(self.longitude), np.cos(self.longitude), 0])
        e_lat = np.array([-np.sin(self.latitude) * np.cos(self.longitude),
                          -np.sin(self.latitude) * np.sin(self.longitude), np.cos(self.latitude)])
        e_h = np.array([np.cos(self.latitude) * np.cos(self.longitude),
                        np.cos(self.latitude) * np.sin(self.longitude), np.sin(self.latitude)])
        if arm == 'x':
            n = np.cos(self.xarm_tilt) * np.cos(self.xarm_azimuth) * e_long + np.cos(self.xarm_tilt) \
                * np.sin(self.xarm_azimuth) * e_lat + np.sin(self.xarm_tilt) * e_h
        elif arm == 'y':
            n = np.cos(self.yarm_tilt) * np.cos(self.yarm_azimuth) * e_long + np.cos(self.yarm_tilt) \
                * np.sin(self.yarm_azimuth) * e_lat + np.sin(self.yarm_tilt) * e_h
        else:
            print('Not a recognized arm, aborting!')
            return
        return n

    def set_spectral_densities(self):
        """
        Set the PSD for the interferometer for a user-specified frequency series, this matches the data provided.

        """
        self.power_spectral_density_array = \
            self.power_spectral_density.power_spectral_density_interpolated(self.frequency_array)
        self.amplitude_spectral_density_array = self.power_spectral_density_array ** 0.5

    def set_data(self, sampling_frequency, duration, from_power_spectral_density=None,
                 frequency_domain_strain=None):
        """
        Set the interferometer frequency-domain stain and accompanying PSD values.

        :param sampling_frequency: sampling frequency
        :param duration: duration of data
        :param from_power_spectral_density: flag, use IFO's PSD object to generate noise
        :param frequency_domain_strain: frequency-domain strain, requires frequencies is also specified
        """

        if frequency_domain_strain is not None:
            logging.info(
                'Setting {} data using provided frequency_domain_strain'
                .format(self.name))
            frequencies = peyote.utils.create_fequency_series(sampling_frequency, duration)
        elif from_power_spectral_density is not None:
            logging.info(
                'Setting {} data using noise realization from provided'
                'power_spectal_density'.format(self.name))
            frequency_domain_strain, frequencies = self.power_spectral_density.get_noise_realisation(sampling_frequency,
                                                                                                     duration)
        else:
            raise ValueError("No method to set data provided.")

        self.data = frequency_domain_strain
        self.frequency_array = frequencies
        self.set_spectral_densities()

        return

    def time_delay_from_geocenter(self, ra, dec, time):
        """
        Calculate the time delay from the geocenter for the interferometer.

        Use the time delay function from utils.

        Input:
        ra - right ascension of source in radians
        dec - declination of source in radians
        time - GPS time
        Output:
        delta_t - time delay from geocenter
        """
        delta_t = peyote.utils.time_delay_geocentric(self.vertex, np.array([0, 0, 0]), ra, dec, time)
        return delta_t

    def vertex_position_geocentric(self):
        """
        Calculate the position of the IFO vertex in geocentric coordinates in meters.

        Based on arXiv:gr-qc/0008066 Eqs. B11-B13 except for the typo in the definition of the local radius.
        See Section 2.1 of LIGO-T980044-10 for the correct expression
        """
        vertex_position = peyote.utils.get_vertex_position_geocentric(self.latitude, self.longitude, self.elevation)
        return vertex_position

    def whiten_data(self):
        self.whitened_data = self.data / self.amplitude_spectral_density_array


class PowerSpectralDensity:

    def __init__(self, asd_file=None, psd_file='aLIGO_ZERO_DET_high_P_psd.txt'):
        """
        Instantiate a new PSD object.

        Only one of the asd_file or psd_file needs to be specified.
        If multiple are given, the first will be used.
        FIXME: Allow reading a frame and then FFT to get PSD, use gwpy?

        :param asd_file: amplitude spectral density, format 'f h_f'
        :param psd_file: power spectral density, format 'f h_f'
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
            if min(self.power_spectral_density) < 1e30:
                print("You specified an amplitude spectral density file.")
                print("{} WARNING {}".format("*" * 30, "*" * 30))
                print("The minimum of the provided curve is {:.2e}.".format(min(self.power_spectral_density)))
                print("You may have intended to provide this as a power spectral density.")
        else:
            self.power_spectral_density_file = psd_file
            self.import_power_spectral_density()
            if min(self.power_spectral_density) > 1e30:
                print("You specified a power spectral density file.")
                print("{} WARNING {}".format("*" * 30, "*" * 30))
                print("The minimum of the provided curve is {:.2e}.".format(min(self.power_spectral_density)))
                print("You may have intended to provide this as an amplitude spectral density.")

    def import_amplitude_spectral_density(self):
        """
        Automagically load one of the amplitude spectral density curves contained in the noise_curves directory.

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
        Automagically load one of the power spectral density curves contained in the noise_curves directory.

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

        :param sampling_frequency: sampling frequency of noise
        :param duration: duration of noise
        :return:  frequency_domain_strain (array), frequency (array)
        """
        white_noise, frequency = peyote.utils.create_white_noise(sampling_frequency, duration)

        interpolated_power_spectral_density = self.power_spectral_density_interpolated(frequency)

        frequency_domain_strain = 0.5 * interpolated_power_spectral_density ** 0.5 * white_noise

        return frequency_domain_strain, frequency


# Detector positions taken from LIGO-T980044-10 for L1/H1 and from arXiv:gr-qc/0008066 [45] for V1/ GEO600
H1 = Interferometer(name='H1', power_spectral_density=PowerSpectralDensity(), length=4,
                    latitude=46 + 27. / 60 + 18.528 / 3600,
                    longitude=-(119 + 24. / 60 + 27.5657 / 3600), elevation=142.554, xarm_azimuth=125.9994,
                    yarm_azimuth=215.994, xarm_tilt=-6.195e-4, yarm_tilt=1.25e-5)
L1 = Interferometer(name='L1', power_spectral_density=PowerSpectralDensity(), length=4,
                    latitude=30 + 33. / 60 + 46.4196 / 3600,
                    longitude=-(90 + 46. / 60 + 27.2654 / 3600), elevation=-6.574, xarm_azimuth=197.7165,
                    yarm_azimuth=287.7165,
                    xarm_tilt=-3.121e-4, yarm_tilt=-6.107e-4)
V1 = Interferometer(name='V1', power_spectral_density=PowerSpectralDensity(psd_file='AdV_psd.txt'), length=3,
                    latitude=43 + 37. / 60 + 53.0921 / 3600, longitude=10 + 30. / 60 + 16.1878 / 3600,
                    elevation=51.884, xarm_azimuth=70.5674, yarm_azimuth=160.5674)
# FIXME: Using initial LIGO noise curve for GEO600, do we really want GEO?
GEO600 = Interferometer(name='GEO600', power_spectral_density=PowerSpectralDensity(psd_file='LIGO_srd_psd.txt'),
                        length=0.6, latitude=52 + 14. / 60 + 42.528 / 3600, longitude=9 + 48. / 60 + 25.894 / 3600,
                        elevation=114.425,
                        xarm_azimuth=115.9431, yarm_azimuth=21.6117)
