from __future__ import division, print_function
import numpy as np
import os
from scipy.interpolate import interp1d
import peyote

class Interferometer:
    """Class for the Interferometer """

    def __init__(self, name, length, latitude, longitude, elevation, xarm_azimuth, yarm_azimuth, xarm_tilt = 0, yarm_tilt = 0):
        """
        Interferometer class
        :param name: interferometer name, e.g., H1
        :param length: length of the interferometer
        :param latitude: latitude North in degrees (South is negative)
        :param longitude: longitude East in degrees (West is negative)
        :param elevation: height above surface in meters
        :param xarm_azimuth: orientation of the x arm in degrees North of East
        :param yarm_azimuth: orientation of the y arm in degrees North of East
        :param xarm_tilt: tilt of the x arm in radians above the horizontal defined by ellipsoid earth model in LIGO-T980044-08
        :param yarm_tilt: tilt of the y arm in radians above the horizontal
        """
        self.name = name
        self.length = length
        self.latitude = latitude * np.pi / 180 #convert to rads
        self.longitude = longitude * np.pi / 180
        self.elevation = elevation
        self.xarm_azimuth = xarm_azimuth * np.pi / 180
        self.yarm_azimuth = yarm_azimuth * np.pi / 180
        self.xarm_tilt = xarm_tilt
        self.yarm_tilt = yarm_tilt
        self.x = self.unit_vector_along_arm('x')
        self.y = self.unit_vector_along_arm('y')
        self.detector_tensor = self.detector_tensor()

    def unit_vector_along_arm(self, arm):
        '''
        Calculates the unit vector pointing along the specified arm of the detector at the given position in cartesian Earth-based coordinates.
        See Eqs. B14-B17 in arXiv:gr-qc/0008066
        Input:
        arm - x or y arm of the detector
        Output:
        n - unit vector along arm in cartesian Earth-based coordinates
        '''
        e_long = np.array([-np.sin(self.longitude), np.cos(self.longitude), 0])
        e_lat = np.array([-np.sin(self.latitude) * np.cos(self.longitude), -np.sin(self.latitude) * np.sin(self.longitude), np.cos(self.latitude)])
        e_h = np.array([np.cos(self.latitude) * np.cos(self.longitude), np.cos(self.latitude) * np.sin(self.longitude), np.sin(self.latitude)])
        if arm == 'x':
            n = np.cos(self.xarm_tilt) * np.cos(self.xarm_azimuth) * e_long + np.cos(self.xarm_tilt) * np.sin(self.xarm_azimuth) * e_lat + np.sin(self.xarm_tilt) * e_h
        elif arm == 'y':
            n = np.cos(self.yarm_tilt) * np.cos(self.yarm_azimuth) * e_long + np.cos(self.yarm_tilt) * np.sin(self.yarm_azimuth) * e_lat + np.sin(self.yarm_tilt) * e_h
        else:
            print('Not a recognized arm, aborting!')
            return
        return n
    

    def detector_tensor(self):
        '''
        Calculate the detector tensor from the unit vectors along each arm of the detector.
        See Eq. B6 of arXiv:gr-qc/0008066
        '''
        detector_tensor = 0.5 * (np.einsum('i,j->ij', self.x, self.x) - np.einsum('i,j->ij', self.y, self.y))
        return detector_tensor 


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
        gmst = peyote.utils.gps_time_to_gmst(time)
        theta, phi = peyote.utils.ra_dec_to_theta_phi(ra, dec, gmst)
        u = np.array([np.cos(phi) * np.cos(theta), np.cos(theta) * np.sin(phi), -np.sin(theta)])
        v = np.array([-np.sin(phi), np.cos(phi), 0])
        m = -u * np.sin(psi) - v * np.cos(psi)
        n = -u * np.cos(psi) + v * np.sin(psi)
        omega = np.cross(m, n)

        if mode == "plus":
            polarization_tensor = np.einsum('i,j->ij', m, m) - np.einsum('i,j->ij', n, n)
        elif mode == "cross":
            polarization_tensor = np.einsum('i,j->ij', m, n) + np.einsum('i,j->ij', n, m)
        elif mode == "breathing":
            polarization_tensor = np.einsum('i,j->ij', m, m) + np.einsum('i,j->ij', n, n)
        elif mode == "longitudinal":
            polarization_tensor = np.sqrt(2) * np.einsum('i,j->ij', omega, omega)
        elif mode == "x":
            polarization_tensor = np.einsum('i,j->ij', m, omega) + np.einsum('i,j->ij', omega, m)
        elif mode == "y":
            polarization_tensor = np.einsum('i,j->ij', n, omega) + np.einsum('i,j->ij', omega, n)
        else:
            print("Not a polarization mode!")
            return None

        detector_response = np.einsum('ij,ij->', self.detector_tensor, polarization_tensor)
        return detector_response


class PowerSpectralDensity:

    def __init__(self):
        self.frequencies = []
        self.power_spectral_density = []
        self.amplitude_spectral_density = []
        self.frequency_noise_realization = []
        self.interpolated_frequency = []

    def import_power_spectral_density(self, spectral_density_file='aLIGO_ZERO_DET_high_P_psd.txt'):
        """
        Automagically load one of the power spectral density or amplitude spectral density
        curves contained in the noise_curves directory
        """
        sd_file = os.path.join(os.path.dirname(__file__), 'noise_curves', spectral_density_file)
        spectral_density = np.genfromtxt(sd_file)
        self.frequencies = spectral_density[:, 0]
        self.power_spectral_density = spectral_density[:, 1]
        self.amplitude_spectral_density = np.sqrt(self.power_spectral_density)
        self.interpolate_power_spectral_density()

    def import_amplitude_spectral_density(self, spectral_density_file='aLIGO_ZERO_DET_high_P_asd.txt'):
        """
        Automagically load one of the amplitude spectral density
        curves contained in the noise_curves directory
        """
        sd_file = os.path.join(os.path.dirname(__file__), 'noise_curves', spectral_density_file)
        spectral_density = np.genfromtxt(sd_file)
        self.frequencies = spectral_density[:, 0]
        self.amplitude_spectral_density = spectral_density[:, 1]
        self.power_spectral_density = self.amplitude_spectral_density**2
        self.interpolate_power_spectral_density()

    def interpolate_power_spectral_density(self):
        self.power_spectral_density_interpolated = interp1d(self.frequencies, self.power_spectral_density,
                                                            bounds_error=False,
                                                            fill_value=max(self.power_spectral_density))

    def noise_realisation(self, sampling_frequency, duration):

        white_noise, frequency = peyote.utils.create_white_noise(sampling_frequency, duration)

        Pf1 = self.power_spectral_density_interpolated(frequency)

        hf = 0.5*(Pf1)**0.5 * white_noise
        self.frequency_noise_realization = hf
        self.interpolated_frequency = frequency

        return hf, frequency

    @staticmethod
    def equally_spaced_frequency_array(deltaF, numFreqs):
        frequency_array = deltaF * np.linspace(1, numFreqs, numFreqs)
        return frequency_array

H1 = Interferometer(name='H1', length=4, latitude=46+27./60+18.528/3600, longitude=-(119+24./60+27.5657/3600),\
                    elevation=142.554, xarm_azimuth=125.9994, yarm_azimuth=215.994, xarm_tilt=-6.195e-4, yarm_tilt=1.25e-5)
L1 = Interferometer(name='L1', length=4, latitude=30+33./60+46.4196/3600, longitude=-(90+46./60+27.2654/3600),\
                    elevation=-6.574, xarm_azimuth=197.7165, yarm_azimuth=287.7165, xarm_tilt=-3.121e-4, yarm_tilt=-6.107e-4)
V1 = Interferometer(name='V1', length=3, latitude=43+37./60+53.0921/3600, longitude=10+30./60+16.1878/3600,\
                    elevation=51.884, xarm_azimuth=70.5674, yarm_azimuth=160.5674)
GEO600 = Interferometer(name='GEO600', length=0.6, latitude=52+14./60+42.528/3600, longitude=9+48./60+25.894/3600,\
                    elevation=114.425, xarm_azimuth=115.9431, yarm_azimuth=21.6117)
