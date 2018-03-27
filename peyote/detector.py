from __future__ import division, print_function
import numpy as np
import os

class Interferometer:
    """Class for the Interferometer """

    def __init__(self, name, x_arm, y_arm, length, power_spectral_density):
        """
        Interferometer class
        :param name: interferometer name, e.g., H1
        :param x_arm: unit vector along one arm in Earth-centered cartesian coordinates
        :param y_arm: unit vector along the other arm in Earth-centered cartesian coordinates
        :param length: length of the interferometer
        """
        self.name = name
        self.x_arm = x_arm
        self.y_arm = y_arm
        self.length = length
        self.detector_tensor = 0.5*(np.tensordot(self.x_arm, self.x_arm, 0)-np.tensordot(self.y_arm, self.y_arm, 0))
        self.power_spectral_density = power_spectral_density
        return

    def antenna_response(self, theta, phi, mode):
        """
        Calculate the antenna response function for a given sky location

        arXiv:0903.0528
        TODO: argue about frames, relative to detector frame, earth-frame?

        :param theta: angle from the north pole
        :param phi: angle from the prime meridian
        :param mode: polarisation mode
        :return: response(theta, phi, mode): antenna response for the specified mode.
        """
        mm = np.array([np.sin(phi), -np.cos(phi), 0])
        nn = np.array([np.cos(phi)*np.cos(theta), np.cos(theta)*np.sin(phi), -np.sin(theta)])
        omega = np.cross(mm, nn)

        if mode=="plus":
            polarisation_tensor = np.tensordot(mm, mm, 0)-np.tensordot(nn, nn, 0)
        elif mode=="cross":
            polarisation_tensor = np.tensordot(mm, nn, 0)+np.tensordot(nn, mm, 0)
        elif mode=="b":
            polarisation_tensor = np.tensordot(mm, mm, 0)+np.tensordot(nn, nn, 0)
        elif mode=="l":
            polarisation_tensor = np.sqrt(2)*np.tensordot(omega, omega, 0)
        elif mode=="x":
            polarisation_tensor = np.tensordot(mm, omega, 0)+np.tensordot(omega, mm, 0)
        elif mode=="y":
            polarisation_tensor = np.tensordot(nn, omega, 0)+np.tensordot(omega, nn, 0)
        else:
            print("Not a polarization mode!")
            return None

        response = np.tensordot(self.detector_tensor, polarisation_tensor, axes=2)
        return response


class PowerSpectralDensity:

    def __init__(self):
        self.frequencies = []
        self.power_spectral_density = []
        self.amplitude_spectral_density = []
        return

    def import_power_spectral_density(self, spectral_density_file='aLIGO_ZERO_DET_high_P_psd.txt'):
        """
        Automagically load the power spectral density 
        curves contained in the noise_curves directory
        """
        sd_file = os.path.join(os.path.dirname(__file__), 'noise_curves', spectral_density_file)
        spectral_density = np.genfromtxt(sd_file)
        self.frequencies = spectral_density[:, 0]
        self.power_spectral_density = spectral_density[:, 1]
        self.amplitude_spectral_density = np.sqrt(self.power_spectral_density)

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
