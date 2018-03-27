from __future__ import division, print_function
import numpy as np
import os

class Interferometer:
    """Class for the Interferometer """

    def __init__(self, name, x, y, length):
        """
        Interferometer class
        :param name: interferometer name, e.g., H1
        :param x: unit vector along one arm in Earth-centered cartesian coordinates
        :param y: unit vector along the other arm in Earth-centered cartesian coordinates
        :param length: length of the interferometer
        """
        self.name = name
        self.x = x
        self.y = y
        self.length = length
        self.D = 0.5*(np.tensordot(self.x,self.x,0)-np.tensordot(self.y,self.y,0))
        return None

    def antenna_response(self, theta, phi, mode):
        '''
        Calculate the antenna response function for a given sky location

        arXiv:0903.0528
        TODO: argue about frames, relative to detector frame, earth-frame?

        :param theta: angle from the north pole
        :param phi: angle from the prime meridian
        :param mode: polarisation mode
        :return: f(theta, phi, mode): antenna response for the specified mode.
        '''
        m = np.array([np.sin(phi),-np.cos(phi),0])
        n = np.array([np.cos(phi)*np.cos(theta),np.cos(theta)*np.sin(phi),-np.sin(theta)])
        omega = np.cross(m, n)

        if mode=="plus":
                e = np.tensordot(m, m, 0)-np.tensordot(n, n, 0)
        elif mode=="cross":
                e = np.tensordot(m, n, 0)+np.tensordot(n, m, 0)
        elif mode=="b":
                e = np.tensordot(m, m, 0)+np.tensordot(n, n, 0)
        elif mode=="l":
                e = np.sqrt(2)*np.tensordot(omega, omega, 0)
        elif mode=="x":
                e = np.tensordot(m, omega, 0)+np.tensordot(omega, m, 0)
        elif mode=="y":
                e = np.tensordot(n, omega, 0)+np.tensordot(omega, n, 0)
        else:
                print("Not a polarization mode!")
                return None

        f = np.tensordot(self.D, e, axes=2)
        return f


class PowerSpectralDensity:

    def __init__(self):
        self.frequencies = []
        self.power_spectral_density = []
        self.amplitude_spectral_density = []
        return None

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

    def import_amplitude_spectral_density(self, spectral_density_file='aLIGO_ZERO_DET_high_P_asd.txt'):
        """
        Automagically load one of the p]amplitude spectral density
        curves contained in the noise_curves directory
        """
        sd_file = os.path.join(os.path.dirname(__file__), 'noise_curves', spectral_density_file)
        spectral_density = np.genfromtxt(sd_file)
        self.frequencies = spectral_density[:, 0]
        self.amplitude_spectral_density = spectral_density[:, 1]
        self.power_spectral_density = self.amplitude_spectral_density**2
