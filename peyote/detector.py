import os
import numpy as np

class Interferometer:
    """Class for the Interferometer """

    def __init__(self, name, x, y, length):
        """
        Interferometer class
        :param name: interferometer name, e.g., H1
        :param x: unit vector along one arm in the geocentric frame
        :param y: unit vector along the other arm in the geocentric frame
        :param length: length of the interferometer
        """
        self.name = name
        self.x = x
        self.y = y
        self.length = length
        return None

    def antenna_response(self):
        '''sylvia does stuff'''
        return None

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

    def import_amplitude_spectral_density(self, spectral_density_file='aLIGO_ZERO_DET_high_P_asd.txt'):
        """
        Automagically load one of the p]amplitude spectral density
        curves contained in the noise_curves directory
        """
        sd_file = os.path.join(os.path.dirname(__file__), 'noise_curves', spectral_density_file)
        spectral_density = np.genfromtxt(sd_file)
        self.frequencies = spectral_density[:, 0]
        self.amplitude_spectral_density = spectral_density[:, 1]

    def convert_psd_to_asd(self):
        """
        Convert a power spectral density to an amplitude spectral spectral_density
        """
        self.amplitude_spectral_density = np.sqrt(self.power_spectral_density)

    def convert_asd_to_psd(self):
        """
        Convert an amplitude spectral density to a power spectral density.
        """
        self.power_spectral_density = self.amplitude_spectral_density**2
