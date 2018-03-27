from __future__ import division, print_function
import numpy as np
import os
from scipy.interpolate import interp1d


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
        self.D = 0.5 * (np.tensordot(self.x, self.x, 0) - np.tensordot(self.y, self.y, 0))
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
        m = np.array([np.sin(phi), -np.cos(phi), 0])
        n = np.array([np.cos(phi) * np.cos(theta), np.cos(theta) * np.sin(phi), -np.sin(theta)])
        omega = np.cross(m, n)

        if mode == "plus":
            e = np.tensordot(m, m, 0) - np.tensordot(n, n, 0)
        elif mode == "cross":
            e = np.tensordot(m, n, 0) + np.tensordot(n, m, 0)
        elif mode == "b":
            e = np.tensordot(m, m, 0) + np.tensordot(n, n, 0)
        elif mode == "l":
            e = np.sqrt(2) * np.tensordot(omega, omega, 0)
        elif mode == "x":
            e = np.tensordot(m, omega, 0) + np.tensordot(omega, m, 0)
        elif mode == "y":
            e = np.tensordot(n, omega, 0) + np.tensordot(omega, n, 0)
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

    def import_amplitude_spectral_density(self, spectral_density_file='aLIGO_ZERO_DET_high_P_asd.txt'):
        """
        Automagically load one of the amplitude spectral density
        curves contained in the noise_curves directory
        """
        sd_file = os.path.join(os.path.dirname(__file__), 'noise_curves', spectral_density_file)
        spectral_density = np.genfromtxt(sd_file)
        self.frequencies = spectral_density[:, 0]
        self.amplitude_spectral_density = spectral_density[:, 1]
        self.power_spectral_density = self.amplitude_spectral_density ** 2

    def noise_realisations(self, sampling_frequency, duration):

        number_of_samples = int(np.round(duration * sampling_frequency))

        # prepare for FFT
        number_of_frequencies = (number_of_samples - 1) // 2
        delta_f = 1. / duration

        f = self.equally_spaced_frequency_array(delta_f, number_of_frequencies)

        pf1_interp_func = interp1d(self.frequencies, self.power_spectral_density, bounds_error=False, fill_value=np.inf)
        pf1 = pf1_interp_func(f)
        if sum(np.isinf(pf1)) > 0:
            pf1[np.isinf(pf1)] = max(pf1[~np.isinf(pf1)])

        norm1 = 0.5 * (pf1 / delta_f) ** 0.5
        re1 = np.random.normal(0, norm1, int(number_of_frequencies))
        im1 = np.random.normal(0, norm1, int(number_of_frequencies))
        z1 = re1 + 1j * im1

        # freq domain solution for htilde1, htilde2 in terms of z1, z2
        htilde1 = z1
        # convolve data with instrument transfer function
        otilde1 = htilde1 * 1.
        # set DC and Nyquist = 0
        # python: we are working entirely with positive frequencies
        if np.mod(number_of_samples, 2) == 0:
            otilde1 = np.concatenate(([0], otilde1, [0]))
            f = np.concatenate(([0], f, [sampling_frequency / 2.]))
        else:
            # no Nyquist frequency when N=odd
            otilde1 = np.concatenate(([0], otilde1))
            f = np.concatenate(([0], f))

        # normalise for positive frequencies and units of strain/rHz
        hf = otilde1
        # python: transpose for use with infft
        hf = np.transpose(hf)
        f = np.transpose(f)

        self.frequency_noise_realization = hf
        self.interpolated_frequency = f

    @staticmethod
    def equally_spaced_frequency_array(deltaF, numFreqs):
        f = deltaF * np.linspace(1, numFreqs, numFreqs)
        return f
