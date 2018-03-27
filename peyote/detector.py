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
        return None

    def import_spectral_density_file(self, spectral_density_file='aLIGO_ZERO_DET_high_P_psd.txt'):
        """
        Automagically load one of the power spectral density or amplitude spectral density
        curves contained in the noise_curves directory
        """
        sd_file = os.path.join(os.path.dirname(__file__), 'noise_curves', spectral_density_file)
        spectral_density = np.genfromtxt(sd_file)
        return spectral_density

    def convert_psd_to_asd(self, power_spectral_density):
        """
        Convert a power spectral density to an amplitude spectral spectral_density
        Return a two-dimensional array of frequency and amplitude spectral density.
        """
        frequencies = self.power_spectral_density[:, 0]
        psd = self.power_spectral_density[:, 1]
        amplitude_spectral_density = np.sqrt(psd)
        return np.c_[frequencies, amplitude_spectral_density]

    def convert_asd_to_psd(self, amplitude_spectral_density):
        """
        Convert an amplitude spectral density to a power spectral density.
        Return two dimensional array: frequency, power spectral density
        """
        frequencies = self.amplitude_spectral_density[:, 0]
        asd = self.amplitude_spectral_density[:, 1]
        power_spectral_density = asd**2.
        return np.c_[frequencies, power_spectral_density]
