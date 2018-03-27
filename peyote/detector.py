class Interferometer:
    """Class for the Interferometer """


    def __init__(self, name, x, y, length):
        '''

        :param name: interferometer name, e.g., H1
        :param x: unit vector along one arm in the geocentric frame
        :param y: unit vector along the other arm in the geocentric frame
        :param length: length of the interferometer
        '''
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

    def import_power_spectral_density_file(self, psd_file='aLIGO_ZERO_DET_high_P_psd.txt'):
        '''
        Automagically load one of the PSD curves contained in the NoiseCurves directory of MonashGWTools
        without having to point to a local directory
        '''
        psd_file = os.path.join(os.path.dirname(__file__), 'noise_curves', psd_file)
        psd = np.genfromtxt(psd_file)
        return psd
