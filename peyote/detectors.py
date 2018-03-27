class Interferometer():
    """Class for the Interferometer """


    def __init__(self, arg):
        super([object Object], self).__init__()
        self.arg = arg


class PowerSpectralDensity(Interferometer):
    def __init__(self, name):
        self.name = name

    def import_power_spectral_density_file(self, psd_file='aLIGO_ZERO_DET_high_P_psd.txt'):
        '''
        Automagically load one of the PSD curves contained in the NoiseCurves directory of MonashGWTools
        without having to point to a local directory
        '''
        psd_file = os.path.join(os.path.dirname(__file__), 'noise_curves', psd_file)
        psd = np.genfromtxt(psd_file)
        return psd




'''
class
- import psd
-
'''
