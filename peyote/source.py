import peyote
import numpy as np
import os.path
from astropy.table import Table
from peyote.utils import sampling_frequency, nfft


class Source:
    def __init__(self, name):
        self.name = name

    def model(self, time):
        return 0


class SimpleSinusoidSource(Source):
    """ A simple example of a sinusoid source

    model takes one parameter `parameters`, a dictionary of Parameters and
    returns the waveform model.

    """

    def model(self, parameters):
        return parameters['A'] * np.sin(
            parameters['f'] * parameters['geocent_time'])


class Glitch(Source):
    def __init__(self, name):
        Source.__init__(self, name)


class AstrophysicalSource(Source):

    def __init__(self, name, right_ascension, declination, luminosity_distance):
        Source.__init__(self, name)
        self.right_ascension = right_ascension
        self.declination = declination
        self.luminosity_distance = luminosity_distance


class CompactBinaryCoalescence(AstrophysicalSource):
    def __init__(self, name, right_ascension, declination, luminosity_distance, mass_1, mass_2, spin_1, spin_2,
                 coalescence_time, inclination_angle, waveform_phase, polarisation_angle, eccentricity):
        AstrophysicalSource.__init__(self, name, right_ascension, declination, luminosity_distance)
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.spin_1 = spin_1
        self.spin_2 = spin_2
        self.coalescence_time = coalescence_time  # tc
        self.inclination_angle = inclination_angle  # iota
        self.waveform_phase = waveform_phase  # phi
        self.polarisation_angle = polarisation_angle  # psi
        self.eccentricity = eccentricity


class Supernova(AstrophysicalSource):
    def __init__(self, name, right_ascension, declination, luminosity_distance):
        AstrophysicalSource.__init__(self, name, right_ascension, declination, luminosity_distance)


class BinaryBlackHole(CompactBinaryCoalescence):
    def __init__(self, name, right_ascension, declination, luminosity_distance, mass_1, mass_2, spin_1, spin_2,
                 coalescence_time, inclination_angle, waveform_phase, polarisation_angle, eccentricity):
        CompactBinaryCoalescence.__init__(self, name, right_ascension, declination, luminosity_distance, mass_1,
                                          mass_2, spin_1, spin_2, coalescence_time, inclination_angle, waveform_phase,
                                          polarisation_angle, eccentricity)


class BinaryNeutronStar(CompactBinaryCoalescence):
    def __init__(self, name, right_ascension, declination, luminosity_distance, mass_1, mass_2, spin_1, spin_2,
                 coalescence_time, inclination_angle, waveform_phase, polarisation_angle, eccentricity,
                 tidal_deformability_1, tidal_deformability_2):
        CompactBinaryCoalescence.__init__(self, name, right_ascension, declination, luminosity_distance, mass_1,
                                          mass_2, spin_1, spin_2, coalescence_time, inclination_angle, waveform_phase,
                                          polarisation_angle, eccentricity)
        self.tidal_deformability_1 = tidal_deformability_1  # lambda parameter for Neutron Star 1
        self.tidal_deformability_2 = tidal_deformability_2  # lambda parameter for Neutron Star 2


class NeutronStarBlackHole(CompactBinaryCoalescence):
    def __init__(self, name, right_ascension, declination, luminosity_distance, mass_1, mass_2, spin_1, spin_2,
                 coalescence_time, inclination_angle, waveform_phase, polarisation_angle, eccentricity,
                 tidal_deformability):
        CompactBinaryCoalescence.__init__(self, name, right_ascension, declination, luminosity_distance, mass_1,
                                          mass_2, spin_1, spin_2, coalescence_time, inclination_angle, waveform_phase,
                                          polarisation_angle, eccentricity)
        self.tidal_deformability = tidal_deformability  # lambda parameter for Neutron Star


class BinaryNeutronStarMergerNumericalRelativity(Source):
    """ Loads in NR simulations of BNS merger

    takes parameters mean_mass, mass_ratio and equation_of_state, directory_path

    returns time,hplus,hcross,freq,Hplus(freq),Hcross(freq)

    """

    def model(self, parameters):
        mean_mass_string = '{:.0f}'.format(parameters['mean_mass'] * 1000)
        eos_string = parameters['equation_of_state']
        mass_ratio_string = '{:.0f}'.format(parameters['mass_ratio'] * 10)
        directory_path = parameters['directory_path']

        file_name = '{}-q{}-M{}.csv'.format(eos_string, mass_ratio_string, mean_mass_string)
        full_filename = '{}/{}'.format(directory_path, file_name)

        if not os.path.isfile(full_filename):
            print('{} does not exist'.format(full_filename)) # add exception
            return(-1)
        else: # ok file exists
            strain_table = Table.read(full_filename)
            Hplus, _ = nfft(strain_table["hplus"], sampling_frequency(strain_table['time']))
            Hcross, frequency = nfft(strain_table["hcross"], sampling_frequency(strain_table['time']))
            return(strain_table['time'],strain_table["hplus"],strain_table["hcross"],frequency,Hplus,Hcross)


