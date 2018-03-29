import peyote
import numpy as np
import os.path

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
                 tidal_deformability_1, tidal_deformability_2 ):

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

    model takes one parameter `parameters`, a dictionary of Parameters and
    returns the waveform model.

    """

    def model(self, parameters):
        mean_mass_string='{:.0f}'.format(self.parameters['mean_mass'].value * 1000)
        eos_string=self.parameters['equation_of_state'].value
        mass_ratio_string='{:.0f}'.format(self.parameters['mass_ratio'].value*10)
        directory_path=self.parameters['directory_path'].value

        file_name='{}-q{}-M{}.csv'.format(eos_string,mass_ratio_string,mean_mass_string)
        full_filename='{}/{}'.format(directory_path,file_name)
        print(full_filename)
        if os.path.isfile(fname):
            return full_filename


