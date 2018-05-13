from __future__ import division, print_function

import logging

try:
    import lalsimulation as lalsim
except ImportError:
    logging.warning("You do not have lalsuite installed currently. You will not be able to use some of the "
                        "prebuilt functions.")

from . import utils


def lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
        iota, phase, waveform_approximant, reference_frequency, ra, dec,
        geocent_time, psi):
    """ A Binary Black Hole waveform model using lalsimulation """
    if mass_2 > mass_1:
        return None

    luminosity_distance = luminosity_distance * 1e6 * utils.parsec
    mass_1 = mass_1 * utils.solar_mass
    mass_2 = mass_2 * utils.solar_mass

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = \
        lalsim.SimInspiralTransformPrecessingNewInitialConditions(iota, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2,
                                                                  mass_1, mass_2, reference_frequency, phase)

    longitude_ascending_nodes = 0.0
    eccentricity = 0.0
    mean_per_ano = 0.0

    waveform_dictionary = None

    approximant = lalsim.GetApproximantFromString(waveform_approximant)

    frequency_minimum = 20
    frequency_maximum = frequency_array[-1]
    delta_frequency = frequency_array[1] - frequency_array[0]

    hplus, hcross = lalsim.SimInspiralChooseFDWaveform(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, luminosity_distance, iota, phase,
        longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
        frequency_minimum, frequency_maximum, reference_frequency,
        waveform_dictionary, approximant)

    h_plus = hplus.data.data
    h_cross = hcross.data.data

    return {'plus': h_plus, 'cross': h_cross}


#class BinaryNeutronStarMergerNumericalRelativity:
#    """Loads in NR simulations of BNS merger
#    takes parameters mean_mass, mass_ratio and equation_of_state, directory_path
#    returns time,hplus,hcross,freq,Hplus(freq),Hcross(freq)
#    """
#    def model(self, parameters):
#        mean_mass_string = '{:.0f}'.format(parameters['mean_mass'] * 1000)
#        eos_string = parameters['equation_of_state']
#        mass_ratio_string = '{:.0f}'.format(parameters['mass_ratio'] * 10)
#        directory_path = parameters['directory_path']
#        file_name = '{}-q{}-M{}.csv'.format(eos_string, mass_ratio_string, mean_mass_string)
#        full_filename = '{}/{}'.format(directory_path, file_name)
#        if not os.path.isfile(full_filename):
#            print('{} does not exist'.format(full_filename))  # add exception
#            return (-1)
#        else:  # ok file exists
#            strain_table = Table.read(full_filename)
#            Hplus, _ = utils.nfft(strain_table["hplus"], utils.get_sampling_frequency(strain_table['time']))
#            Hcross, frequency = utils.nfft(strain_table["hcross"], utils.get_sampling_frequency(strain_table['time']))
#            return (strain_table['time'], strain_table["hplus"], strain_table["hcross"], frequency, Hplus, Hcross)
