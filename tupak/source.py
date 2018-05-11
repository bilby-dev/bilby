from __future__ import division, print_function

import numpy as np

try:
    import lalsimulation as lalsim
except ImportError:
    raise ImportWarning("You do not have lalsuite installed currently. You will not be able to use some of the "
                        "prebuilt functions.")

from . import utils


def convert_from_lal_binary_black_hole_parameters(parameters, search_keys):
    """
    Convert parameters we have into parameters we need.

    This is defined by the parameters of tupak.source.lal_binary_black_hole()


    Mass: mass_1, mass_2
    Spin: a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl
    Extrinsic: luminosity_distance, theta_jn, phase, ra, dec, geocent_time, psi

    This involves popping a lot of things from parameters.
    The keys in ignored_keys should be popped after evaluating the waveform.

    Parameters
    ----------
    parameters: dict
        dictionary of parameter values to convert into the required parameters
    search_keys: list
        parameters which are needed for the waveform generation
    """

    if 'mass_1' in parameters.keys() and 'mass_2' in parameters.keys():
        if 'chirp_mass' in search_keys:
            parameters['chirp_mass'] = (parameters['mass_1'] * parameters['mass_2'])**0.6\
                                       / (parameters['mass_1'] + parameters['mass_2'])**0.2
        if 'total_mass' in search_keys:
            parameters['total_mass'] = parameters['mass_1'] + parameters['mass_2']
        if 'symmetric_mass_ratio' in search_keys:
            parameters['symmetric_mass_ratio'] = (parameters['mass_1'] * parameters['mass_2'])\
                                                 / (parameters['mass_1'] + parameters['mass_2'])**2
        if 'mass_ratio' in search_keys:
            parameters['mass_ratio'] = parameters['mass_2'] / parameters['mass_1']

    if 'tilt_1' in parameters.keys():
        if 'cos_tilt_1' in search_keys:
            parameters['cos_tilt_1'] = np.cos(parameters['tilt_1'])
    if 'tilt_2' in parameters.keys():
        if 'cos_tilt_2' in search_keys:
            parameters['cos_tilt_2'] = np.cos(parameters['tilt_2'])

    if 'iota' in parameters.keys():
        parameters['cos_iota'] = np.arccos(parameters['iota'])


def convert_to_lal_binary_black_hole_parameters(parameters, search_keys):
    """
    Convert parameters we have into parameters we need.

    This is defined by the parameters of tupak.source.lal_binary_black_hole()


    Mass: mass_1, mass_2
    Spin: a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl
    Extrinsic: luminosity_distance, theta_jn, phase, ra, dec, geocent_time, psi

    This involves popping a lot of things from parameters.
    The keys in ignored_keys should be popped after evaluating the waveform.

    Parameters
    ----------
    parameters: dict
        dictionary of parameter values to convert into the required parameters
    search_keys: list
        parameters which are needed for the waveform generation

    Return
    ------
    ignored_keys: list
        keys which are added to parameters during function call
    """

    ignored_keys = []

    if 'mass_1' not in search_keys and 'mass_2' not in search_keys:
        if 'chirp_mass' in parameters.keys():
            if 'total_mass' in parameters.keys():
                # chirp_mass, total_mass to total_mass, symmetric_mass_ratio
                parameters['symmetric_mass_ratio'] = (parameters['chirp_mass'] / parameters['total_mass'])**(5 / 3)
                parameters.pop('chirp_mass')
            if 'symmetric_mass_ratio' in parameters.keys():
                # symmetric_mass_ratio to mass_ratio
                temp = (1 / parameters['symmetric_mass_ratio'] / 2 - 1)
                parameters['mass_ratio'] = temp - (temp**2 - 1)**0.5
                parameters.pop('symmetric_mass_ratio')
            if 'mass_ratio' in parameters.keys():
                if 'total_mass' not in parameters.keys():
                    parameters['total_mass'] = parameters['chirp_mass'] * (1 + parameters['mass_ratio'])**1.2\
                                               / parameters['mass_ratio']**0.6
                    parameters.pop('chirp_mass')
                # total_mass, mass_ratio to component masses
                parameters['mass_1'] = parameters['total_mass'] / (1 + parameters['mass_ratio'])
                parameters['mass_2'] = parameters['mass_1'] * parameters['mass_ratio']
                parameters.pop('total_mass')
                parameters.pop('mass_ratio')
            ignored_keys.append('mass_1')
            ignored_keys.append('mass_2')
        elif 'total_mass' in parameters.keys():
            if 'symmetric_mass_ratio' in parameters.keys():
                # symmetric_mass_ratio to mass_ratio
                temp = (1 / parameters['symmetric_mass_ratio'] / 2 - 1)
                parameters['mass_ratio'] = temp - (temp**2 - 1)**0.5
                parameters.pop('symmetric_mass_ratio')
            if 'mass_ratio' in parameters.keys():
                # total_mass, mass_ratio to component masses
                parameters['mass_1'] = parameters['total_mass'] / (1 + parameters['mass_ratio'])
                parameters['mass_2'] = parameters['mass_1'] * parameters['mass_ratio']
                parameters.pop('total_mass')
                parameters.pop('mass_ratio')
            ignored_keys.append('mass_1')
            ignored_keys.append('mass_2')

    if 'cos_tilt_1' in parameters.keys():
        ignored_keys.append('tilt_1')
        parameters['tilt_1'] = np.arccos(parameters['cos_tilt_1'])
        parameters.pop('cos_tilt_1')
    if 'cos_tilt_2' in parameters.keys():
        ignored_keys.append('tilt_2')
        parameters['tilt_2'] = np.arccos(parameters['cos_tilt_2'])
        parameters.pop('cos_tilt_2')

    if 'cos_iota' in parameters.keys():
        parameters['iota'] = np.arccos(parameters['cos_iota'])
        parameters.pop('cos_iota')

    return ignored_keys


def lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
        iota, phase, waveform_approximant, reference_frequency, ra, dec, geocent_time, psi):
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
