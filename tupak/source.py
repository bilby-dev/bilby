from __future__ import division, print_function

import logging
import numpy as np

try:
    import lalsimulation as lalsim
except ImportError:
    logging.warning("You do not have lalsuite installed currently. You will "
                    " not be able to use some of the prebuilt functions.")

from . import utils


def lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
        iota, phase, waveform_approximant, reference_frequency, ra, dec, geocent_time, psi):
    """ A Binary Black Hole waveform model using lalsimulation """
    if mass_2 > mass_1:
        return None

    luminosity_distance = luminosity_distance * 1e6 * utils.parsec
    mass_1 = mass_1 * utils.solar_mass
    mass_2 = mass_2 * utils.solar_mass

    if tilt_1 == 0 and tilt_2 == 0:
        spin_1x = 0
        spin_1y = 0
        spin_1z = a_1
        spin_2x = 0
        spin_2y = 0
        spin_2z = a_2
    else:
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


def sinegaussian(
        frequency_array, hrss, Q, frequency, ra, dec, geocent_time, psi):

    tau  = Q / (np.sqrt(2.0)*np.pi*frequency)
    temp = Q / (4.0*np.sqrt(np.pi)*frequency)
    t = geocent_time
    fm = frequency_array - frequency
    fp = frequency_array + frequency

    h_plus = ((hrss / np.sqrt(temp * (1+np.exp(-Q**2))))
              * ((np.sqrt(np.pi)*tau)/2.0)
              * (np.exp(-fm**2 * np.pi**2 * tau**2)
                  + np.exp(-fp**2 * np.pi**2 * tau**2)))

    h_cross = (-1j*(hrss / np.sqrt(temp * (1-np.exp(-Q**2))))
               * ((np.sqrt(np.pi)*tau)/2.0)
               * (np.exp(-fm**2 * np.pi**2 * tau**2)
                  - np.exp(-fp**2 * np.pi**2 * tau**2)))

    return{'plus': h_plus, 'cross': h_cross}


def supernova(
        frequency_array, realPCs, imagPCs, file_path, luminosity_distance, ra,
        dec, geocent_time, psi):
    """ A supernova NR simulation for injections """

    realhplus, imaghplus, realhcross, imaghcross = np.loadtxt(
        file_path, usecols=(0, 1, 2, 3), unpack=True)

    # waveform in file at 10kpc
    scaling = 1e-3 * (10.0 / luminosity_distance)

    h_plus = scaling * (realhplus + 1.0j*imaghplus)
    h_cross = scaling * (realhcross + 1.0j*imaghcross)
    return {'plus': h_plus, 'cross': h_cross}


def supernova_pca_model(
        frequency_array, realPCs, imagPCs, pc_coeff1, pc_coeff2, pc_coeff3,
        pc_coeff4, pc_coeff5, luminosity_distance, ra, dec, geocent_time, psi):
    """ Supernova signal model """

    pc1 = realPCs[:, 0] + 1.0j*imagPCs[:, 0]
    pc2 = realPCs[:, 1] + 1.0j*imagPCs[:, 1]
    pc3 = realPCs[:, 2] + 1.0j*imagPCs[:, 2]
    pc4 = realPCs[:, 3] + 1.0j*imagPCs[:, 3]
    pc5 = realPCs[:, 4] + 1.0j*imagPCs[:, 5]

    # file at 10kpc
    scaling = 1e-23 * (10.0 / luminosity_distance)

    h_plus = scaling * (pc_coeff1*pc1 + pc_coeff2*pc2 + pc_coeff3*pc3
                        + pc_coeff4*pc4 + pc_coeff5*pc5)
    h_cross = scaling * (pc_coeff1*pc1 + pc_coeff2*pc2 + pc_coeff3*pc3
                         + pc_coeff4*pc4 + pc_coeff5*pc5)

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
