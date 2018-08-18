from __future__ import division, print_function

import numpy as np

from tupak.core.utils import logger
from tupak.core import utils

try:
    import lalsimulation as lalsim
except ImportError:
    logger.warning("You do not have lalsuite installed currently. You will "
                   " not be able to use some of the prebuilt functions.")


def lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
        iota, phase, ra, dec, geocent_time, psi, **kwargs):
    """ A Binary Black Hole waveform model using lalsimulation

    Parameters
    ----------
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float

    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float

    iota: float
        Orbital inclination
    phase: float
        The phase at coalescence
    ra: float
        The right ascension of the binary
    dec: float
        The declination of the object
    geocent_time: float
        The time at coalescence
    psi: float
        Orbital polarisation
    kwargs: dict
        Optional keyword arguments

    Returns
    -------
    dict: A dictionary with the plus and cross polarisation strain modes
    """

    waveform_kwargs = dict(waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
                           minimum_frequency=20.0)
    waveform_kwargs.update(kwargs)
    waveform_approximant = waveform_kwargs['waveform_approximant']
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']

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
            lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                iota, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2, reference_frequency, phase)

    longitude_ascending_nodes = 0.0
    eccentricity = 0.0
    mean_per_ano = 0.0

    waveform_dictionary = None

    approximant = lalsim.GetApproximantFromString(waveform_approximant)

    maximum_frequency = frequency_array[-1]
    delta_frequency = frequency_array[1] - frequency_array[0]

    hplus, hcross = lalsim.SimInspiralChooseFDWaveform(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, luminosity_distance, iota, phase,
        longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
        minimum_frequency, maximum_frequency, reference_frequency,
        waveform_dictionary, approximant)

    h_plus = hplus.data.data
    h_cross = hcross.data.data

    h_plus = h_plus[:len(frequency_array)]
    h_cross = h_cross[:len(frequency_array)]

    return {'plus': h_plus, 'cross': h_cross}

def lal_eccentric_binary_black_hole_no_spins(
        frequency_array, mass_1, mass_2, eccentricity, luminosity_distance, iota, phase, ra, dec, 
        geocent_time, psi, **kwargs):        
    """ Eccentric binary black hole waveform model using lalsimulation (EccentricFD)

    Parameters
    ----------
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    eccentricity: float
        The orbital eccentricity of the system
    luminosity_distance: float
        The luminosity distance in megaparsec
    iota: float
        Orbital inclination
    phase: float
        The phase at coalescence
    ra: float
        The right ascension of the binary
    dec: float
        The declination of the object
    geocent_time: float
        The time at coalescence
    psi: float
        Orbital polarisation
    kwargs: dict
        Optional keyword arguments

    Returns
    -------
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    
    waveform_kwargs = dict(waveform_approximant='EccentricFD', reference_frequency=10.0,
                           minimum_frequency=10.0)
    waveform_kwargs.update(kwargs)
    waveform_approximant = waveform_kwargs['waveform_approximant']
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']

    if mass_2 > mass_1:
        return None

    luminosity_distance = luminosity_distance * 1e6 * utils.parsec
    mass_1 = mass_1 * utils.solar_mass
    mass_2 = mass_2 * utils.solar_mass
    
    spin_1x = 0.0
    spin_1y = 0.0
    spin_1z = 0.0 
    spin_2x = 0.0
    spin_2y = 0.0
    spin_2z = 0.0
    
    longitude_ascending_nodes = 0.0
    mean_per_ano = 0.0

    waveform_dictionary = None

    approximant = lalsim.GetApproximantFromString(waveform_approximant)

    maximum_frequency = frequency_array[-1]
    delta_frequency = frequency_array[1] - frequency_array[0]

    hplus, hcross = lalsim.SimInspiralChooseFDWaveform(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, luminosity_distance, iota, phase,
        longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
        minimum_frequency, maximum_frequency, reference_frequency,
        waveform_dictionary, approximant)

    h_plus = hplus.data.data
    h_cross = hcross.data.data

    return {'plus': h_plus, 'cross': h_cross}


def sinegaussian(frequency_array, hrss, Q, frequency, ra, dec, geocent_time, psi):
    tau = Q / (np.sqrt(2.0)*np.pi*frequency)
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
        frequency_array, pc_coeff1, pc_coeff2, pc_coeff3, pc_coeff4, pc_coeff5,
        luminosity_distance, ra, dec, geocent_time, psi, **kwargs):
    """ Supernova signal model """

    realPCs = kwargs['realPCs']
    imagPCs = kwargs['imagPCs']

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
