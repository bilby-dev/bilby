from __future__ import division, print_function

import numpy as np

from ..core import utils
from ..core.utils import logger
from .utils import (lalsim_SimInspiralTransformPrecessingNewInitialConditions,
                    lalsim_GetApproximantFromString,
                    lalsim_SimInspiralChooseFDWaveform,
                    lalsim_SimInspiralWaveformParamsInsertTidalLambda1,
                    lalsim_SimInspiralWaveformParamsInsertTidalLambda2,
                    lalsim_SimIMRPhenomPCalculateModelParametersFromSourceFrame,
                    lalsim_SimIMRPhenomPFrequencySequence)

try:
    import lal
    import lalsimulation as lalsim
except ImportError:
    logger.warning("You do not have lalsuite installed currently. You will"
                   " not be able to use some of the prebuilt functions.")


def lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
        iota, phase, **kwargs):
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
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = (
            lalsim_SimInspiralTransformPrecessingNewInitialConditions(
                iota, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1,
                mass_2, reference_frequency, phase))

    longitude_ascending_nodes = 0.0
    eccentricity = 0.0
    mean_per_ano = 0.0

    waveform_dictionary = None

    approximant = lalsim_GetApproximantFromString(waveform_approximant)

    maximum_frequency = frequency_array[-1]
    delta_frequency = frequency_array[1] - frequency_array[0]

    hplus, hcross = lalsim_SimInspiralChooseFDWaveform(
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
        frequency_array, mass_1, mass_2, eccentricity, luminosity_distance, iota, phase, **kwargs):
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

    approximant = lalsim_GetApproximantFromString(waveform_approximant)

    maximum_frequency = frequency_array[-1]
    delta_frequency = frequency_array[1] - frequency_array[0]

    hplus, hcross = lalsim_SimInspiralChooseFDWaveform(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, luminosity_distance, iota, phase,
        longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
        minimum_frequency, maximum_frequency, reference_frequency,
        waveform_dictionary, approximant)

    h_plus = hplus.data.data
    h_cross = hcross.data.data

    return {'plus': h_plus, 'cross': h_cross}


def sinegaussian(frequency_array, hrss, Q, frequency, **kwargs):
    tau = Q / (np.sqrt(2.0) * np.pi * frequency)
    temp = Q / (4.0 * np.sqrt(np.pi) * frequency)
    fm = frequency_array - frequency
    fp = frequency_array + frequency

    h_plus = ((hrss / np.sqrt(temp * (1 + np.exp(-Q**2)))) *
              ((np.sqrt(np.pi) * tau) / 2.0) *
              (np.exp(-fm**2 * np.pi**2 * tau**2) +
              np.exp(-fp**2 * np.pi**2 * tau**2)))

    h_cross = (-1j * (hrss / np.sqrt(temp * (1 - np.exp(-Q**2)))) *
               ((np.sqrt(np.pi) * tau) / 2.0) *
               (np.exp(-fm**2 * np.pi**2 * tau**2) -
               np.exp(-fp**2 * np.pi**2 * tau**2)))

    return{'plus': h_plus, 'cross': h_cross}


def supernova(
        frequency_array, realPCs, imagPCs, file_path, luminosity_distance, **kwargs):
    """ A supernova NR simulation for injections """

    realhplus, imaghplus, realhcross, imaghcross = np.loadtxt(
        file_path, usecols=(0, 1, 2, 3), unpack=True)

    # waveform in file at 10kpc
    scaling = 1e-3 * (10.0 / luminosity_distance)

    h_plus = scaling * (realhplus + 1.0j * imaghplus)
    h_cross = scaling * (realhcross + 1.0j * imaghcross)
    return {'plus': h_plus, 'cross': h_cross}


def supernova_pca_model(
        frequency_array, pc_coeff1, pc_coeff2, pc_coeff3, pc_coeff4, pc_coeff5,
        luminosity_distance, **kwargs):
    """ Supernova signal model """

    realPCs = kwargs['realPCs']
    imagPCs = kwargs['imagPCs']

    pc1 = realPCs[:, 0] + 1.0j * imagPCs[:, 0]
    pc2 = realPCs[:, 1] + 1.0j * imagPCs[:, 1]
    pc3 = realPCs[:, 2] + 1.0j * imagPCs[:, 2]
    pc4 = realPCs[:, 3] + 1.0j * imagPCs[:, 3]
    pc5 = realPCs[:, 4] + 1.0j * imagPCs[:, 5]

    # file at 10kpc
    scaling = 1e-23 * (10.0 / luminosity_distance)

    h_plus = scaling * (pc_coeff1 * pc1 + pc_coeff2 * pc2 + pc_coeff3 * pc3 +
                        pc_coeff4 * pc4 + pc_coeff5 * pc5)
    h_cross = scaling * (pc_coeff1 * pc1 + pc_coeff2 * pc2 + pc_coeff3 * pc3 +
                         pc_coeff4 * pc4 + pc_coeff5 * pc5)

    return {'plus': h_plus, 'cross': h_cross}


def lal_binary_neutron_star(
        frequency_array, mass_1, mass_2, luminosity_distance, chi_1, chi_2,
        iota, phase, lambda_1, lambda_2, **kwargs):
    """ A Binary Neutron Star waveform model using lalsimulation

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
    chi_1: float
        Dimensionless aligned spin
    chi_2: float
        Dimensionless aligned spin
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
    lambda_1: float
        Dimensionless tidal deformability of mass_1
    lambda_2: float
        Dimensionless tidal deformability of mass_2

    kwargs: dict
        Optional keyword arguments

    Returns
    -------
    dict: A dictionary with the plus and cross polarisation strain modes
    """

    waveform_kwargs = dict(waveform_approximant='TaylorF2', reference_frequency=50.0,
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

    spin_1x = 0
    spin_1y = 0
    spin_1z = chi_1
    spin_2x = 0
    spin_2y = 0
    spin_2z = chi_2

    longitude_ascending_nodes = 0.0
    eccentricity = 0.0
    mean_per_ano = 0.0

    waveform_dictionary = lal.CreateDict()
    lalsim_SimInspiralWaveformParamsInsertTidalLambda1(waveform_dictionary, lambda_1)
    lalsim_SimInspiralWaveformParamsInsertTidalLambda2(waveform_dictionary, lambda_2)

    approximant = lalsim_GetApproximantFromString(waveform_approximant)

    maximum_frequency = frequency_array[-1]
    delta_frequency = frequency_array[1] - frequency_array[0]

    hplus, hcross = lalsim_SimInspiralChooseFDWaveform(
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


def roq(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, iota, phase, **waveform_arguments):
    """
    See https://git.ligo.org/lscsoft/lalsuite/blob/master/lalsimulation/src/LALSimInspiral.c#L1460

    Parameters
    ----------
    frequency_array: np.array
        This input is ignored for the roq source model
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

    Waveform arguments
    ------------------
    Non-sampled extra data used in the source model calculation
    frequency_nodes_linear: np.array
    frequency_nodes_quadratic: np.array
    reference_frequency: float
    version: str

    Note: for the frequency_nodes_linear and frequency_nodes_quadratic arguments,
    if using data from https://git.ligo.org/lscsoft/ROQ_data, this should be
    loaded as `np.load(filename).T`.

    Returns
    -------
    waveform_polarizations: dict
        Dict containing plus and cross modes evaluated at the linear and
        quadratic frequency nodes.

    """
    if mass_2 > mass_1:
        return None

    frequency_nodes_linear = waveform_arguments['frequency_nodes_linear']
    frequency_nodes_quadratic = waveform_arguments['frequency_nodes_quadratic']
    reference_frequency = getattr(waveform_arguments,
                                  'reference_frequency', 20.0)
    versions = dict(IMRPhenomPv2=lalsim.IMRPhenomPv2_V)
    version = versions[getattr(waveform_arguments, 'version', 'IMRPhenomPv2')]

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
            lalsim_SimInspiralTransformPrecessingNewInitialConditions(
                iota, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
                reference_frequency, phase)

    chi_1_l, chi_2_l, chi_p, theta_jn, alpha, phase_aligned, zeta =\
        lalsim_SimIMRPhenomPCalculateModelParametersFromSourceFrame(
            mass_1, mass_2, reference_frequency, phase, iota, spin_1x,
            spin_1y, spin_1z, spin_2x, spin_2y, spin_2z, version)

    waveform_polarizations = dict()

    h_linear_plus, h_linear_cross = lalsim_SimIMRPhenomPFrequencySequence(
        frequency_nodes_linear, chi_1_l, chi_2_l, chi_p, theta_jn,
        mass_1, mass_2, luminosity_distance,
        alpha, phase_aligned, reference_frequency, version)
    h_quadratic_plus, h_quadratic_cross = lalsim_SimIMRPhenomPFrequencySequence(
        frequency_nodes_quadratic, chi_1_l, chi_2_l, chi_p, theta_jn,
        mass_1, mass_2, luminosity_distance,
        alpha, phase_aligned, reference_frequency, version)

    waveform_polarizations['linear'] = dict(
        plus=(np.cos(2 * zeta) * h_linear_plus.data.data +
              np.sin(2 * zeta) * h_linear_cross.data.data),
        cross=(np.cos(2 * zeta) * h_linear_cross.data.data -
               np.sin(2 * zeta) * h_linear_plus.data.data))

    waveform_polarizations['quadratic'] = dict(
        plus=(np.cos(2 * zeta) * h_quadratic_plus.data.data +
              np.sin(2 * zeta) * h_quadratic_cross.data.data),
        cross=(np.cos(2 * zeta) * h_quadratic_cross.data.data -
               np.sin(2 * zeta) * h_quadratic_plus.data.data))

    return waveform_polarizations
