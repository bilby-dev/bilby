from __future__ import division
import tupak
import numpy as np
import pandas as pd

from tupak.core.utils import logger

try:
    from astropy.cosmology import z_at_value, Planck15
    import astropy.units as u
except ImportError:
    logger.warning("You do not have astropy installed currently. You will"
                   " not be able to use some of the prebuilt functions.")

try:
    import lalsimulation as lalsim
except ImportError:
    logger.warning("You do not have lalsuite installed currently. You will"
                   " not be able to use some of the prebuilt functions.")


def redshift_to_luminosity_distance(redshift):
    return Planck15.luminosity_distance(redshift).value


def redshift_to_comoving_distance(redshift):
    return Planck15.comoving_distance(redshift).value


@np.vectorize
def luminosity_distance_to_redshift(distance):
    return z_at_value(Planck15.luminosity_distance, distance * u.Mpc)


@np.vectorize
def comoving_distance_to_redshift(distance):
    return z_at_value(Planck15.comoving_distance, distance * u.Mpc)


def comoving_distance_to_luminosity_distance(distance):
    redshift = comoving_distance_to_redshift(distance)
    return redshift_to_luminosity_distance(redshift)


def luminosity_distance_to_comoving_distance(distance):
    redshift = luminosity_distance_to_redshift(distance)
    return redshift_to_comoving_distance(redshift)


def convert_to_lal_binary_black_hole_parameters(parameters):
    """
    Convert parameters we have into parameters we need.

    This is defined by the parameters of tupak.source.lal_binary_black_hole()


    Mass: mass_1, mass_2
    Spin: a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl
    Extrinsic: luminosity_distance, theta_jn, phase, ra, dec, geocent_time, psi

    This involves popping a lot of things from parameters.
    The keys in added_keys should be popped after evaluating the waveform.

    Parameters
    ----------
    parameters: dict
        dictionary of parameter values to convert into the required parameters

    Return
    ------
    converted_parameters: dict
        dict of the required parameters
    added_keys: list
        keys which are added to parameters during function call
    """

    converted_parameters = parameters.copy()
    original_keys = list(converted_parameters.keys())

    if 'chirp_mass' in converted_parameters.keys():
        if 'total_mass' in converted_parameters.keys():
            converted_parameters['symmetric_mass_ratio'] =\
                chirp_mass_and_total_mass_to_symmetric_mass_ratio(
                    converted_parameters['chirp_mass'],
                    converted_parameters['total_mass'])
        if 'symmetric_mass_ratio' in converted_parameters.keys():
            converted_parameters['mass_ratio'] =\
                symmetric_mass_ratio_to_mass_ratio(
                    converted_parameters['symmetric_mass_ratio'])
        if 'total_mass' not in converted_parameters.keys():
            converted_parameters['total_mass'] =\
                chirp_mass_and_mass_ratio_to_total_mass(
                    converted_parameters['chirp_mass'],
                    converted_parameters['mass_ratio'])
        converted_parameters['mass_1'], converted_parameters['mass_2'] = \
            total_mass_and_mass_ratio_to_component_masses(
                converted_parameters['mass_ratio'],
                converted_parameters['total_mass'])
    elif 'total_mass' in converted_parameters.keys():
        if 'symmetric_mass_ratio' in converted_parameters.keys():
            converted_parameters['mass_ratio'] = \
                symmetric_mass_ratio_to_mass_ratio(
                    converted_parameters['symmetric_mass_ratio'])
        if 'mass_ratio' in converted_parameters.keys():
            converted_parameters['mass_1'], converted_parameters['mass_2'] =\
                total_mass_and_mass_ratio_to_component_masses(
                    converted_parameters['mass_ratio'],
                    converted_parameters['total_mass'])
        elif 'mass_1' in converted_parameters.keys():
            converted_parameters['mass_2'] =\
                converted_parameters['total_mass'] -\
                converted_parameters['mass_1']
        elif 'mass_2' in converted_parameters.keys():
            converted_parameters['mass_1'] = \
                converted_parameters['total_mass'] - \
                converted_parameters['mass_2']
    elif 'symmetric_mass_ratio' in converted_parameters.keys():
        converted_parameters['mass_ratio'] =\
            symmetric_mass_ratio_to_mass_ratio(
                converted_parameters['symmetric_mass_ratio'])
        if 'mass_1' in converted_parameters.keys():
            converted_parameters['mass_2'] =\
                converted_parameters['mass_1'] *\
                converted_parameters['mass_ratio']
        elif 'mass_2' in converted_parameters.keys():
            converted_parameters['mass_1'] =\
                converted_parameters['mass_2'] /\
                converted_parameters['mass_ratio']
    elif 'mass_ratio' in converted_parameters.keys():
        if 'mass_1' in converted_parameters.keys():
            converted_parameters['mass_2'] =\
                converted_parameters['mass_1'] *\
                converted_parameters['mass_ratio']
        if 'mass_2' in converted_parameters.keys():
            converted_parameters['mass_1'] = \
                converted_parameters['mass_2'] /\
                converted_parameters['mass_ratio']

    for angle in ['tilt_1', 'tilt_2', 'iota']:
        cos_angle = str('cos_' + angle)
        if cos_angle in converted_parameters.keys():
            converted_parameters[angle] =\
                np.arccos(converted_parameters[cos_angle])

    if 'redshift' in converted_parameters.keys():
        converted_parameters['luminosity_distance'] =\
            redshift_to_luminosity_distance(parameters['redshift'])
    elif 'comoving_distance' in converted_parameters.keys():
        converted_parameters['luminosity_distance'] = \
            comoving_distance_to_luminosity_distance(
                parameters['comoving_distance'])

    added_keys = [key for key in converted_parameters.keys()
                  if key not in original_keys]

    return converted_parameters, added_keys


def convert_to_lal_binary_neutron_star_parameters(parameters):
    """
    Convert parameters we have into parameters we need.

    This is defined by the parameters of tupak.source.lal_binary_black_hole()


    Mass: mass_1, mass_2
    Spin: a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl
    Extrinsic: luminosity_distance, theta_jn, phase, ra, dec, geocent_time, psi

    This involves popping a lot of things from parameters.
    The keys in added_keys should be popped after evaluating the waveform.

    Parameters
    ----------
    parameters: dict
        dictionary of parameter values to convert into the required parameters

    Return
    ------
    converted_parameters: dict
        dict of the required parameters
    added_keys: list
        keys which are added to parameters during function call
    """
    converted_parameters = parameters.copy()
    original_keys = list(converted_parameters.keys())
    converted_parameters, added_keys =\
        convert_to_lal_binary_black_hole_parameters(converted_parameters)

    if 'delta_lambda' in converted_parameters.keys():
        converted_parameters['lambda_1'], converted_parameters['lambda_2'] =\
            lambda_tilde_delta_lambda_to_lambda_1_lambda_2(
                converted_parameters['lambda_tilde'],
                parameters['delta_lambda'], converted_parameters['mass_1'],
                converted_parameters['mass_2'])
    elif 'lambda_tilde' in converted_parameters.keys():
        converted_parameters['lambda_1'], converted_parameters['lambda_2'] =\
            lambda_tilde_to_lambda_1_lambda_2(
                converted_parameters['lambda_tilde'],
                converted_parameters['mass_1'], converted_parameters['mass_2'])
    if 'lambda_2' not in converted_parameters.keys():
        converted_parameters['lambda_2'] =\
            converted_parameters['lambda_1']\
            * converted_parameters['mass_1']**5\
            / converted_parameters['mass_2']**5
    elif converted_parameters['lambda_2'] is None:
        converted_parameters['lambda_2'] =\
            converted_parameters['lambda_1']\
            * converted_parameters['mass_1']**5\
            / converted_parameters['mass_2']**5

    added_keys = [key for key in converted_parameters.keys()
                  if key not in original_keys]

    return converted_parameters, added_keys


def total_mass_and_mass_ratio_to_component_masses(mass_ratio, total_mass):
    """
    Convert total mass and mass ratio of a binary to its component masses.

    Parameters
    ----------
    mass_ratio: float
        Mass ratio (mass_2/mass_1) of the binary
    total_mass: float
        Total mass of the binary

    Return
    ------
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object
    """

    mass_1 = total_mass / (1 + mass_ratio)
    mass_2 = mass_1 * mass_ratio
    return mass_1, mass_2


def symmetric_mass_ratio_to_mass_ratio(symmetric_mass_ratio):
    """
    Convert the symmetric mass ratio to the normal mass ratio.

    Parameters
    ----------
    symmetric_mass_ratio: float
        Symmetric mass ratio of the binary

    Return
    ------
    mass_ratio: float
        Mass ratio of the binary
    """

    temp = (1 / symmetric_mass_ratio / 2 - 1)
    return temp - (temp ** 2 - 1) ** 0.5


def chirp_mass_and_total_mass_to_symmetric_mass_ratio(chirp_mass, total_mass):
    """
    Convert chirp mass and total mass of a binary to its symmetric mass ratio.

    Parameters
    ----------
    chirp_mass: float
        Chirp mass of the binary
    total_mass: float
        Total mass of the binary

    Return
    ------
    symmetric_mass_ratio: float
        Symmetric mass ratio of the binary
    """

    return (chirp_mass / total_mass) ** (5 / 3)


def chirp_mass_and_mass_ratio_to_total_mass(chirp_mass, mass_ratio):
    """
    Convert chirp mass and mass ratio of a binary to its total mass.

    Parameters
    ----------
    chirp_mass: float
        Chirp mass of the binary
    mass_ratio: float
        Mass ratio (mass_2/mass_1) of the binary

    Return
    ------
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object
    """

    return chirp_mass * (1 + mass_ratio) ** 1.2 / mass_ratio ** 0.6


def component_masses_to_chirp_mass(mass_1, mass_2):
    """
    Convert the component masses of a binary to its chirp mass.

    Parameters
    ----------
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object

    Return
    ------
    chirp_mass: float
        Chirp mass of the binary
    """

    return (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2


def component_masses_to_total_mass(mass_1, mass_2):
    """
    Convert the component masses of a binary to its total mass.

    Parameters
    ----------
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object

    Return
    ------
    total_mass: float
        Total mass of the binary
    """

    return mass_1 + mass_2


def component_masses_to_symmetric_mass_ratio(mass_1, mass_2):
    """
    Convert the component masses of a binary to its symmetric mass ratio.

    Parameters
    ----------
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object

    Return
    ------
    symmetric_mass_ratio: float
        Symmetric mass ratio of the binary
    """

    return (mass_1 * mass_2) / (mass_1 + mass_2) ** 2


def component_masses_to_mass_ratio(mass_1, mass_2):
    """
    Convert the component masses of a binary to its chirp mass.

    Parameters
    ----------
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object

    Return
    ------
    mass_ratio: float
        Mass ratio of the binary
    """

    return mass_2 / mass_1


def mass_1_and_chirp_mass_to_mass_ratio(mass_1, chirp_mass):
    """
    Calculate mass ratio from mass_1 and chirp_mass.

    This involves solving mc = m1 * q**(3/5) / (1 + q)**(1/5).

    Parameters
    ----------
    mass_1: float
        Mass of the heavier object
    chirp_mass: float
        Mass of the lighter object

    Return
    ------
    mass_ratio: float
        Mass ratio of the binary
    """
    temp = (chirp_mass / mass_1) ** 5
    mass_ratio = (2 / 3 / (3 ** 0.5 * (27 * temp ** 2 - 4 * temp ** 3) ** 0.5 +
                           9 * temp)) ** (1 / 3) * temp + \
                 ((3 ** 0.5 * (27 * temp ** 2 - 4 * temp ** 3) ** 0.5 +
                   9 * temp) / (2 * 3 ** 2)) ** (1 / 3)
    return mass_ratio


def lambda_tilde_delta_lambda_to_lambda_1_lambda_2(
        lambda_tilde, delta_lambda, mass_1, mass_2):
    """
    Convert from dominant tidal terms to individual tidal parameters.

    See, e.g., Wade et al., https://arxiv.org/pdf/1402.5156.pdf.

    Parameters
    ----------
    lambda_tilde: float
        Dominant tidal term.
    delta_lambda: float
        Secondary tidal term.
    mass_1: float
        Mass of more massive neutron star.
    mass_2: float
        Mass of less massive neutron star.

    Return
    ------
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.
    """
    eta = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)
    coefficient_1 = (1 + 7 * eta - 31 * eta**2)
    coefficient_2 = (1 - 4 * eta)**0.5 * (1 + 9 * eta - 11 * eta**2)
    coefficient_3 = (1 - 4 * eta)**0.5 *\
                    (1 - 13272 / 1319 * eta + 8944 / 1319 * eta**2)
    coefficient_4 = (1 - 15910 / 1319 * eta + 32850 / 1319 * eta**2 +
                     3380 / 1319 * eta**3)
    lambda_1 =\
        (13 * lambda_tilde / 8 * (coefficient_3 - coefficient_4) -
         2 * delta_lambda * (coefficient_1 - coefficient_2))\
        / ((coefficient_1 + coefficient_2) * (coefficient_3 - coefficient_4) -
           (coefficient_1 - coefficient_2) * (coefficient_3 + coefficient_4))
    lambda_2 =\
        (13 * lambda_tilde / 8 * (coefficient_3 + coefficient_4) -
         2 * delta_lambda * (coefficient_1 + coefficient_2)) \
        / ((coefficient_1 - coefficient_2) * (coefficient_3 + coefficient_4) -
           (coefficient_1 + coefficient_2) * (coefficient_3 - coefficient_4))
    return lambda_1, lambda_2


def lambda_tilde_to_lambda_1_lambda_2(
        lambda_tilde, mass_1, mass_2):
    """
    Convert from dominant tidal term to individual tidal parameters
    assuming lambda_1 * mass_1**5 = lambda_2 * mass_2**5.

    See, e.g., Wade et al., https://arxiv.org/pdf/1402.5156.pdf.

    Parameters
    ----------
    lambda_tilde: float
        Dominant tidal term.
    mass_1: float
        Mass of more massive neutron star.
    mass_2: float
        Mass of less massive neutron star.

    Return
    ------
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.
    """
    eta = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)
    q = mass_2 / mass_1
    lambda_1 = 13 / 8 * lambda_tilde / (
        (1 + 7 * eta - 31 * eta**2) * (1 + q**-5) +
        (1 - 4 * eta)**0.5 * (1 + 9 * eta - 11 * eta**2) * (1 - q**-5))
    lambda_2 = lambda_1 / q**5
    return lambda_1, lambda_2


def generate_all_bbh_parameters(sample, likelihood=None, priors=None):
    """
    From either a single sample or a set of samples fill in all missing
    BBH parameters, in place.

    Parameters
    ----------
    sample: dict or pandas.DataFrame
        Samples to fill in with extra parameters, this may be either an
        injection or posterior samples.
    likelihood: tupak.gw.likelihood.GravitationalWaveTransient, optional
        GravitationalWaveTransient used for sampling, used for waveform and
        likelihood.interferometers.
    priors: dict, optional
        Dictionary of prior objects, used to fill in non-sampled parameters.
    """
    output_sample = sample.copy()
    if likelihood is not None:
        output_sample['reference_frequency'] =\
            likelihood.waveform_generator.waveform_arguments[
                'reference_frequency']
        output_sample['waveform_approximant'] =\
            likelihood.waveform_generator.waveform_arguments[
                'waveform_approximant']

    output_sample = fill_from_fixed_priors(output_sample, priors)
    output_sample, _ =\
        convert_to_lal_binary_black_hole_parameters(output_sample)
    output_sample = generate_non_standard_parameters(output_sample)
    output_sample = generate_component_spins(output_sample)
    compute_snrs(output_sample, likelihood)
    return output_sample


def fill_from_fixed_priors(sample, priors):
    """Add parameters with delta function prior to the data frame/dictionary.

    Parameters
    ----------
    sample: dict
        A dictionary or data frame
    priors: dict
        A dictionary of priors

    Returns
    -------
    dict:
    """
    output_sample = sample.copy()
    if priors is not None:
        for name in priors:
            if isinstance(priors[name], tupak.core.prior.DeltaFunction):
                output_sample[name] = priors[name].peak
    return output_sample


def generate_non_standard_parameters(sample):
    """
    Add the known non-standard parameters to the data frame/dictionary.

    We add:
        chirp mass, total mass, symmetric mass ratio, mass ratio, cos tilt 1,
        cos tilt 2, cos iota

    Parameters
    ----------
    sample : dict
        The input dictionary with component masses 'mass_1' and 'mass_2'

    Returns
    -------
    dict: The updated dictionary

    """
    output_sample = sample.copy()
    output_sample['chirp_mass'] =\
        component_masses_to_chirp_mass(sample['mass_1'], sample['mass_2'])
    output_sample['total_mass'] =\
        component_masses_to_total_mass(sample['mass_1'], sample['mass_2'])
    output_sample['symmetric_mass_ratio'] =\
        component_masses_to_symmetric_mass_ratio(sample['mass_1'],
                                                 sample['mass_2'])
    output_sample['mass_ratio'] =\
        component_masses_to_mass_ratio(sample['mass_1'], sample['mass_2'])

    output_sample['cos_tilt_1'] = np.cos(output_sample['tilt_1'])
    output_sample['cos_tilt_2'] = np.cos(output_sample['tilt_2'])
    output_sample['cos_iota'] = np.cos(output_sample['iota'])

    output_sample['redshift'] =\
        luminosity_distance_to_redshift(sample['luminosity_distance'])
    return output_sample


def generate_component_spins(sample):
    """
    Add the component spins to the data frame/dictionary.

    This function uses a lalsimulation function to transform the spins.

    Parameters
    ----------
    sample: A dictionary with the necessary spin conversion parameters:
    'iota', 'phi_jl', 'tilt_1', 'tilt_2', 'phi_12', 'a_1', 'a_2', 'mass_1',
    'mass_2', 'reference_frequency', 'phase'

    Returns
    -------
    dict: The updated dictionary

    """
    output_sample = sample.copy()
    spin_conversion_parameters =\
        ['iota', 'phi_jl', 'tilt_1', 'tilt_2', 'phi_12', 'a_1', 'a_2', 'mass_1',
         'mass_2', 'reference_frequency', 'phase']
    if all(key in output_sample.keys() for key in spin_conversion_parameters)\
            and isinstance(output_sample, dict):
        output_sample['iota'], output_sample['spin_1x'],\
            output_sample['spin_1y'], output_sample['spin_1z'], \
            output_sample['spin_2x'], output_sample['spin_2y'],\
            output_sample['spin_2z'] =\
            lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                output_sample['iota'], output_sample['phi_jl'],
                output_sample['tilt_1'], output_sample['tilt_2'],
                output_sample['phi_12'], output_sample['a_1'],
                output_sample['a_2'],
                output_sample['mass_1'] * tupak.core.utils.solar_mass,
                output_sample['mass_2'] * tupak.core.utils.solar_mass,
                output_sample['reference_frequency'], output_sample['phase'])

        output_sample['phi_1'] =\
            np.arctan(output_sample['spin_1y'] / output_sample['spin_1x'])
        output_sample['phi_2'] =\
            np.arctan(output_sample['spin_2y'] / output_sample['spin_2x'])

    elif all(key in output_sample.keys() for key in spin_conversion_parameters)\
            and isinstance(output_sample, pd.DataFrame):
        logger.debug('Extracting component spins.')
        new_spin_parameters =\
            ['spin_1x', 'spin_1y', 'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z']
        new_spins =\
            {name: np.zeros(len(output_sample)) for name in new_spin_parameters}

        for ii in range(len(output_sample)):
            new_spins['iota'], new_spins['spin_1x'][ii],\
                new_spins['spin_1y'][ii], new_spins['spin_1z'][ii], \
                new_spins['spin_2x'][ii], new_spins['spin_2y'][ii],\
                new_spins['spin_2z'][ii] =\
                lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                    output_sample['iota'][ii], output_sample['phi_jl'][ii],
                    output_sample['tilt_1'][ii], output_sample['tilt_2'][ii],
                    output_sample['phi_12'][ii], output_sample['a_1'][ii],
                    output_sample['a_2'][ii],
                    output_sample['mass_1'][ii] * tupak.core.utils.solar_mass,
                    output_sample['mass_2'][ii] * tupak.core.utils.solar_mass,
                    output_sample['reference_frequency'][ii],
                    output_sample['phase'][ii])

        for name in new_spin_parameters:
            output_sample[name] = new_spins[name]

        output_sample['phi_1'] =\
            np.arctan(output_sample['spin_1y'] / output_sample['spin_1x'])
        output_sample['phi_2'] =\
            np.arctan(output_sample['spin_2y'] / output_sample['spin_2x'])

    else:
        logger.warning("Component spin extraction failed.")

    return output_sample


def compute_snrs(sample, likelihood):
    """
    Compute the optimal and matched filter snrs of all posterior samples
    and print it out.

    Parameters
    ----------
    sample: dict or array_like

    likelihood: tupak.gw.likelihood.GravitationalWaveTransient
        Likelihood function to be applied on the posterior

    """
    temp_sample = sample
    if likelihood is not None:
        if isinstance(temp_sample, dict):
            for key in likelihood.waveform_generator.parameters.keys():
                likelihood.waveform_generator.parameters[key] = temp_sample[key]
            signal_polarizations =\
                likelihood.waveform_generator.frequency_domain_strain()
            for ifo in likelihood.interferometers:
                signal = ifo.get_detector_response(
                    signal_polarizations,
                    likelihood.waveform_generator.parameters)
                sample['{}_matched_filter_snr'.format(ifo.name)] =\
                    ifo.matched_filter_snr_squared(signal=signal) ** 0.5
                sample['{}_optimal_snr'.format(ifo.name)] = \
                    ifo.optimal_snr_squared(signal=signal) ** 0.5
        else:
            logger.info(
                'Computing SNRs for every sample, this may take some time.')
            all_interferometers = likelihood.interferometers
            matched_filter_snrs = {ifo.name: [] for ifo in all_interferometers}
            optimal_snrs = {ifo.name: [] for ifo in all_interferometers}
            for ii in range(len(temp_sample)):
                for key in set(temp_sample.keys()).intersection(
                        likelihood.waveform_generator.parameters.keys()):
                    likelihood.waveform_generator.parameters[key] =\
                        temp_sample[key][ii]
                signal_polarizations =\
                    likelihood.waveform_generator.frequency_domain_strain()
                for ifo in all_interferometers:
                    signal = ifo.get_detector_response(
                        signal_polarizations,
                        likelihood.waveform_generator.parameters)
                    matched_filter_snrs[ifo.name].append(
                        ifo.matched_filter_snr_squared(signal=signal) ** 0.5)
                    optimal_snrs[ifo.name].append(
                        ifo.optimal_snr_squared(signal=signal) ** 0.5)

            for ifo in likelihood.interferometers:
                sample['{}_matched_filter_snr'.format(ifo.name)] =\
                    matched_filter_snrs[ifo.name]
                sample['{}_optimal_snr'.format(ifo.name)] =\
                    optimal_snrs[ifo.name]

            likelihood.interferometers = all_interferometers

    else:
        logger.debug('Not computing SNRs.')
