import tupak
import numpy as np
import pandas as pd
import logging

try:
    import lalsimulation as lalsim
except ImportError:
    logging.warning("You do not have lalsuite installed currently. You will "
                    " not be able to use some of the prebuilt functions.")


def convert_to_lal_binary_black_hole_parameters(parameters, search_keys, remove=True):
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
    remove: bool, optional
        Whether or not to remove the extra key, necessary for sampling, default=True.

    Return
    ------
    ignored_keys: list
        keys which are added to parameters during function call
    """

    ignored_keys = []
    converted_parameters = parameters.copy()

    if 'mass_1' not in search_keys and 'mass_2' not in search_keys:
        if 'chirp_mass' in converted_parameters.keys():
            if 'total_mass' in converted_parameters.keys():
                # chirp_mass, total_mass to total_mass, symmetric_mass_ratio
                converted_parameters['symmetric_mass_ratio'] = chirp_mass_and_total_mass_to_symmetric_mass_ratio(
                    converted_parameters['chirp_mass'], converted_parameters['total_mass'])
                if remove:
                    converted_parameters.pop('chirp_mass')
            if 'symmetric_mass_ratio' in converted_parameters.keys():
                # symmetric_mass_ratio to mass_ratio
                converted_parameters['mass_ratio'] = symmetric_mass_ratio_to_mass_ratio(converted_parameters['symmetric_mass_ratio'])
                if remove:
                    converted_parameters.pop('symmetric_mass_ratio')
            if 'mass_ratio' in converted_parameters.keys():
                if 'total_mass' not in converted_parameters.keys():
                    converted_parameters['total_mass'] = chirp_mass_and_mass_ratio_to_total_mass(
                        converted_parameters['chirp_mass'], converted_parameters['mass_ratio'])
                    converted_parameters.pop('chirp_mass')
                # total_mass, mass_ratio to component masses
                converted_parameters['mass_1'], converted_parameters['mass_2'] = \
                    total_mass_and_mass_ratio_to_component_masses(converted_parameters['mass_ratio'],
                                                                  converted_parameters['total_mass'])
                if remove:
                    converted_parameters.pop('total_mass')
                    converted_parameters.pop('mass_ratio')
            ignored_keys.append('mass_1')
            ignored_keys.append('mass_2')
        elif 'total_mass' in converted_parameters.keys():
            if 'symmetric_mass_ratio' in converted_parameters.keys():
                # symmetric_mass_ratio to mass_ratio
                converted_parameters['mass_ratio'] = symmetric_mass_ratio_to_mass_ratio(converted_parameters['symmetric_mass_ratio'])
                if remove:
                    converted_parameters.pop('symmetric_mass_ratio')
            if 'mass_ratio' in converted_parameters.keys():
                # total_mass, mass_ratio to component masses
                converted_parameters['mass_1'], converted_parameters['mass_2'] = total_mass_and_mass_ratio_to_component_masses(converted_parameters)
                if remove:
                    converted_parameters.pop('total_mass')
                    converted_parameters.pop('mass_ratio')
            ignored_keys.append('mass_1')
            ignored_keys.append('mass_2')

    for cos_angle in ['cos_tilt_1', 'cos_tilt_2', 'cos_iota']:
        if str(cos_angle) in converted_parameters.keys():
            cos_angle_to_angle(cos_angle)
            if remove:
                converted_parameters.pop(cos_angle)

    return converted_paramters, ignored_keys


def total_mass_and_mass_ratio_to_component_masses(mass_ratio, total_mass):
    mass_1 = total_mass / (1 + mass_ratio)
    mass_2 = mass_1 * mass_ratio
    return mass_1, mass_2


def symmetric_mass_ratio_to_mass_ratio(symmetric_mass_ratio):
    temp = (1 / symmetric_mass_ratio / 2 - 1)
    return temp - (temp ** 2 - 1) ** 0.5


def chirp_mass_and_total_mass_to_symmetric_mass_ratio(chirp_mass, total_mass):
    return (chirp_mass / total_mass) ** (5 / 3)


def chirp_mass_and_mass_ratio_to_total_mass(chirp_mass, mass_ratio):
    return chirp_mass * (1 + mass_ratio) ** 1.2 / mass_ratio ** 0.6


def component_masses_to_chirp_mass(mass_1, mass_2):
    return (mass_1 * mass_2) ** 0.6 / (component_masses_to_total_mass(mass_1, mass_2)) ** 0.2


def component_masses_to_total_mass(mass_1, mass_2):
    return mass_1 + mass_2


def component_masses_to_symmetric_mass_ratio(mass_1, mass_2):
    return (mass_1 * mass_2) / (mass_1 + mass_2) ** 2


def component_masses_to_mass_ratio(mass_1, mass_2):
    return mass_2 / mass_1


def cos_angle_to_angle(cos_angle):
    return np.arccos(cos_angle)


def angle_to_cos_angle(angle):
    return np.cos(angle)


def generate_all_bbh_parameters(sample, likelihood=None, priors=None):
    """
    From either a single sample or a set of samples fill in all missing BBH parameters, in place.

    Parameters
    ----------
    sample: dict or pandas.DataFrame
        Samples to fill in with extra parameters, this may be either an injection or posterior samples.
    likelihood: tupak.GravitationalWaveTransient
        GravitationalWaveTransient used for sampling, used for waveform and likelihood.interferometers.
    priors: dict, optional
        Dictionary of prior objects, used to fill in non-sampled parameters.
    """
    output_sample = sample.copy()
    if likelihood is not None:
        output_sample['reference_frequency'] = likelihood.waveform_generator.parameters['reference_frequency']
        output_sample['waveform_approximant'] = likelihood.waveform_generator.parameters['waveform_approximant']

    output_sample = fill_from_fixed_priors(output_sample, priors)
    output_sample, _ = convert_to_lal_binary_black_hole_parameters(output_sample, [key for key in output_sample.keys()], remove=False)
    output_sample = generate_non_standard_parameters(output_sample)
    output_sample = generate_component_spins(output_sample)
    compute_snrs(output_sample, likelihood)
    return output_sample


def fill_from_fixed_priors(sample, priors):
    """Add parameters with delta function prior to the data frame/dictionary."""
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
        chirp mass, total mass, symmetric mass ratio, mass ratio, cos tilt 1, cos tilt 2, cos iota
    """
    output_sample = sample.copy()
    output_sample['chirp_mass'] = component_masses_to_chirp_mass(sample['mass_1'], sample['mass_2'])
    output_sample['total_mass'] = component_masses_to_total_mass(sample['mass_1'], sample['mass_2'])
    output_sample['symmetric_mass_ratio'] = component_masses_to_symmetric_mass_ratio(sample['mass_1'],
                                                                                     sample['mass_2'])
    output_sample['mass_ratio'] = component_masses_to_mass_ratio(sample['mass_1'], sample['mass_2'])

    output_sample['cos_tilt_1'] = angle_to_cos_angle(output_sample['tilt_1'])
    output_sample['cos_tilt_2'] = angle_to_cos_angle(output_sample['tilt_2'])
    output_sample['cos_iota'] = angle_to_cos_angle(output_sample['iota'])
    return output_sample


def generate_component_spins(sample):
    """
    Add the component spins to the data frame/dictionary.

    This function uses a lalsimulation function to transform the spins.
    """
    output_sample = sample.copy()
    spin_conversion_parameters = ['iota', 'phi_jl', 'tilt_1', 'tilt_2', 'phi_12', 'a_1', 'a_2', 'mass_1',
                                  'mass_2', 'reference_frequency', 'phase']
    if all(key in output_sample.keys() for key in spin_conversion_parameters) and isinstance(output_sample, dict):
        output_sample['iota'], output_sample['spin_1x'], output_sample['spin_1y'], output_sample['spin_1z'], output_sample['spin_2x'], \
        output_sample['spin_2y'], output_sample['spin_2z'] = \
            lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                output_sample['iota'], output_sample['phi_jl'], output_sample['tilt_1'], output_sample['tilt_2'], output_sample['phi_12'], output_sample['a_1'],
                output_sample['a_2'], output_sample['mass_1'] * tupak.core.utils.solar_mass, output_sample['mass_2'] * tupak.core.utils.solar_mass,
                output_sample['reference_frequency'], output_sample['phase'])

        output_sample['phi_1'] = np.arctan(output_sample['spin_1y'] / output_sample['spin_1x'])
        output_sample['phi_2'] = np.arctan(output_sample['spin_2y'] / output_sample['spin_2x'])

    elif all(key in output_sample.keys() for key in spin_conversion_parameters) and isinstance(output_sample, pd.DataFrame):
        logging.info('Extracting component spins.')
        new_spin_parameters = ['spin_1x', 'spin_1y', 'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z']
        new_spins = {name: np.zeros(len(output_sample)) for name in new_spin_parameters}

        for ii in range(len(output_sample)):
            new_spins['iota'], new_spins['spin_1x'][ii], new_spins['spin_1y'][ii], new_spins['spin_1z'][ii], \
                new_spins['spin_2x'][ii], new_spins['spin_2y'][ii], new_spins['spin_2z'][ii] = \
                lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                    output_sample['iota'][ii], output_sample['phi_jl'][ii], output_sample['tilt_1'][ii], output_sample['tilt_2'][ii],
                    output_sample['phi_12'][ii], output_sample['a_1'][ii], output_sample['a_2'][ii],
                    output_sample['mass_1'][ii] * tupak.core.utils.solar_mass, output_sample['mass_2'][ii] * tupak.core.utils.solar_mass,
                    output_sample['reference_frequency'][ii], output_sample['phase'][ii])

        for name in new_spin_parameters:
            output_sample[name] = new_spins[name]

        output_sample['phi_1'] = np.arctan(output_sample['spin_1y'] / output_sample['spin_1x'])
        output_sample['phi_2'] = np.arctan(output_sample['spin_2y'] / output_sample['spin_2x'])

    else:
        logging.warning("Component spin extraction failed.")

    return output_sample


def compute_snrs(sample, likelihood):
    """Compute the optimal and matched filter snrs of all posterior samples."""
    temp_sample = sample
    if likelihood is not None:
        if isinstance(temp_sample, dict):
            for key in likelihood.waveform_generator.parameters.keys():
                likelihood.waveform_generator.parameters[key] = temp_sample[key]
            signal_polarizations = likelihood.waveform_generator.frequency_domain_strain()
            for interferometer in likelihood.interferometers:
                signal = interferometer.get_detector_response(signal_polarizations,
                                                              likelihood.waveform_generator.parameters)
                sample['{}_matched_filter_snr'.format(interferometer.name)] = \
                    tupak.gw.utils.matched_filter_snr_squared(signal, interferometer,
                                                                likelihood.waveform_generator.time_duration) ** 0.5
                sample['{}_optimal_snr'.format(interferometer.name)] = tupak.gw.utils.optimal_snr_squared(
                    signal, interferometer, likelihood.waveform_generator.time_duration) ** 0.5
        else:
            logging.info('Computing SNRs for every sample, this may take some time.')
            all_interferometers = likelihood.interferometers
            matched_filter_snrs = {interferometer.name: [] for interferometer in all_interferometers}
            optimal_snrs = {interferometer.name: [] for interferometer in all_interferometers}
            for ii in range(len(temp_sample)):
                for key in set(temp_sample.keys()).intersection(likelihood.waveform_generator.parameters.keys()):
                    likelihood.waveform_generator.parameters[key] = temp_sample[key][ii]
                for key in likelihood.waveform_generator.non_standard_sampling_parameter_keys:
                    likelihood.waveform_generator.parameters[key] = temp_sample[key][ii]
                signal_polarizations = likelihood.waveform_generator.frequency_domain_strain()
                for interferometer in all_interferometers:
                    signal = interferometer.get_detector_response(signal_polarizations,
                                                                  likelihood.waveform_generator.parameters)
                    matched_filter_snrs[interferometer.name].append(tupak.gw.utils.matched_filter_snr_squared(
                        signal, interferometer, likelihood.waveform_generator.time_duration) ** 0.5)
                    optimal_snrs[interferometer.name].append(tupak.gw.utils.optimal_snr_squared(
                        signal, interferometer, likelihood.waveform_generator.time_duration) ** 0.5)

            for interferometer in likelihood.interferometers:
                sample['{}_matched_filter_snr'.format(interferometer.name)] = matched_filter_snrs[interferometer.name]
                sample['{}_optimal_snr'.format(interferometer.name)] = optimal_snrs[interferometer.name]

            likelihood.interferometers = all_interferometers
            print([interferometer.name for interferometer in likelihood.interferometers])

    else:
        logging.info('Not computing SNRs.')
