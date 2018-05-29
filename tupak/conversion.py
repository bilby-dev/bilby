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

    if 'mass_1' not in search_keys and 'mass_2' not in search_keys:
        if 'chirp_mass' in parameters.keys():
            if 'total_mass' in parameters.keys():
                # chirp_mass, total_mass to total_mass, symmetric_mass_ratio
                parameters['symmetric_mass_ratio'] = (parameters['chirp_mass'] / parameters['total_mass'])**(5 / 3)
                if remove:
                    parameters.pop('chirp_mass')
            if 'symmetric_mass_ratio' in parameters.keys():
                # symmetric_mass_ratio to mass_ratio
                temp = (1 / parameters['symmetric_mass_ratio'] / 2 - 1)
                parameters['mass_ratio'] = temp - (temp**2 - 1)**0.5
                if remove:
                    parameters.pop('symmetric_mass_ratio')
            if 'mass_ratio' in parameters.keys():
                if 'total_mass' not in parameters.keys():
                    parameters['total_mass'] = parameters['chirp_mass'] * (1 + parameters['mass_ratio'])**1.2 \
                                               / parameters['mass_ratio']**0.6
                    parameters.pop('chirp_mass')
                # total_mass, mass_ratio to component masses
                parameters['mass_1'] = parameters['total_mass'] / (1 + parameters['mass_ratio'])
                parameters['mass_2'] = parameters['mass_1'] * parameters['mass_ratio']
                if remove:
                    parameters.pop('total_mass')
                    parameters.pop('mass_ratio')
            ignored_keys.append('mass_1')
            ignored_keys.append('mass_2')
        elif 'total_mass' in parameters.keys():
            if 'symmetric_mass_ratio' in parameters.keys():
                # symmetric_mass_ratio to mass_ratio
                temp = (1 / parameters['symmetric_mass_ratio'] / 2 - 1)
                parameters['mass_ratio'] = temp - (temp**2 - 1)**0.5
                if remove:
                    parameters.pop('symmetric_mass_ratio')
            if 'mass_ratio' in parameters.keys():
                # total_mass, mass_ratio to component masses
                parameters['mass_1'] = parameters['total_mass'] / (1 + parameters['mass_ratio'])
                parameters['mass_2'] = parameters['mass_1'] * parameters['mass_ratio']
                if remove:
                    parameters.pop('total_mass')
                    parameters.pop('mass_ratio')
            ignored_keys.append('mass_1')
            ignored_keys.append('mass_2')

    if 'cos_tilt_1' in parameters.keys():
        ignored_keys.append('tilt_1')
        parameters['tilt_1'] = np.arccos(parameters['cos_tilt_1'])
        if remove:
            parameters.pop('cos_tilt_1')
    if 'cos_tilt_2' in parameters.keys():
        ignored_keys.append('tilt_2')
        parameters['tilt_2'] = np.arccos(parameters['cos_tilt_2'])
        if remove:
            parameters.pop('cos_tilt_2')

    if 'cos_iota' in parameters.keys():
        parameters['iota'] = np.arccos(parameters['cos_iota'])
        if remove:
            parameters.pop('cos_iota')

    return ignored_keys


def generate_all_bbh_parameters(sample, likelihood=None, priors=None):
    """
    From either a single sample or a set of samples fill in all missing BBH parameters, in place.

    Parameters
    ----------
    sample: dict or pandas.DataFrame
        Samples to fill in with extra parameters, this may be either an injection or posterior samples.
    likelihood: tupak.likelihood.GravitationalWaveTransient
        GravitationalWaveTransient used for sampling, used for waveform and likelihood.interferometers.
    priors: dict, optional
        Dictionary of prior objects, used to fill in non-sampled parameters.
    """

    if likelihood is not None:
        sample['reference_frequency'] = likelihood.waveform_generator.parameters['reference_frequency']
        sample['waveform_approximant'] = likelihood.waveform_generator.parameters['waveform_approximant']

    fill_from_fixed_priors(sample, priors)
    convert_to_lal_binary_black_hole_parameters(sample, [key for key in sample.keys()], remove=False)
    generate_non_standard_parameters(sample)
    generate_component_spins(sample)
    compute_snrs(sample, likelihood)


def fill_from_fixed_priors(sample, priors):
    """Add parameters with delta function prior to the data frame/dictionary."""
    if priors is not None:
        for name in priors:
            if isinstance(priors[name], tupak.prior.DeltaFunction):
                sample[name] = priors[name].peak


def generate_non_standard_parameters(sample):
    """
    Add the known non-standard parameters to the data frame/dictionary.

    We add:
        chirp mass, total mass, symmetric mass ratio, mass ratio, cos tilt 1, cos tilt 2, cos iota
    """
    sample['chirp_mass'] = (sample['mass_1'] * sample['mass_2'])**0.6 / (sample['mass_1'] + sample['mass_2'])**0.2
    sample['total_mass'] = sample['mass_1'] + sample['mass_2']
    sample['symmetric_mass_ratio'] = (sample['mass_1'] * sample['mass_2']) / (sample['mass_1'] + sample['mass_2'])**2
    sample['mass_ratio'] = sample['mass_2'] / sample['mass_1']

    sample['cos_tilt_1'] = np.cos(sample['tilt_1'])
    sample['cos_tilt_2'] = np.cos(sample['tilt_2'])
    sample['cos_iota'] = np.cos(sample['iota'])


def generate_component_spins(sample):
    """
    Add the component spins to the data frame/dictionary.

    This function uses a lalsimulation function to transform the spins.
    """
    spin_conversion_parameters = ['iota', 'phi_jl', 'tilt_1', 'tilt_2', 'phi_12', 'a_1', 'a_2', 'mass_1',
                                  'mass_2', 'reference_frequency', 'phase']
    if all(key in sample.keys() for key in spin_conversion_parameters) and isinstance(sample, dict):
        sample['iota'], sample['spin_1x'], sample['spin_1y'], sample['spin_1z'], sample['spin_2x'], \
            sample['spin_2y'], sample['spin_2z'] = \
            lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                sample['iota'], sample['phi_jl'], sample['tilt_1'], sample['tilt_2'], sample['phi_12'], sample['a_1'],
                sample['a_2'], sample['mass_1'] * tupak.utils.solar_mass, sample['mass_2'] * tupak.utils.solar_mass,
                sample['reference_frequency'], sample['phase'])

        sample['phi_1'] = np.arctan(sample['spin_1y'] / sample['spin_1x'])
        sample['phi_2'] = np.arctan(sample['spin_2y'] / sample['spin_2x'])

    elif all(key in sample.keys() for key in spin_conversion_parameters) and isinstance(sample, pd.DataFrame):
        logging.info('Extracting component spins.')
        new_spin_parameters = ['spin_1x', 'spin_1y', 'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z']
        new_spins = {name: np.zeros(len(sample)) for name in new_spin_parameters}

        for ii in range(len(sample)):
            new_spins['iota'], new_spins['spin_1x'][ii], new_spins['spin_1y'][ii], new_spins['spin_1z'][ii], \
                new_spins['spin_2x'][ii], new_spins['spin_2y'][ii], new_spins['spin_2z'][ii] = \
                lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                    sample['iota'][ii], sample['phi_jl'][ii], sample['tilt_1'][ii], sample['tilt_2'][ii],
                    sample['phi_12'][ii], sample['a_1'][ii], sample['a_2'][ii],
                    sample['mass_1'][ii] * tupak.utils.solar_mass, sample['mass_2'][ii] * tupak.utils.solar_mass,
                    sample['reference_frequency'][ii], sample['phase'][ii])

        for name in new_spin_parameters:
            sample[name] = new_spins[name]

        sample['phi_1'] = np.arctan(sample['spin_1y'] / sample['spin_1x'])
        sample['phi_2'] = np.arctan(sample['spin_2y'] / sample['spin_2x'])

    else:
        logging.warning("Component spin extraction failed.")


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
                    tupak.utils.matched_filter_snr_squared(signal, interferometer,
                                                           likelihood.waveform_generator.time_duration)**0.5
                sample['{}_optimal_snr'.format(interferometer.name)] = tupak.utils.optimal_snr_squared(
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
                    matched_filter_snrs[interferometer.name].append(tupak.utils.matched_filter_snr_squared(
                        signal, interferometer, likelihood.waveform_generator.time_duration)**0.5)
                    optimal_snrs[interferometer.name].append(tupak.utils.optimal_snr_squared(
                        signal, interferometer, likelihood.waveform_generator.time_duration) ** 0.5)

            for interferometer in likelihood.interferometers:
                sample['{}_matched_filter_snr'.format(interferometer.name)] = matched_filter_snrs[interferometer.name]
                sample['{}_optimal_snr'.format(interferometer.name)] = optimal_snrs[interferometer.name]

            likelihood.interferometers = all_interferometers
            print([interferometer.name for interferometer in likelihood.interferometers])

    else:
        logging.info('Not computing SNRs.')
