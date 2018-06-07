import logging

import numpy as np

import tupak.core.prior


def create_default_prior(name):
    """
    Make a default prior for a parameter with a known name.

    This is currently set up for binary black holes.

    Parameters
    ----------
    name: str
        Parameter name

    Return
    ------
    prior: Prior
        Default prior distribution for that parameter, if unknown None is returned.
    """
    default_priors = {
        'mass_1': tupak.core.prior.Uniform(name=name, minimum=20, maximum=100),
        'mass_2': tupak.core.prior.Uniform(name=name, minimum=20, maximum=100),
        'chirp_mass': tupak.core.prior.Uniform(name=name, minimum=25, maximum=100),
        'total_mass': tupak.core.prior.Uniform(name=name, minimum=10, maximum=200),
        'mass_ratio': tupak.core.prior.Uniform(name=name, minimum=0.125, maximum=1),
        'symmetric_mass_ratio': tupak.core.prior.Uniform(name=name, minimum=8 / 81, maximum=0.25),
        'a_1': tupak.core.prior.Uniform(name=name, minimum=0, maximum=0.8),
        'a_2': tupak.core.prior.Uniform(name=name, minimum=0, maximum=0.8),
        'tilt_1': tupak.core.prior.Sine(name=name),
        'tilt_2': tupak.core.prior.Sine(name=name),
        'cos_tilt_1': tupak.core.prior.Uniform(name=name, minimum=-1, maximum=1),
        'cos_tilt_2': tupak.core.prior.Uniform(name=name, minimum=-1, maximum=1),
        'phi_12': tupak.core.prior.Uniform(name=name, minimum=0, maximum=2 * np.pi),
        'phi_jl': tupak.core.prior.Uniform(name=name, minimum=0, maximum=2 * np.pi),
        'luminosity_distance': tupak.core.prior.UniformComovingVolume(name=name, minimum=1e2, maximum=5e3),
        'dec': tupak.core.prior.Cosine(name=name),
        'ra': tupak.core.prior.Uniform(name=name, minimum=0, maximum=2 * np.pi),
        'iota': tupak.core.prior.Sine(name=name),
        'cos_iota': tupak.core.prior.Uniform(name=name, minimum=-1, maximum=1),
        'psi': tupak.core.prior.Uniform(name=name, minimum=0, maximum=2 * np.pi),
        'phase': tupak.core.prior.Uniform(name=name, minimum=0, maximum=2 * np.pi)
    }
    if name in default_priors.keys():
        prior = default_priors[name]
    else:
        logging.info(
            "No default prior found for variable {}.".format(name))
        prior = None
    return prior


def test_redundancy(key, prior):
    """
    Test whether adding the key would add be redundant.

    Parameters
    ----------
    key: str
        The string to test.
    prior: dict
        Current prior dictionary.

    Return
    ------
    redundant: bool
        Whether the key is redundant
    """
    redundant = False
    mass_parameters = {'mass_1', 'mass_2', 'chirp_mass', 'total_mass', 'mass_ratio', 'symmetric_mass_ratio'}
    spin_magnitude_parameters = {'a_1', 'a_2'}
    spin_tilt_1_parameters = {'tilt_1', 'cos_tilt_1'}
    spin_tilt_2_parameters = {'tilt_2', 'cos_tilt_2'}
    spin_azimuth_parameters = {'phi_1', 'phi_2', 'phi_12', 'phi_jl'}
    inclination_parameters = {'iota', 'cos_iota'}
    distance_parameters = {'luminosity_distance', 'comoving_distance', 'redshift'}

    for parameter_set in [mass_parameters, spin_magnitude_parameters, spin_azimuth_parameters]:
        if key in parameter_set:
            if len(parameter_set.intersection(prior.keys())) > 2:
                redundant = True
                logging.warning('{} in prior. This may lead to unexpected behaviour.'.format(
                    parameter_set.intersection(prior.keys())))
                break
            elif len(parameter_set.intersection(prior.keys())) == 2:
                redundant = True
                break
    for parameter_set in [inclination_parameters, distance_parameters, spin_tilt_1_parameters, spin_tilt_2_parameters]:
        if key in parameter_set:
            if len(parameter_set.intersection(prior.keys())) > 1:
                redundant = True
                logging.warning('{} in prior. This may lead to unexpected behaviour.'.format(
                    parameter_set.intersection(prior.keys())))
                break
            elif len(parameter_set.intersection(prior.keys())) == 1:
                redundant = True
                break

    return redundant