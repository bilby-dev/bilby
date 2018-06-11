from tupak.core.prior import *


class UniformComovingVolume(FromFile):

    def __init__(self, minimum=None, maximum=None, name=None, latex_label=None):
        FromFile.__init__(self, file_name='comoving.txt', minimum=minimum, maximum=maximum, name=name,
                          latex_label=latex_label)

    def __repr__(self, subclass_keys=list(), subclass_names=list()):
        return FromFile.__repr__(self)


class BBHPriorSet(PriorSet):
    def __init__(self, dictionary=None, filename=None):
        if dictionary is None and filename is None:
            filename = os.path.join(os.path.dirname(__file__), 'prior_files', 'binary_black_holes.prior')
            logging.info('No prior given, using default BBH priors in {}.'.format(filename))
        PriorSet.__init__(dictionary=dictionary, filename=filename)
        self.test_redundancy = test_cbc_redundancy


def test_cbc_redundancy(key, prior):
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
