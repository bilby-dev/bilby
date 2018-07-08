import os
from tupak.core.prior import PriorSet, FromFile, Prior

from tupak.core.utils import logger


class UniformComovingVolume(FromFile):

    def __init__(self, minimum=None, maximum=None, name='luminosity distance', latex_label='$d_L$'):
        """

        Parameters
        ----------
        minimum: float, optional
            See superclass
        maximum: float, optional
            See superclass
        name: str, optional
            See superclass
        latex_label: str, optional
            See superclass
        """
        file_name = os.path.join(os.path.dirname(__file__), 'prior_files', 'comoving.txt')
        FromFile.__init__(self, file_name=file_name, minimum=minimum, maximum=maximum, name=name,
                          latex_label=latex_label)


class BBHPriorSet(PriorSet):
    def __init__(self, dictionary=None, filename=None):
        """ Initialises a Prior set for Binary Black holes

        Parameters
        ----------
        dictionary: dict, optional
            See superclass
        filename: str, optional
            See superclass
        """
        if dictionary is None and filename is None:
            filename = os.path.join(os.path.dirname(__file__), 'prior_files', 'binary_black_holes.prior')
            logger.info('No prior given, using default BBH priors in {}.'.format(filename))
        elif not os.path.isfile(filename):
            filename = os.path.join(os.path.dirname(__file__), 'prior_files', filename)
        PriorSet.__init__(self, dictionary=dictionary, filename=filename)

    def test_redundancy(self, key):
        """
        Test whether adding the key would add be redundant.

        Parameters
        ----------
        key: str
            The key to test.

        Return
        ------
        redundant: bool
            Whether the key is redundant or not
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
                if len(parameter_set.intersection(self)) > 2:
                    redundant = True
                    logger.warning('{} in prior. This may lead to unexpected behaviour.'.format(
                        parameter_set.intersection(self)))
                    break
            elif len(parameter_set.intersection(self)) == 2:
                redundant = True
                break
        for parameter_set in [inclination_parameters, distance_parameters, spin_tilt_1_parameters,
                              spin_tilt_2_parameters]:
            if key in parameter_set:
                if len(parameter_set.intersection(self)) > 1:
                    redundant = True
                    logger.warning('{} in prior. This may lead to unexpected behaviour.'.format(
                        parameter_set.intersection(self)))
                    break
                elif len(parameter_set.intersection(self)) == 1:
                    redundant = True
                    break

        return redundant


Prior._default_latex_labels = {
    'mass_1': '$m_1$',
    'mass_2': '$m_2$',
    'total_mass': '$M$',
    'chirp_mass': '$\mathcal{M}$',
    'mass_ratio': '$q$',
    'symmetric_mass_ratio': '$\eta$',
    'a_1': '$a_1$',
    'a_2': '$a_2$',
    'tilt_1': '$\\theta_1$',
    'tilt_2': '$\\theta_2$',
    'cos_tilt_1': '$\cos\\theta_1$',
    'cos_tilt_2': '$\cos\\theta_2$',
    'phi_12': '$\Delta\phi$',
    'phi_jl': '$\phi_{JL}$',
    'luminosity_distance': '$d_L$',
    'dec': '$\mathrm{DEC}$',
    'ra': '$\mathrm{RA}$',
    'iota': '$\iota$',
    'cos_iota': '$\cos\iota$',
    'psi': '$\psi$',
    'phase': '$\phi$',
    'geocent_time': '$t_c$'}
