import os

import numpy as np
from scipy.interpolate import UnivariateSpline

from ..core.prior import (PriorDict, Uniform, Prior, DeltaFunction, Gaussian,
                          Interped, Constraint)
from ..core.utils import infer_args_from_method, logger
from .conversion import (
    convert_to_lal_binary_black_hole_parameters,
    convert_to_lal_binary_neutron_star_parameters, generate_mass_parameters,
    generate_tidal_parameters, fill_from_fixed_priors)
from .cosmology import get_cosmology

try:
    from astropy import cosmology as cosmo, units
except ImportError:
    logger.debug("You do not have astropy installed currently. You will"
                 " not be able to use some of the prebuilt functions.")


class Cosmological(Interped):

    @property
    def _default_args_dict(self):
        return dict(
            redshift=dict(name='redshift', latex_label='$z$', unit=None),
            luminosity_distance=dict(
                name='luminosity_distance', latex_label='$d_L$', unit=units.Mpc),
            comoving_distance=dict(
                name='comoving_distance', latex_label='$d_C$', unit=units.Mpc))

    def __init__(self, minimum, maximum, cosmology=None, name=None,
                 latex_label=None, unit=None, boundary=None):
        self.cosmology = get_cosmology(cosmology)
        if name not in self._default_args_dict:
            raise ValueError(
                "Name {} not recognised. Must be one of luminosity_distance, "
                "comoving_distance, redshift".format(name))
        self.name = name
        label_args = self._default_args_dict[self.name]
        if latex_label is not None:
            label_args['latex_label'] = latex_label
        if unit is not None:
            if isinstance(unit, str):
                unit = units.__dict__[unit]
            label_args['unit'] = unit
        self.unit = label_args['unit']
        self._minimum = dict()
        self._maximum = dict()
        self.minimum = minimum
        self.maximum = maximum
        if name == 'redshift':
            xx, yy = self._get_redshift_arrays()
        elif name == 'comoving_distance':
            xx, yy = self._get_comoving_distance_arrays()
        elif name == 'luminosity_distance':
            xx, yy = self._get_luminosity_distance_arrays()
        else:
            raise ValueError('Name {} not recognized.'.format(name))
        Interped.__init__(self, xx=xx, yy=yy, minimum=minimum, maximum=maximum,
                          boundary=boundary, **label_args)

    @property
    def minimum(self):
        return self._minimum[self.name]

    @minimum.setter
    def minimum(self, minimum):
        cosmology = get_cosmology(self.cosmology)
        self._minimum[self.name] = minimum
        if self.name == 'redshift':
            self._minimum['luminosity_distance'] =\
                cosmology.luminosity_distance(minimum).value
            self._minimum['comoving_distance'] =\
                cosmology.comoving_distance(minimum).value
        elif self.name == 'luminosity_distance':
            if minimum == 0:
                self._minimum['redshift'] = 0
            else:
                self._minimum['redshift'] = cosmo.z_at_value(
                    cosmology.luminosity_distance, minimum * self.unit)
            self._minimum['comoving_distance'] = self._minimum['redshift']
        elif self.name == 'comoving_distance':
            if minimum == 0:
                self._minimum['redshift'] = 0
            else:
                self._minimum['redshift'] = cosmo.z_at_value(
                    cosmology.comoving_distance, minimum * self.unit)
            self._minimum['luminosity_distance'] = self._minimum['redshift']
        try:
            self._update_instance()
        except (AttributeError, KeyError):
            pass

    @property
    def maximum(self):
        return self._maximum[self.name]

    @maximum.setter
    def maximum(self, maximum):
        cosmology = get_cosmology(self.cosmology)
        self._maximum[self.name] = maximum
        if self.name == 'redshift':
            self._maximum['luminosity_distance'] = \
                cosmology.luminosity_distance(maximum).value
            self._maximum['comoving_distance'] = \
                cosmology.comoving_distance(maximum).value
        elif self.name == 'luminosity_distance':
            self._maximum['redshift'] = cosmo.z_at_value(
                cosmology.luminosity_distance, maximum * self.unit)
            self._maximum['comoving_distance'] = self._maximum['redshift']
        elif self.name == 'comoving_distance':
            self._maximum['redshift'] = cosmo.z_at_value(
                cosmology.comoving_distance, maximum * self.unit)
            self._maximum['luminosity_distance'] = self._maximum['redshift']
        try:
            self._update_instance()
        except (AttributeError, KeyError):
            pass

    def get_corresponding_prior(self, name=None, unit=None):
        subclass_args = infer_args_from_method(self.__init__)
        args_dict = {key: getattr(self, key) for key in subclass_args}
        self._convert_to(new=name, args_dict=args_dict)
        if unit is not None:
            args_dict['unit'] = unit
        return self.__class__(**args_dict)

    def _convert_to(self, new, args_dict):
        args_dict.update(self._default_args_dict[new])
        args_dict['minimum'] = self._minimum[args_dict['name']]
        args_dict['maximum'] = self._maximum[args_dict['name']]

    def _get_comoving_distance_arrays(self):
        zs, p_dz = self._get_redshift_arrays()
        dc_of_z = self.cosmology.comoving_distance(zs).value
        ddc_dz = np.gradient(dc_of_z, zs)
        p_dc = p_dz / ddc_dz
        return dc_of_z, p_dc

    def _get_luminosity_distance_arrays(self):
        zs, p_dz = self._get_redshift_arrays()
        dl_of_z = self.cosmology.luminosity_distance(zs).value
        ddl_dz = np.gradient(dl_of_z, zs)
        p_dl = p_dz / ddl_dz
        return dl_of_z, p_dl

    def _get_redshift_arrays(self):
        raise NotImplementedError

    @classmethod
    def from_repr(cls, string):
        if "FlatLambdaCDM" in string:
            logger.warning(
                "Cosmological priors cannot be loaded from a string. "
                "If the prior has a name, use that instead."
            )
            return string
        else:
            return cls._from_repr(string)

    @property
    def _repr_dict(self):
        """
        Get a dictionary containing the arguments needed to reproduce this object.
        """
        dict_with_properties = super(Cosmological, self)._repr_dict
        if isinstance(dict_with_properties['cosmology'], cosmo.core.Cosmology):
            if dict_with_properties['cosmology'].name is not None:
                dict_with_properties['cosmology'] = dict_with_properties['cosmology'].name
        if isinstance(dict_with_properties['unit'], units.Unit):
            dict_with_properties['unit'] = dict_with_properties['unit'].to_string()
        return dict_with_properties


class UniformComovingVolume(Cosmological):

    def _get_redshift_arrays(self):
        zs = np.linspace(self._minimum['redshift'] * 0.99,
                         self._maximum['redshift'] * 1.01, 1000)
        p_dz = self.cosmology.differential_comoving_volume(zs).value
        return zs, p_dz


class UniformSourceFrame(Cosmological):
    """
    Prior for redshift which is uniform in comoving volume and source frame
    time.

    For redshift this is p(z) \propto dVc/dz 1 / (1 + z), where the extra 1+z
    is due to doppler shifting of the source frame time.
    """

    def _get_redshift_arrays(self):
        zs = np.linspace(self._minimum['redshift'] * 0.99,
                         self._maximum['redshift'] * 1.01, 1000)
        p_dz = self.cosmology.differential_comoving_volume(zs).value / (1 + zs)
        return zs, p_dz


class AlignedSpin(Interped):

    def __init__(self, a_prior=Uniform(0, 1), z_prior=Uniform(-1, 1),
                 name=None, latex_label=None, unit=None, boundary=None):
        """
        Prior distribution for the aligned (z) component of the spin.

        This takes prior distributions for the magnitude and cosine of the tilt
        and forms a compound prior.

        This is useful when using aligned-spin only waveform approximants.

        This is an extension of e.g., (A7) of https://arxiv.org/abs/1805.10457.

        Parameters
        ----------
        a_prior: Prior
            Prior distribution for spin magnitude
        z_prior: Prior
            Prior distribution for cosine spin tilt
        name: see superclass
        latex_label: see superclass
        unit: see superclass
        """
        self.a_prior = a_prior
        self.z_prior = z_prior
        chi_min = min(a_prior.maximum * z_prior.minimum,
                      a_prior.minimum * z_prior.maximum)
        chi_max = a_prior.maximum * z_prior.maximum
        xx = np.linspace(chi_min, chi_max, 800)
        aas = np.linspace(a_prior.minimum, a_prior.maximum, 1000)
        yy = [np.trapz(np.nan_to_num(a_prior.prob(aas) / aas *
                                     z_prior.prob(x / aas)), aas) for x in xx]
        Interped.__init__(self, xx=xx, yy=yy, name=name,
                          latex_label=latex_label, unit=unit,
                          boundary=boundary)


class BBHPriorDict(PriorDict):
    def __init__(self, dictionary=None, filename=None, aligned_spin=False,
                 conversion_function=None):
        """ Initialises a Prior set for Binary Black holes

        Parameters
        ----------
        dictionary: dict, optional
            See superclass
        filename: str, optional
            See superclass
        conversion_function: func
            Function to convert between sampled parameters and constraints.
            By default this generates many additional parameters, see
            BBHPriorDict.default_conversion_function
        """
        basedir = os.path.join(os.path.dirname(__file__), 'prior_files')
        if dictionary is None and filename is None:
            fname = 'binary_black_holes.prior'
            if aligned_spin:
                fname = 'aligned_spin_' + fname
                logger.info('Using aligned spin prior')
            filename = os.path.join(basedir, fname)
            logger.info('No prior given, using default BBH priors in {}.'.format(filename))
        elif filename is not None:
            if not os.path.isfile(filename):
                filename = os.path.join(os.path.dirname(__file__), 'prior_files', filename)
        PriorDict.__init__(self, dictionary=dictionary, filename=filename,
                           conversion_function=conversion_function)

    def default_conversion_function(self, sample):
        """
        Default parameter conversion function for BBH signals.

        This generates:
        - the parameters passed to source.lal_binary_black_hole
        - all mass parameters

        It does not generate:
        - component spins
        - source-frame parameters

        Parameters
        ----------
        sample: dict
            Dictionary to convert

        Returns
        -------
        sample: dict
            Same as input
        """
        out_sample = fill_from_fixed_priors(sample, self)
        out_sample, _ = convert_to_lal_binary_black_hole_parameters(out_sample)
        out_sample = generate_mass_parameters(out_sample)

        return out_sample

    def test_redundancy(self, key, disable_logging=False):
        """
        Test whether adding the key would add be redundant.
        Already existing keys return True.

        Parameters
        ----------
        key: str
            The key to test.
        disable_logging: bool, optional
            Disable logging in this function call. Default is False.

        Return
        ------
        redundant: bool
            Whether the key is redundant or not
        """
        if key in self:
            logger.debug('{} already in prior'.format(key))
            return True

        sampling_parameters = {key for key in self if not isinstance(
            self[key], (DeltaFunction, Constraint))}

        mass_parameters = {'mass_1', 'mass_2', 'chirp_mass', 'total_mass', 'mass_ratio', 'symmetric_mass_ratio'}
        spin_tilt_1_parameters = {'tilt_1', 'cos_tilt_1'}
        spin_tilt_2_parameters = {'tilt_2', 'cos_tilt_2'}
        spin_azimuth_parameters = {'phi_1', 'phi_2', 'phi_12', 'phi_jl'}
        inclination_parameters = {'theta_jn', 'cos_theta_jn'}
        distance_parameters = {'luminosity_distance', 'comoving_distance', 'redshift'}

        for independent_parameters, parameter_set in \
                zip([2, 2, 1, 1, 1, 1],
                    [mass_parameters, spin_azimuth_parameters,
                     spin_tilt_1_parameters, spin_tilt_2_parameters,
                     inclination_parameters, distance_parameters]):
            if key in parameter_set:
                if len(parameter_set.intersection(
                        sampling_parameters)) >= independent_parameters:
                    logger.disabled = disable_logging
                    logger.warning('{} already in prior. '
                                   'This may lead to unexpected behaviour.'
                                   .format(parameter_set.intersection(self)))
                    logger.disabled = False
                    return True
        return False


class BNSPriorDict(PriorDict):

    def __init__(self, dictionary=None, filename=None, aligned_spin=True,
                 conversion_function=None):
        """ Initialises a Prior set for Binary Neutron Stars

        Parameters
        ----------
        dictionary: dict, optional
            See superclass
        filename: str, optional
            See superclass
        conversion_function: func
            Function to convert between sampled parameters and constraints.
            By default this generates many additional parameters, see
            BNSPriorDict.default_conversion_function
        """
        if aligned_spin:
            default_file = 'binary_neutron_stars.prior'
        else:
            default_file = 'precessing_binary_neutron_stars.prior'
        if dictionary is None and filename is None:
            filename = os.path.join(os.path.dirname(__file__), 'prior_files', default_file)
            logger.info('No prior given, using default BNS priors in {}.'.format(filename))
        elif filename is not None:
            if not os.path.isfile(filename):
                filename = os.path.join(os.path.dirname(__file__), 'prior_files', filename)
        PriorDict.__init__(self, dictionary=dictionary, filename=filename,
                           conversion_function=conversion_function)

    def default_conversion_function(self, sample):
        """
        Default parameter conversion function for BNS signals.

        This generates:
        - the parameters passed to source.lal_binary_neutron_star
        - all mass parameters
        - all tidal parameters

        It does not generate:
        - component spins
        - source-frame parameters

        Parameters
        ----------
        sample: dict
            Dictionary to convert

        Returns
        -------
        sample: dict
            Same as input
        """
        out_sample = fill_from_fixed_priors(sample, self)
        out_sample, _ = convert_to_lal_binary_neutron_star_parameters(out_sample)
        out_sample = generate_mass_parameters(out_sample)
        out_sample = generate_tidal_parameters(out_sample)
        return out_sample

    def test_redundancy(self, key, disable_logging=False):
        logger.disabled = disable_logging
        logger.info("Performing redundancy check using BBHPriorDict(self).test_redundancy")
        logger.disabled = False
        bbh_redundancy = BBHPriorDict(self).test_redundancy(key)

        if bbh_redundancy:
            return True
        redundant = False

        sampling_parameters = {key for key in self if not isinstance(
            self[key], (DeltaFunction, Constraint))}

        tidal_parameters = \
            {'lambda_1', 'lambda_2', 'lambda_tilde', 'delta_lambda_tilde'}

        if key in tidal_parameters:
            if len(tidal_parameters.intersection(sampling_parameters)) > 2:
                redundant = True
                logger.disabled = disable_logging
                logger.warning('{} already in prior. '
                               'This may lead to unexpected behaviour.'
                               .format(tidal_parameters.intersection(self)))
                logger.disabled = False
            elif len(tidal_parameters.intersection(sampling_parameters)) == 2:
                redundant = True
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
    'theta_jn': '$\\theta_{JN}$',
    'cos_theta_jn': '$\cos\\theta_{JN}$',
    'psi': '$\psi$',
    'phase': '$\phi$',
    'geocent_time': '$t_c$',
    'lambda_1': '$\\Lambda_1$',
    'lambda_2': '$\\Lambda_2$',
    'lambda_tilde': '$\\tilde{\\Lambda}$',
    'delta_lambda_tilde': '$\\delta\\tilde{\\Lambda}$'}


class CalibrationPriorDict(PriorDict):

    def __init__(self, dictionary=None, filename=None):
        """ Initialises a Prior set for Binary Black holes

        Parameters
        ----------
        dictionary: dict, optional
            See superclass
        filename: str, optional
            See superclass
        """
        if dictionary is None and filename is not None:
            filename = os.path.join(os.path.dirname(__file__),
                                    'prior_files', filename)
        PriorDict.__init__(self, dictionary=dictionary, filename=filename)
        self.source = None

    def to_file(self, outdir, label):
        """
        Write the prior to file. This includes information about the source if
        possible.

        Parameters
        ----------
        outdir: str
            Output directory.
        label: str
            Label for prior.
        """
        PriorDict.to_file(self, outdir=outdir, label=label)
        if self.source is not None:
            prior_file = os.path.join(outdir, "{}.prior".format(label))
            with open(prior_file, "a") as outfile:
                outfile.write("# prior source file is {}".format(self.source))

    @staticmethod
    def from_envelope_file(envelope_file, minimum_frequency,
                           maximum_frequency, n_nodes, label):
        """
        Load in the calibration envelope.

        This is a text file with columns:
            frequency median-amplitude median-phase -1-sigma-amplitude
            -1-sigma-phase +1-sigma-amplitude +1-sigma-phase

        Parameters
        ----------
        envelope_file: str
            Name of file to read in.
        minimum_frequency: float
            Minimum frequency for the spline.
        maximum_frequency: float
            Minimum frequency for the spline.
        n_nodes: int
            Number of nodes for the spline.
        label: str
            Label for the names of the parameters, e.g., recalib_H1_

        Returns
        -------
        prior: PriorDict
            Priors for the relevant parameters.
            This includes the frequencies of the nodes which are _not_ sampled.
        """
        calibration_data = np.genfromtxt(envelope_file).T
        frequency_array = calibration_data[0]
        amplitude_median = calibration_data[1] - 1
        phase_median = calibration_data[2]
        amplitude_sigma = (calibration_data[5] - calibration_data[3]) / 2
        phase_sigma = (calibration_data[6] - calibration_data[4]) / 2

        nodes = np.logspace(np.log10(minimum_frequency),
                            np.log10(maximum_frequency), n_nodes)

        amplitude_mean_nodes = \
            UnivariateSpline(frequency_array, amplitude_median)(nodes)
        amplitude_sigma_nodes = \
            UnivariateSpline(frequency_array, amplitude_sigma)(nodes)
        phase_mean_nodes = \
            UnivariateSpline(frequency_array, phase_median)(nodes)
        phase_sigma_nodes = \
            UnivariateSpline(frequency_array, phase_sigma)(nodes)

        prior = CalibrationPriorDict()
        for ii in range(n_nodes):
            name = "recalib_{}_amplitude_{}".format(label, ii)
            latex_label = "$A^{}_{}$".format(label, ii)
            prior[name] = Gaussian(mu=amplitude_mean_nodes[ii],
                                   sigma=amplitude_sigma_nodes[ii],
                                   name=name, latex_label=latex_label,
                                   boundary=None)
        for ii in range(n_nodes):
            name = "recalib_{}_phase_{}".format(label, ii)
            latex_label = "$\\phi^{}_{}$".format(label, ii)
            prior[name] = Gaussian(mu=phase_mean_nodes[ii],
                                   sigma=phase_sigma_nodes[ii],
                                   name=name, latex_label=latex_label,
                                   boundary=None)
        for ii in range(n_nodes):
            name = "recalib_{}_frequency_{}".format(label, ii)
            latex_label = "$f^{}_{}$".format(label, ii)
            prior[name] = DeltaFunction(peak=nodes[ii], name=name,
                                        latex_label=latex_label)
        prior.source = os.path.abspath(envelope_file)
        return prior

    @staticmethod
    def constant_uncertainty_spline(
            amplitude_sigma, phase_sigma, minimum_frequency, maximum_frequency,
            n_nodes, label):
        """
        Make prior assuming constant in frequency calibration uncertainty.

        This assumes Gaussian fluctuations about 0.

        Parameters
        ----------
        amplitude_sigma: float
            Uncertainty in the amplitude.
        phase_sigma: float
            Uncertainty in the phase.
        minimum_frequency: float
            Minimum frequency for the spline.
        maximum_frequency: float
            Minimum frequency for the spline.
        n_nodes: int
            Number of nodes for the spline.
        label: str
            Label for the names of the parameters, e.g., recalib_H1_

        Returns
        -------
        prior: PriorDict
            Priors for the relevant parameters.
            This includes the frequencies of the nodes which are _not_ sampled.
        """
        nodes = np.logspace(np.log10(minimum_frequency),
                            np.log10(maximum_frequency), n_nodes)

        amplitude_mean_nodes = [0] * n_nodes
        amplitude_sigma_nodes = [amplitude_sigma] * n_nodes
        phase_mean_nodes = [0] * n_nodes
        phase_sigma_nodes = [phase_sigma] * n_nodes

        prior = CalibrationPriorDict()
        for ii in range(n_nodes):
            name = "recalib_{}_amplitude_{}".format(label, ii)
            latex_label = "$A^{}_{}$".format(label, ii)
            prior[name] = Gaussian(mu=amplitude_mean_nodes[ii],
                                   sigma=amplitude_sigma_nodes[ii],
                                   name=name, latex_label=latex_label,
                                   boundary=None)
        for ii in range(n_nodes):
            name = "recalib_{}_phase_{}".format(label, ii)
            latex_label = "$\\phi^{}_{}$".format(label, ii)
            prior[name] = Gaussian(mu=phase_mean_nodes[ii],
                                   sigma=phase_sigma_nodes[ii],
                                   name=name, latex_label=latex_label,
                                   boundary=None)
        for ii in range(n_nodes):
            name = "recalib_{}_frequency_{}".format(label, ii)
            latex_label = "$f^{}_{}$".format(label, ii)
            prior[name] = DeltaFunction(peak=nodes[ii], name=name,
                                        latex_label=latex_label)

        return prior
