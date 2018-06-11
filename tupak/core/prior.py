#!/bin/python
from __future__ import division

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.special import erf, erfinv
import scipy.stats
import logging
import os


class PriorSet(dict):
    def __init__(self, dictionary=None, filename=None):
        """ A set of priors

        Parameters
        ----------
        dictionary: dict, None
            If given, a dictionary to generate the prior set.
        filename: str, None
            If given, a file containing the prior to generate the prior set.
        """

        if type(dictionary) is dict:
            self.update(dictionary)
        elif filename:
            self.read_in_file(filename)

    def write_to_file(self, outdir, label):
        """
        Write the prior distribution to file.

        Parameters
        ----------
        priors: dict
            priors used
        outdir, label: str
            output directory and label
        """

        prior_file = os.path.join(outdir, "{}_prior.txt".format(label))
        logging.debug("Writing priors to {}".format(prior_file))
        with open(prior_file, "w") as outfile:
            for key in self.keys():
                outfile.write(
                    "{} = {}\n".format(key, self[key]))

    def read_in_file(self, filename):
        """
        Reads in a prior from a file specification
        """

        prior = {}
        with open(filename, 'r') as f:
            for line in f:
                elements = line.split('=')
                key = elements[0].replace(' ', '')
                val = '='.join(elements[1:])
                prior[key] = eval(val)
        self.update(prior)

    def fill_priors(self, likelihood, default_priors_file=None):
        """
        Fill dictionary of priors based on required parameters of likelihood

        Any floats in prior will be converted to delta function prior. Any
        required, non-specified parameters will use the default.

        Parameters
        ----------
        prior: dict
            dictionary of prior objects and floats
        likelihood: tupak.likelihood.GravitationalWaveTransient instance
            Used to infer the set of parameters to fill the prior with
        default_priors_file: str
            If given, a file containing the default priors; otherwise defaults
            to the tupak defaults for a binary black hole.


        Note: if `likelihood` has `non_standard_sampling_parameter_keys`, then
        this will set-up default priors for those as well.

        Returns
        -------
        prior: dict
            The filled prior dictionary

        """

        for key in self:
            if isinstance(self[key], Prior):
                continue
            elif isinstance(self[key], float) or isinstance(self[key], int):
                self[key] = DeltaFunction(self[key])
                logging.info(
                    "{} converted to delta function prior.".format(key))
            else:
                logging.info(
                    "{} cannot be converted to delta function prior."
                    .format(key))

        missing_keys = set(likelihood.parameters) - set(self.keys())

        if getattr(likelihood, 'non_standard_sampling_parameter_keys', None) is not None:
            for parameter in likelihood.non_standard_sampling_parameter_keys:
                self[parameter] = create_default_prior(parameter, default_priors_file)

        for missing_key in missing_keys:
            default_prior = create_default_prior(missing_key, default_priors_file)
            if default_prior is None:
                set_val = likelihood.parameters[missing_key]
                logging.warning(
                    "Parameter {} has no default prior and is set to {}, this"
                    " will not be sampled and may cause an error."
                    .format(missing_key, set_val))
            else:
                if not self.test_redundancy(missing_key):
                    self[missing_key] = default_prior

        for key in self:
            self.test_redundancy(key)

    def sample(self, size=None):
        """Draw samples from the prior set"""
        samples = dict()
        for key in self:
            if isinstance(self[key], Prior):
                samples[key] = self[key].sample(size=size)
        return samples

    def rescale(self, keys, theta):
        """Rescale samples from unit cube to prior"""
        rescaled_sample = [self[key].rescale(sample) for key, sample in zip(keys, theta)]
        return rescaled_sample

    def test_redundancy(self, key):
        """Empty redundancy test, should be overwritten"""
        return False


def create_default_prior(name, default_priors_file=None):
    """
    Make a default prior for a parameter with a known name.

    Parameters
    ----------
    name: str
        Parameter name
    default_priors_file: str
        If given, a file containing the default priors; otherwise defaults to
        the tupak defaults for a binary black hole.

    Return
    ------
    prior: Prior
        Default prior distribution for that parameter, if unknown None is
        returned.
    """

    if default_priors_file is None:
        logging.info(
            "No prior file given.")
        prior = None
    else:
        default_priors = PriorSet(filename=default_priors_file)
        if name in default_priors.keys():
            prior = default_priors[name]
        else:
            logging.info(
                "No default prior found for variable {}.".format(name))
            prior = None
    return prior


class Prior(object):
    """
    Prior class

    Methods
    -------
    __init__:
        Instantiate a prior object.
    __call__:
        Draw a single sample from the prior.
    __repr__:
        Print prior type and parameters.
    sample(size=None):
        Draw samples of size size from the prior.
    rescale(val):
        Rescale samples from a uniform distribution on [0, 1] to samples from the prior.
    test_valid_for_recaling(val):
        Test whether val is in [0, 1] and hence valid for rescaling.

    Parameters
    ----------
    name: str
        Name associated with prior.
    latex_label: str
        Latex label associated with prior, used for plotting.
    minimum: float, optional
        Minimum of the domain, default=-np.inf
    maximum: float, optional
        Maximum of the domain, default=np.inf
    """

    def __init__(self, name=None, latex_label=None, minimum=-np.inf, maximum=np.inf):
        self.name = name
        self.latex_label = latex_label
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self):
        return self.sample()

    def sample(self, size=None):
        """Draw a sample from the prior """
        return self.rescale(np.random.uniform(0, 1, size))

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior.

        This should be overwritten by each subclass.
        """
        return None

    @staticmethod
    def test_valid_for_rescaling(val):
        """Test if 0 < val < 1"""
        val = np.atleast_1d(val)
        tests = (val < 0) + (val > 1)
        if np.any(tests):
            raise ValueError("Number to be rescaled should be in [0, 1]")

    def __repr__(self):
        return self._subclass_repr_helper()

    def _subclass_repr_helper(self, subclass_args=list()):
        prior_name = self.__class__.__name__
        args = ['name', 'latex_label', 'minimum', 'maximum']
        args.extend(subclass_args)

        property_names = [p for p in dir(self.__class__) if isinstance(getattr(self.__class__, p), property)]
        dict_with_properties = self.__dict__.copy()
        for key in property_names:
            dict_with_properties[key] = getattr(self, key)

        args = ', '.join(['{}={}'.format(key, repr(dict_with_properties[key])) for key in args])
        return "{}({})".format(prior_name, args)

    @property
    def is_fixed(self):
        return isinstance(self, DeltaFunction)

    @property
    def latex_label(self):
        return self.__latex_label

    @latex_label.setter
    def latex_label(self, latex_label=None):
        if latex_label is None:
            self.__latex_label = self.__default_latex_label
        else:
            self.__latex_label = latex_label

    @property
    def minimum(self):
        return self.__minimum

    @minimum.setter
    def minimum(self, minimum):
        self.__minimum = minimum

    @property
    def maximum(self):
        return self.__maximum

    @maximum.setter
    def maximum(self, maximum):
        self.__maximum = maximum

    @property
    def __default_latex_label(self):
        default_labels = {
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
            'geocent_time': '$t_c$'
        }
        if self.name in default_labels.keys():
            label = default_labels[self.name]
        else:
            label = self.name
        return label


class DeltaFunction(Prior):
    """Dirac delta function prior, this always returns peak."""

    def __init__(self, peak, name=None, latex_label=None):
        Prior.__init__(self, name, latex_label, minimum=peak, maximum=peak)
        self.peak = peak

    def rescale(self, val):
        """Rescale everything to the peak with the correct shape."""
        Prior.test_valid_for_rescaling(val)
        return self.peak * val ** 0

    def prob(self, val):
        """Return the prior probability of val"""
        if self.peak == val:
            return np.inf
        else:
            return 0

    def __repr__(self):
        return Prior._subclass_repr_helper(self, subclass_args=['peak'])


class PowerLaw(Prior):
    """Power law prior distribution"""

    def __init__(self, alpha, minimum, maximum, name=None, latex_label=None):
        """Power law with bounds and alpha, spectral index"""
        Prior.__init__(self, name, latex_label, minimum, maximum)
        self.alpha = alpha

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the power-law prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        if self.alpha == -1:
            return self.minimum * np.exp(val * np.log(self.maximum / self.minimum))
        else:
            return (self.minimum ** (1 + self.alpha) + val *
                    (self.maximum ** (1 + self.alpha) - self.minimum ** (1 + self.alpha))) ** (1. / (1 + self.alpha))

    def prob(self, val):
        """Return the prior probability of val"""
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        if self.alpha == -1:
            return np.nan_to_num(1 / val / np.log(self.maximum / self.minimum)) * in_prior
        else:
            return np.nan_to_num(val ** self.alpha * (1 + self.alpha) / (self.maximum ** (1 + self.alpha)
                                                                         - self.minimum ** (1 + self.alpha))) * in_prior

    def lnprob(self, val):
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        normalising = (1 + self.alpha) / (self.maximum ** (1 + self.alpha)
                                          - self.minimum ** (1 + self.alpha))
        return self.alpha * np.log(val) * np.log(normalising) * in_prior

    def __repr__(self):
        return Prior._subclass_repr_helper(self, subclass_args=['alpha'])


class Uniform(Prior):
    """Uniform prior"""

    def __init__(self, minimum, maximum, name=None, latex_label=None):
        Prior.__init__(self, name, latex_label, minimum, maximum)

    def rescale(self, val):
        Prior.test_valid_for_rescaling(val)
        return self.minimum + val * (self.maximum - self.minimum)

    def prob(self, val):
        return scipy.stats.uniform.pdf(val, loc=self.minimum,
                                       scale=self.maximum-self.minimum)


class LogUniform(PowerLaw):
    """Log Uniform prior"""

    def __init__(self, minimum, maximum, name=None, latex_label=None):
        PowerLaw.__init__(self, name=name, latex_label=latex_label, minimum=minimum, maximum=maximum, alpha=-1)
        if self.minimum <= 0:
            logging.warning('You specified a uniform-in-log prior with minimum={}'.format(self.minimum))

    def __repr__(self, subclass_keys=list(), subclass_names=list()):
        return Prior._subclass_repr_helper(self)


class Cosine(Prior):

    def __init__(self, name=None, latex_label=None, minimum=-np.pi / 2, maximum=np.pi / 2):
        Prior.__init__(self, name, latex_label, minimum, maximum)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to a uniform in cosine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        return np.arcsin(-1 + val * 2)

    def prob(self, val):
        """Return the prior probability of val, defined over [-pi/2, pi/2]"""
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        return np.cos(val) / 2 * in_prior

    def __repr__(self, subclass_keys=list(), subclass_names=list()):
        return Prior._subclass_repr_helper(self)


class Sine(Prior):

    def __init__(self, name=None, latex_label=None, minimum=0, maximum=np.pi):
        Prior.__init__(self, name, latex_label, minimum, maximum)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to a uniform in sine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        return np.arccos(1 - val * 2)

    def prob(self, val):
        """Return the prior probability of val, defined over [0, pi]"""
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        return np.sin(val) / 2 * in_prior

    def __repr__(self, subclass_keys=list(), subclass_names=list()):
        return Prior._subclass_repr_helper(self)


class Gaussian(Prior):
    """Gaussian prior"""

    def __init__(self, mu, sigma, name=None, latex_label=None):
        """Power law with bounds and alpha, spectral index"""
        Prior.__init__(self, name, latex_label)
        self.mu = mu
        self.sigma = sigma

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Gaussian prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        return self.mu + erfinv(2 * val - 1) * 2 ** 0.5 * self.sigma

    def prob(self, val):
        """Return the prior probability of val"""
        return np.exp(-(self.mu - val) ** 2 / (2 * self.sigma ** 2)) / (2 * np.pi) ** 0.5 / self.sigma

    def lnprob(self, val):
        return -0.5 * ((self.mu - val) ** 2 / self.sigma ** 2 + np.log(2 * np.pi * self.sigma ** 2))

    def __repr__(self):
        return Prior._subclass_repr_helper(self, subclass_args=['mu', 'sigma'])


class TruncatedGaussian(Prior):
    """
    Truncated Gaussian prior

    https://en.wikipedia.org/wiki/Truncated_normal_distribution
    """

    def __init__(self, mu, sigma, minimum, maximum, name=None, latex_label=None):
        """Power law with bounds and alpha, spectral index"""
        Prior.__init__(self, name=name, latex_label=latex_label, minimum=minimum, maximum=maximum)
        self.mu = mu
        self.sigma = sigma

        self.normalisation = (erf((self.maximum - self.mu) / 2 ** 0.5 / self.sigma) - erf(
            (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) / 2

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate truncated Gaussian prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        return erfinv(2 * val * self.normalisation + erf(
            (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) * 2 ** 0.5 * self.sigma + self.mu

    def prob(self, val):
        """Return the prior probability of val"""
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        return np.exp(-(self.mu - val) ** 2 / (2 * self.sigma ** 2)) / (
                2 * np.pi) ** 0.5 / self.sigma / self.normalisation * in_prior

    def __repr__(self):
        return Prior._subclass_repr_helper(self, subclass_args=['mu', 'sigma'])


class Interped(Prior):

    def __init__(self, xx, yy, minimum=np.nan, maximum=np.nan, name=None, latex_label=None):
        """Initialise object from arrays of x and y=p(x)"""
        self.xx = xx
        self.yy = yy
        self.all_interpolated = interp1d(x=xx, y=yy, bounds_error=False, fill_value=0)
        Prior.__init__(self, name, latex_label,
                       minimum=np.nanmax(np.array((min(xx), minimum))),
                       maximum=np.nanmin(np.array((max(xx), maximum))))
        self.__initialize_attributes()

    def prob(self, val):
        """Return the prior probability of val"""
        return self.probability_density(val)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior.

        This maps to the inverse CDF. This is done using interpolation.
        """
        Prior.test_valid_for_rescaling(val)
        rescaled = self.inverse_cumulative_distribution(val)
        if rescaled.shape == ():
            rescaled = float(rescaled)
        return rescaled

    def __repr__(self):
        return Prior._subclass_repr_helper(self, subclass_args=['xx', 'yy'])

    @property
    def minimum(self):
        return self.__minimum

    @minimum.setter
    def minimum(self, minimum):
        self.__minimum = minimum
        if '_Interped__maximum' in self.__dict__ and self.__maximum < np.inf:
            self.__update_instance()

    @property
    def maximum(self):
        return self.__maximum

    @maximum.setter
    def maximum(self, maximum):
        self.__maximum = maximum
        if '_Interped__minimum' in self.__dict__ and self.__minimum < np.inf:
            self.__update_instance()

    def __update_instance(self):
        self.xx = np.linspace(self.minimum, self.maximum, len(self.xx))
        self.yy = self.all_interpolated(self.xx)
        self.__initialize_attributes()

    def __initialize_attributes(self):
        if np.trapz(self.yy, self.xx) != 1:
            logging.info('Supplied PDF for {} is not normalised, normalising.'.format(self.name))
        self.yy /= np.trapz(self.yy, self.xx)
        self.YY = cumtrapz(self.yy, self.xx, initial=0)
        # Need last element of cumulative distribution to be exactly one.
        self.YY[-1] = 1
        self.probability_density = interp1d(x=self.xx, y=self.yy, bounds_error=False, fill_value=0)
        self.cumulative_distribution = interp1d(x=self.xx, y=self.YY, bounds_error=False, fill_value=0)
        self.inverse_cumulative_distribution = interp1d(x=self.YY, y=self.xx, bounds_error=True)


class FromFile(Interped):

    def __init__(self, file_name, minimum=None, maximum=None, name=None, latex_label=None):
        try:
            self.id = file_name
            if '/' not in self.id:
                self.id = os.path.join(os.path.dirname(__file__), 'prior_files', self.id)
            xx, yy = np.genfromtxt(self.id).T
            Interped.__init__(self, xx=xx, yy=yy, minimum=minimum, maximum=maximum, name=name, latex_label=latex_label)
        except IOError:
            logging.warning("Can't load {}.".format(self.id))
            logging.warning("Format should be:")
            logging.warning(r"x\tp(x)")

    def __repr__(self, subclass_keys=list(), subclass_names=list()):
        return Prior._subclass_repr_helper(self, subclass_args=['id'])


class UniformComovingVolume(FromFile):

    def __init__(self, minimum=None, maximum=None, name=None, latex_label=None):
        FromFile.__init__(self, file_name='comoving.txt', minimum=minimum, maximum=maximum, name=name,
                          latex_label=latex_label)

    def __repr__(self, subclass_keys=list(), subclass_names=list()):
        return FromFile.__repr__(self)
