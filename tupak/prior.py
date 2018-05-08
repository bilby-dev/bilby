#!/bin/python
from __future__ import division

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.special import erf, erfinv
import logging
import os


class Prior(object):
    """Prior class"""

    def __init__(self, name=None, latex_label=None):
        self.name = name
        self.latex_label = latex_label

    def __call__(self):
        return self.sample(1)

    def sample(self, n_samples=None):
        """Draw a sample from the prior, this rescales a unit line element according to the rescaling function"""
        return self.rescale(np.random.uniform(0, 1, n_samples))

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior.

        This should be overwritten by each subclass.
        """
        return None

    @staticmethod
    def test_valid_for_rescaling(val):
        """Test if 0 < val < 1"""
        if (val < 0) or (val > 1):
            raise ValueError("Number to be rescaled should be in [0, 1]")

    def __repr__(self):
        prior_name = self.__class__.__name__
        prior_args = ', '.join(
            ['{}={}'.format(k, v) for k, v in self.__dict__.items()])
        return "{}({})".format(prior_name, prior_args)

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
    def __default_latex_label(self):
        if self.name == 'mass_1':
            return '$m_1$'
        elif self.name == 'mass_2':
            return '$m_2$'
        elif self.name == 'mchirp':
            return '$\mathcal{M}$'
        elif self.name == 'q':
            return '$q$'
        elif self.name == 'a_1':
            return '$a_1$'
        elif self.name == 'a_2':
            return '$a_2$'
        elif self.name == 'tilt_1':
            return '$\\theta_1$'
        elif self.name == 'tilt_2':
            return '$\\theta_2$'
        elif self.name == 'phi_12':
            return '$\Delta\phi$'
        elif self.name == 'phi_jl':
            return '$\phi_{JL}$'
        elif self.name == 'luminosity_distance':
            return '$d_L$'
        elif self.name == 'dec':
            return '$\mathrm{DEC}$'
        elif self.name == 'ra':
            return '$\mathrm{RA}$'
        elif self.name == 'iota':
            return '$\iota$'
        elif self.name == 'psi':
            return '$\psi$'
        elif self.name == 'phase':
            return '$\phi$'
        elif self.name == 'tc':
            return '$t_c$'
        elif self.name == 'geocent_time':
            return '$t_c$'
        else:
            return self.name


class Uniform(Prior):
    """Uniform prior"""

    def __init__(self, minimum, maximum, name=None, latex_label=None):
        Prior.__init__(self, name, latex_label)
        self.minimum = minimum
        self.maximum = maximum
        self.support = maximum - minimum

    def rescale(self, val):
        Prior.test_valid_for_rescaling(val)
        return self.minimum + val * self.support

    def prob(self, val):
        """Return the prior probability of val"""
        if (self.minimum < val) and (val < self.maximum):
            return 1 / self.support
        else:
            return 0


class DeltaFunction(Prior):
    """Dirac delta function prior, this always returns peak."""

    def __init__(self, peak, name=None, latex_label=None):
        Prior.__init__(self, name, latex_label)
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


class PowerLaw(Prior):
    """Power law prior distribution"""

    def __init__(self, alpha, minimum, maximum, name=None, latex_label=None):
        """Power law with bounds and alpha, spectral index"""
        Prior.__init__(self, name, latex_label)
        self.alpha = alpha
        self.minimum = minimum
        self.maximum = maximum

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
        if (val > self.minimum) and (val < self.maximum):
            if self.alpha == -1:
                return 1 / val / np.log(self.maximum / self.minimum)
            else:
                return val ** self.alpha * (1 + self.alpha) / (self.maximum ** (1 + self.alpha)
                                                               - self.minimum ** (1 + self.alpha))
        else:
            return 0


class Cosine(Prior):

    def __init__(self, name=None, latex_label=None):
        Prior.__init__(self, name, latex_label)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to a uniform in cosine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        return np.arcsin(-1 + val * 2)

    @staticmethod
    def prob(val):
        """Return the prior probability of val, defined over [-pi/2, pi/2]"""
        if (val > -np.pi / 2) and (val < np.pi / 2):
            return np.cos(val) / 2
        else:
            return 0


class Sine(Prior):

    def __init__(self, name=None, latex_label=None):
        Prior.__init__(self, name, latex_label)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to a uniform in sine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        return np.arccos(-1 + val * 2)

    @staticmethod
    def prob(val):
        """Return the prior probability of val, defined over [0, pi]"""
        if (val > 0) and (val < np.pi):
            return np.sin(val) / 2
        else:
            return 0


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
        return self.mu + erfinv(2 * val - 1) * 2**0.5 * self.sigma

    def prob(self, val):
        """Return the prior probability of val"""
        return np.exp(-(self.mu - val)**2 / (2 * self.sigma**2)) / (2 * np.pi)**0.5 / self.sigma


class TruncatedGaussian(Prior):
    """
    Truncated Gaussian prior

    https://en.wikipedia.org/wiki/Truncated_normal_distribution
    """

    def __init__(self, mu, sigma, minimum, maximum, name=None, latex_label=None):
        """Power law with bounds and alpha, spectral index"""
        Prior.__init__(self, name, latex_label)
        self.mu = mu
        self.sigma = sigma
        self.minimum = minimum
        self.maximum = maximum

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
        if (val > self.minimum) & (val < self.maximum):
            return np.exp(-(self.mu - val) ** 2 / (2 * self.sigma ** 2)) / (
                        2 * np.pi) ** 0.5 / self.sigma / self.normalisation
        else:
            return 0


class Interped(Prior):

    def __init__(self, xx, yy, minimum=None, maximum=None, name=None, latex_label=None):
        """Initialise object from arrays of x and y=p(x)"""
        Prior.__init__(self, name, latex_label)
        if minimum is None or minimum < min(xx):
            self.minimum = min(xx)
        else:
            self.minimum = minimum
        if maximum is None or maximum > max(xx):
            self.maximum = max(xx)
        else:
            self.maximum = maximum
        self.xx = xx[(xx > self.minimum) & (xx < self.maximum)]
        self.yy = yy[(xx > self.minimum) & (xx < self.maximum)]
        if np.trapz(self.yy, self.xx) != 1:
            logging.info('Supplied PDF is not normalised, normalising.')
        self.yy /= np.trapz(self.yy, self.xx)
        self.YY = cumtrapz(self.yy, self.xx, initial=0)
        self.probability_density = interp1d(x=self.xx, y=self.yy, bounds_error=False, fill_value=0)
        self.cumulative_distribution = interp1d(x=self.xx, y=self.YY, bounds_error=False, fill_value=0)
        self.inverse_cumulative_distribution = interp1d(x=self.YY, y=self.xx, bounds_error=True)

    def prob(self, val):
        """Return the prior probability of val"""
        if (val > self.minimum) & (val < self.maximum):
            return self.probability_density(val)
        else:
            return 0

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior.

        This maps to the inverse CDF. This is done using interpolation.
        """
        Prior.test_valid_for_rescaling(val)
        return self.inverse_cumulative_distribution(val)

    def __repr__(self):
        prior_name = self.__class__.__name__
        prior_args = ', '.join(
            ['{}={}'.format(key, self.__dict__[key]) for key in ['xx', 'yy', '_Prior__latex_label']])
        return "{}({})".format(prior_name, prior_args)


class FromFile(Interped):

    def __init__(self, file_name, minimum=None, maximum=None, name=None, latex_label=None):
        try:
            self.id = file_name
            if '/' not in self.id:
                self.id = '{}/tupak/prior_files/{}'.format(os.getcwd(), self.id)
            xx, yy = np.genfromtxt(self.id).T
            Interped.__init__(self, xx=xx, yy=yy, minimum=minimum, maximum=maximum, name=name, latex_label=latex_label)
        except IOError:
            logging.warning("Can't load {}.".format(self.id))
            logging.warning("Format should be:")
            logging.warning(r"x\tp(x)")

    def __repr__(self):
        prior_name = self.__class__.__name__
        prior_args = ', '.join(
            ['{}={}'.format(key, self.__dict__[key]) for key in ['id', 'minimum', 'maximum', '_Prior__latex_label']])
        return "{}({})".format(prior_name, prior_args)


class UniformComovingVolume(FromFile):

    def __init__(self, minimum=None, maximum=None, name=None, latex_label=None):
        FromFile.__init__(self, file_name='comoving.txt', minimum=minimum, maximum=maximum, name=name,
                          latex_label=latex_label)


def fix(prior, value=None):
    if value is None or np.isnan(value):
        raise ValueError("You can't fix the value to be np.nan. You need to assign it a legal value")
    prior = DeltaFunction(name=prior.name, latex_label=prior.latex_label, peak=value)
    return prior


def create_default_prior(name):
    if name == 'mass_1':
        prior = PowerLaw(name=name, alpha=0, minimum=5, maximum=100)
    elif name == 'mass_2':
        prior = PowerLaw(name=name, alpha=0, minimum=5, maximum=100)
    elif name == 'mchirp':
        prior = Uniform(name=name, minimum=5, maximum=100)
    elif name == 'q':
        prior = Uniform(name=name, minimum=0, maximum=1)
    elif name == 'a_1':
        prior = Uniform(name=name, minimum=0, maximum=0.8)
    elif name == 'a_2':
        prior = Uniform(name=name, minimum=0, maximum=0.8)
    elif name == 'tilt_1':
        prior = Sine(name=name)
    elif name == 'tilt_2':
        prior = Sine(name=name)
    elif name == 'phi_12':
        prior = Uniform(name=name, minimum=0, maximum=2 * np.pi)
    elif name == 'phi_jl':
        prior = Uniform(name=name, minimum=0, maximum=2 * np.pi)
    elif name == 'luminosity_distance':
        prior = UniformComovingVolume(minimum=1e2, maximum=5e3)
    elif name == 'dec':
        prior = Cosine(name=name)
    elif name == 'ra':
        prior = Uniform(name=name, minimum=0, maximum=2 * np.pi)
    elif name == 'iota':
        prior = Sine(name=name)
    elif name == 'psi':
        prior = Uniform(name=name, minimum=0, maximum=2 * np.pi)
    elif name == 'phase':
        prior = Uniform(name=name, minimum=0, maximum=2 * np.pi)
    else:
        prior = None
    return prior


def parse_floats_to_fixed_priors(old_parameters):
    parameters = old_parameters.copy()
    for key in parameters:
        if type(parameters[key]) is not float and type(parameters[key]) is not int \
                and type(parameters[key]) is not Prior:
            logging.info("Expected parameter " + str(key) + " to be a float or int but was "
                         + str(type(parameters[key])) + " instead. Will not be converted.")
            continue
        elif type(parameters[key]) is Prior:
            continue
        parameters[key] = DeltaFunction(name=key, latex_label=None, peak=old_parameters[key])
    return parameters


def parse_keys_to_parameters(keys):
    parameters = {}
    for key in keys:
        parameters[key] = create_default_prior(key)
    return parameters


def fill_priors(prior, waveform_generator):
    """
    Fill dictionary of priors based on required parameters for waveform generator

    Any floats in prior will be converted to delta function prior.
    Any required, non-specified parameters will use the default.
    Parameters
    ----------
    prior: dict
        dictionary of prior objects and floats
    waveform_generator: WaveformGenerator
        waveform generator to be used for inference
    """
    bad_keys = []
    for key in prior:
        if isinstance(prior[key], Prior):
            continue
        elif isinstance(prior[key], float) or isinstance(prior[key], int):
            prior[key] = DeltaFunction(prior[key])
            logging.info("{} converted to delta function prior.".format(key))
        else:
            logging.warning("{} cannot be converted to delta function prior.".format(key))
            logging.warning("If required the default prior will be used.")
            bad_keys.append(key)

    missing_keys = set(waveform_generator.parameters) - set(prior.keys())

    for missing_key in missing_keys:
        prior[missing_key] = create_default_prior(missing_key)
        if prior[missing_key] is None:
            logging.warning("No default prior found for unspecified variable {}.".format(missing_key))
            logging.warning("This variable will NOT be sampled.")
            bad_keys.append(missing_key)

    for key in bad_keys:
        prior.pop(key)


def write_priors_to_file(priors, outdir):
    """
    Write the prior distribtuion to file.

    Parameters
    ----------
    priors: dict
        priors used
    outdir: str
        output directory
    """
    if outdir[-1] != "/":
        outdir += "/"
    prior_file = outdir + "prior.txt"
    print("Writing priors to {}".format(prior_file))
    with open(prior_file, "w") as outfile:
        for key in priors:
            outfile.write("prior['{}'] = {}\n".format(key, priors[key]))
