#!/bin/python

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz


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
        'Rescale' a sample from the unit line element to the prior, does nothing.

        This maps to the inverse CDF.
        """
        return val

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
            return 'q'
        elif self.name == 'a1':
            return 'a_1'
        elif self.name == 'a2':
            return 'a_2'
        elif self.name == 'tilt1':
            return 'tilt_1'
        elif self.name == 'tilt2':
            return 'tilt_2'
        elif self.name == 'phi1':
            return '$\phi_1$'
        elif self.name == 'phi2':
            return '$\phi_2$'
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

    def __init__(self, lower, upper, name=None, latex_label=None):
        Prior.__init__(self, name, latex_label)
        self.lower = lower
        self.upper = upper
        self.support = upper - lower

    def rescale(self, val):
        return self.lower + val * self.support

    def prob(self, val):
        """Return the prior probability of val"""
        if (self.lower < val) and (val < self.upper):
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
        return self.peak * val ** 0

    def prob(self, val):
        """Return the prior probability of val"""
        if self.peak == val:
            return np.inf
        else:
            return 0


class PowerLaw(Prior):
    """Power law prior distribution"""

    def __init__(self, alpha, bounds, name=None, latex_label=None):
        """Power law with bounds and alpha, spectral index"""
        Prior.__init__(self, name, latex_label)
        self.alpha = alpha
        self.low, self.high = bounds

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the power-law prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        if self.alpha == -1:
            return self.low * np.exp(val * np.log(self.high / self.low))
        else:
            return (self.low ** (1 + self.alpha) + val *
                    (self.high ** (1 + self.alpha) - self.low ** (1 + self.alpha))) ** (1. / (1 + self.alpha))

    def prob(self, val):
        """Return the prior probability of val"""
        if (val > self.low) and (val < self.high):
            if self.alpha == -1:
                return 1 / val / np.log(self.high / self.low)
            else:
                return val ** self.alpha * (1 + self.alpha) / (self.high ** (1 + self.alpha)
                                                               - self.low ** (1 + self.alpha))
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
        return np.arcsin(-1 + val * 2)

    @staticmethod
    def prob(val):
        """Return the prior probability of val"""
        return np.cos(val) / 2


class Sine(Prior):

    def __init__(self, name=None, latex_label=None):
        Prior.__init__(self, name, latex_label)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to a uniform in sine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        return np.arccos(-1 + val * 2)

    @staticmethod
    def prob(val):
        """Return the prior probability of val"""
        return np.sin(val) / 2


class Interped(Prior):

    def __init__(self, xx, yy, name=None, latex_label=None):
        """Initialise object from arrays of x and y=p(x)"""
        Prior.__init__(self, name, latex_label)
        self.xx = xx
        self.low = min(self.xx)
        self.high = max(self.xx)
        self.yy = yy
        if np.trapz(self.yy, self.xx) != 0:
            print('Supplied PDF is not normalised, normalising.')
        self.yy /= np.trapz(self.yy, self.xx)
        self.YY = cumtrapz(self.yy, self.xx, initial=0)
        self.probability_density = interp1d(x=self.xx, y=self.yy, bounds_error=False, fill_value=min(self.yy))
        self.cumulative_distribution = interp1d(x=self.xx, y=self.YY, bounds_error=False, fill_value=0)
        self.invervse_cumulative_distribution = interp1d(x=self.YY, y=self.xx, bounds_error=False,
                                                         fill_value=(min(self.xx), max(self.xx)))

    def prob(self, val):
        """Return the prior probability of val"""
        return self.probability_density(val)

    def rescale(self, x):
        """
        'Rescale' a sample from the unit line element to the prior.

        This maps to the inverse CDF. This is done using interpolation.
        """
        return self.invervse_cumulative_distribution(x)


class FromFile(Interped):

    def __init__(self, file_name):
        try:
            self.id = file_name
            xx, yy = np.genfromtxt(file_name).T
            Interped.__init__(self, xx, yy)
        except IOError:
            print("Can't load {}.".format(file_name))
            print("Format should be:")
            print(r"x\tp(x)")


def fix(prior, value=None):
    if value is None or np.isnan(value):
        raise ValueError("You can't fix the value to be np.nan. You need to assign it a legal value")
    prior = DeltaFunction(name=prior.name,
                             latex_label=prior.latex_label,
                             peak=value)
    return prior


def create_default_prior(name):
    if name == 'mass_1':
        prior = PowerLaw(name=name, alpha=0, bounds=(5, 100))
    elif name == 'mass_2':
        prior = PowerLaw(name=name, alpha=0, bounds=(5, 100))
    elif name == 'mchirp':
        prior = PowerLaw(name=name, alpha=0, bounds=(5, 100))
    elif name == 'q':
        prior = PowerLaw(name=name, alpha=0, bounds=(0, 1))
    elif name == 'a_1':
        prior = PowerLaw(name=name, alpha=0, bounds=(0, 1))
    elif name == 'a_2':
        prior = PowerLaw(name=name, alpha=0, bounds=(0, 1))
    elif name == 'tilt_1':
        prior = Sine(name=name)
    elif name == 'tilt_2':
        prior = Sine(name=name)
    elif name == 'phi_1':
        prior = PowerLaw(name=name, alpha=0, bounds=(0, 2 * np.pi))
    elif name == 'phi_2':
        prior = PowerLaw(name=name, alpha=0, bounds=(0, 2 * np.pi))
    elif name == 'luminosity_distance':
        prior = PowerLaw(name=name, alpha=2, bounds=(1e2, 5e3))
    elif name == 'dec':
        prior = Cosine(name=name)
    elif name == 'ra':
        prior = PowerLaw(name=name, alpha=0, bounds=(0, 2 * np.pi))
    elif name == 'iota':
        prior = Sine(name=name)
    elif name == 'psi':
        prior = PowerLaw(name=name, alpha=0, bounds=(0, 2 * np.pi))
    elif name == 'phase':
        prior = PowerLaw(name=name, alpha=0, bounds=(0, 2 * np.pi))
    else:
        prior = None
    return prior


def parse_floats_to_fixed_priors(old_parameters):
    parameters = old_parameters.copy()
    for key in parameters:
        if type(parameters[key]) is not float and type(parameters[key]) is not int \
                and type(parameters[key]) is not Prior:
            print("Expected parameter " + str(key) + " to be a float or int but was " + str(type(parameters[key]))
                  + " instead. Will not be converted.")
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
