#!/bin/python

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz


class Prior(object):
    """Prior class"""

    def __init__(self, **kwargs):
        return

    def __call__(self):
        return self.sample(1)

    def sample(self, n_samples=1):
        """Draw a sample from the prior, this rescales a unit line element according to the rescaling function"""
        return self.rescale(np.random.uniform(0, 1, n_samples))

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior, does nothing.

        This maps to the inverse CDF.
        """
        return val


class DeltaFunction(Prior):
    """Dirac delta function prior, this always returns peak."""

    def __init__(self, peak):
        Prior.__init__(self)
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

    def __init__(self, alpha, bounds):
        """Power law with bounds and alpha, spectral index"""
        Prior.__init__(self)
        self.alpha = alpha
        self.low, self.high = bounds

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the power-law prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        if self.alpha == -1:
            return self.low * np.exp(val * np.log(self.low / self.high))
        else:
            return (self.low ** (1 + self.alpha) + val *
                    (self.high ** (1 + self.alpha) - self.low ** (1 + self.alpha))) ** (1. / (1 + self.alpha))

    def prob(self, val):
        """Return the prior probability of val"""
        return val ** self.alpha * (1 + self.alpha) / (self.high ** (1 + self.alpha) -
                                                       self.low ** (1 + self.alpha))


class Cosine(Prior):

    def __init__(self):
        Prior.__init__(self)

    @staticmethod
    def rescale(val):
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

    def __init__(self):
        Prior.__init__(self)

    @staticmethod
    def rescale(val):
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

    def __init__(self, xx, yy):
        """Initialise object from arrays of x and y=p(x)"""
        Prior.__init__(self)
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
