#!/bin/python

import numpy as np
import scipy.interpolate as sip
import scipy.integrate as sit


class Prior(object):

    def __init__(self, **kwargs):
        return

    def __call__(self):
        return self.sample(1)

    def sample(self, Nsamples=1):
        return self.rescale(np.random.uniform(0, 1, Nsamples))


class DeltaFunction(Prior):

    def __init__(self, peak):
        Prior.__init__(self)
        self.peak = peak

    def rescale(self, val):
        return self.peak * val ** 0

    def prob(self, val):
        if self.peak == val:
            return np.inf
        else:
            return 0


class PowerLaw(Prior):

    def __init__(self, alpha, bounds):
        Prior.__init__(self)
        self.alpha = alpha
        self.low, self.high = bounds

    def rescale(self, val):
        if self.alpha == -1:
            return self.low * np.exp(val * np.log(self.low / self.high))
        else:
            return (self.low ** (1 + self.alpha) + val *
                    (self.high ** (1 + self.alpha) - self.low ** (1 + self.alpha))) ** (1. / (1 + self.alpha))

    def prob(self, val):
        return val ** self.alpha * (1 + self.alpha) / (self.high ** (1 + self.alpha) -
                                                       self.low ** (1 + self.alpha))


class Cosine(Prior):

    def __init__(self):
        Prior.__init__(self)

    def rescale(self, val):
        return np.arcsin(-1 + val * 2)

    def prob(self, val):
        return np.cos(val) / 2


class Sine(Prior):

    def __init__(self):
        Prior.__init__(self)

    def rescale(self, val):
        return np.arccos(-1 + val * 2)

    def prob(self, val):
        return np.sin(val) / 2


class Interped(Prior):

    def __init__(self, xx, yy):
        Prior.__init__(self)
        self.xx = xx
        self.low = min(self.xx)
        self.high = max(self.xx)
        self.yy = yy
        if np.trapz(self.yy, self.xx) != 0:
            print('Supplied PDF is not normalised, normalising.')
        self.yy /= np.trapz(self.yy, self.xx)
        self.YY = sit.cumtrapz(self.yy, self.xx, initial=0)
        self.interpolate()

    def interpolate(self):
        self.PDF = sip.interp1d(x=self.xx, y=self.yy, bounds_error=False, fill_value=min(self.yy))
        self.CDF = sip.interp1d(x=self.xx, y=self.YY, bounds_error=False, fill_value=0)
        self.inv_CDF = sip.interp1d(x=self.YY, y=self.xx, bounds_error=False, fill_value=(min(self.xx), max(self.xx)))

    def rescale(self, x):
        return self.inv_CDF(x)


class FromFile(Interped):

    def __init__(self, fID):
        try:
            self.id = fID
            xx, yy = np.genfromtxt(fID).T
            Interped.__init__(self, xx, yy)
        except:
            print("Can't load {}.".format(fID))
            print("Format should be:")
            print("x\tp(x)")
        return
