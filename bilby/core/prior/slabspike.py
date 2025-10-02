from numbers import Number
import numpy as np

from .base import Prior
from ..utils import logger


class SlabSpikePrior(Prior):

    def __init__(self, slab, spike_location=None, spike_height=0):
        """'Slab-and-spike' prior, see e.g. https://arxiv.org/abs/1812.07259
        This prior is composed of a `slab`, i.e. any common prior distribution,
        and a Dirac spike at a fixed location. This can effectively be used
        to emulate sampling in the number of dimensions (similar to reversible-
        jump MCMC).

        `SymmetricLogUniform` and `FermiDirac` are currently not supported.

        Parameters
        ==========
        slab: Prior
            Any instance of a bilby prior class. All general prior attributes
            from the slab are copied into the SlabSpikePrior.
            Note that this hasn't been tested for conditional priors.
        spike_location: float, optional
            Location of the Dirac spike. Must be between minimum and maximum
            of the slab. Defaults to the minimum of the slab
        spike_height: float, optional
            Relative weight of the spike compared to the slab. Must be
            between 0 and 1. Defaults to 0, i.e. the prior is just the slab.

        """
        self.slab = slab
        super().__init__(name=self.slab.name, latex_label=self.slab.latex_label, unit=self.slab.unit,
                         minimum=self.slab.minimum, maximum=self.slab.maximum,
                         check_range_nonzero=self.slab.check_range_nonzero, boundary=self.slab.boundary)
        self.spike_location = spike_location
        self.spike_height = spike_height
        try:
            self.inverse_cdf_below_spike = self._find_inverse_cdf_fraction_before_spike()
        except Exception as e:
            logger.warning("Disregard the following warning when running tests:\n {}".format(e))

    @property
    def spike_location(self):
        return self._spike_loc

    @spike_location.setter
    def spike_location(self, spike_loc):
        if spike_loc is None:
            spike_loc = self.minimum
        if not self.minimum <= spike_loc <= self.maximum:
            raise ValueError("Spike location {} not within prior domain ".format(spike_loc))
        self._spike_loc = spike_loc

    @property
    def spike_height(self):
        return self._spike_height

    @spike_height.setter
    def spike_height(self, spike_height):
        if 0 <= spike_height <= 1:
            self._spike_height = spike_height
        else:
            raise ValueError("Spike height must be between 0 and 1, but is {}".format(spike_height))

    @property
    def slab_fraction(self):
        """ Relative prior weight of the slab. """
        return 1 - self.spike_height

    def _find_inverse_cdf_fraction_before_spike(self):
        return float(self.slab.cdf(self.spike_location)) * self.slab_fraction

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior.

        Parameters
        ==========
        val: Union[float, int, array_like]
            A random number between 0 and 1

        Returns
        =======
        array_like: Associated prior value with input value.
        """
        original_is_number = isinstance(val, Number)
        val = np.atleast_1d(val)

        lower_indices = val < self.inverse_cdf_below_spike
        intermediate_indices = np.logical_and(
            self.inverse_cdf_below_spike <= val,
            val <= (self.inverse_cdf_below_spike + self.spike_height))
        higher_indices = val > (self.inverse_cdf_below_spike + self.spike_height)

        res = np.zeros(len(val))
        res[lower_indices] = self._contracted_rescale(val[lower_indices])
        res[intermediate_indices] = self.spike_location
        res[higher_indices] = self._contracted_rescale(val[higher_indices] - self.spike_height)
        if original_is_number:
            try:
                res = res[0]
            except (KeyError, TypeError):
                logger.warning("Based on inputs, a number should be output\
                               but this could not be accessed from what was computed")
        return res

    def _contracted_rescale(self, val):
        """
        Contracted version of the rescale function that implements the `rescale` function
        on the pure slab part of the prior.

        Parameters
        ==========
        val: Union[float, int, array_like]
            A random number between 0 and self.slab_fraction

        Returns
        =======
        array_like: Associated prior value with input value.
        """
        return self.slab.rescale(val / self.slab_fraction)

    def prob(self, val):
        """Return the prior probability of val.
        Returns np.inf for the spike location

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        array_like: Prior probability of val
        """
        original_is_number = isinstance(val, Number)
        res = self.slab.prob(val) * self.slab_fraction
        res = np.atleast_1d(res)
        res[val == self.spike_location] = np.inf
        if original_is_number:
            try:
                res = res[0]
            except (KeyError, TypeError):
                logger.warning("Based on inputs, a number should be output\
                               but this could not be accessed from what was computed")
        return res

    def ln_prob(self, val):
        """Return the Log prior probability of val.
        Returns np.inf for the spike location

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        array_like: Prior probability of val
        """
        original_is_number = isinstance(val, Number)
        res = self.slab.ln_prob(val) + np.log(self.slab_fraction)
        res = np.atleast_1d(res)
        res[val == self.spike_location] = np.inf
        if original_is_number:
            try:
                res = res[0]
            except (KeyError, TypeError):
                logger.warning("Based on inputs, a number should be output\
                               but this could not be accessed from what was computed")
        return res

    def cdf(self, val):
        """ Return the CDF of the prior.
        This calls to the slab CDF and adds a discrete step
        at the spike location.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        array_like: CDF value of val

        """
        res = self.slab.cdf(val) * self.slab_fraction
        res += self.spike_height * (val > self.spike_location)
        return res
