import numpy as np
from scipy.special import logsumexp


class SamplesSummary(object):
    """ Object to store a set of samples and calculate summary statistics

    Parameters
    ==========
    samples: array_like
        Array of samples
    average: str {'median', 'mean'}
        Use either a median average or mean average when calculating relative
        uncertainties
    level: float
        The default confidence interval level, defaults t0 0.9

    """
    def __init__(self, samples, average='median', confidence_level=.9):
        self.samples = samples
        self.average = average
        self.confidence_level = confidence_level

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        self._samples = samples

    @property
    def confidence_level(self):
        return self._confidence_level

    @confidence_level.setter
    def confidence_level(self, confidence_level):
        if 0 < confidence_level and confidence_level < 1:
            self._confidence_level = confidence_level
        else:
            raise ValueError("Confidence level must be between 0 and 1")

    @property
    def average(self):
        if self._average == 'mean':
            return self.mean
        elif self._average == 'median':
            return self.median

    @average.setter
    def average(self, average):
        allowed_averages = ['mean', 'median']
        if average in allowed_averages:
            self._average = average
        else:
            raise ValueError("Average {} not in allowed averages".format(average))

    @property
    def median(self):
        return np.median(self.samples, axis=0)

    @property
    def mean(self):
        return np.mean(self.samples, axis=0)

    @property
    def _lower_level(self):
        """ The credible interval lower quantile value """
        return (1 - self.confidence_level) / 2.

    @property
    def _upper_level(self):
        """ The credible interval upper quantile value """
        return (1 + self.confidence_level) / 2.

    @property
    def lower_absolute_credible_interval(self):
        """ Absolute lower value of the credible interval """
        return np.quantile(self.samples, self._lower_level, axis=0)

    @property
    def upper_absolute_credible_interval(self):
        """ Absolute upper value of the credible interval """
        return np.quantile(self.samples, self._upper_level, axis=0)

    @property
    def lower_relative_credible_interval(self):
        """ Relative (to average) lower value of the credible interval """
        return self.lower_absolute_credible_interval - self.average

    @property
    def upper_relative_credible_interval(self):
        """ Relative (to average) upper value of the credible interval """
        return self.upper_absolute_credible_interval - self.average


def kish_log_effective_sample_size(ln_weights):
    """ Calculate the Kish effective sample size from the natural-log weights

    See https://en.wikipedia.org/wiki/Effective_sample_size for details

    Parameters
    ==========
    ln_weights: array
        An array of the ln-weights

    Returns
    =======
    ln_n_eff:
        The natural-log of the effective sample size

    """
    return 2 * logsumexp(ln_weights) - logsumexp(2 * ln_weights)


def reflect(u):
    """
    Iteratively reflect a number until it is contained in [0, 1].

    This is for priors with a reflective boundary condition, all numbers in the
    set `u = 2n +/- x` should be mapped to x.

    For the `+` case we just take `u % 1`.
    For the `-` case we take `1 - (u % 1)`.

    E.g., -0.9, 1.1, and 2.9 should all map to 0.9.

    Parameters
    ==========
    u: array-like
        The array of points to map to the unit cube

    Returns
    =======
    u: array-like
       The input array, modified in place.
    """
    idxs_even = np.mod(u, 2) < 1
    u[idxs_even] = np.mod(u[idxs_even], 1)
    u[~idxs_even] = 1 - np.mod(u[~idxs_even], 1)
    return u
