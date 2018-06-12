from __future__ import division, print_function

import numpy as np

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp


class Likelihood(object):
    """ Empty likelihood class to be subclassed by other likelihoods """

    def __init__(self, parameters=None):
        """

        :param parameters:
        """
        self.parameters = parameters

    def log_likelihood(self):
        """

        :return:
        """
        return np.nan

    def noise_log_likelihood(self):
        """

        :return:
        """
        return np.nan

    def log_likelihood_ratio(self):
        """

        :return:
        """
        return self.log_likelihood() - self.noise_log_likelihood()


