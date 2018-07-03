from __future__ import division, print_function

import inspect
import numpy as np


class Likelihood(object):

    def __init__(self, parameters=None):
        """Empty likelihood class to be subclassed by other likelihoods

        Parameters
        ----------
        parameters:
        """
        self.parameters = parameters

    def log_likelihood(self):
        """

        Returns
        -------
        float
        """
        return np.nan

    def noise_log_likelihood(self):
        """

        Returns
        -------
        float
        """
        return np.nan

    def log_likelihood_ratio(self):
        """Difference between log likelihood and noise log likelihood

        Returns
        -------
        float
        """
        return self.log_likelihood() - self.noise_log_likelihood()


class GaussianLikelihood(Likelihood):
    def __init__(self, x, y, function, sigma=None):
        """
        A general Gaussian likelihood for known or unknown noise - the model
        parameters are inferred from the arguments of function

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        sigma: None, float, array_like
            If None, the standard deviation of the noise is unknown and will be
            estimated (note: this requires a prior to be given for sigma). If
            not None, this defined the standard-deviation of the data points.
            This can either be a single float, or an array with length equal
            to that for `x` and `y`.
        """
        self.x = x
        self.y = y
        self.N = len(x)
        self.sigma = sigma
        self.function = function

        # These lines of code infer the parameters from the provided function
        parameters = inspect.getargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        self.function_keys = self.parameters.keys()
        if self.sigma is None:
            self.parameters['sigma'] = None

    def log_likelihood(self):
        sigma = self.parameters.get('sigma', self.sigma)
        model_parameters = {k: self.parameters[k] for k in self.function_keys}
        res = self.y - self.function(self.x, **model_parameters)
        return -0.5 * (np.sum((res / sigma)**2)
                       + self.N*np.log(2*np.pi*sigma**2))
