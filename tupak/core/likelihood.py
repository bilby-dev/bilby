from __future__ import division, print_function

import inspect
import numpy as np
from scipy.special import gammaln

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
            not None, this defines the standard-deviation of the data points.
            This can either be a single float, or an array with length equal
            to that for `x` and `y`.
        """

        parameters = self._infer_parameters_from_function(function)
        Likelihood.__init__(self, dict.fromkeys(parameters))

        self.x = x
        self.y = y
        self.sigma = sigma
        self.function = function

        # Check if sigma was provided, if not it is a parameter
        self.function_keys = list(self.parameters.keys())
        if self.sigma is None:
            self.parameters['sigma'] = None

    @staticmethod
    def _infer_parameters_from_function(func):
        """ Infers the arguments of function (except the first arg which is
            assumed to be the dep. variable
        """
        parameters = inspect.getargspec(func).args
        parameters.pop(0)
        return parameters

    @property
    def N(self):
        """ The number of data points """
        return len(self.x)

    def log_likelihood(self):
        # This checks if sigma has been set in parameters. If so, that value
        # will be used. Otherwise, the attribute sigma is used. The logic is
        # that if sigma is not in parameters the attribute is used which was
        # given at init (i.e. the known sigma as either a float or array).
        sigma = self.parameters.get('sigma', self.sigma)

        # This sets up the function only parameters (i.e. not sigma)
        model_parameters = {k: self.parameters[k] for k in self.function_keys}

        # Calculate the residual
        res = self.y - self.function(self.x, **model_parameters)

        # Return the summed log likelihood
        return -0.5 * (np.sum((res / sigma)**2)
                       + self.N * np.log(2 * np.pi * sigma**2))


class PoissonLikelihood(Likelihood):
    def __init__(self, x, function):
        """
        A general Poisson likelihood for a rate - the model parameters are
        inferred from the arguments of function, which provides a rate.

        Note: This can only be used for fitting a single rate and not a
        changing rate through a generalised linear model (GLM).

        Parameters
        ----------
        x: array_like
            The data to analyse - this must be a set of non-negative integers,
            each being the number of events within some interval.
        function:
            The python function providing the rate of events per interval to
            fit to the data. The arguments will require priors and will be
            sampled over (unless a fixed value is given).
        """

        parameters = self._infer_parameters_from_function(function)
        Likelihood.__init__(self, dict.fromkeys(parameters))

        self.x = x

        # check values are non-negative integers
        if isinstance(self.x, int):
            # convert to numpy array if passing a single integer
            self.x = np.array(self.x)

        try:
            # use bincount to check all values are non-negative integers
            _ = np.bincount(self.x)
        except ValueError:
            raise ValueError("Data must be non-negative integers")

        # save sum of log factorial of counts
        self.sumlogfactorial = np.sum(gammaln(self.x + 1))

        self.function = function

        # Check if sigma was provided, if not it is a parameter
        self.function_keys = list(self.parameters.keys())

    @staticmethod
    def _infer_parameters_from_function(func):
        """ Infers the arguments of function (except the first arg which is
            assumed to be the dep. variable
        """
        parameters = inspect.getargspec(func).args
        parameters.pop(0)
        return parameters

    @property
    def N(self):
        """ The number of data points """
        return len(self.x)

    def log_likelihood(self):
        # This sets up the function only parameters (i.e. not sigma)
        model_parameters = {k: self.parameters[k] for k in self.function_keys}

        # Calculate the rate
        rate = self.function(**model_parameters)

        # check rate is positive
        if rate < 0.:
            raise ValueError("Poisson rate function returns a negative value!")

        # Return the summed log likelihood
        return -self.N*rate + np.sum(self.x*np.log(rate)) - self.sumlogfactorial
