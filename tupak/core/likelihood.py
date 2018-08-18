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
            assumed to be the dep. variable)
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
    def __init__(self, x, y, func):
        """
        A general Poisson likelihood for a rate - the model parameters are
        inferred from the arguments of function, which provides a rate.

        Parameters
        ----------

        x: array_like
            A dependent variable at which the Poisson rates will be calculated
        y: array_like
            The data to analyse - this must be a set of non-negative integers,
            each being the number of events within some interval.
        func:
            The python function providing the rate of events per interval to
            fit to the data. The function must be defined with the first
            argument being a dependent parameter (although this does not have
            to be used by the function if not required). The subsequent
            arguments will require priors and will be sampled over (unless a
            fixed value is given).
        """

        parameters = self._infer_parameters_from_function(func)
        Likelihood.__init__(self, dict.fromkeys(parameters))

        self.x = x           # the dependent variable
        self.y = y           # the counts

        # check values are non-negative integers
        if isinstance(self.y, int):
            # convert to numpy array if passing a single integer
            self.y = np.array([self.y])

        # check array is an integer array
        if self.y.dtype.kind not in 'ui':
            raise ValueError("Data must be non-negative integers")

        # check for non-negative integers
        if np.any(self.y < 0):
            raise ValueError("Data must be non-negative integers")

        self.function = func

        # Check if sigma was provided, if not it is a parameter
        self.function_keys = list(self.parameters.keys())

    @staticmethod
    def _infer_parameters_from_function(func):
        """ Infers the arguments of function (except the first arg which is
            assumed to be the dep. variable)
        """
        parameters = inspect.getargspec(func).args
        parameters.pop(0)
        return parameters

    @property
    def N(self):
        """ The number of data points """
        return len(self.y)

    def log_likelihood(self):
        # This sets up the function only parameters (i.e. not sigma)
        model_parameters = {k: self.parameters[k] for k in self.function_keys}

        # Calculate the rate
        rate = self.function(self.x, **model_parameters)

        # sum of log factorial of counts
        sumlogfactorial = np.sum(gammaln(self.y + 1))

        # check if rate is a single value
        if isinstance(rate, float):
            # check rate is positive
            if rate < 0.:
                raise ValueError(("Poisson rate function returns a negative ",
                                  "value!"))

            if rate == 0.:
                return -np.inf
            else:
                # Return the summed log likelihood
                return (-self.N*rate + np.sum(self.y*np.log(rate))
                        -sumlogfactorial)
        elif isinstance(rate, np.ndarray):
            # check rates are positive
            if np.any(rate < 0.):
                raise ValueError(("Poisson rate function returns a negative",
                                  " value!"))

            if np.any(rate == 0.):
                return -np.inf
            else:
                return (np.sum(-rate + self.counts*np.log(rate))
                        -sumlogfactorial)
        else:
            raise ValueError("Poisson rate function returns wrong value type!")
