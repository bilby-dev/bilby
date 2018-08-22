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


class Analytical1DLikelihood(Likelihood):

    def __init__(self, x, y, func):
        parameters = self._infer_parameters_from_function(func)
        Likelihood.__init__(self, dict.fromkeys(parameters))
        self.x = x
        self.y = y
        self.__func = func
        self.__function_keys = list(self.parameters.keys())

    @property
    def func(self):
        return self.__func

    @property
    def function(self):
        """Alias"""
        return self.__func

    @property
    def model_parameters(self):
        # This sets up the function only parameters (i.e. not sigma for the GaussianLikelihood)
        return {k: self.parameters[k] for k in self.function_keys}

    @property
    def function_keys(self):
        return self.__function_keys

    @staticmethod
    def _infer_parameters_from_function(func):
        """ Infers the arguments of function (except the first arg which is
            assumed to be the dep. variable)
        """
        parameters = inspect.getargspec(func).args
        parameters.pop(0)
        return parameters

    @property
    def n(self):
        """ The number of data points """
        return len(self.x)


class GaussianLikelihood(Analytical1DLikelihood):
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

        Analytical1DLikelihood.__init__(self, x=x, y=y, func=function)
        self.sigma = sigma

        # Check if sigma was provided, if not it is a parameter
        if self.sigma is None:
            self.parameters['sigma'] = None

    def log_likelihood(self):
        # This checks if sigma has been set in parameters. If so, that value
        # will be used. Otherwise, the attribute sigma is used. The logic is
        # that if sigma is not in parameters the attribute is used which was
        # given at init (i.e. the known sigma as either a float or array).
        sigma = self.parameters.get('sigma', self.sigma)

        # Calculate the residual
        res = self.y - self.func(self.x, **self.model_parameters)

        # Return the summed log likelihood
        return -0.5 * (np.sum((res / sigma) ** 2)
                       + self.n * np.log(2 * np.pi * sigma ** 2))


class PoissonLikelihood(Analytical1DLikelihood):
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

        Analytical1DLikelihood.__init__(self, x=x, y=y, func=func)

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

    def log_likelihood(self):
        # Calculate the rate
        rate = self.func(self.x, **self.model_parameters)

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
                return (-self.n * rate + np.sum(self.y * np.log(rate))
                        - sumlogfactorial)
        elif isinstance(rate, np.ndarray):
            # check rates are positive
            if np.any(rate < 0.):
                raise ValueError(("Poisson rate function returns a negative",
                                  " value!"))

            if np.any(rate == 0.):
                return -np.inf
            else:
                return (np.sum(-rate + self.counts * np.log(rate))
                        - sumlogfactorial)
        else:
            raise ValueError("Poisson rate function returns wrong value type!")


class ExponentialLikelihood(Analytical1DLikelihood):
    def __init__(self, x, y, func):
        """
        An exponential likelihood function.

        Parameters
        ----------

        x, y: array_like
            The data to analyse
        func:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given). The model should return the expected mean of
            the exponential distribution for each data point.
        """
        Analytical1DLikelihood.__init__(self, x=x, y=y, func=func)

        # check for non-negative values
        if np.any(self.y < 0):
            raise ValueError("Data must be non-negative")

    def log_likelihood(self):
        # Calculate the mean of the distribution
        mu = self.func(self.x, **self.model_parameters)

        # return -inf if any mean values are negative
        if np.any(mu < 0.):
            return -np.inf

        # Return the summed log likelihood
        return -np.sum(np.log(mu) + (self.y / mu))


class StudentTLikelihood(Analytical1DLikelihood):
    def __init__(self, x, y, func, nu=None, sigma=1.):
        """
        A general Student's t-likelihood for known or unknown number of degrees
        of freedom, and known or unknown scale (which tends toward the standard
        deviation for large numbers of degrees of freedom) - the model
        parameters are inferred from the arguments of function

        https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        func:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        nu: None, float
            If None, the number of degrees of freedom of the noise is unknown
            and will be estimated (note: this requires a prior to be given for
            nu). If not None, this defines the number of degrees of freedom of
            the data points. As an example a `nu` of `len(x)-2` is equivalent
            to having marginalised a Gaussian distribution over an unknown
            standard deviation parameter using a uniform prior.
        sigma: 1.0, float
            Set the scale of the distribution. If not given then this defaults
            to 1, which specifies a standard (central) Student's t-distribution
        """
        Analytical1DLikelihood.__init__(self, x=x, y=y, func=func)

        self.nu = nu
        self.sigma = sigma

        # Check if nu was provided, if not it is a parameter
        if self.nu is None:
            self.parameters['nu'] = None

    def log_likelihood(self):
        # This checks if nu or sigma have been set in parameters. If so, those
        # values will be used. Otherwise, the attribute sigma is used. The logic is
        # that if nu is not in parameters the attribute is used which was
        # given at init (i.e. the known nu as a float).
        nu = self.parameters.get('nu', self.nu)

        if nu <= 0.:
            raise ValueError("Number of degrees of freedom for Student's t-likelihood must be positive")

        # Calculate the residual
        res = self.y - self.func(self.x, **self.model_parameters)

        # convert "scale" to "precision"
        lam = 1. / self.sigma ** 2

        # Return the summed log likelihood
        return (self.n * (gammaln((nu + 1.0) / 2.0)
                          + .5 * np.log(lam / (nu * np.pi))
                          - gammaln(nu / 2.0))
                - (nu + 1.0) / 2.0 * np.sum(np.log1p(lam * res ** 2 / nu)))
