from __future__ import division, print_function

import numpy as np
from scipy.special import gammaln
from tupak.core.utils import infer_parameters_from_function


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
    """
    A general class for 1D analytical functions. The model
    parameters are inferred from the arguments of function

    Parameters
    ----------
    x, y: array_like
        The data to analyse
    func:
        The python function to fit to the data. Note, this must take the
        dependent variable as its first argument. The other arguments
        will require a prior and will be sampled over (unless a fixed
        value is given).
    """

    def __init__(self, x, y, func):
        parameters = infer_parameters_from_function(func)
        Likelihood.__init__(self, dict.fromkeys(parameters))
        self.x = x
        self.y = y
        self.__func = func
        self.__function_keys = list(self.parameters.keys())

    @property
    def func(self):
        """ Make func read-only """
        return self.__func

    @property
    def model_parameters(self):
        """ This sets up the function only parameters (i.e. not sigma for the GaussianLikelihood) """
        return {key: self.parameters[key] for key in self.function_keys}

    @property
    def function_keys(self):
        """ Makes function_keys read_only """
        return self.__function_keys

    @property
    def n(self):
        """ The number of data points """
        return len(self.x)

    @property
    def x(self):
        """ The independent variable. Setter assures that single numbers will be converted to arrays internally """
        return self.__x

    @x.setter
    def x(self, x):
        if isinstance(x, int) or isinstance(x, float):
            x = np.array([x])
        self.__x = x

    @property
    def y(self):
        """ The dependent variable. Setter assures that single numbers will be converted to arrays internally """
        return self.__y

    @y.setter
    def y(self, y):
        if isinstance(y, int) or isinstance(y, float):
            y = np.array([y])
        self.__y = y

    @property
    def residual(self):
        """ Residual of the function against the data. """
        return self.y - self.func(self.x, **self.model_parameters)


class GaussianLikelihood(Analytical1DLikelihood):
    def __init__(self, x, y, func, sigma=None):
        """
        A general Gaussian likelihood for known or unknown noise - the model
        parameters are inferred from the arguments of function

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        func:
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

        Analytical1DLikelihood.__init__(self, x=x, y=y, func=func)
        self.sigma = sigma

        # Check if sigma was provided, if not it is a parameter
        if self.sigma is None:
            self.parameters['sigma'] = None

    def log_likelihood(self):
        return self.__summed_log_likelihood(sigma=self.__get_sigma())

    def __get_sigma(self):
        """
        This checks if sigma has been set in parameters. If so, that value
        will be used. Otherwise, the attribute sigma is used. The logic is
        that if sigma is not in parameters the attribute is used which was
        given at init (i.e. the known sigma as either a float or array).
        """
        return self.parameters.get('sigma', self.sigma)

    def __summed_log_likelihood(self, sigma):
        return -0.5 * (np.sum((self.residual / sigma) ** 2)
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

    @property
    def y(self):
        """ Property assures that y-value is a positive integer. """
        return self.__y

    @y.setter
    def y(self, y):
        if not isinstance(y, np.ndarray):
            y = np.array([y])
        # check array is a non-negative integer array
        if y.dtype.kind not in 'ui' or np.any(y < 0):
            raise ValueError("Data must be non-negative integers")
        self.__y = y

    def log_likelihood(self):
        rate = self.func(self.x, **self.model_parameters)
        if not isinstance(rate, np.ndarray):
            raise ValueError("Poisson rate function returns wrong value type! "
                             "Is {} when it should be numpy.ndarray".format(type(rate)))
        elif np.any(rate < 0.):
            raise ValueError(("Poisson rate function returns a negative",
                              " value!"))
        elif np.any(rate == 0.):
            return -np.inf
        else:
            return np.sum(-rate + self.y * np.log(rate)) - np.sum(gammaln(self.y + 1))


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

    @property
    def y(self):
        """ Property assures that y-value is positive. """
        return self.__y

    @y.setter
    def y(self, y):
        if not isinstance(y, np.ndarray):
            y = np.array([y])
        if np.any(y < 0):
            raise ValueError("Data must be non-negative")
        self.__y = y

    def log_likelihood(self):
        mu = self.func(self.x, **self.model_parameters)
        if np.any(mu < 0.):
            return -np.inf
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

    @property
    def lam(self):
        """ Converts 'scale' to 'precision' """
        return 1. / self.sigma ** 2

    def log_likelihood(self):
        if self.__get_nu() <= 0.:
            raise ValueError("Number of degrees of freedom for Student's t-likelihood must be positive")

        return self.__summed_log_likelihood(self.__get_nu())

    def __get_nu(self):
        """ This checks if nu or sigma have been set in parameters. If so, those
        values will be used. Otherwise, the attribute nu is used. The logic is
        that if nu is not in parameters the attribute is used which was
        given at init (i.e. the known nu as a float)."""
        return self.parameters.get('nu', self.nu)

    def __summed_log_likelihood(self, nu):
        return self.n * (gammaln((nu + 1.0) / 2.0) + .5 * np.log(self.lam / (nu * np.pi)) - gammaln(nu / 2.0)) \
               - (nu + 1.0) / 2.0 * np.sum(np.log1p(self.lam * self.residual ** 2 / nu))
