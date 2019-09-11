from __future__ import division, print_function
import copy

import numpy as np
from scipy.special import gammaln

from .utils import infer_parameters_from_function


class Likelihood(object):

    def __init__(self, parameters=None):
        """Empty likelihood class to be subclassed by other likelihoods

        Parameters
        ----------
        parameters: dict
            A dictionary of the parameter names and associated values
        """
        self.parameters = parameters
        self._meta_data = None

    def __repr__(self):
        return self.__class__.__name__ + '(parameters={})'.format(self.parameters)

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

    @property
    def meta_data(self):
        return getattr(self, '_meta_data', None)

    @meta_data.setter
    def meta_data(self, meta_data):
        if isinstance(meta_data, dict):
            self._meta_data = meta_data
        else:
            raise ValueError("The meta_data must be an instance of dict")


class ZeroLikelihood(Likelihood):
    """ A special test-only class which already returns zero likelihood

    Parameters
    ----------
    likelihood: bilby.core.likelihood.Likelihood
        A likelihood object to mimic

    """

    def __init__(self, likelihood):
        Likelihood.__init__(self, dict.fromkeys(likelihood.parameters))
        self.parameters = likelihood.parameters
        self._parent = likelihood

    def log_likelihood(self):
        return 0

    def noise_log_likelihood(self):
        return 0

    def __getattr__(self, name):
        return getattr(self._parent, name)


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
        self._func = func
        self._function_keys = list(self.parameters.keys())

    def __repr__(self):
        return self.__class__.__name__ + '(x={}, y={}, func={})'.format(self.x, self.y, self.func.__name__)

    @property
    def func(self):
        """ Make func read-only """
        return self._func

    @property
    def model_parameters(self):
        """ This sets up the function only parameters (i.e. not sigma for the GaussianLikelihood) """
        return {key: self.parameters[key] for key in self.function_keys}

    @property
    def function_keys(self):
        """ Makes function_keys read_only """
        return self._function_keys

    @property
    def n(self):
        """ The number of data points """
        return len(self.x)

    @property
    def x(self):
        """ The independent variable. Setter assures that single numbers will be converted to arrays internally """
        return self._x

    @x.setter
    def x(self, x):
        if isinstance(x, int) or isinstance(x, float):
            x = np.array([x])
        self._x = x

    @property
    def y(self):
        """ The dependent variable. Setter assures that single numbers will be converted to arrays internally """
        return self._y

    @y.setter
    def y(self, y):
        if isinstance(y, int) or isinstance(y, float):
            y = np.array([y])
        self._y = y

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
        log_l = np.sum(- (self.residual / self.sigma)**2 / 2 -
                       np.log(2 * np.pi * self.sigma**2) / 2)
        return log_l

    def __repr__(self):
        return self.__class__.__name__ + '(x={}, y={}, func={}, sigma={})' \
            .format(self.x, self.y, self.func.__name__, self.sigma)

    @property
    def sigma(self):
        """
        This checks if sigma has been set in parameters. If so, that value
        will be used. Otherwise, the attribute sigma is used. The logic is
        that if sigma is not in parameters the attribute is used which was
        given at init (i.e. the known sigma as either a float or array).
        """
        return self.parameters.get('sigma', self._sigma)

    @sigma.setter
    def sigma(self, sigma):
        if sigma is None:
            self._sigma = sigma
        elif isinstance(sigma, float) or isinstance(sigma, int):
            self._sigma = sigma
        elif len(sigma) == self.n:
            self._sigma = sigma
        else:
            raise ValueError('Sigma must be either float or array-like x.')


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

    def log_likelihood(self):
        rate = self.func(self.x, **self.model_parameters)
        if not isinstance(rate, np.ndarray):
            raise ValueError(
                "Poisson rate function returns wrong value type! "
                "Is {} when it should be numpy.ndarray".format(type(rate)))
        elif np.any(rate < 0.):
            raise ValueError(("Poisson rate function returns a negative",
                              " value!"))
        elif np.any(rate == 0.):
            return -np.inf
        else:
            return np.sum(-rate + self.y * np.log(rate) - gammaln(self.y + 1))

    def __repr__(self):
        return Analytical1DLikelihood.__repr__(self)

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

    def log_likelihood(self):
        mu = self.func(self.x, **self.model_parameters)
        if np.any(mu < 0.):
            return -np.inf
        return -np.sum(np.log(mu) + (self.y / mu))

    def __repr__(self):
        return Analytical1DLikelihood.__repr__(self)

    @property
    def y(self):
        """ Property assures that y-value is positive. """
        return self._y

    @y.setter
    def y(self, y):
        if not isinstance(y, np.ndarray):
            y = np.array([y])
        if np.any(y < 0):
            raise ValueError("Data must be non-negative")
        self._y = y


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
        if self.nu <= 0.:
            raise ValueError("Number of degrees of freedom for Student's "
                             "t-likelihood must be positive")

        nu = self.nu
        log_l =\
            np.sum(- (nu + 1) * np.log1p(self.lam * self.residual**2 / nu) / 2 +
                   np.log(self.lam / (nu * np.pi)) / 2 +
                   gammaln((nu + 1) / 2) - gammaln(nu / 2))
        return log_l

    def __repr__(self):
        base_string = '(x={}, y={}, func={}, nu={}, sigma={})'
        return self.__class__.__name__ + base_string.format(
            self.x, self.y, self.func.__name__, self.nu, self.sigma)

    @property
    def lam(self):
        """ Converts 'scale' to 'precision' """
        return 1. / self.sigma ** 2

    @property
    def nu(self):
        """ This checks if nu or sigma have been set in parameters. If so, those
        values will be used. Otherwise, the attribute nu is used. The logic is
        that if nu is not in parameters the attribute is used which was
        given at init (i.e. the known nu as a float)."""
        return self.parameters.get('nu', self._nu)

    @nu.setter
    def nu(self, nu):
        self._nu = nu


class JointLikelihood(Likelihood):
    def __init__(self, *likelihoods):
        """
        A likelihood for combining pre-defined likelihoods.
        The parameters dict is automagically combined through parameters dicts
        of the given likelihoods. If parameters have different values have
        initially different values across different likelihoods, the value
        of the last given likelihood is chosen. This does not matter when
        using the JointLikelihood for sampling, because the parameters will be
        set consistently

        Parameters
        ----------
        *likelihoods: bilby.core.likelihood.Likelihood
            likelihoods to be combined parsed as arguments
        """
        self.likelihoods = likelihoods
        Likelihood.__init__(self, parameters={})
        self.__sync_parameters()

    def __sync_parameters(self):
        """ Synchronizes parameters between the likelihoods
        so that all likelihoods share a single parameter dict."""
        for likelihood in self.likelihoods:
            self.parameters.update(likelihood.parameters)
        for likelihood in self.likelihoods:
            likelihood.parameters = self.parameters

    @property
    def likelihoods(self):
        """ The list of likelihoods """
        return self._likelihoods

    @likelihoods.setter
    def likelihoods(self, likelihoods):
        likelihoods = copy.deepcopy(likelihoods)
        if isinstance(likelihoods, tuple) or isinstance(likelihoods, list):
            if all(isinstance(likelihood, Likelihood) for likelihood in likelihoods):
                self._likelihoods = list(likelihoods)
            else:
                raise ValueError('Try setting the JointLikelihood like this\n'
                                 'JointLikelihood(first_likelihood, second_likelihood, ...)')
        elif isinstance(likelihoods, Likelihood):
            self._likelihoods = [likelihoods]
        else:
            raise ValueError('Input likelihood is not a list of tuple. You need to set multiple likelihoods.')

    def log_likelihood(self):
        """ This is just the sum of the log likelihoods of all parts of the joint likelihood"""
        return sum([likelihood.log_likelihood() for likelihood in self.likelihoods])

    def noise_log_likelihood(self):
        """ This is just the sum of the noise likelihoods of all parts of the joint likelihood"""
        return sum([likelihood.noise_log_likelihood() for likelihood in self.likelihoods])
