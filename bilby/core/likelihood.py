import copy
import inspect
import os
from warnings import warn

import numpy as np
from scipy.special import gammaln, xlogy
from scipy.stats import multivariate_normal

from .utils import infer_parameters_from_function, infer_args_from_function_except_n_args, logger

PARAMETERS_AS_STATE = os.environ.get("BILBY_ALLOW_PARAMETERS_AS_STATE", "TRUE")


def set_parameters_as_state(level):
    global PARAMETERS_AS_STATE

    level = str(level).upper()
    if level not in ["TRUE", "FALSE", "WARN"]:
        raise ValueError("Allowed levels for parameters as state are 'TRUE', 'FALSE', and 'WARN'")
    PARAMETERS_AS_STATE = level


def _fallback_to_parameters(obj, parameters):

    if parameters is None:
        msg = "No parameters provided in likelihood call, falling back to values stored in {obj}"
        if PARAMETERS_AS_STATE == "FALSE":
            raise LikelihoodParameterError(msg)
        elif PARAMETERS_AS_STATE == "WARN":
            warn(msg, FutureWarning)
        else:
            logger.debug(msg)
        parameters = copy.deepcopy(obj.parameters)

    return parameters


def _safe_likelihood_call(likelihood, parameters=None, use_ratio=False):
    if use_ratio:
        method = likelihood.log_likelihood_ratio
    else:
        method = likelihood.log_likelihood

    if "parameters" in inspect.signature(method).parameters:
        logl = method(parameters=parameters)
    else:
        if PARAMETERS_AS_STATE == "FALSE":
            raise LikelihoodParameterError(
                f"Unable to call {likelihood} with {parameters} as an argument"
            )
        elif PARAMETERS_AS_STATE == "WARN":
            warn(f"Using parameters as state for {likelihood}", FutureWarning)
        likelihood.parameters.update(parameters)
        logl = method()
    return logl


class Likelihood:

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if (
            "parameters" not in inspect.signature(cls.log_likelihood).parameters
            or "parameters" not in inspect.signature(cls.log_likelihood_ratio).parameters
        ):
            warn(
                f"{cls} log_likelihood or log_likelihood_ratio method does not "
                "accept 'parameters' as an argument. This is deprecated behaviour "
                "and will be removed in a future release. See "
                "https://bilby-dev.github.io/bilby/parameters for more details.",
                DeprecationWarning,
            )

    def __init__(self, parameters=None):
        """Empty likelihood class to be subclassed by other likelihoods

        Parameters
        ==========
        parameters: dict
            A dictionary of the parameter names and associated values
        """
        self.parameters = parameters
        self._meta_data = None
        self._marginalized_parameters = []

    @property
    def parameters(self):
        msg = f"Parameter attribute queried for {self.__class__}"
        if PARAMETERS_AS_STATE == "FALSE":
            raise LikelihoodParameterError(msg)
        elif PARAMETERS_AS_STATE == "WARN":
            warn(msg, DeprecationWarning)
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if parameters is not None:
            msg = f"Setting non-trivial parameters for {self.__class__}"
            if PARAMETERS_AS_STATE == "FALSE":
                raise LikelihoodParameterError(msg)
            elif PARAMETERS_AS_STATE == "WARN":
                warn(msg, FutureWarning)
        else:
            parameters = dict()
        self._parameters = parameters

    def __repr__(self):
        return self.__class__.__name__

    def log_likelihood(self, parameters=None):
        """

        Returns
        =======
        float
        """
        return np.nan

    def noise_log_likelihood(self):
        """

        Returns
        =======
        float
        """
        return np.nan

    def log_likelihood_ratio(self, parameters=None):
        """Difference between log likelihood and noise log likelihood

        Returns
        =======
        float
        """
        try:
            return self.log_likelihood(parameters=parameters) - self.noise_log_likelihood()
        except TypeError:
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

    @property
    def marginalized_parameters(self):
        return self._marginalized_parameters


class ZeroLikelihood(Likelihood):
    """ A special test-only class which already returns zero likelihood

    Parameters
    ==========
    likelihood: bilby.core.likelihood.Likelihood
        A likelihood object to mimic

    """

    def __init__(self, likelihood):
        super(ZeroLikelihood, self).__init__()
        self.parameters = likelihood.parameters
        self._parent = likelihood

    def log_likelihood(self, parameters=None):
        return 0

    def noise_log_likelihood(self, parameters=None):
        return 0

    def __getattr__(self, name):
        return getattr(self._parent, name)


class Analytical1DLikelihood(Likelihood):
    """
    A general class for 1D analytical functions. The model
    parameters are inferred from the arguments of function

    Parameters
    ==========
    x, y: array_like
        The data to analyse
    func:
        The python function to fit to the data. Note, this must take the
        dependent variable as its first argument. The other arguments
        will require a prior and will be sampled over (unless a fixed
        value is given).
    """

    def __init__(self, x, y, func, **kwargs):
        parameters = infer_parameters_from_function(func)
        super(Analytical1DLikelihood, self).__init__()
        self.x = x
        self.y = y
        self._func = func
        self._function_keys = [key for key in parameters if key not in kwargs]
        self.kwargs = kwargs

    def __repr__(self):
        return self.__class__.__name__ + '(x={}, y={}, func={})'.format(self.x, self.y, self.func.__name__)

    @property
    def func(self):
        """ Make func read-only """
        return self._func

    def model_parameters(self, parameters=None):
        """ This sets up the function only parameters (i.e. not sigma for the GaussianLikelihood) """
        parameters = _fallback_to_parameters(self, parameters)
        return {key: parameters[key] for key in self.function_keys}

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

    def residual(self, parameters=None):
        """ Residual of the function against the data. """
        return self.y - self.func(self.x, **self.model_parameters(parameters=parameters), **self.kwargs)


class GaussianLikelihood(Analytical1DLikelihood):
    def __init__(self, x, y, func, sigma=None, **kwargs):
        """
        A general Gaussian likelihood for known or unknown noise - the model
        parameters are inferred from the arguments of function

        Parameters
        ==========
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

        super(GaussianLikelihood, self).__init__(x=x, y=y, func=func, **kwargs)
        self.sigma = sigma

    def log_likelihood(self, parameters=None):
        parameters = _fallback_to_parameters(self, parameters)
        sigma = parameters.get("sigma", self.sigma)
        log_l = np.sum(- (self.residual(parameters) / sigma)**2 / 2 -
                       np.log(2 * np.pi * sigma**2) / 2)
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
        if PARAMETERS_AS_STATE == "FALSE":
            return self._sigma
        else:
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
    def __init__(self, x, y, func, **kwargs):
        """
        A general Poisson likelihood for a rate - the model parameters are
        inferred from the arguments of function, which provides a rate.

        Parameters
        ==========

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

        super(PoissonLikelihood, self).__init__(x=x, y=y, func=func, **kwargs)

    def log_likelihood(self, parameters=None):
        rate = self.func(self.x, **self.model_parameters(parameters=parameters), **self.kwargs)
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
    def __init__(self, x, y, func, **kwargs):
        """
        An exponential likelihood function.

        Parameters
        ==========

        x, y: array_like
            The data to analyse
        func:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given). The model should return the expected mean of
            the exponential distribution for each data point.
        """
        super(ExponentialLikelihood, self).__init__(x=x, y=y, func=func, **kwargs)

    def log_likelihood(self, parameters=None):
        mu = self.func(self.x, **self.model_parameters(parameters=parameters), **self.kwargs)
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
    def __init__(self, x, y, func, nu=None, sigma=1, **kwargs):
        """
        A general Student's t-likelihood for known or unknown number of degrees
        of freedom, and known or unknown scale (which tends toward the standard
        deviation for large numbers of degrees of freedom) - the model
        parameters are inferred from the arguments of function

        https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution

        Parameters
        ==========
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
        super(StudentTLikelihood, self).__init__(x=x, y=y, func=func, **kwargs)

        self.nu = nu
        self.sigma = sigma

    def log_likelihood(self, parameters=None):
        parameters = _fallback_to_parameters(self, parameters)
        nu = parameters.get("nu", self.nu)
        if nu <= 0.:
            raise ValueError("Number of degrees of freedom for Student's "
                             "t-likelihood must be positive")

        log_l =\
            np.sum(- (nu + 1) * np.log1p(self.lam * self.residual(parameters=parameters)**2 / nu) / 2 +
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
        if PARAMETERS_AS_STATE == "FALSE":
            return self._nu
        else:
            return self.parameters.get('nu', self._nu)

    @nu.setter
    def nu(self, nu):
        self._nu = nu


class Multinomial(Likelihood):
    """
    Likelihood for system with N discrete possibilities.
    """

    def __init__(self, data, n_dimensions, base="parameter_"):
        """

        Parameters
        ==========
        data: array-like
            The number of objects in each class
        n_dimensions: int
            The number of classes
        base: str
            The base of the parameter labels
        """
        self.data = np.array(data)
        self._total = np.sum(self.data)
        super(Multinomial, self).__init__()
        self.n = n_dimensions
        self.base = base
        self._nll = None

    def log_likelihood(self, parameters=None):
        """
        Since n - 1 parameters are sampled, the last parameter is 1 - the rest
        """
        parameters = _fallback_to_parameters(self, parameters)
        probs = [parameters[self.base + str(ii)]
                 for ii in range(self.n - 1)]
        probs.append(1 - sum(probs))
        return self._multinomial_ln_pdf(probs=probs)

    def noise_log_likelihood(self):
        """
        Our null hypothesis is that all bins have probability 1 / nbins, i.e.,
        no bin is preferred over any other.
        """
        if self._nll is None:
            self._nll = self._multinomial_ln_pdf(probs=1 / self.n)
        return self._nll

    def _multinomial_ln_pdf(self, probs):
        """Lifted from scipy.stats.multinomial._logpdf"""
        ln_prob = gammaln(self._total + 1) + np.sum(
            xlogy(self.data, probs) - gammaln(self.data + 1), axis=-1)
        return ln_prob


class AnalyticalMultidimensionalCovariantGaussian(Likelihood):
    """
        A multivariate Gaussian likelihood
        with known analytic solution.

        Parameters
        ==========
        mean: array_like
            Array with the mean values of distribution
        cov: array_like
            The ndim*ndim covariance matrix
        """

    def __init__(self, mean, cov):
        self.cov = np.atleast_2d(cov)
        self.mean = np.atleast_1d(mean)
        self.sigma = np.sqrt(np.diag(self.cov))
        self.pdf = multivariate_normal(mean=self.mean, cov=self.cov)
        super(AnalyticalMultidimensionalCovariantGaussian, self).__init__()

    @property
    def dim(self):
        return len(self.cov[0])

    def log_likelihood(self, parameters=None):
        parameters = _fallback_to_parameters(self, parameters)
        x = np.array([parameters["x{0}".format(i)] for i in range(self.dim)])
        return self.pdf.logpdf(x)


class AnalyticalMultidimensionalBimodalCovariantGaussian(Likelihood):
    """
        A multivariate Gaussian likelihood
        with known analytic solution.

        Parameters
        ==========
        mean_1: array_like
            Array with the mean value of the first mode
        mean_2: array_like
            Array with the mean value of the second mode
        cov: array_like
        """

    def __init__(self, mean_1, mean_2, cov):
        self.cov = np.atleast_2d(cov)
        self.sigma = np.sqrt(np.diag(self.cov))
        self.mean_1 = np.atleast_1d(mean_1)
        self.mean_2 = np.atleast_1d(mean_2)
        self.pdf_1 = multivariate_normal(mean=self.mean_1, cov=self.cov)
        self.pdf_2 = multivariate_normal(mean=self.mean_2, cov=self.cov)
        super(AnalyticalMultidimensionalBimodalCovariantGaussian, self).__init__()

    @property
    def dim(self):
        return len(self.cov[0])

    def log_likelihood(self, parameters=None):
        parameters = _fallback_to_parameters(self, parameters)
        x = np.array([parameters["x{0}".format(i)] for i in range(self.dim)])
        return -np.log(2) + np.logaddexp(self.pdf_1.logpdf(x), self.pdf_2.logpdf(x))


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
        ==========
        *likelihoods: bilby.core.likelihood.Likelihood
            likelihoods to be combined parsed as arguments
        """
        self.likelihoods = likelihoods
        super(JointLikelihood, self).__init__(parameters={})
        self.__sync_parameters()

    def __sync_parameters(self):
        """ Synchronizes parameters between the likelihoods
        so that all likelihoods share a single parameter dict."""
        if PARAMETERS_AS_STATE != "FALSE":
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

    def log_likelihood(self, parameters=None):
        """ This is just the sum of the log likelihoods of all parts of the joint likelihood"""
        return sum([likelihood.log_likelihood(parameters=parameters) for likelihood in self.likelihoods])

    def noise_log_likelihood(self, parameters=None):
        """ This is just the sum of the noise likelihoods of all parts of the joint likelihood"""
        return sum([likelihood.noise_log_likelihood(parameters=parameters) for likelihood in self.likelihoods])


def function_to_celerite_mean_model(func):
    from celerite.modeling import Model as CeleriteModel
    return _function_to_gp_model(func, CeleriteModel)


def function_to_george_mean_model(func):
    from celerite.modeling import Model as GeorgeModel
    return _function_to_gp_model(func, GeorgeModel)


def _function_to_gp_model(func, cls):
    class MeanModel(cls):
        parameter_names = tuple(infer_args_from_function_except_n_args(func=func, n=1))

        def get_value(self, t):
            params = {name: getattr(self, name) for name in self.parameter_names}
            return func(t, **params)

        def compute_gradient(self, *args, **kwargs):
            pass

    return MeanModel


class _GPLikelihood(Likelihood):

    def __init__(self, kernel, mean_model, t, y, yerr=1e-6, gp_class=None):
        """
            Basic Gaussian Process likelihood interface for `celerite` and `george`.
            For `celerite` documentation see: https://celerite.readthedocs.io/en/stable/
            For `george` documentation see: https://george.readthedocs.io/en/latest/

            Parameters
            ==========
            kernel: Union[celerite.term.Term, george.kernels.Kernel]
                `celerite` or `george` kernel. See the respective package documentation about the usage.
            mean_model: Union[celerite.modeling.Model, george.modeling.Model]
                Mean model
            t: array_like
                The `times` or `x` values of the data set.
            y: array_like
                The `y` values of the data set.
            yerr: float, int, array_like, optional
                The error values on the y-values. If a single value is given, it is assumed that the value
                applies for all y-values. Default is 1e-6, effectively assuming that no y-errors are present.
            gp_class: type, None, optional
                GPClass to use. This is determined by the child class used to instantiate the GP. Should usually
                not be given by the user and is mostly used for testing
        """
        self.kernel = kernel
        self.mean_model = mean_model
        self.t = np.array(t)
        self.y = np.array(y)
        self.yerr = np.array(yerr)
        self.GPClass = gp_class
        self.gp = self.GPClass(kernel=self.kernel, mean=self.mean_model, fit_mean=True, fit_white_noise=True)
        self.gp.compute(self.t, yerr=self.yerr)
        if PARAMETERS_AS_STATE == "FALSE":
            parameters = None
        else:
            parameters = self.gp.get_parameter_dict()
        super().__init__(parameters=parameters)

    def set_parameters(self, parameters):
        """
        Safely set a set of parameters to the internal instances of the `gp` and `mean_model`, as well as the
        `parameters` dict.

        Parameters
        ==========
        parameters: dict, pandas.DataFrame
            The set of parameters we would like to set.
        """
        for name, value in parameters.items():
            try:
                self.gp.set_parameter(name=name, value=value)
            except ValueError:
                pass
            if PARAMETERS_AS_STATE != "FALSE":
                self.parameters[name] = value


class CeleriteLikelihood(_GPLikelihood):

    def __init__(self, kernel, mean_model, t, y, yerr=1e-6):
        """
            Basic Gaussian Process likelihood interface for `celerite` and `george`.
            For `celerite` documentation see: https://celerite.readthedocs.io/en/stable/
            For `george` documentation see: https://george.readthedocs.io/en/latest/

            Parameters
            ==========
            kernel: celerite.term.Term
                `celerite` or `george` kernel. See the respective package documentation about the usage.
            mean_model: celerite.modeling.Model
                Mean model
            t: array_like
                The `times` or `x` values of the data set.
            y: array_like
                The `y` values of the data set.
            yerr: float, int, array_like, optional
                The error values on the y-values. If a single value is given, it is assumed that the value
                applies for all y-values. Default is 1e-6, effectively assuming that no y-errors are present.
        """
        import celerite
        super().__init__(kernel=kernel, mean_model=mean_model, t=t, y=y, yerr=yerr, gp_class=celerite.GP)

    def log_likelihood(self, parameters=None):
        """
        Calculate the log-likelihood for the Gaussian process given the current parameters.

        Returns
        =======
        float: The log-likelihood value.
        """
        parameters = _fallback_to_parameters(self, parameters)
        self.gp.set_parameter_vector(vector=np.array(list(parameters.values())))
        try:
            return self.gp.log_likelihood(self.y)
        except Exception:
            return -np.inf


class GeorgeLikelihood(_GPLikelihood):

    def __init__(self, kernel, mean_model, t, y, yerr=1e-6):
        """
            Basic Gaussian Process likelihood interface for `celerite` and `george`.
            For `celerite` documentation see: https://celerite.readthedocs.io/en/stable/
            For `george` documentation see: https://george.readthedocs.io/en/latest/

            Parameters
            ==========
            kernel: george.kernels.Kernel
                `celerite` or `george` kernel. See the respective package documentation about the usage.
            mean_model: george.modeling.Model
                Mean model
            t: array_like
                The `times` or `x` values of the data set.
            y: array_like
                The `y` values of the data set.
            yerr: float, int, array_like, optional
                The error values on the y-values. If a single value is given, it is assumed that the value
                applies for all y-values. Default is 1e-6, effectively assuming that no y-errors are present.
        """
        import george
        super().__init__(kernel=kernel, mean_model=mean_model, t=t, y=y, yerr=yerr, gp_class=george.GP)

    def log_likelihood(self, parameters=None):
        """
        Calculate the log-likelihood for the Gaussian process given the current parameters.

        Returns
        =======
        float: The log-likelihood value.
        """
        parameters = _fallback_to_parameters(self, parameters)
        for name, value in parameters.items():
            try:
                self.gp.set_parameter(name=name, value=value)
            except ValueError:
                raise ValueError(f"Parameter {name} not a valid parameter for the GP.")
        try:
            return self.gp.log_likelihood(self.y)
        except Exception:
            return -np.inf


class LikelihoodParameterError(Exception):
    pass


class MarginalizedLikelihoodReconstructionError(Exception):
    pass
