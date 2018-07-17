from __future__ import division

import tupak
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.special import erf, erfinv
import scipy.stats
import os

from tupak.core.utils import logger


class PriorSet(dict):
    def __init__(self, dictionary=None, filename=None):
        """ A set of priors

        Parameters
        ----------
        dictionary: dict, None
            If given, a dictionary to generate the prior set.
        filename: str, None
            If given, a file containing the prior to generate the prior set.
        """
        dict.__init__(self)
        if type(dictionary) is dict:
            self.update(dictionary)
        elif filename:
            self.read_in_file(filename)

    def write_to_file(self, outdir, label):
        """ Write the prior distribution to file.

        Parameters
        ----------
        outdir: str
            output directory name
        label: str
            Output file naming scheme
        """

        prior_file = os.path.join(outdir, "{}_prior.txt".format(label))
        logger.debug("Writing priors to {}".format(prior_file))
        with open(prior_file, "w") as outfile:
            for key in self.keys():
                outfile.write(
                    "{} = {}\n".format(key, self[key]))

    def read_in_file(self, filename):
        """ Reads in a prior from a file specification

        Parameters
        -------
        filename: str
            Name of the file to be read in
        """

        prior = {}
        with open(filename, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                elements = line.split('=')
                key = elements[0].replace(' ', '')
                val = '='.join(elements[1:])
                prior[key] = eval(val)
        self.update(prior)

    def convert_floats_to_delta_functions(self):
        """ Convert all float parameters to delta functions """
        for key in self:
            if isinstance(self[key], Prior):
                continue
            elif isinstance(self[key], float) or isinstance(self[key], int):
                self[key] = DeltaFunction(self[key])
                logger.debug(
                    "{} converted to delta function prior.".format(key))
            else:
                logger.debug(
                    "{} cannot be converted to delta function prior."
                    .format(key))

    def fill_priors(self, likelihood, default_priors_file=None):
        """
        Fill dictionary of priors based on required parameters of likelihood

        Any floats in prior will be converted to delta function prior. Any
        required, non-specified parameters will use the default.

        Note: if `likelihood` has `non_standard_sampling_parameter_keys`, then
        this will set-up default priors for those as well.

        Parameters
        ----------
        likelihood: tupak.likelihood.GravitationalWaveTransient instance
            Used to infer the set of parameters to fill the prior with
        default_priors_file: str, optional
            If given, a file containing the default priors; otherwise defaults
            to the tupak defaults for a binary black hole.


        Returns
        -------
        prior: dict
            The filled prior dictionary

        """

        self.convert_floats_to_delta_functions()

        missing_keys = set(likelihood.parameters) - set(self.keys())

        if getattr(likelihood, 'non_standard_sampling_parameter_keys', None) is not None:
            for parameter in likelihood.non_standard_sampling_parameter_keys:
                if parameter in self:
                    continue
                self[parameter] = create_default_prior(parameter, default_priors_file)

        for missing_key in missing_keys:
            if not self.test_redundancy(missing_key):
                default_prior = create_default_prior(missing_key, default_priors_file)
                if default_prior is None:
                    set_val = likelihood.parameters[missing_key]
                    logger.warning(
                        "Parameter {} has no default prior and is set to {}, this"
                        " will not be sampled and may cause an error."
                        .format(missing_key, set_val))
                else:
                    self[missing_key] = default_prior

        for key in self:
            self.test_redundancy(key)

    def sample(self, size=None):
        """Draw samples from the prior set

        Parameters
        ----------
        size: int or tuple of ints, optional
            See numpy.random.uniform docs

        Returns
        -------
        dict: Dictionary of the samples
        """
        self.convert_floats_to_delta_functions()
        samples = dict()
        for key in self:
            if isinstance(self[key], Prior):
                samples[key] = self[key].sample(size=size)
        return samples

    def sample_subset(self, keys=list(), size=None):
        """Draw samples from the prior set for parameters which are not a DeltaFunction

        Parameters
        ----------
        keys: list
            List of prior keys to draw samples from
        size: int or tuple of ints, optional
            See numpy.random.uniform docs

        Returns
        -------
        dict: Dictionary of the drawn samples
        """
        samples = dict()
        for key in keys:
            if isinstance(self[key], Prior):
                samples[key] = self[key].sample(size=size)
            else:
                logger.debug('{} not a known prior.'.format(key))
        return samples

    def prob(self, sample):
        """

        Parameters
        ----------
        sample: dict
            Dictionary of the samples of which we want to have the probability of

        Returns
        -------
        float: Joint probability of all individual sample probabilities

        """
        return np.product([self[key].prob(sample[key]) for key in sample])

    def ln_prob(self, sample):
        """

        Parameters
        ----------
        sample: dict
            Dictionary of the samples of which we want to have the log probability of

        Returns
        -------
        float: Joint log probability of all the individual sample probabilities

        """
        return np.sum([self[key].ln_prob(sample[key]) for key in sample])

    def rescale(self, keys, theta):
        """Rescale samples from unit cube to prior

        Parameters
        ----------
        keys: list
            List of prior keys to be rescaled
        theta: list
            List of randomly drawn values on a unit cube associated with the prior keys

        Returns
        -------
        list: List of floats containing the rescaled sample
        """
        return [self[key].rescale(sample) for key, sample in zip(keys, theta)]

    def test_redundancy(self, key):
        """Empty redundancy test, should be overwritten"""
        return False


def create_default_prior(name, default_priors_file=None):
    """Make a default prior for a parameter with a known name.

    Parameters
    ----------
    name: str
        Parameter name
    default_priors_file: str, optional
        If given, a file containing the default priors; otherwise defaults to
        the tupak defaults for a binary black hole.

    Return
    ------
    prior: Prior
        Default prior distribution for that parameter, if unknown None is
        returned.
    """

    if default_priors_file is None:
        logger.debug(
            "No prior file given.")
        prior = None
    else:
        default_priors = PriorSet(filename=default_priors_file)
        if name in default_priors.keys():
            prior = default_priors[name]
        else:
            logger.debug(
                "No default prior found for variable {}.".format(name))
            prior = None
    return prior


class Prior(object):

    def __init__(self, name=None, latex_label=None, minimum=-np.inf, maximum=np.inf):
        """ Implements a Prior object

        Parameters
        ----------
        name: str, optional
            Name associated with prior.
        latex_label: str, optional
            Latex label associated with prior, used for plotting.
        minimum: float, optional
            Minimum of the domain, default=-np.inf
        maximum: float, optional
            Maximum of the domain, default=np.inf

        """
        self.name = name
        self.latex_label = latex_label
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self):
        """Overrides the __call__ special method. Calls the sample method.

        Returns
        -------
        float: The return value of the sample method.
        """
        return self.sample()

    def sample(self, size=None):
        """Draw a sample from the prior

        Parameters
        ----------
        size: int or tuple of ints, optional
            See numpy.random.uniform docs

        Returns
        -------
        float: A random number between 0 and 1, rescaled to match the distribution of this Prior

        """
        return self.rescale(np.random.uniform(0, 1, size))

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior.

        This should be overwritten by each subclass.

        Parameters
        ----------
        val: float
            A random number between 0 and 1

        Returns
        -------
        None

        """
        return None

    def prob(self, val):
        """Return the prior probability of val, this should be overwritten

        Parameters
        ----------
        val: float

        Returns
        -------
        np.nan

        """
        return np.nan

    def ln_prob(self, val):
        """Return the prior ln probability of val, this should be overwritten

        Parameters
        ----------
        val: float

        Returns
        -------
        np.nan

        """
        return np.log(self.prob(val))

    @staticmethod
    def test_valid_for_rescaling(val):
        """Test if 0 < val < 1

        Parameters
        ----------
        val: float

        Raises
        -------
        ValueError: If val is not between 0 and 1
        """
        val = np.atleast_1d(val)
        tests = (val < 0) + (val > 1)
        if np.any(tests):
            raise ValueError("Number to be rescaled should be in [0, 1]")

    def __repr__(self):
        """Overrides the special method __repr__.

        Should return a representation of this instance that resembles how it is instantiated

        Returns
        -------
        str: A string representation of this instance

        """
        return self._subclass_repr_helper()

    def _subclass_repr_helper(self, subclass_args=list()):
        """Helps out subclass _repr__ methods by creating a common template

        Parameters
        ----------
        subclass_args: list, optional
            List of attributes in the subclass instance

        Returns
        -------
        str: A string representation for this subclass instance.

        """
        prior_name = self.__class__.__name__
        args = ['name', 'latex_label', 'minimum', 'maximum']
        args.extend(subclass_args)

        property_names = [p for p in dir(self.__class__) if isinstance(getattr(self.__class__, p), property)]
        dict_with_properties = self.__dict__.copy()
        for key in property_names:
            dict_with_properties[key] = getattr(self, key)

        args = ', '.join(['{}={}'.format(key, repr(dict_with_properties[key])) for key in args])
        return "{}({})".format(prior_name, args)

    @property
    def is_fixed(self):
        """
        Returns True if the prior is fixed and should not be used in the sampler. Does this by checking if this instance
        is an instance of DeltaFunction.


        Returns
        -------
        bool: Whether it's fixed or not!

        """
        return isinstance(self, DeltaFunction)

    @property
    def latex_label(self):
        """Latex label that can be used for plots.

        Draws from a set of default labels if no label is given

        Returns
        -------
        str: A latex representation for this prior

        """
        return self.__latex_label

    @latex_label.setter
    def latex_label(self, latex_label=None):
        if latex_label is None:
            self.__latex_label = self.__default_latex_label
        else:
            self.__latex_label = latex_label

    @property
    def minimum(self):
        return self.__minimum

    @minimum.setter
    def minimum(self, minimum):
        self.__minimum = minimum

    @property
    def maximum(self):
        return self.__maximum

    @maximum.setter
    def maximum(self, maximum):
        self.__maximum = maximum

    @property
    def __default_latex_label(self):
        if self.name in self._default_latex_labels.keys():
            label = self._default_latex_labels[self.name]
        else:
            label = self.name
        return label

    _default_latex_labels = dict()


class DeltaFunction(Prior):

    def __init__(self, peak, name=None, latex_label=None):
        """Dirac delta function prior, this always returns peak.

        Parameters
        ----------
        peak: float
            Peak value of the delta function
        name: str
            See superclass
        latex_label: str
            See superclass

        """
        Prior.__init__(self, name, latex_label, minimum=peak, maximum=peak)
        self.peak = peak

    def rescale(self, val):
        """Rescale everything to the peak with the correct shape.

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Rescaled probability, equivalent to peak
        """
        Prior.test_valid_for_rescaling(val)
        return self.peak * val ** 0

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ----------
        val: float

        Returns
        -------
        float: np.inf if val = peak, 0 otherwise

        """
        if self.peak == val:
            return np.inf
        else:
            return 0

    def __repr__(self):
        """Call to helper method in the super class."""
        return Prior._subclass_repr_helper(self, subclass_args=['peak'])


class PowerLaw(Prior):

    def __init__(self, alpha, minimum, maximum, name=None, latex_label=None):
        """Power law with bounds and alpha, spectral index

        Parameters
        ----------
        alpha: float
            Power law exponent parameter
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        """
        Prior.__init__(self, name, latex_label, minimum, maximum)
        self.alpha = alpha

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the power-law prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ----------
        val: float
            Uniform probability

        Returns
        -------
        float: Rescaled probability
        """
        Prior.test_valid_for_rescaling(val)
        if self.alpha == -1:
            return self.minimum * np.exp(val * np.log(self.maximum / self.minimum))
        else:
            return (self.minimum ** (1 + self.alpha) + val *
                    (self.maximum ** (1 + self.alpha) - self.minimum ** (1 + self.alpha))) ** (1. / (1 + self.alpha))

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        if self.alpha == -1:
            return np.nan_to_num(1 / val / np.log(self.maximum / self.minimum)) * in_prior
        else:
            return np.nan_to_num(val ** self.alpha * (1 + self.alpha) / (self.maximum ** (1 + self.alpha)
                                                                         - self.minimum ** (1 + self.alpha))) * in_prior

    def lnprob(self, val):
        """Return the logarithmic prior probability of val

        Parameters
        ----------
        val: float

        Returns
        -------
        float:

        """
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        normalising = (1 + self.alpha) / (self.maximum ** (1 + self.alpha)
                                          - self.minimum ** (1 + self.alpha))
        return self.alpha * np.log(val) * np.log(normalising) * in_prior

    def __repr__(self):
        """Call to helper method in the super class."""
        return Prior._subclass_repr_helper(self, subclass_args=['alpha'])


class Uniform(Prior):

    def __init__(self, minimum, maximum, name=None, latex_label=None):
        """Uniform prior with bounds

        Parameters
        ----------
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        """
        Prior.__init__(self, name, latex_label, minimum, maximum)

    def rescale(self, val):
        Prior.test_valid_for_rescaling(val)
        return self.minimum + val * (self.maximum - self.minimum)

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """
        return scipy.stats.uniform.pdf(val, loc=self.minimum,
                                       scale=self.maximum-self.minimum)

    def ln_prob(self, val):
        """Return the log prior probability of val

        Parameters
        ----------
        val: float

        Returns
        -------
        float: log probability of val
        """
        return scipy.stats.uniform.logpdf(val, loc=self.minimum,
                                          scale=self.maximum-self.minimum)


class LogUniform(PowerLaw):

    def __init__(self, minimum, maximum, name=None, latex_label=None):
        """Log-Uniform prior with bounds

        Parameters
        ----------
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        """
        PowerLaw.__init__(self, name=name, latex_label=latex_label, minimum=minimum, maximum=maximum, alpha=-1)
        if self.minimum <= 0:
            logger.warning('You specified a uniform-in-log prior with minimum={}'.format(self.minimum))

    def __repr__(self):
        """Call to helper method in the super class."""
        return Prior._subclass_repr_helper(self)


class Cosine(Prior):

    def __init__(self, name=None, latex_label=None, minimum=-np.pi / 2, maximum=np.pi / 2):
        """Cosine prior with bounds

        Parameters
        ----------
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        """
        Prior.__init__(self, name, latex_label, minimum, maximum)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to a uniform in cosine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        return np.arcsin(-1 + val * 2)

    def prob(self, val):
        """Return the prior probability of val. Defined over [-pi/2, pi/2].

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        return np.cos(val) / 2 * in_prior


class Sine(Prior):

    def __init__(self, name=None, latex_label=None, minimum=0, maximum=np.pi):
        """Sine prior with bounds

        Parameters
        ----------
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        """
        Prior.__init__(self, name, latex_label, minimum, maximum)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to a uniform in sine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        return np.arccos(1 - val * 2)

    def prob(self, val):
        """Return the prior probability of val. Defined over [0, pi].

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        return np.sin(val) / 2 * in_prior


class Gaussian(Prior):
    def __init__(self, mu, sigma, name=None, latex_label=None):
        """Gaussian prior with mean mu and width sigma

        Parameters
        ----------
        mu: float
            Mean of the Gaussian prior
        sigma:
            Width/Standard deviation of the Gaussian prior
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        """
        Prior.__init__(self, name, latex_label)
        self.mu = mu
        self.sigma = sigma

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Gaussian prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        return self.mu + erfinv(2 * val - 1) * 2 ** 0.5 * self.sigma

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """
        return np.exp(-(self.mu - val) ** 2 / (2 * self.sigma ** 2)) / (2 * np.pi) ** 0.5 / self.sigma

    def lnprob(self, val):
        return -0.5 * ((self.mu - val) ** 2 / self.sigma ** 2 + np.log(2 * np.pi * self.sigma ** 2))

    def __repr__(self):
        """Call to helper method in the super class."""
        return Prior._subclass_repr_helper(self, subclass_args=['mu', 'sigma'])


class TruncatedGaussian(Prior):

    def __init__(self, mu, sigma, minimum, maximum, name=None, latex_label=None):
        """Truncated Gaussian prior with mean mu and width sigma

        https://en.wikipedia.org/wiki/Truncated_normal_distribution

        Parameters
        ----------
        mu: float
            Mean of the Gaussian prior
        sigma:
            Width/Standard deviation of the Gaussian prior
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, minimum=minimum, maximum=maximum)
        self.mu = mu
        self.sigma = sigma

    @property
    def normalisation(self):
        """ Calculates the proper normalisation of the truncated Gaussian

        Returns
        -------
        float: Proper normalisation of the truncated Gaussian
        """
        return (erf((self.maximum - self.mu) / 2 ** 0.5 / self.sigma) - erf(
            (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) / 2

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate truncated Gaussian prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        return erfinv(2 * val * self.normalisation + erf(
            (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) * 2 ** 0.5 * self.sigma + self.mu

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        return np.exp(-(self.mu - val) ** 2 / (2 * self.sigma ** 2)) / (
                2 * np.pi) ** 0.5 / self.sigma / self.normalisation * in_prior

    def __repr__(self):
        """Call to helper method in the super class."""
        return Prior._subclass_repr_helper(self, subclass_args=['mu', 'sigma'])


class Interped(Prior):

    def __init__(self, xx, yy, minimum=np.nan, maximum=np.nan, name=None, latex_label=None):
        """Creates an interpolated prior function from arrays of xx and yy=p(xx)

        Parameters
        ----------
        xx: array_like
            x values for the to be interpolated prior function
        yy: array_like
            p(xx) values for the to be interpolated prior function
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass

        Attributes
        -------
        probability_density: scipy.interpolate.interp1d
            Interpolated prior probability distribution
        cumulative_distribution: scipy.interpolate.interp1d
            Interpolated cumulative prior probability distribution
        inverse_cumulative_distribution: scipy.interpolate.interp1d
            Inverted cumulative prior probability distribution
        YY: array_like
            Cumulative prior probability distribution

        """
        self.xx = xx
        self.yy = yy
        self.__all_interpolated = interp1d(x=xx, y=yy, bounds_error=False, fill_value=0)
        Prior.__init__(self, name, latex_label,
                       minimum=np.nanmax(np.array((min(xx), minimum))),
                       maximum=np.nanmin(np.array((max(xx), maximum))))
        self.__initialize_attributes()

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """
        return self.probability_density(val)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior.

        This maps to the inverse CDF. This is done using interpolation.
        """
        Prior.test_valid_for_rescaling(val)
        rescaled = self.inverse_cumulative_distribution(val)
        if rescaled.shape == ():
            rescaled = float(rescaled)
        return rescaled

    def __repr__(self):
        """Call to helper method in the super class."""
        return Prior._subclass_repr_helper(self, subclass_args=['xx', 'yy'])

    @property
    def minimum(self):
        """Return minimum of the prior distribution.

        Updates the prior distribution if minimum is set to a different value.

        Returns
        -------
        float: Minimum of the prior distribution

        """
        return self.__minimum

    @minimum.setter
    def minimum(self, minimum):
        self.__minimum = minimum
        if '_Interped__maximum' in self.__dict__ and self.__maximum < np.inf:
            self.__update_instance()

    @property
    def maximum(self):
        """Return maximum of the prior distribution.

        Updates the prior distribution if maximum is set to a different value.

        Returns
        -------
        float: Maximum of the prior distribution

        """
        return self.__maximum

    @maximum.setter
    def maximum(self, maximum):
        self.__maximum = maximum
        if '_Interped__minimum' in self.__dict__ and self.__minimum < np.inf:
            self.__update_instance()

    def __update_instance(self):
        self.xx = np.linspace(self.minimum, self.maximum, len(self.xx))
        self.yy = self.__all_interpolated(self.xx)
        self.__initialize_attributes()

    def __initialize_attributes(self):
        if np.trapz(self.yy, self.xx) != 1:
            logger.debug('Supplied PDF for {} is not normalised, normalising.'.format(self.name))
        self.yy /= np.trapz(self.yy, self.xx)
        self.YY = cumtrapz(self.yy, self.xx, initial=0)
        # Need last element of cumulative distribution to be exactly one.
        self.YY[-1] = 1
        self.probability_density = interp1d(x=self.xx, y=self.yy, bounds_error=False, fill_value=0)
        self.cumulative_distribution = interp1d(x=self.xx, y=self.YY, bounds_error=False, fill_value=0)
        self.inverse_cumulative_distribution = interp1d(x=self.YY, y=self.xx, bounds_error=True)


class FromFile(Interped):

    def __init__(self, file_name, minimum=None, maximum=None, name=None, latex_label=None):
        """Creates an interpolated prior function from arrays of xx and yy=p(xx) extracted from a file

        Parameters
        ----------
        file_name: str
            Name of the file containing the xx and yy arrays
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass

        Attributes
        -------
        all_interpolated: scipy.interpolate.interp1d
            Interpolated prior function

        """
        try:
            self.id = file_name
            xx, yy = np.genfromtxt(self.id).T
            Interped.__init__(self, xx=xx, yy=yy, minimum=minimum, maximum=maximum, name=name, latex_label=latex_label)
        except IOError:
            logger.warning("Can't load {}.".format(self.id))
            logger.warning("Format should be:")
            logger.warning(r"x\tp(x)")

    def __repr__(self):
        """Call to helper method in the super class."""
        return Prior._subclass_repr_helper(self, subclass_args=['id'])
