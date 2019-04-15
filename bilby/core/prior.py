from __future__ import division

import os
from collections import OrderedDict
from future.utils import iteritems

import numpy as np
import scipy.stats
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.special import erf, erfinv

# Keep import bilby statement, it is necessary for some eval() statements
import bilby  # noqa
from .utils import logger, infer_args_from_method, check_directory_exists_and_if_not_mkdir


class PriorDict(OrderedDict):
    def __init__(self, dictionary=None, filename=None,
                 conversion_function=None):
        """ A set of priors

        Parameters
        ----------
        dictionary: dict, None
            If given, a dictionary to generate the prior set.
        filename: str, None
            If given, a file containing the prior to generate the prior set.
        conversion_function: func
            Function to convert between sampled parameters and constraints.
            Default is no conversion.
        """
        OrderedDict.__init__(self)
        if isinstance(dictionary, dict):
            self.from_dictionary(dictionary)
        elif type(dictionary) is str:
            logger.debug('Argument "dictionary" is a string.' +
                         ' Assuming it is intended as a file name.')
            self.from_file(dictionary)
        elif type(filename) is str:
            self.from_file(filename)
        elif dictionary is not None:
            raise ValueError("PriorDict input dictionary not understood")

        self.convert_floats_to_delta_functions()

        if conversion_function is not None:
            self.conversion_function = conversion_function
        else:
            self.conversion_function = self.default_conversion_function

    def evaluate_constraints(self, sample):
        out_sample = self.conversion_function(sample)
        prob = 1
        for key in self:
            if isinstance(self[key], Constraint) and key in out_sample:
                prob *= self[key].prob(out_sample[key])
        return prob

    def default_conversion_function(self, sample):
        """
        Placeholder parameter conversion function.

        Parameters
        ----------
        sample: dict
            Dictionary to convert

        Returns
        -------
        sample: dict
            Same as input
        """
        return sample

    def to_file(self, outdir, label):
        """ Write the prior distribution to file.

        Parameters
        ----------
        outdir: str
            output directory name
        label: str
            Output file naming scheme
        """

        check_directory_exists_and_if_not_mkdir(outdir)
        prior_file = os.path.join(outdir, "{}.prior".format(label))
        logger.debug("Writing priors to {}".format(prior_file))
        with open(prior_file, "w") as outfile:
            for key in self.keys():
                outfile.write(
                    "{} = {}\n".format(key, self[key]))

    def from_file(self, filename):
        """ Reads in a prior from a file specification

        Parameters
        ----------
        filename: str
            Name of the file to be read in

        Notes
        -----
        Lines beginning with '#' or empty lines will be ignored.
        """

        comments = ['#', '\n']
        prior = dict()
        with open(filename, 'r') as f:
            for line in f:
                if line[0] in comments:
                    continue
                elements = line.split('=')
                key = elements[0].replace(' ', '')
                val = '='.join(elements[1:])
                prior[key] = eval(val)
        self.update(prior)

    def from_dictionary(self, dictionary):
        for key, val in iteritems(dictionary):
            if isinstance(val, str):
                try:
                    prior = eval(val)
                    if isinstance(prior, (Prior, float, int, str)):
                        val = prior
                except (NameError, SyntaxError, TypeError):
                    logger.debug(
                        "Failed to load dictionary value {} correctlty"
                        .format(key))
                    pass
            self[key] = val

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
        likelihood: bilby.likelihood.GravitationalWaveTransient instance
            Used to infer the set of parameters to fill the prior with
        default_priors_file: str, optional
            If given, a file containing the default priors.


        Returns
        -------
        prior: dict
            The filled prior dictionary

        """

        self.convert_floats_to_delta_functions()

        missing_keys = set(likelihood.parameters) - set(self.keys())

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
        return self.sample_subset_constrained(keys=list(self.keys()), size=size)

    def sample_subset(self, keys=iter([]), size=None):
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
        self.convert_floats_to_delta_functions()
        samples = dict()
        for key in keys:
            if isinstance(self[key], Prior):
                if isinstance(self[key], Constraint):
                    continue
                else:
                    samples[key] = self[key].sample(size=size)
            else:
                logger.debug('{} not a known prior.'.format(key))
        return samples

    def sample_subset_constrained(self, keys=iter([]), size=None):
        if size is None or size == 1:
            while True:
                sample = self.sample_subset(keys=keys, size=size)
                if self.evaluate_constraints(sample):
                    return sample
        else:
            needed = np.prod(size)
            all_samples = {key: np.array([]) for key in keys}
            _first_key = list(all_samples.keys())[0]
            while len(all_samples[_first_key]) <= needed:
                samples = self.sample_subset(keys=keys, size=needed)
                keep = np.array(self.evaluate_constraints(samples), dtype=bool)
                for key in samples:
                    all_samples[key] = np.hstack(
                        [all_samples[key], samples[key][keep].flatten()])
            all_samples = {key: np.reshape(all_samples[key][:needed], size)
                           for key in all_samples
                           if not isinstance(self[key], Constraint)}
            return all_samples

    def prob(self, sample, **kwargs):
        """

        Parameters
        ----------
        sample: dict
            Dictionary of the samples of which we want to have the probability of
        kwargs:
            The keyword arguments are passed directly to `np.product`

        Returns
        -------
        float: Joint probability of all individual sample probabilities

        """
        prob = np.product([self[key].prob(sample[key])
                           for key in sample], **kwargs)

        if np.all(prob == 0.):
            return prob
        else:
            if isinstance(prob, float):
                if self.evaluate_constraints(sample):
                    return prob
                else:
                    return 0.
            else:
                constrained_prob = np.zeros_like(prob)
                keep = np.array(self.evaluate_constraints(sample), dtype=bool)
                constrained_prob[keep] = prob[keep]
                return constrained_prob

    def ln_prob(self, sample, axis=None):
        """

        Parameters
        ----------
        sample: dict
            Dictionary of the samples of which to calculate the log probability
        axis: None or int
            Axis along which the summation is performed

        Returns
        -------
        float or ndarray:
            Joint log probability of all the individual sample probabilities

        """
        ln_prob = np.sum([self[key].ln_prob(sample[key])
                          for key in sample], axis=axis)

        if np.all(np.isinf(ln_prob)):
            return ln_prob
        else:
            if isinstance(ln_prob, float):
                if self.evaluate_constraints(sample):
                    return ln_prob
                else:
                    return -np.inf
            else:
                constrained_ln_prob = -np.inf * np.ones_like(ln_prob)
                keep = np.array(self.evaluate_constraints(sample), dtype=bool)
                constrained_ln_prob[keep] = ln_prob[keep]
                return constrained_ln_prob

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

    def test_redundancy(self, key, disable_logging=False):
        """Empty redundancy test, should be overwritten in subclasses"""
        return False

    def test_has_redundant_keys(self):
        """
        Test whether there are redundant keys in self.

        Return
        ------
        bool: Whether there are redundancies or not
        """
        redundant = False
        for key in self:
            if isinstance(self[key], Constraint):
                continue
            temp = self.copy()
            del temp[key]
            if temp.test_redundancy(key, disable_logging=True):
                logger.warning('{} is a redundant key in this {}.'
                               .format(key, self.__class__.__name__))
                redundant = True
        return redundant

    def copy(self):
        """
        We have to overwrite the copy method as it fails due to the presence of
        defaults.
        """
        return self.__class__(dictionary=OrderedDict(self))


class PriorSet(PriorDict):

    def __init__(self, dictionary=None, filename=None):
        """ DEPRECATED: USE PriorDict INSTEAD"""
        logger.warning("The name 'PriorSet' is deprecated use 'PriorDict' instead")
        super(PriorSet, self).__init__(dictionary, filename)


def create_default_prior(name, default_priors_file=None):
    """Make a default prior for a parameter with a known name.

    Parameters
    ----------
    name: str
        Parameter name
    default_priors_file: str, optional
        If given, a file containing the default priors.

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
        default_priors = PriorDict(filename=default_priors_file)
        if name in default_priors.keys():
            prior = default_priors[name]
        else:
            logger.debug(
                "No default prior found for variable {}.".format(name))
            prior = None
    return prior


class Prior(object):
    _default_latex_labels = dict()

    def __init__(self, name=None, latex_label=None, unit=None, minimum=-np.inf,
                 maximum=np.inf, periodic_boundary=False):
        """ Implements a Prior object

        Parameters
        ----------
        name: str, optional
            Name associated with prior.
        latex_label: str, optional
            Latex label associated with prior, used for plotting.
        unit: str, optional
            If given, a Latex string describing the units of the parameter.
        minimum: float, optional
            Minimum of the domain, default=-np.inf
        maximum: float, optional
            Maximum of the domain, default=np.inf
        periodic_boundary: bool, optional
            Whether or not the boundary condition is periodic. Not available in all samplers.
        """
        self.name = name
        self.latex_label = latex_label
        self.unit = unit
        self.minimum = minimum
        self.maximum = maximum
        self.periodic_boundary = periodic_boundary

    def __call__(self):
        """Overrides the __call__ special method. Calls the sample method.

        Returns
        -------
        float: The return value of the sample method.
        """
        return self.sample()

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if sorted(self.__dict__.keys()) != sorted(other.__dict__.keys()):
            return False
        for key in self.__dict__:
            if type(self.__dict__[key]) is np.ndarray:
                if not np.array_equal(self.__dict__[key], other.__dict__[key]):
                    return False
            else:
                if not self.__dict__[key] == other.__dict__[key]:
                    return False
        return True

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

    def is_in_prior_range(self, val):
        """Returns True if val is in the prior boundaries, zero otherwise

        Parameters
        ----------
        val: float

        Returns
        -------
        np.nan

        """
        return (val >= self.minimum) & (val <= self.maximum)

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

        Returns a representation of this instance that resembles how it is instantiated.
        Works correctly for all child classes

        Returns
        -------
        str: A string representation of this instance

        """
        subclass_args = infer_args_from_method(self.__init__)
        prior_name = self.__class__.__name__

        property_names = [p for p in dir(self.__class__) if isinstance(getattr(self.__class__, p), property)]
        dict_with_properties = self.__dict__.copy()
        for key in property_names:
            dict_with_properties[key] = getattr(self, key)
        args = ', '.join(['{}={}'.format(key, repr(dict_with_properties[key])) for key in subclass_args])
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
        return isinstance(self, (Constraint, DeltaFunction))

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
    def unit(self):
        return self.__unit

    @unit.setter
    def unit(self, unit):
        self.__unit = unit

    @property
    def latex_label_with_unit(self):
        """ If a unit is specifed, returns a string of the latex label and unit """
        if self.unit is not None:
            return "{} [{}]".format(self.latex_label, self.unit)
        else:
            return self.latex_label

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        self._minimum = minimum

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        self._maximum = maximum

    @property
    def periodic_boundary(self):
        return self._periodic_boundary

    @periodic_boundary.setter
    def periodic_boundary(self, periodic_boundary):
        if type(periodic_boundary) is not bool:
            raise ValueError('{} is not a valid setting for prior boundaries'.format(periodic_boundary))
        self._periodic_boundary = periodic_boundary

    @property
    def __default_latex_label(self):
        if self.name in self._default_latex_labels.keys():
            label = self._default_latex_labels[self.name]
        else:
            label = self.name
        return label


class Constraint(Prior):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None):
        Prior.__init__(self, minimum=minimum, maximum=maximum, name=name,
                       latex_label=latex_label, unit=unit)

    def prob(self, val):
        return (val > self.minimum) & (val < self.maximum)

    def ln_prob(self, val):
        return np.log((val > self.minimum) & (val < self.maximum))


class DeltaFunction(Prior):

    def __init__(self, peak, name=None, latex_label=None, unit=None):
        """Dirac delta function prior, this always returns peak.

        Parameters
        ----------
        peak: float
            Peak value of the delta function
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass

        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit,
                       minimum=peak, maximum=peak)
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
        at_peak = (val == self.peak)
        return np.nan_to_num(np.multiply(at_peak, np.inf))


class PowerLaw(Prior):

    def __init__(self, alpha, minimum, maximum, name=None, latex_label=None,
                 unit=None, periodic_boundary=False):
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
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label,
                       minimum=minimum, maximum=maximum, unit=unit,
                       periodic_boundary=periodic_boundary)
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
        if self.alpha == -1:
            return np.nan_to_num(1 / val / np.log(self.maximum / self.minimum)) * self.is_in_prior_range(val)
        else:
            return np.nan_to_num(val ** self.alpha * (1 + self.alpha) /
                                 (self.maximum ** (1 + self.alpha) -
                                  self.minimum ** (1 + self.alpha))) * self.is_in_prior_range(val)

    def ln_prob(self, val):
        """Return the logarithmic prior probability of val

        Parameters
        ----------
        val: float

        Returns
        -------
        float:

        """
        if self.alpha == -1:
            normalising = 1. / np.log(self.maximum / self.minimum)
        else:
            normalising = (1 + self.alpha) / (self.maximum ** (1 + self.alpha) -
                                              self.minimum ** (1 + self.alpha))

        return (self.alpha * np.nan_to_num(np.log(val)) + np.log(normalising)) + np.log(
            1. * self.is_in_prior_range(val))


class Uniform(Prior):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None, periodic_boundary=False):
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
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label,
                       minimum=minimum, maximum=maximum, unit=unit,
                       periodic_boundary=periodic_boundary)

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
                                       scale=self.maximum - self.minimum)

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
                                          scale=self.maximum - self.minimum)


class LogUniform(PowerLaw):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None, periodic_boundary=False):
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
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        PowerLaw.__init__(self, name=name, latex_label=latex_label, unit=unit,
                          minimum=minimum, maximum=maximum, alpha=-1, periodic_boundary=periodic_boundary)
        if self.minimum <= 0:
            logger.warning('You specified a uniform-in-log prior with minimum={}'.format(self.minimum))


class SymmetricLogUniform(Prior):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None, periodic_boundary=False):
        """Symmetric Log-Uniform distribtions with bounds

        This is identical to a Log-Uniform distribition, but mirrored about
        the zero-axis and subsequently normalized. As such, the distribution
        has support on the two regions [-maximum, -minimum] and [minimum,
        maximum].

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
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label,
                       minimum=minimum, maximum=maximum, unit=unit,
                       periodic_boundary=periodic_boundary)

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
        if val < 0.5:
            return -self.maximum * np.exp(-2 * val * np.log(self.maximum / self.minimum))
        elif val > 0.5:
            return self.minimum * np.exp(np.log(self.maximum / self.minimum) * (2 * val - 1))
        else:
            raise ValueError("Rescale not valid for val=0.5")

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """
        return (
            np.nan_to_num(0.5 / np.abs(val) / np.log(self.maximum / self.minimum)) *
            self.is_in_prior_range(val))

    def ln_prob(self, val):
        """Return the logarithmic prior probability of val

        Parameters
        ----------
        val: float

        Returns
        -------
        float:

        """
        return np.nan_to_num(- np.log(2 * np.abs(val)) - np.log(np.log(self.maximum / self.minimum)))


class Cosine(Prior):

    def __init__(self, name=None, latex_label=None, unit=None,
                 minimum=-np.pi / 2, maximum=np.pi / 2, periodic_boundary=False):
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
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit,
                       minimum=minimum, maximum=maximum, periodic_boundary=periodic_boundary)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to a uniform in cosine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        norm = 1 / (np.sin(self.maximum) - np.sin(self.minimum))
        return np.arcsin(val / norm + np.sin(self.minimum))

    def prob(self, val):
        """Return the prior probability of val. Defined over [-pi/2, pi/2].

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """
        return np.cos(val) / 2 * self.is_in_prior_range(val)


class Sine(Prior):

    def __init__(self, name=None, latex_label=None, unit=None, minimum=0,
                 maximum=np.pi, periodic_boundary=False):
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
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit,
                       minimum=minimum, maximum=maximum, periodic_boundary=periodic_boundary)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to a uniform in sine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        norm = 1 / (np.cos(self.minimum) - np.cos(self.maximum))
        return np.arccos(np.cos(self.minimum) - val / norm)

    def prob(self, val):
        """Return the prior probability of val. Defined over [0, pi].

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """
        return np.sin(val) / 2 * self.is_in_prior_range(val)


class Gaussian(Prior):

    def __init__(self, mu, sigma, name=None, latex_label=None, unit=None, periodic_boundary=False):
        """Gaussian prior with mean mu and width sigma

        Parameters
        ----------
        mu: float
            Mean of the Gaussian prior
        sigma:
            Width/Standard deviation of the Gaussian prior
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit, periodic_boundary=periodic_boundary)
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

    def ln_prob(self, val):
        return -0.5 * ((self.mu - val) ** 2 / self.sigma ** 2 + np.log(2 * np.pi * self.sigma ** 2))


class Normal(Gaussian):

    def __init__(self, mu, sigma, name=None, latex_label=None, unit=None, periodic_boundary=False):
        """A synonym for the Gaussian distribution.

        Parameters
        ----------
        mu: float
            Mean of the Gaussian prior
        sigma: float
            Width/Standard deviation of the Gaussian prior
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Gaussian.__init__(self, mu=mu, sigma=sigma, name=name, latex_label=latex_label,
                          unit=unit, periodic_boundary=periodic_boundary)


class TruncatedGaussian(Prior):

    def __init__(self, mu, sigma, minimum, maximum, name=None,
                 latex_label=None, unit=None, periodic_boundary=False):
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
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit,
                       minimum=minimum, maximum=maximum, periodic_boundary=periodic_boundary)
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
        return np.exp(-(self.mu - val) ** 2 / (2 * self.sigma ** 2)) / \
            (2 * np.pi) ** 0.5 / self.sigma / self.normalisation * self.is_in_prior_range(val)


class TruncatedNormal(TruncatedGaussian):

    def __init__(self, mu, sigma, minimum, maximum, name=None,
                 latex_label=None, unit=None, periodic_boundary=False):
        """A synonym for the TruncatedGaussian distribution.

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
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        TruncatedGaussian.__init__(self, mu=mu, sigma=sigma, minimum=minimum,
                                   maximum=maximum, name=name, latex_label=latex_label,
                                   unit=unit, periodic_boundary=periodic_boundary)


class HalfGaussian(TruncatedGaussian):
    def __init__(self, sigma, name=None, latex_label=None, unit=None, periodic_boundary=False):
        """A Gaussian with its mode at zero, and truncated to only be positive.

        Parameters
        ----------
        sigma: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        TruncatedGaussian.__init__(self, 0., sigma, minimum=0., maximum=np.inf,
                                   name=name, latex_label=latex_label,
                                   unit=unit, periodic_boundary=periodic_boundary)


class HalfNormal(HalfGaussian):
    def __init__(self, sigma, name=None, latex_label=None, unit=None, periodic_boundary=False):
        """A synonym for the HalfGaussian distribution.

        Parameters
        ----------
        sigma: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        HalfGaussian.__init__(self, sigma=sigma, name=name,
                              latex_label=latex_label, unit=unit,
                              periodic_boundary=periodic_boundary)


class LogNormal(Prior):
    def __init__(self, mu, sigma, name=None, latex_label=None, unit=None, periodic_boundary=False):
        """Log-normal prior with mean mu and width sigma

        https://en.wikipedia.org/wiki/Log-normal_distribution

        Parameters
        ----------
        mu: float
            Mean of the Gaussian prior
        sigma:
            Width/Standard deviation of the Gaussian prior
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Prior.__init__(self, name=name, minimum=0., latex_label=latex_label,
                       unit=unit, periodic_boundary=periodic_boundary)

        if sigma <= 0.:
            raise ValueError("For the LogGaussian prior the standard deviation must be positive")

        self.mu = mu
        self.sigma = sigma

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate LogNormal prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        return scipy.stats.lognorm.ppf(val, self.sigma, scale=np.exp(self.mu))

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """

        return scipy.stats.lognorm.pdf(val, self.sigma, scale=np.exp(self.mu))

    def ln_prob(self, val):
        return scipy.stats.lognorm.logpdf(val, self.sigma, scale=np.exp(self.mu))


class LogGaussian(LogNormal):
    def __init__(self, mu, sigma, name=None, latex_label=None, unit=None, periodic_boundary=False):
        """Synonym of LogNormal prior

        https://en.wikipedia.org/wiki/Log-normal_distribution

        Parameters
        ----------
        mu: float
            Mean of the Gaussian prior
        sigma:
            Width/Standard deviation of the Gaussian prior
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        LogNormal.__init__(self, mu=mu, sigma=sigma, name=name,
                           latex_label=latex_label, unit=unit, periodic_boundary=periodic_boundary)


class Exponential(Prior):
    def __init__(self, mu, name=None, latex_label=None, unit=None, periodic_boundary=False):
        """Exponential prior with mean mu

        Parameters
        ----------
        mu: float
            Mean of the Exponential prior
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Prior.__init__(self, name=name, minimum=0., latex_label=latex_label,
                       unit=unit, periodic_boundary=periodic_boundary)
        self.mu = mu

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Exponential prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)
        return scipy.stats.expon.ppf(val, scale=self.mu)

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """

        return scipy.stats.expon.pdf(val, scale=self.mu)

    def ln_prob(self, val):
        return scipy.stats.expon.logpdf(val, scale=self.mu)


class StudentT(Prior):
    def __init__(self, df, mu=0., scale=1., name=None, latex_label=None,
                 unit=None, periodic_boundary=False):
        """Student's t-distribution prior with number of degrees of freedom df,
        mean mu and scale

        https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution

        Parameters
        ----------
        df: float
            Number of degrees of freedom for distribution
        mu: float
            Mean of the Student's t-prior
        scale:
            Width of the Student's t-prior
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit, periodic_boundary=periodic_boundary)

        if df <= 0. or scale <= 0.:
            raise ValueError("For the StudentT prior the number of degrees of freedom and scale must be positive")

        self.df = df
        self.mu = mu
        self.scale = scale

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Student's t-prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)

        # use scipy distribution percentage point function (ppf)
        return scipy.stats.t.ppf(val, self.df, loc=self.mu, scale=self.scale)

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """
        return scipy.stats.t.pdf(val, self.df, loc=self.mu, scale=self.scale)

    def ln_prob(self, val):
        return scipy.stats.t.logpdf(val, self.df, loc=self.mu, scale=self.scale)


class Beta(Prior):
    def __init__(self, alpha, beta, minimum=0, maximum=1, name=None,
                 latex_label=None, unit=None, periodic_boundary=False):
        """Beta distribution

        https://en.wikipedia.org/wiki/Beta_distribution

        This wraps around
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

        Parameters
        ----------
        alpha: float
            first shape parameter
        beta: float
            second shape parameter
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        if alpha <= 0. or beta <= 0.:
            raise ValueError("alpha and beta must both be positive values")

        self._alpha = alpha
        self._beta = beta
        self._minimum = minimum
        self._maximum = maximum
        Prior.__init__(self, minimum=minimum, maximum=maximum, name=name,
                       latex_label=latex_label, unit=unit, periodic_boundary=periodic_boundary)
        self._set_dist()

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Beta prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)

        # use scipy distribution percentage point function (ppf)
        return self._dist.ppf(val)

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """

        spdf = self._dist.pdf(val)
        if np.all(np.isfinite(spdf)):
            return spdf

        # deal with the fact that if alpha or beta are < 1 you get infinities at 0 and 1
        if isinstance(val, np.ndarray):
            pdf = np.zeros(len(val))
            pdf[np.isfinite(spdf)] = spdf[np.isfinite]
            return spdf
        else:
            return 0.

    def ln_prob(self, val):
        spdf = self._dist.logpdf(val)
        if np.all(np.isfinite(spdf)):
            return spdf

        if isinstance(val, np.ndarray):
            pdf = -np.inf * np.ones(len(val))
            pdf[np.isfinite(spdf)] = spdf[np.isfinite]
            return spdf
        else:
            return -np.inf

    def _set_dist(self):
        """Try/except to stop it falling over at instantiation"""
        self._dist = scipy.stats.beta(
            a=self.alpha, b=self.beta, loc=self.minimum,
            scale=(self.maximum - self.minimum))

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        self._maximum = maximum
        self._set_dist()

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        self._minimum = minimum
        self._set_dist()

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        self._set_dist()

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        self._set_dist()


class Logistic(Prior):
    def __init__(self, mu, scale, name=None, latex_label=None, unit=None, periodic_boundary=False):
        """Logistic distribution

        https://en.wikipedia.org/wiki/Logistic_distribution

        Parameters
        ----------
        mu: float
            Mean of the distribution
        scale: float
            Width of the distribution
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit, periodic_boundary=periodic_boundary)

        if scale <= 0.:
            raise ValueError("For the Logistic prior the scale must be positive")

        self.mu = mu
        self.scale = scale

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Logistic prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)

        # use scipy distribution percentage point function (ppf)
        return scipy.stats.logistic.ppf(val, loc=self.mu, scale=self.scale)

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """
        return scipy.stats.logistic.pdf(val, loc=self.mu, scale=self.scale)

    def ln_prob(self, val):
        return scipy.stats.logistic.logpdf(val, loc=self.mu, scale=self.scale)


class Cauchy(Prior):
    def __init__(self, alpha, beta, name=None, latex_label=None, unit=None, periodic_boundary=False):
        """Cauchy distribution

        https://en.wikipedia.org/wiki/Cauchy_distribution

        Parameters
        ----------
        alpha: float
            Location parameter
        beta: float
            Scale parameter
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit, periodic_boundary=periodic_boundary)

        if beta <= 0.:
            raise ValueError("For the Cauchy prior the scale must be positive")

        self.alpha = alpha
        self.beta = beta

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Cauchy prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)

        # use scipy distribution percentage point function (ppf)
        return scipy.stats.cauchy.ppf(val, loc=self.alpha, scale=self.beta)

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """
        return scipy.stats.cauchy.pdf(val, loc=self.alpha, scale=self.beta)

    def ln_prob(self, val):
        return scipy.stats.cauchy.logpdf(val, loc=self.alpha, scale=self.beta)


class Lorentzian(Cauchy):
    def __init__(self, alpha, beta, name=None, latex_label=None, unit=None, periodic_boundary=False):
        """Synonym for the Cauchy distribution

        https://en.wikipedia.org/wiki/Cauchy_distribution

        Parameters
        ----------
        alpha: float
            Location parameter
        beta: float
            Scale parameter
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Cauchy.__init__(self, alpha=alpha, beta=beta, name=name,
                        latex_label=latex_label, unit=unit, periodic_boundary=periodic_boundary)


class Gamma(Prior):
    def __init__(self, k, theta=1., name=None, latex_label=None, unit=None, periodic_boundary=False):
        """Gamma distribution

        https://en.wikipedia.org/wiki/Gamma_distribution

        Parameters
        ----------
        k: float
            The shape parameter
        theta: float
            The scale parameter
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """
        Prior.__init__(self, name=name, minimum=0., latex_label=latex_label,
                       unit=unit, periodic_boundary=periodic_boundary)

        if k <= 0 or theta <= 0:
            raise ValueError("For the Gamma prior the shape and scale must be positive")

        self.k = k
        self.theta = theta

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Gamma prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        Prior.test_valid_for_rescaling(val)

        # use scipy distribution percentage point function (ppf)
        return scipy.stats.gamma.ppf(val, self.k, loc=0., scale=self.theta)

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """

        return scipy.stats.gamma.pdf(val, self.k, loc=0., scale=self.theta)

    def ln_prob(self, val):
        return scipy.stats.gamma.logpdf(val, self.k, loc=0., scale=self.theta)


class ChiSquared(Gamma):
    def __init__(self, nu, name=None, latex_label=None, unit=None, periodic_boundary=False):
        """Chi-squared distribution

        https://en.wikipedia.org/wiki/Chi-squared_distribution

        Parameters
        ----------
        nu: int
            Number of degrees of freedom
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass
        """

        if nu <= 0 or not isinstance(nu, int):
            raise ValueError("For the ChiSquared prior the number of degrees of freedom must be a positive integer")

        Gamma.__init__(self, name=name, k=nu / 2., theta=2.,
                       latex_label=latex_label, unit=unit, periodic_boundary=periodic_boundary)

    @property
    def nu(self):
        return int(self.k * 2)

    @nu.setter
    def nu(self, nu):
        self.k = nu / 2.


class Interped(Prior):

    def __init__(self, xx, yy, minimum=np.nan, maximum=np.nan, name=None,
                 latex_label=None, unit=None, periodic_boundary=False):
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
        unit: str
            See superclass
        periodic_boundary: bool
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
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit,
                       minimum=np.nanmax(np.array((min(xx), minimum))),
                       maximum=np.nanmin(np.array((max(xx), maximum))),
                       periodic_boundary=periodic_boundary)
        self._update_instance()

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if np.array_equal(self.xx, other.xx) and np.array_equal(self.yy, other.yy):
            return True
        return False

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

    @property
    def minimum(self):
        """Return minimum of the prior distribution.

        Updates the prior distribution if minimum is set to a different value.

        Returns
        -------
        float: Minimum of the prior distribution

        """
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        self._minimum = minimum
        if '_maximum' in self.__dict__ and self._maximum < np.inf:
            self._update_instance()

    @property
    def maximum(self):
        """Return maximum of the prior distribution.

        Updates the prior distribution if maximum is set to a different value.

        Returns
        -------
        float: Maximum of the prior distribution

        """
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        self._maximum = maximum
        if '_minimum' in self.__dict__ and self._minimum < np.inf:
            self._update_instance()

    def _update_instance(self):
        self.xx = np.linspace(self.minimum, self.maximum, len(self.xx))
        self.yy = self.__all_interpolated(self.xx)
        self._initialize_attributes()

    def _initialize_attributes(self):
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

    def __init__(self, file_name, minimum=None, maximum=None, name=None,
                 latex_label=None, unit=None, periodic_boundary=False):
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
        unit: str
            See superclass
        periodic_boundary: bool
            See superclass

        Attributes
        -------
        all_interpolated: scipy.interpolate.interp1d
            Interpolated prior function

        """
        try:
            self.id = file_name
            xx, yy = np.genfromtxt(self.id).T
            Interped.__init__(self, xx=xx, yy=yy, minimum=minimum,
                              maximum=maximum, name=name, latex_label=latex_label,
                              unit=unit, periodic_boundary=periodic_boundary)
        except IOError:
            logger.warning("Can't load {}.".format(self.id))
            logger.warning("Format should be:")
            logger.warning(r"x\tp(x)")


class FermiDirac(Prior):
    def __init__(self, sigma, mu=None, r=None, name=None, latex_label=None,
                 unit=None):
        """A Fermi-Dirac type prior, with a fixed lower boundary at zero
        (see, e.g. Section 2.3.5 of [1]_). The probability distribution
        is defined by Equation 22 of [1]_.

        Parameters
        ----------
        sigma: float (required)
            The range over which the attenuation of the distribution happens
        mu: float
            The point at which the distribution falls to 50% of its maximum
            value
        r: float
            A value giving mu/sigma. This can be used instead of specifying
            mu.
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass

        References
        ----------

        .. [1] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
           <https:arxiv.org/abs/1705.08978v1>`_, 2017.
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit, minimum=0.)

        self.sigma = sigma

        if mu is None and r is None:
            raise ValueError("For the Fermi-Dirac prior either a 'mu' value or 'r' "
                             "value must be given.")

        if r is None and mu is not None:
            self.mu = mu
            self.r = self.mu / self.sigma
        else:
            self.r = r
            self.mu = self.sigma * self.r

        if self.r <= 0. or self.sigma <= 0.:
            raise ValueError("For the Fermi-Dirac prior the values of sigma and r "
                             "must be positive.")

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Fermi-Dirac prior.

        This maps to the inverse CDF. This has been analytically solved for this case,
        see Equation 24 of [1]_.

        References
        ----------

        .. [1] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
           <https:arxiv.org/abs/1705.08978v1>`_, 2017.
        """
        Prior.test_valid_for_rescaling(val)

        inv = (-np.exp(-1. * self.r) + (1. + np.exp(self.r))**-val +
               np.exp(-1. * self.r) * (1. + np.exp(self.r))**-val)

        # if val is 1 this will cause inv to be negative (due to numerical
        # issues), so return np.inf
        if isinstance(val, (float, int)):
            if inv < 0:
                return np.inf
            else:
                return -self.sigma * np.log(inv)
        else:
            idx = inv >= 0.
            tmpinv = np.inf * np.ones(len(val))
            tmpinv[idx] = -self.sigma * np.log(inv[idx])
            return tmpinv

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: float

        Returns
        -------
        float: Prior probability of val
        """
        return np.exp(self.ln_prob(val))

    def ln_prob(self, val):
        norm = -np.log(self.sigma * np.log(1. + np.exp(self.r)))
        if isinstance(val, (float, int)):
            if val < self.minimum:
                return -np.inf
            else:
                return norm - np.logaddexp((val / self.sigma) - self.r, 0.)
        else:
            lnp = -np.inf * np.ones(len(val))
            idx = val >= self.minimum
            lnp[idx] = norm - np.logaddexp((val[idx] / self.sigma) - self.r, 0.)
            return lnp
