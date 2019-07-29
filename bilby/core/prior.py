from __future__ import division

import re
from importlib import import_module
import os
from collections import OrderedDict
from future.utils import iteritems
from matplotlib.cbook import flatten

import numpy as np
import scipy.stats
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.special import erf, erfinv

from .utils import (
    logger, infer_args_from_method, check_directory_exists_and_if_not_mkdir)


class PriorDict(OrderedDict):
    def __init__(self, dictionary=None, filename=None,
                 conversion_function=None):
        """ A set of priors

        Parameters
        ----------
        dictionary: Union[dict, str, None]
            If given, a dictionary to generate the prior set.
        filename: Union[str, None]
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
        mvgs = []
        with open(prior_file, "w") as outfile:
            for key in self.keys():
                if isinstance(self[key], MultivariateGaussian):
                    mvgname = '_'.join(self[key].mvg.names) + '_mvg'
                    if mvgname not in mvgs:
                        mvgs.append(mvgname)
                        outfile.write(
                            "{} = {}\n".format(mvgname, self[key].mvg))
                    mvgstr = repr(self[key].mvg)
                    priorstr = repr(self[key])
                    outfile.write(
                        "{} = {}\n".format(key, priorstr.replace(mvgstr,
                                                                 mvgname)))
                else:
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
        Priors can be loaded from:
            bilby.core.prior as, e.g.,    foo = Uniform(minimum=0, maximum=1)
            floats, e.g.,                 foo = 1
            bilby.gw.prior as, e.g.,      foo = bilby.gw.prior.AlignedSpin()
            other external modules, e.g., foo = my.module.CustomPrior(...)
        """

        comments = ['#', '\n']
        prior = dict()
        mvgdict = dict(inf=np.inf)  # evaluate inf as np.inf
        with open(filename, 'r') as f:
            for line in f:
                if line[0] in comments:
                    continue
                line.replace(' ', '')
                elements = line.split('=')
                key = elements[0].replace(' ', '')
                val = '='.join(elements[1:]).strip()
                cls = val.split('(')[0]
                args = '('.join(val.split('(')[1:])[:-1]
                try:
                    prior[key] = DeltaFunction(peak=float(cls))
                    logger.debug("{} converted ot DeltaFunction prior".format(
                        key))
                    continue
                except ValueError:
                    pass
                if "." in cls:
                    module = '.'.join(cls.split('.')[:-1])
                    cls = cls.split('.')[-1]
                else:
                    module = __name__
                cls = getattr(import_module(module), cls)
                if key.lower() == "conversion_function":
                    setattr(self, key, cls)
                elif (cls.__name__ in ['MultivariateGaussianDist',
                                       'MultivariateNormalDist']):
                    if key not in mvgdict:
                        mvgdict[key] = eval(val, None, mvgdict)
                elif (cls.__name__ in ['MultivariateGaussian',
                                       'MultivariateNormal']):
                    prior[key] = eval(val, None, mvgdict)
                else:
                    try:
                        prior[key] = cls.from_repr(args)
                    except TypeError as e:
                        raise TypeError(
                            "Unable to parse dictionary file {}, bad line: {} "
                            "= {}. Error message {}".format(
                                filename, key, val, e))
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
                        "Failed to load dictionary value {} correctly"
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
        return list(flatten([self[key].rescale(sample) for key, sample in zip(keys, theta)]))

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
                 maximum=np.inf, boundary=None):
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
        boundary: str, optional
            The boundary condition of the prior, can be 'periodic', 'reflective'
            Currently implemented in cpnest, dynesty and pymultinest.
        """
        self.name = name
        self.latex_label = latex_label
        self.unit = unit
        self.minimum = minimum
        self.maximum = maximum
        self.boundary = boundary

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
            elif isinstance(self.__dict__[key], type(scipy.stats.beta(1., 1.))):
                continue
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
        val: Union[float, int, array_like]
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
        val: Union[float, int, array_like]

        Returns
        -------
        np.nan

        """
        return np.nan

    def cdf(self, val):
        """ Generic method to calculate CDF, can be overwritten in subclass """
        if np.any(np.isinf([self.minimum, self.maximum])):
            raise ValueError(
                "Unable to use the generic CDF calculation for priors with"
                "infinite support")
        x = np.linspace(self.minimum, self.maximum, 1000)
        pdf = self.prob(x)
        cdf = cumtrapz(pdf, x, initial=0)
        interp = interp1d(x, cdf, assume_sorted=True, bounds_error=False,
                          fill_value=(0, 1))
        return interp(val)

    def ln_prob(self, val):
        """Return the prior ln probability of val, this should be overwritten

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        np.nan

        """
        return np.log(self.prob(val))

    def is_in_prior_range(self, val):
        """Returns True if val is in the prior boundaries, zero otherwise

        Parameters
        ----------
        val: Union[float, int, array_like]

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
        val: Union[float, int, array_like]

        Raises
        -------
        ValueError: If val is not between 0 and 1
        """
        valarray = np.atleast_1d(val)
        tests = (valarray < 0) + (valarray > 1)
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
        prior_name = self.__class__.__name__
        args = ', '.join(['{}={}'.format(key, repr(self._repr_dict[key])) for key in self._repr_dict])
        return "{}({})".format(prior_name, args)

    @property
    def _repr_dict(self):
        """
        Get a dictionary containing the arguments needed to reproduce this object.
        """
        property_names = {p for p in dir(self.__class__) if isinstance(getattr(self.__class__, p), property)}
        subclass_args = infer_args_from_method(self.__init__)
        dict_with_properties = self.__dict__.copy()
        for key in property_names.intersection(subclass_args):
            dict_with_properties[key] = getattr(self, key)
        return {key: dict_with_properties[key] for key in subclass_args}

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
        """ If a unit is specified, returns a string of the latex label and unit """
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
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        if boundary not in ['periodic', 'reflective', None]:
            raise ValueError('{} is not a valid setting for prior boundaries'.format(boundary))
        self._boundary = boundary

    @property
    def __default_latex_label(self):
        if self.name in self._default_latex_labels.keys():
            label = self._default_latex_labels[self.name]
        else:
            label = self.name
        return label

    @classmethod
    def from_repr(cls, string):
        """Generate the prior from it's __repr__"""
        return cls._from_repr(string)

    @classmethod
    def _from_repr(cls, string):
        subclass_args = infer_args_from_method(cls.__init__)

        string = string.replace(' ', '')
        kwargs = cls._split_repr(string)
        for key in kwargs:
            val = kwargs[key]
            if key not in subclass_args:
                raise AttributeError('Unknown argument {} for class {}'.format(
                    key, cls.__name__))
            else:
                kwargs[key] = cls._parse_argument_string(val)
        return cls(**kwargs)

    @classmethod
    def _split_repr(cls, string):
        subclass_args = infer_args_from_method(cls.__init__)
        args = string.split(',')
        remove = list()
        for ii, key in enumerate(args):
            if '(' in key:
                jj = ii
                while ')' not in args[jj]:
                    jj += 1
                    args[ii] = ','.join([args[ii], args[jj]]).strip()
                    remove.append(jj)
        remove.reverse()
        for ii in remove:
            del args[ii]
        kwargs = dict()
        for ii, arg in enumerate(args):
            if '=' not in arg:
                logger.debug(
                    'Reading priors with non-keyword arguments is dangerous!')
                key = subclass_args[ii]
                val = arg
            else:
                split_arg = arg.split('=')
                key = split_arg[0]
                val = '='.join(split_arg[1:])
            kwargs[key] = val
        return kwargs

    @classmethod
    def _parse_argument_string(cls, val):
        if re.sub(r'\'.*\'', '', val) in ['r', 'u']:
            # If the val is a latex label like "r"\log(x)"' then ignore it
            pass
        elif '(' in val:
            other_cls = val.split('(')[0]
            vals = '('.join(val.split('(')[1:])[:-1]
            if "." in other_cls:
                module = '.'.join(other_cls.split('.')[:-1])
                other_cls = other_cls.split('.')[-1]
            else:
                module = __name__
            other_cls = getattr(import_module(module), other_cls)
            val = other_cls.from_repr(vals)
        elif "'" in val:
            val = val.strip("'")
        elif val == 'None':
            val = None
        else:
            try:
                val = eval(val, dict(), dict(np=np))
            except NameError:
                raise TypeError()
        return val


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
        val: Union[float, int, array_like]

        Returns
        -------
        float: Rescaled probability, equivalent to peak
        """
        self.test_valid_for_rescaling(val)
        return self.peak * val ** 0

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
         Union[float, array_like]: np.inf if val = peak, 0 otherwise

        """
        at_peak = (val == self.peak)
        return np.nan_to_num(np.multiply(at_peak, np.inf))

    def cdf(self, val):
        return np.ones_like(val) * (val > self.peak)


class PowerLaw(Prior):

    def __init__(self, alpha, minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label,
                       minimum=minimum, maximum=maximum, unit=unit,
                       boundary=boundary)
        self.alpha = alpha

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the power-law prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ----------
        val: Union[float, int, array_like]
            Uniform probability

        Returns
        -------
        Union[float, array_like]: Rescaled probability
        """
        self.test_valid_for_rescaling(val)
        if self.alpha == -1:
            return self.minimum * np.exp(val * np.log(self.maximum / self.minimum))
        else:
            return (self.minimum ** (1 + self.alpha) + val *
                    (self.maximum ** (1 + self.alpha) - self.minimum ** (1 + self.alpha))) ** (1. / (1 + self.alpha))

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ----------
        val: Union[float, int, array_like]

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
        val: Union[float, int, array_like]

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

    def cdf(self, val):
        if self.alpha == -1:
            _cdf = (
                np.log(val / self.minimum) /
                np.log(self.maximum / self.minimum))
        else:
            _cdf = np.atleast_1d(
                val**(self.alpha + 1) - self.minimum**(self.alpha + 1)
            ) / (
                self.maximum**(self.alpha + 1) - self.minimum**(self.alpha + 1))
        _cdf = np.minimum(_cdf, 1)
        _cdf = np.maximum(_cdf, 0)
        return _cdf


class Uniform(Prior):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label,
                       minimum=minimum, maximum=maximum, unit=unit,
                       boundary=boundary)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the power-law prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ----------
        val: Union[float, int, array_like]
            Uniform probability

        Returns
        -------
        Union[float, array_like]: Rescaled probability
        """
        self.test_valid_for_rescaling(val)
        return self.minimum + val * (self.maximum - self.minimum)

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ----------
        val: Union[float, int, array_like]

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
        val: Union[float, int, array_like]

        Returns
        -------
        float: log probability of val
        """
        return scipy.stats.uniform.logpdf(val, loc=self.minimum,
                                          scale=self.maximum - self.minimum)

    def cdf(self, val):
        _cdf = (val - self.minimum) / (self.maximum - self.minimum)
        _cdf = np.minimum(_cdf, 1)
        _cdf = np.maximum(_cdf, 0)
        return _cdf


class LogUniform(PowerLaw):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        PowerLaw.__init__(self, name=name, latex_label=latex_label, unit=unit,
                          minimum=minimum, maximum=maximum, alpha=-1, boundary=boundary)
        if self.minimum <= 0:
            logger.warning('You specified a uniform-in-log prior with minimum={}'.format(self.minimum))


class SymmetricLogUniform(Prior):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
        """Symmetric Log-Uniform distribtions with bounds

        This is identical to a Log-Uniform distribution, but mirrored about
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
        boundary: str
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label,
                       minimum=minimum, maximum=maximum, unit=unit,
                       boundary=boundary)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the power-law prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ----------
        val: Union[float, int, array_like]
            Uniform probability

        Returns
        -------
        Union[float, array_like]: Rescaled probability
        """
        self.test_valid_for_rescaling(val)
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
        val: Union[float, int, array_like]

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
        val: Union[float, int, array_like]

        Returns
        -------
        float:

        """
        return np.nan_to_num(- np.log(2 * np.abs(val)) - np.log(np.log(self.maximum / self.minimum)))


class Cosine(Prior):

    def __init__(self, name=None, latex_label=None, unit=None,
                 minimum=-np.pi / 2, maximum=np.pi / 2, boundary=None):
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
        boundary: str
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit,
                       minimum=minimum, maximum=maximum, boundary=boundary)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to a uniform in cosine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        self.test_valid_for_rescaling(val)
        norm = 1 / (np.sin(self.maximum) - np.sin(self.minimum))
        return np.arcsin(val / norm + np.sin(self.minimum))

    def prob(self, val):
        """Return the prior probability of val. Defined over [-pi/2, pi/2].

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        float: Prior probability of val
        """
        return np.cos(val) / 2 * self.is_in_prior_range(val)

    def cdf(self, val):
        _cdf = np.atleast_1d((np.sin(val) - np.sin(self.minimum)) /
                             (np.sin(self.maximum) - np.sin(self.minimum)))
        _cdf[val > self.maximum] = 1
        _cdf[val < self.minimum] = 0
        return _cdf


class Sine(Prior):

    def __init__(self, name=None, latex_label=None, unit=None, minimum=0,
                 maximum=np.pi, boundary=None):
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
        boundary: str
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit,
                       minimum=minimum, maximum=maximum, boundary=boundary)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to a uniform in sine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        self.test_valid_for_rescaling(val)
        norm = 1 / (np.cos(self.minimum) - np.cos(self.maximum))
        return np.arccos(np.cos(self.minimum) - val / norm)

    def prob(self, val):
        """Return the prior probability of val. Defined over [0, pi].

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
        """
        return np.sin(val) / 2 * self.is_in_prior_range(val)

    def cdf(self, val):
        _cdf = np.atleast_1d((np.cos(val) - np.cos(self.minimum)) /
                             (np.cos(self.maximum) - np.cos(self.minimum)))
        _cdf[val > self.maximum] = 1
        _cdf[val < self.minimum] = 0
        return _cdf


class Gaussian(Prior):

    def __init__(self, mu, sigma, name=None, latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit, boundary=boundary)
        self.mu = mu
        self.sigma = sigma

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Gaussian prior.

        Parameters
        ----------
        val: Union[float, int, array_like]

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        self.test_valid_for_rescaling(val)
        return self.mu + erfinv(2 * val - 1) * 2 ** 0.5 * self.sigma

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
        """
        return np.exp(-(self.mu - val) ** 2 / (2 * self.sigma ** 2)) / (2 * np.pi) ** 0.5 / self.sigma

    def ln_prob(self, val):
        """Return the Log prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
        """

        return -0.5 * ((self.mu - val) ** 2 / self.sigma ** 2 + np.log(2 * np.pi * self.sigma ** 2))

    def cdf(self, val):
        return (1 - erf((self.mu - val) / 2**0.5 / self.sigma)) / 2


class Normal(Gaussian):

    def __init__(self, mu, sigma, name=None, latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        Gaussian.__init__(self, mu=mu, sigma=sigma, name=name, latex_label=latex_label,
                          unit=unit, boundary=boundary)


class TruncatedGaussian(Prior):

    def __init__(self, mu, sigma, minimum, maximum, name=None,
                 latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit,
                       minimum=minimum, maximum=maximum, boundary=boundary)
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
        self.test_valid_for_rescaling(val)
        return erfinv(2 * val * self.normalisation + erf(
            (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) * 2 ** 0.5 * self.sigma + self.mu

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        float: Prior probability of val
        """
        return np.exp(-(self.mu - val) ** 2 / (2 * self.sigma ** 2)) / \
            (2 * np.pi) ** 0.5 / self.sigma / self.normalisation * self.is_in_prior_range(val)

    def cdf(self, val):
        _cdf = (erf((val - self.mu) / 2 ** 0.5 / self.sigma) - erf(
            (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) / 2 / self.normalisation
        _cdf[val > self.maximum] = 1
        _cdf[val < self.minimum] = 0
        return _cdf


class TruncatedNormal(TruncatedGaussian):

    def __init__(self, mu, sigma, minimum, maximum, name=None,
                 latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        TruncatedGaussian.__init__(self, mu=mu, sigma=sigma, minimum=minimum,
                                   maximum=maximum, name=name, latex_label=latex_label,
                                   unit=unit, boundary=boundary)


class HalfGaussian(TruncatedGaussian):
    def __init__(self, sigma, name=None, latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        TruncatedGaussian.__init__(self, 0., sigma, minimum=0., maximum=np.inf,
                                   name=name, latex_label=latex_label,
                                   unit=unit, boundary=boundary)


class HalfNormal(HalfGaussian):
    def __init__(self, sigma, name=None, latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        HalfGaussian.__init__(self, sigma=sigma, name=name,
                              latex_label=latex_label, unit=unit,
                              boundary=boundary)


class LogNormal(Prior):
    def __init__(self, mu, sigma, name=None, latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        Prior.__init__(self, name=name, minimum=0., latex_label=latex_label,
                       unit=unit, boundary=boundary)

        if sigma <= 0.:
            raise ValueError("For the LogGaussian prior the standard deviation must be positive")

        self.mu = mu
        self.sigma = sigma

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate LogNormal prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        self.test_valid_for_rescaling(val)
        return scipy.stats.lognorm.ppf(val, self.sigma, scale=np.exp(self.mu))

    def prob(self, val):
        """Returns the prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
        """

        return scipy.stats.lognorm.pdf(val, self.sigma, scale=np.exp(self.mu))

    def ln_prob(self, val):
        """Returns the log prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
        """

        return scipy.stats.lognorm.logpdf(val, self.sigma, scale=np.exp(self.mu))

    def cdf(self, val):
        return scipy.stats.lognorm.cdf(val, self.sigma, scale=np.exp(self.mu))


class LogGaussian(LogNormal):
    def __init__(self, mu, sigma, name=None, latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        LogNormal.__init__(self, mu=mu, sigma=sigma, name=name,
                           latex_label=latex_label, unit=unit, boundary=boundary)


class Exponential(Prior):
    def __init__(self, mu, name=None, latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        Prior.__init__(self, name=name, minimum=0., latex_label=latex_label,
                       unit=unit, boundary=boundary)
        self.mu = mu

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Exponential prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        self.test_valid_for_rescaling(val)
        return scipy.stats.expon.ppf(val, scale=self.mu)

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
        """

        return scipy.stats.expon.pdf(val, scale=self.mu)

    def ln_prob(self, val):
        """Returns the log prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
        """

        return scipy.stats.expon.logpdf(val, scale=self.mu)

    def cdf(self, val):
        return scipy.stats.expon.cdf(val, scale=self.mu)


class StudentT(Prior):
    def __init__(self, df, mu=0., scale=1., name=None, latex_label=None,
                 unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit, boundary=boundary)

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
        self.test_valid_for_rescaling(val)

        # use scipy distribution percentage point function (ppf)
        return scipy.stats.t.ppf(val, self.df, loc=self.mu, scale=self.scale)

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
        """
        return scipy.stats.t.pdf(val, self.df, loc=self.mu, scale=self.scale)

    def ln_prob(self, val):
        """Returns the log prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
        """

        return scipy.stats.t.logpdf(val, self.df, loc=self.mu, scale=self.scale)

    def cdf(self, val):
        return scipy.stats.t.cdf(val, self.df, loc=self.mu, scale=self.scale)


class Beta(Prior):
    def __init__(self, alpha, beta, minimum=0, maximum=1, name=None,
                 latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        if alpha <= 0. or beta <= 0.:
            raise ValueError("alpha and beta must both be positive values")

        self._alpha = alpha
        self._beta = beta
        self._minimum = minimum
        self._maximum = maximum
        Prior.__init__(self, minimum=minimum, maximum=maximum, name=name,
                       latex_label=latex_label, unit=unit, boundary=boundary)
        self._set_dist()

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Beta prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        self.test_valid_for_rescaling(val)

        # use scipy distribution percentage point function (ppf)
        return self._dist.ppf(val)

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
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
        """Returns the log prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
        """

        spdf = self._dist.logpdf(val)
        if np.all(np.isfinite(spdf)):
            return spdf

        if isinstance(val, np.ndarray):
            pdf = -np.inf * np.ones(len(val))
            pdf[np.isfinite(spdf)] = spdf[np.isfinite]
            return spdf
        else:
            return -np.inf

    def cdf(self, val):
        return self._dist.cdf(val)

    def _set_dist(self):
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
    def __init__(self, mu, scale, name=None, latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit, boundary=boundary)

        if scale <= 0.:
            raise ValueError("For the Logistic prior the scale must be positive")

        self.mu = mu
        self.scale = scale

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Logistic prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        self.test_valid_for_rescaling(val)

        # use scipy distribution percentage point function (ppf)
        return scipy.stats.logistic.ppf(val, loc=self.mu, scale=self.scale)

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
        """
        return scipy.stats.logistic.pdf(val, loc=self.mu, scale=self.scale)

    def ln_prob(self, val):
        """Returns the log prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
        """

        return scipy.stats.logistic.logpdf(val, loc=self.mu, scale=self.scale)

    def cdf(self, val):
        return scipy.stats.logistic.cdf(val, loc=self.mu, scale=self.scale)


class Cauchy(Prior):
    def __init__(self, alpha, beta, name=None, latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit, boundary=boundary)

        if beta <= 0.:
            raise ValueError("For the Cauchy prior the scale must be positive")

        self.alpha = alpha
        self.beta = beta

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Cauchy prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        self.test_valid_for_rescaling(val)

        # use scipy distribution percentage point function (ppf)
        return scipy.stats.cauchy.ppf(val, loc=self.alpha, scale=self.beta)

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
        """
        return scipy.stats.cauchy.pdf(val, loc=self.alpha, scale=self.beta)

    def ln_prob(self, val):
        """Return the log prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Log prior probability of val
        """
        return scipy.stats.cauchy.logpdf(val, loc=self.alpha, scale=self.beta)

    def cdf(self, val):
        return scipy.stats.cauchy.cdf(val, loc=self.alpha, scale=self.beta)


class Lorentzian(Cauchy):
    def __init__(self, alpha, beta, name=None, latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        Cauchy.__init__(self, alpha=alpha, beta=beta, name=name,
                        latex_label=latex_label, unit=unit, boundary=boundary)


class Gamma(Prior):
    def __init__(self, k, theta=1., name=None, latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """
        Prior.__init__(self, name=name, minimum=0., latex_label=latex_label,
                       unit=unit, boundary=boundary)

        if k <= 0 or theta <= 0:
            raise ValueError("For the Gamma prior the shape and scale must be positive")

        self.k = k
        self.theta = theta

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Gamma prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        self.test_valid_for_rescaling(val)

        # use scipy distribution percentage point function (ppf)
        return scipy.stats.gamma.ppf(val, self.k, loc=0., scale=self.theta)

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val:  Union[float, int, array_like]

        Returns
        -------
         Union[float, array_like]: Prior probability of val
        """

        return scipy.stats.gamma.pdf(val, self.k, loc=0., scale=self.theta)

    def ln_prob(self, val):
        """Returns the log prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Prior probability of val
        """

        return scipy.stats.gamma.logpdf(val, self.k, loc=0., scale=self.theta)

    def cdf(self, val):
        return scipy.stats.gamma.cdf(val, self.k, loc=0., scale=self.theta)


class ChiSquared(Gamma):
    def __init__(self, nu, name=None, latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass
        """

        if nu <= 0 or not isinstance(nu, int):
            raise ValueError("For the ChiSquared prior the number of degrees of freedom must be a positive integer")

        Gamma.__init__(self, name=name, k=nu / 2., theta=2.,
                       latex_label=latex_label, unit=unit, boundary=boundary)

    @property
    def nu(self):
        return int(self.k * 2)

    @nu.setter
    def nu(self, nu):
        self.k = nu / 2.


class Interped(Prior):

    def __init__(self, xx, yy, minimum=np.nan, maximum=np.nan, name=None,
                 latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass

        Attributes
        ----------
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
        self.YY = None
        self.probability_density = None
        self.cumulative_distribution = None
        self.inverse_cumulative_distribution = None
        self.__all_interpolated = interp1d(x=xx, y=yy, bounds_error=False, fill_value=0)
        minimum = float(np.nanmax(np.array((min(xx), minimum))))
        maximum = float(np.nanmin(np.array((max(xx), maximum))))
        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit,
                       minimum=minimum, maximum=maximum, boundary=boundary)
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
        val:  Union[float, int, array_like]

        Returns
        -------
         Union[float, array_like]: Prior probability of val
        """
        return self.probability_density(val)

    def cdf(self, val):
        return self.cumulative_distribution(val)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior.

        This maps to the inverse CDF. This is done using interpolation.
        """
        self.test_valid_for_rescaling(val)
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
        self.cumulative_distribution = interp1d(x=self.xx, y=self.YY, bounds_error=False, fill_value=(0, 1))
        self.inverse_cumulative_distribution = interp1d(x=self.YY, y=self.xx, bounds_error=True)


class FromFile(Interped):

    def __init__(self, file_name, minimum=None, maximum=None, name=None,
                 latex_label=None, unit=None, boundary=None):
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
        boundary: str
            See superclass

        """
        try:
            self.id = file_name
            xx, yy = np.genfromtxt(self.id).T
            Interped.__init__(self, xx=xx, yy=yy, minimum=minimum,
                              maximum=maximum, name=name, latex_label=latex_label,
                              unit=unit, boundary=boundary)
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

        Parameters
        ----------
        val: Union[float, int, array_like]

        This maps to the inverse CDF. This has been analytically solved for this case,
        see Equation 24 of [1]_.

        References
        ----------

        .. [1] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
           <https:arxiv.org/abs/1705.08978v1>`_, 2017.
        """
        self.test_valid_for_rescaling(val)

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
            tmpinv = np.inf * np.ones(len(np.atleast_1d(val)))
            tmpinv[idx] = -self.sigma * np.log(inv[idx])
            return tmpinv

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        float: Prior probability of val
        """
        return np.exp(self.ln_prob(val))

    def ln_prob(self, val):
        """Return the log prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        Union[float, array_like]: Log prior probability of val
        """

        norm = -np.log(self.sigma * np.log(1. + np.exp(self.r)))
        if isinstance(val, (float, int)):
            if val < self.minimum:
                return -np.inf
            else:
                return norm - np.logaddexp((val / self.sigma) - self.r, 0.)
        else:
            val = np.atleast_1d(val)
            lnp = -np.inf * np.ones(len(val))
            idx = val >= self.minimum
            lnp[idx] = norm - np.logaddexp((val[idx] / self.sigma) - self.r, 0.)
            return lnp


class MultivariateGaussianDist(object):

    def __init__(self, names, nmodes=1, mus=None, sigmas=None, corrcoefs=None,
                 covs=None, weights=None, bounds=None):
        """
        A class defining a multi-variate Gaussian, allowing multiple modes for
        a Gaussian mixture model.

        Note: if using a multivariate Gaussian prior, with bounds, this can
        lead to biases in the marginal likelihood estimate and posterior
        estimate for nested samplers routines that rely on sampling from a unit
        hypercube and having a prior transform, e.g., nestle, dynesty and
        MultiNest.

        Parameters
        ----------
        names: list
            A list of the parameter names in the multivariate Gaussian. The
            listed parameters must have the same order that they appear in
            the lists of means, standard deviations, and the correlation
            coefficient, or covariance, matrices.
        nmodes: int
            The number of modes for the mixture model. This defaults to 1,
            which will be checked against the shape of the other inputs.
        mus: array_like
            A list of lists of means of each mode in a multivariate Gaussian
            mixture model. A single list can be given for a single mode. If
            this is None then means at zero will be assumed.
        sigmas: array_like
            A list of lists of the standard deviations of each mode of the
            multivariate Gaussian. If supplying a correlation coefficient
            matrix rather than a covariance matrix these values must be given.
            If this is None unit variances will be assumed.
        corrcoefs: array
            A list of square matrices containing the correlation coefficients
            of the parameters for each mode. If this is None it will be assumed
            that the parameters are uncorrelated.
        covs: array
            A list of square matrices containing the covariance matrix of the
            multivariate Gaussian.
        weights: list
            A list of weights (relative probabilities) for each mode of the
            multivariate Gaussian. This will default to equal weights for each
            mode.
        bounds: list
            A list of bounds on each parameter. The defaults are for bounds at
            +/- infinity.
        """

        if not isinstance(names, list):
            self.names = [names]
        else:
            self.names = names

        self.num_vars = len(self.names)  # the number of parameters

        # set the bounds for each parameter
        if isinstance(bounds, list):
            if len(bounds) != len(self):
                raise ValueError("Wrong number of parameter bounds")

            # check bounds
            for bound in bounds:
                if isinstance(bounds, (list, tuple, np.ndarray)):
                    if len(bound) != 2:
                        raise ValueError("Bounds must contain an upper and "
                                         "lower value.")
                    else:
                        if bound[1] <= bound[0]:
                            raise ValueError("Bounds are not properly set")
                else:
                    raise TypeError("Bound must be a list")

                logger.warning("If using bounded ranges on the multivariate "
                               "Gaussian this will lead to biased posteriors "
                               "for nested sampling routines that require "
                               "a prior transform.")
        else:
            bounds = [(-np.inf, np.inf) for _ in self.names]

        # set bounds as dictionary
        self.bounds = {name: val for name, val in zip(self.names, bounds)}

        self.mus = []
        self.covs = []
        self.corrcoefs = []
        self.sigmas = []
        self.weights = []
        self.eigvalues = []
        self.eigvectors = []
        self.sqeigvalues = []  # square root of the eigenvalues
        self.mvn = []  # list of multivariate normal distributions

        self._current_sample = {}  # initialise empty sample
        self._uncorrelated = None
        self._current_lnprob = None

        # put values in lists if required
        if nmodes == 1:
            if mus is not None:
                if len(np.shape(mus)) == 1:
                    mus = [mus]
                elif len(np.shape(mus)) == 0:
                    raise ValueError("Must supply a list of means")
            if sigmas is not None:
                if len(np.shape(sigmas)) == 1:
                    sigmas = [sigmas]
                elif len(np.shape(sigmas)) == 0:
                    raise ValueError("Must supply a list of standard "
                                     "deviations")
            if covs is not None:
                if isinstance(covs, np.ndarray):
                    covs = [covs]
                elif isinstance(covs, list):
                    if len(np.shape(covs)) == 2:
                        covs = [np.array(covs)]
                    elif len(np.shape(covs)) != 3:
                        raise TypeError("List of covariances the wrong shape")
                else:
                    raise TypeError("Must pass a list of covariances")
            if corrcoefs is not None:
                if isinstance(corrcoefs, np.ndarray):
                    corrcoefs = [corrcoefs]
                elif isinstance(corrcoefs, list):
                    if len(np.shape(corrcoefs)) == 2:
                        corrcoefs = [np.array(corrcoefs)]
                    elif len(np.shape(corrcoefs)) != 3:
                        raise TypeError("List of correlation coefficients the wrong shape")
                elif not isinstance(corrcoefs, list):
                    raise TypeError("Must pass a list of correlation "
                                    "coefficients")
            if weights is not None:
                if isinstance(weights, (int, float)):
                    weights = [weights]
                elif isinstance(weights, list):
                    if len(weights) != 1:
                        raise ValueError("Wrong number of weights given")

        for val in [mus, sigmas, covs, corrcoefs, weights]:
            if val is not None and not isinstance(val, list):
                raise TypeError("Value must be a list")
            else:
                if val is not None and len(val) != nmodes:
                    raise ValueError("Wrong number of modes given")

        # add the modes
        self.nmodes = 0
        for i in range(nmodes):
            mu = mus[i] if mus is not None else None
            sigma = sigmas[i] if sigmas is not None else None
            corrcoef = corrcoefs[i] if corrcoefs is not None else None
            cov = covs[i] if covs is not None else None
            weight = weights[i] if weights is not None else 1.

            self.add_mode(mu, sigma, corrcoef, cov, weight)

        # a dictionary of the parameters as requested by the prior
        self.requested_parameters = OrderedDict()
        self.reset_request()

        # a dictionary of the rescaled parameters
        self.rescale_parameters = OrderedDict()
        self.reset_rescale()

        # a list of sampled parameters
        self.reset_sampled()

    def reset_sampled(self):
        self.sampled_parameters = []
        self.current_sample = {}

    def filled_request(self):
        """
        Check if all requested parameters have been filled.
        """

        return not np.any([val is None for val in
                          self.requested_parameters.values()])

    def reset_request(self):
        """
        Reset the requested parameters to None.
        """

        for name in self.names:
            self.requested_parameters[name] = None

    def filled_rescale(self):
        """
        Check is all the rescaled parameters have been filled.
        """

        return not np.any([val is None for val in
                          self.rescale_parameters.values()])

    def reset_rescale(self):
        """
        Reset the rescaled parameters to None.
        """

        for name in self.names:
            self.rescale_parameters[name] = None

    def add_mode(self, mus=None, sigmas=None, corrcoef=None, cov=None,
                 weight=1.):
        """
        Add a new mode.
        """

        # add means
        if mus is not None:
            try:
                self.mus.append(list(mus))  # means
            except TypeError:
                raise TypeError("'mus' must be a list")
        else:
            self.mus.append(np.zeros(self.num_vars))

        # add the covariances if supplied
        if cov is not None:
            self.covs.append(np.asarray(cov))

            if len(self.covs[-1].shape) != 2:
                raise ValueError("Covariance matrix must be a 2d array")

            if (self.covs[-1].shape[0] != self.covs[-1].shape[1] or
                    self.covs[-1].shape[0] != self.num_vars):
                raise ValueError("Covariance shape is inconsistent")

            # check matrix is symmetric
            if not np.allclose(self.covs[-1], self.covs[-1].T):
                raise ValueError("Covariance matrix is not symmetric")

            self.sigmas.append(np.sqrt(np.diag(self.covs[-1])))  # standard deviations

            # convert covariance into a correlation coefficient matrix
            D = self.sigmas[-1] * np.identity(self.covs[-1].shape[0])
            Dinv = np.linalg.inv(D)
            self.corrcoefs.append(np.dot(np.dot(Dinv, self.covs[-1]), Dinv))
        elif corrcoef is not None and sigmas is not None:
            self.corrcoefs.append(np.asarray(corrcoef))

            if len(self.corrcoefs[-1].shape) != 2:
                raise ValueError("Correlation coefficient matrix must be a 2d "
                                 "array.")

            if (self.corrcoefs[-1].shape[0] != self.corrcoefs[-1].shape[1] or
                    self.corrcoefs[-1].shape[0] != self.num_vars):
                raise ValueError("Correlation coefficient matrix shape is "
                                 "inconsistent")

            # check matrix is symmetric
            if not np.allclose(self.corrcoefs[-1], self.corrcoefs[-1].T):
                raise ValueError("Correlation coefficient matrix is not "
                                 "symmetric")

            # check diagonal is all ones
            if not np.all(np.diag(self.corrcoefs[-1]) == 1.):
                raise ValueError("Correlation coefficient matrix is not"
                                 "correct")

            try:
                self.sigmas.append(list(sigmas))  # standard deviations
            except TypeError:
                raise TypeError("'sigmas' must be a list")

            if len(self.sigmas[-1]) != self.num_vars:
                raise ValueError("Number of standard deviations must be the "
                                 "same as the number of parameters.")

            # convert correlation coefficients to covariance matrix
            D = self.sigmas[-1] * np.identity(self.corrcoefs[-1].shape[0])
            self.covs.append(np.dot(D, np.dot(self.corrcoefs[-1], D)))
        else:
            # set unit variance uncorrelated covariance
            self.corrcoefs.append(np.eye(self.num_vars))
            self.covs.append(np.eye(self.num_vars))
            self.sigmas.append(np.ones(self.num_vars))

        # get eigen values and vectors
        try:
            evals, evecs = np.linalg.eig(self.corrcoefs[-1])
            self.eigvalues.append(evals)
            self.eigvectors.append(evecs)
        except Exception as e:
            raise RuntimeError("Problem getting eigenvalues and vectors: "
                               "{}".format(e))

        # check eigenvalues are positive
        if np.any(self.eigvalues[-1] <= 0.):
            raise ValueError("Correlation coefficient matrix is not positive "
                             "definite")
        self.sqeigvalues.append(np.sqrt(self.eigvalues[-1]))

        # set the weights
        if weight is None:
            self.weights.append(1.)
        else:
            self.weights.append(weight)

        # set the cumulative relative weights
        self.cumweights = np.cumsum(self.weights) / np.sum(self.weights)

        # add the mode
        self.nmodes += 1

        # add multivariate Gaussian
        self.mvn.append(scipy.stats.multivariate_normal(mean=self.mus[-1],
                                                        cov=self.covs[-1]))

    def rescale(self, value, mode=None):
        """
        Rescale from a unit hypercube to multivariate Gaussian. Note that no
        bounds are applied in the rescale function.

        Parameters
        ----------
        value: array
            A 1d vector sample (one for each parameter) drawn from a uniform
            distribution between 0 and 1, or a 2d NxM array of samples where
            N is the number of samples and M is the number of parameters.
        mode: int
            Specify which mode to sample from. If not set then a mode is
            chosen randomly based on its weight.

        Returns
        -------
        array:
            An vector sample drawn from the multivariate Gaussian
            distribution.
        """

        # pick a mode (with a probability given by their weights)
        if mode is None:
            if self.nmodes == 1:
                mode = 0
            else:
                mode = np.argwhere(self.cumweights - np.random.rand() > 0)[0][0]

        samp = np.asarray(value)
        if len(samp.shape) == 1:
            samp = samp.reshape(1, self.num_vars)

        if len(samp.shape) != 2:
            raise ValueError("Array is the wrong shape")
        elif samp.shape[1] != self.num_vars:
            raise ValueError("Array is the wrong shape")

        # draw points from unit variance, uncorrelated Gaussian
        samp = erfinv(2. * samp - 1) * 2. ** 0.5

        # rotate and scale to the multivariate normal shape
        samp = self.mus[mode] + self.sigmas[mode] * np.einsum('ij,kj->ik',
                                                              samp * self.sqeigvalues[mode],
                                                              self.eigvectors[mode])

        return np.squeeze(samp)

    def sample(self, size=1, mode=None):
        """
        Draw, and set, a sample from the multivariate Gaussian.

        Parameters
        ----------
        mode: int
            Specify which mode to sample from. If not set then a mode is
            chosen randomly based on its weight.
        """

        if size is None:
            size = 1

        # samples drawn from unit variance uncorrelated multivariate Gaussian
        samps = np.zeros((size, len(self)))
        for i in range(size):
            inbound = False
            while not inbound:
                # sample the multivariate Gaussian keys
                vals = np.random.uniform(0, 1, len(self))

                samp = self.rescale(vals, mode=mode)
                samps[i, :] = samp

                # check sample is in bounds (otherwise perform another draw)
                outbound = False
                for name, val in zip(self.names, samp):
                    if val < self.bounds[name][0] or val > self.bounds[name][1]:
                        outbound = True
                        break

                if not outbound:
                    inbound = True

        for i, name in enumerate(self.names):
            if size == 1:
                self.current_sample[name] = samps[:, i].flatten()[0]
            else:
                self.current_sample[name] = samps[:, i].flatten()

    def ln_prob(self, value):
        """
        Get the log-probability of a sample. For bounded priors the
        probability will not be properly normalised.

        Parameters
        ----------
        value: array_like
            A 1d vector of the sample, or 2d array of sample values with shape
            NxM, where N is the number of samples and M is the number of
            parameters.
        """

        samp = np.asarray(value)
        if len(samp.shape) == 1:
            samp = samp.reshape(1, self.num_vars)

        if len(samp.shape) != 2:
            raise ValueError("Array is the wrong shape")
        elif samp.shape[1] != self.num_vars:
            raise ValueError("Array is the wrong shape")

        # check sample(s) is within bounds
        outbounds = np.ones(samp.shape[0], dtype=np.bool)
        for s, bound in zip(samp.T, self.bounds.values()):
            outbounds = (s < bound[0]) | (s > bound[1])
            if np.any(outbounds):
                break

        lnprob = -np.inf * np.ones(samp.shape[0])
        for j in range(samp.shape[0]):
            # loop over the modes and sum the probabilities
            for i in range(self.nmodes):
                lnprob[j] = np.logaddexp(lnprob[j], self.mvn[i].logpdf(samp[j]))

        # set out-of-bounds values to -inf
        lnprob[outbounds] = -np.inf

        if samp.shape[0] == 1:
            return lnprob[0]
        else:
            return lnprob

    def prob(self, samp):
        """
        Get the probability of a sample. For bounded priors the
        probability will not be properly normalised.
        """

        return np.exp(self.ln_prob(samp))

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        """Overrides the special method __repr__.

        Returns a representation of this instance that resembles how it is instantiated.
        Works correctly for all child classes

        Returns
        -------
        str: A string representation of this instance

        """
        subclass_args = infer_args_from_method(self.__init__)
        dist_name = self.__class__.__name__

        property_names = [p for p in dir(self.__class__) if isinstance(getattr(self.__class__, p), property)]
        dict_with_properties = self.__dict__.copy()
        for key in property_names:
            dict_with_properties[key] = getattr(self, key)

        argslist = []
        for key in subclass_args:
            # make sure lists containing arrays are returned just as lists
            if isinstance(dict_with_properties[key], list):
                argsval = np.asarray(dict_with_properties[key]).tolist()
            else:
                argsval = dict_with_properties[key]
            argslist.append('{}={}'.format(key, repr(argsval)))
        args = ', '.join(argslist)
        return "{}({})".format(dist_name, args)

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if sorted(self.__dict__.keys()) != sorted(other.__dict__.keys()):
            return False
        for key in self.__dict__:
            if key == 'mvn':
                if len(self.__dict__[key]) != len(other.__dict__[key]):
                    return False
                for thismvn, othermvn in zip(self.__dict__[key], other.__dict__[key]):
                    if (not isinstance(thismvn, scipy.stats._multivariate.multivariate_normal_frozen) or
                            not isinstance(othermvn, scipy.stats._multivariate.multivariate_normal_frozen)):
                        return False
            elif isinstance(self.__dict__[key], (np.ndarray, list)):
                thisarr = np.asarray(self.__dict__[key])
                otherarr = np.asarray(other.__dict__[key])
                if thisarr.dtype == np.float and otherarr.dtype == np.float:
                    fin1 = np.isfinite(np.asarray(self.__dict__[key]))
                    fin2 = np.isfinite(np.asarray(other.__dict__[key]))
                    if not np.array_equal(fin1, fin2):
                        return False
                    if not np.allclose(thisarr[fin1], otherarr[fin2], atol=1e-15):
                        return False
                else:
                    if not np.array_equal(thisarr, otherarr):
                        return False
            else:
                if not self.__dict__[key] == other.__dict__[key]:
                    return False
        return True


class MultivariateNormalDist(MultivariateGaussianDist):

    def __init__(self, names, nmodes=1, mus=None, sigmas=None, corrcoefs=None,
                 covs=None, weights=None, bounds=None):
        """
        A synonym for the :class:`~bilby.core.prior.MultivariateGaussianDist`
        distribution.

        Parameters
        ----------
        names: list
            A list of the parameter names in the multivariate Gaussian. The
            listed parameters must have the same order that they appear in
            the lists of means, standard deviations, and the correlation
            coefficient, or covariance, matrices.
        nmodes: int
            The number of modes for the mixture model. This defaults to 1,
            which will be checked against the shape of the other inputs.
        mus: array_like
            A list of lists of means of each mode in a multivariate Gaussian
            mixture model. A single list can be given for a single mode. If
            this is None then means at zero will be assumed.
        sigmas: array_like
            A list of lists of the standard deviations of each mode of the
            multivariate Gaussian. If supplying a correlation coefficient
            matrix rather than a covariance matrix these values must be given.
            If this is None unit variances will be assumed.
        corrcoefs: array
            A list of square matrices containing the correlation coefficients
            of the parameters for each mode. If this is None it will be assumed
            that the parameters are uncorrelated.
        covs: array
            A list of square matrices containing the covariance matrix of the
            multivariate Gaussian.
        weights: list
            A list of weights (relative probabilities) for each mode of the
            multivariate Gaussian. This will default to equal weights for each
            mode.
        bounds: list
            A list of bounds on each parameter. The defaults are for bounds at
            +/- infinity.
        """
        MultivariateGaussianDist.__init__(self, names, nmodes=nmodes,
                                          mus=mus, sigmas=sigmas,
                                          corrcoefs=corrcoefs, covs=covs,
                                          weights=weights, bounds=bounds)


class MultivariateGaussian(Prior):

    def __init__(self, mvg, name=None, latex_label=None, unit=None):
        """
        A prior class for a multivariate Gaussian (mixture model) prior.

        Parameters
        ----------
        mvg: MultivariateGaussianDist
            A :class:`bilby.core.prior.MultivariateGaussianDist` object defining
            the multivariate Gaussian distribution. This object is not copied,
            as it needs to be shared across multiple priors, and as such its
            contents will be altered by the prior.
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass

        """

        if not isinstance(mvg, MultivariateGaussianDist):
            raise TypeError("Must supply a multivariate Gaussian object")

        # check name is in the MultivariateGaussianDist class
        if name not in mvg.names:
            raise ValueError("'{}' is not a parameter in the multivariate "
                             "Gaussian")
        self.mvg = mvg

        Prior.__init__(self, name=name, latex_label=latex_label, unit=unit,
                       minimum=mvg.bounds[name][0],
                       maximum=mvg.bounds[name][1])

    def rescale(self, val, mode=None):
        """
        Scale a unit hypercube sample to the prior.

        Parameters
        ----------
        mode: int
            Specify which mode to sample from. If not set then a mode is
            chosen randomly based on its weight.
        """

        Prior.test_valid_for_rescaling(val)

        # add parameter value to multivariate Gaussian
        self.mvg.rescale_parameters[self.name] = val

        if self.mvg.filled_rescale():
            values = np.array(list(self.mvg.rescale_parameters.values())).T
            samples = self.mvg.rescale(values, mode=mode)
            self.mvg.reset_rescale()
            return samples
        else:
            return []  # return empty list

    def sample(self, size=1, mode=None):
        """
        Draw a sample from the prior.

        Parameters
        ----------
        mode: int
            Specify which mode to sample from. If not set then a mode is
            chosen randomly based on its weight.

        Returns
        -------
        float:
            A sample from the prior paramter.
        """

        if self.name in self.mvg.sampled_parameters:
            logger.warning("You have already drawn a sample from parameter "
                           "'{}'. The same sample will be "
                           "returned".format(self.name))

        if len(self.mvg.current_sample) == 0:
            # generate a sample
            self.mvg.sample(size=size, mode=mode)

        sample = self.mvg.current_sample[self.name]

        if self.name not in self.mvg.sampled_parameters:
            self.mvg.sampled_parameters.append(self.name)

        if len(self.mvg.sampled_parameters) == len(self.mvg):
            # reset samples
            self.mvg.reset_sampled()

        return sample

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ----------
        val: float

        Returns
        -------
        float:

        """

        return np.exp(self.ln_prob(val))

    def ln_prob(self, val):
        """
        Return the natural logarithm of the prior probability. Note that this
        will not be correctly normalised if there are bounds on the
        distribution.
        """

        # add parameter value to multivariate Gaussian
        self.mvg.requested_parameters[self.name] = val

        if self.mvg.filled_request():
            # all required parameters have been set
            values = list(self.mvg.requested_parameters.values())

            # check for the same number of values for each parameter
            for i in range(len(self.mvg) - 1):
                if (isinstance(values[i], (list, np.ndarray)) or
                        isinstance(values[i + 1], (list, np.ndarray))):
                    if (isinstance(values[i], (list, np.ndarray)) and
                            isinstance(values[i + 1], (list, np.ndarray))):
                        if len(values[i]) != len(values[i + 1]):
                            raise ValueError("Each parameter must have the same "
                                             "number of requested values.")
                    else:
                        raise ValueError("Each parameter must have the same "
                                         "number of requested values.")

            lnp = self.mvg.ln_prob(np.asarray(values).T)

            # reset the requested parameters
            self.mvg.reset_request()

            return lnp
        else:
            # if not all parameters have been requested yet, just return 0
            if isinstance(val, (float, int)):
                return 0.
            else:
                try:
                    # check value has a length
                    len(val)
                except Exception as e:
                    raise TypeError('Invalid type for ln_prob: {}'.format(e))

                if len(val) == 1:
                    return 0.
                else:
                    return np.zeros_like(val)

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        self._minimum = minimum

        # update the bounds in the MultivariateGaussianDist
        self.mvg.bounds[self.name] = (minimum, self.mvg.bounds[self.name][1])

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        self._maximum = maximum

        # update the bounds in the MultivariateGaussianDist
        self.mvg.bounds[self.name] = (self.mvg.bounds[self.name][0], maximum)


class MultivariateNormal(MultivariateGaussian):

    def __init__(self, mvg, name=None, latex_label=None, unit=None):
        """A synonym for the :class:`bilby.core.prior.MultivariateGaussian`
        prior distribution.

        Parameters
        ----------
        mvg: MultivariateGaussianDist
            A :class:`bilby.core.prior.MultivariateGaussianDist` object
            defining the multivariate Gaussian distribution. This object is not
            copied, as it needs to be shared across multiple priors, and as
            such its contents will be altered by the prior.
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        """
        MultivariateGaussian.__init__(self, mvg, name=name,
                                      latex_label=latex_label, unit=unit)
