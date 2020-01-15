from importlib import import_module
from io import open as ioopen
import json
import numpy as np
import os

from future.utils import iteritems
from matplotlib.cbook import flatten

# keep 'import *' to make eval() statement further down work consistently
from bilby.core.prior.analytical import *  # noqa
from bilby.core.prior.analytical import DeltaFunction
from bilby.core.prior.base import Prior, Constraint
from bilby.core.prior.joint import JointPrior
from bilby.core.utils import logger, check_directory_exists_and_if_not_mkdir, BilbyJsonEncoder, decode_bilby_json


class PriorDict(dict):
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
        super(PriorDict, self).__init__()
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
        joint_dists = []
        with open(prior_file, "w") as outfile:
            for key in self.keys():
                if JointPrior in self[key].__class__.__mro__:
                    distname = '_'.join(self[key].dist.names) + '_{}'.format(self[key].dist.distname)
                    if distname not in joint_dists:
                        joint_dists.append(distname)
                        outfile.write(
                            "{} = {}\n".format(distname, self[key].dist))
                    diststr = repr(self[key].dist)
                    priorstr = repr(self[key])
                    outfile.write(
                        "{} = {}\n".format(key, priorstr.replace(diststr,
                                                                 distname)))
                else:
                    outfile.write(
                        "{} = {}\n".format(key, self[key]))

    def _get_json_dict(self):
        self.convert_floats_to_delta_functions()
        total_dict = {key: json.loads(self[key].to_json()) for key in self}
        total_dict["__prior_dict__"] = True
        total_dict["__module__"] = self.__module__
        total_dict["__name__"] = self.__class__.__name__
        return total_dict

    def to_json(self, outdir, label):
        check_directory_exists_and_if_not_mkdir(outdir)
        prior_file = os.path.join(outdir, "{}_prior.json".format(label))
        logger.debug("Writing priors to {}".format(prior_file))
        with open(prior_file, "w") as outfile:
            json.dump(self._get_json_dict(), outfile, cls=BilbyJsonEncoder,
                      indent=2)

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
        with ioopen(filename, 'r', encoding='unicode_escape') as f:
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
                    logger.debug("{} converted to DeltaFunction prior".format(
                        key))
                    continue
                except ValueError:
                    pass
                if "." in cls:
                    module = '.'.join(cls.split('.')[:-1])
                    cls = cls.split('.')[-1]
                else:
                    module = __name__.replace('.' + os.path.basename(__file__).replace('.py', ''), '')
                cls = getattr(import_module(module), cls, cls)
                if key.lower() in ["conversion_function", "condition_func"]:
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

    @classmethod
    def _get_from_json_dict(cls, prior_dict):
        try:
            cls == getattr(
                import_module(prior_dict["__module__"]),
                prior_dict["__name__"])
        except ImportError:
            logger.debug("Cannot import prior module {}.{}".format(
                prior_dict["__module__"], prior_dict["__name__"]
            ))
        except KeyError:
            logger.debug("Cannot find module name to load")
        for key in ["__module__", "__name__", "__prior_dict__"]:
            if key in prior_dict:
                del prior_dict[key]
        obj = cls(dict())
        obj.from_dictionary(prior_dict)
        return obj

    @classmethod
    def from_json(cls, filename):
        """ Reads in a prior from a json file

        Parameters
        ----------
        filename: str
            Name of the file to be read in
        """
        with open(filename, "r") as ff:
            obj = json.load(ff, object_hook=decode_bilby_json)
        return obj

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
            elif isinstance(val, dict):
                logger.warning(
                    'Cannot convert {} into a prior object. '
                    'Leaving as dictionary.'.format(key))
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
            if isinstance(self[key], Constraint):
                continue
            elif isinstance(self[key], Prior):
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
            constraint_keys = list()
            for ii, key in enumerate(keys[-1::-1]):
                if isinstance(self[key], Constraint):
                    constraint_keys.append(-ii - 1)
            for ii in constraint_keys[-1::-1]:
                del keys[ii]
            all_samples = {key: np.array([]) for key in keys}
            _first_key = list(all_samples.keys())[0]
            while len(all_samples[_first_key]) < needed:
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
        return self.__class__(dictionary=dict(self))


class PriorSet(PriorDict):

    def __init__(self, dictionary=None, filename=None):
        """ DEPRECATED: USE PriorDict INSTEAD"""
        logger.warning("The name 'PriorSet' is deprecated use 'PriorDict' instead")
        super(PriorSet, self).__init__(dictionary, filename)


class PriorDictException(Exception):
    """ General base class for all prior dict exceptions """


class ConditionalPriorDict(PriorDict):

    def __init__(self, dictionary=None, filename=None, conversion_function=None):
        """

        Parameters
        ----------
        dictionary: dict
            See parent class
        filename: str
            See parent class
        """
        self._conditional_keys = []
        self._unconditional_keys = []
        self._rescale_keys = []
        self._rescale_indexes = []
        self._least_recently_rescaled_keys = []
        super(ConditionalPriorDict, self).__init__(
            dictionary=dictionary, filename=filename,
            conversion_function=conversion_function
        )
        self._resolved = False
        self._resolve_conditions()

    def _resolve_conditions(self):
        """
        Resolves how priors depend on each other and automatically
        sorts them into the right order.
        1. All unconditional priors are put in front in arbitrary order
        2. We loop through all the unsorted conditional priors to find
        which one can go next
        3. We repeat step 2 len(self) number of times to make sure that
        all conditional priors will be sorted in order
        4. We set the `self._resolved` flag to True if all conditional
        priors were added in the right order
        """
        self._unconditional_keys = [key for key in self.keys() if not hasattr(self[key], 'condition_func')]
        conditional_keys_unsorted = [key for key in self.keys() if hasattr(self[key], 'condition_func')]
        self._conditional_keys = []
        for _ in range(len(self)):
            for key in conditional_keys_unsorted[:]:
                if self._check_conditions_resolved(key, self.sorted_keys):
                    self._conditional_keys.append(key)
                    conditional_keys_unsorted.remove(key)

        self._resolved = True
        if len(conditional_keys_unsorted) != 0:
            self._resolved = False

    def _check_conditions_resolved(self, key, sampled_keys):
        """ Checks if all required variables have already been sampled so we can sample this key """
        conditions_resolved = True
        for k in self[key].required_variables:
            if k not in sampled_keys:
                conditions_resolved = False
        return conditions_resolved

    def sample_subset(self, keys=iter([]), size=None):
        self.convert_floats_to_delta_functions()
        subset_dict = ConditionalPriorDict({key: self[key] for key in keys})
        if not subset_dict._resolved:
            raise IllegalConditionsException("The current set of priors contains unresolvable conditions.")
        samples = dict()
        for key in subset_dict.sorted_keys:
            if isinstance(self[key], Constraint):
                continue
            elif isinstance(self[key], Prior):
                try:
                    samples[key] = subset_dict[key].sample(size=size, **subset_dict.get_required_variables(key))
                except ValueError:
                    # Some prior classes can not handle an array of conditional parameters (e.g. alpha for PowerLaw)
                    # If that is the case, we sample each sample individually.
                    required_variables = subset_dict.get_required_variables(key)
                    samples[key] = np.zeros(size)
                    for i in range(size):
                        rvars = {key: value[i] for key, value in required_variables.items()}
                        samples[key][i] = subset_dict[key].sample(**rvars)
            else:
                logger.debug('{} not a known prior.'.format(key))
        return samples

    def get_required_variables(self, key):
        """ Returns the required variables to sample a given conditional key.

        Parameters
        ----------
        key : str
            Name of the key that we want to know the required variables for

        Returns
        ----------
        dict: key/value pairs of the required variables
        """
        return {k: self[k].least_recently_sampled for k in getattr(self[key], 'required_variables', [])}

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
        self._check_resolved()
        for key, value in sample.items():
            self[key].least_recently_sampled = value
        res = [self[key].prob(sample[key], **self.get_required_variables(key)) for key in sample]
        return np.product(res, **kwargs)

    def ln_prob(self, sample, axis=None):
        """

        Parameters
        ----------
        sample: dict
            Dictionary of the samples of which we want to have the log probability of
        axis: Union[None, int]
            Axis along which the summation is performed

        Returns
        -------
        float: Joint log probability of all the individual sample probabilities

        """
        self._check_resolved()
        for key, value in sample.items():
            self[key].least_recently_sampled = value
        res = [self[key].ln_prob(sample[key], **self.get_required_variables(key)) for key in sample]
        return np.sum(res, axis=axis)

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
        self._check_resolved()
        self._update_rescale_keys(keys)
        result = dict()
        for key, index in zip(self.sorted_keys_without_fixed_parameters, self._rescale_indexes):
            required_variables = {k: result[k] for k in getattr(self[key], 'required_variables', [])}
            result[key] = self[key].rescale(theta[index], **required_variables)
        return [result[key] for key in keys]

    def _update_rescale_keys(self, keys):
        if not keys == self._least_recently_rescaled_keys:
            self._rescale_indexes = [keys.index(element) for element in self.sorted_keys_without_fixed_parameters]
            self._least_recently_rescaled_keys = keys

    def _check_resolved(self):
        if not self._resolved:
            raise IllegalConditionsException("The current set of priors contains unresolveable conditions.")

    @property
    def conditional_keys(self):
        return self._conditional_keys

    @property
    def unconditional_keys(self):
        return self._unconditional_keys

    @property
    def sorted_keys(self):
        return self.unconditional_keys + self.conditional_keys

    @property
    def sorted_keys_without_fixed_parameters(self):
        return [key for key in self.sorted_keys if not isinstance(self[key], (DeltaFunction, Constraint))]

    def __setitem__(self, key, value):
        super(ConditionalPriorDict, self).__setitem__(key, value)
        self._resolve_conditions()

    def __delitem__(self, key):
        super(ConditionalPriorDict, self).__delitem__(key)
        self._resolve_conditions()


class ConditionalPriorDictException(PriorDictException):
    """ General base class for all conditional prior dict exceptions """


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


class IllegalConditionsException(ConditionalPriorDictException):
    """ Exception class to handle prior dicts that contain unresolvable conditions. """
