from __future__ import absolute_import
import datetime
import numpy as np

from pandas import DataFrame

from ..utils import logger, command_line_args, Counter
from ..prior import Prior, PriorDict, DeltaFunction, Constraint
from ..result import Result, read_in_result


class Sampler(object):
    """ A sampler object to aid in setting up an inference run

    Parameters
    ----------
    likelihood: likelihood.Likelihood
        A  object with a log_l method
    priors: bilby.core.prior.PriorDict, dict
        Priors to be used in the search.
        This has attributes for each parameter to be sampled.
    external_sampler: str, Sampler, optional
        A string containing the module name of the sampler or an instance of
        this class
    outdir: str, optional
        Name of the output directory
    label: str, optional
        Naming scheme of the output files
    use_ratio: bool, optional
        Switch to set whether or not you want to use the log-likelihood ratio
        or just the log-likelihood
    plot: bool, optional
        Switch to set whether or not you want to create traceplots
    injection_parameters:
        A dictionary of the injection parameters
    meta_data:
        A dictionary of extra meta data to store in the result
    result_class: bilby.core.result.Result, or child of
        The result class to use. By default, `bilby.core.result.Result` is used,
        but objects which inherit from this class can be given providing
        additional methods.
    **kwargs: dict
        Additional keyword arguments

    Attributes
    ----------
    likelihood: likelihood.Likelihood
        A  object with a log_l method
    priors: bilby.core.prior.PriorDict
        Priors to be used in the search.
        This has attributes for each parameter to be sampled.
    external_sampler: Module
        An external module containing an implementation of a sampler.
    outdir: str
        Name of the output directory
    label: str
        Naming scheme of the output files
    use_ratio: bool
        Switch to set whether or not you want to use the log-likelihood ratio
        or just the log-likelihood
    plot: bool
        Switch to set whether or not you want to create traceplots
    skip_import_verification: bool
        Skips the check if the sampler is installed if true. This is
        only advisable for testing environments
    result: bilby.core.result.Result
        Container for the results of the sampling run
    kwargs: dict
        Dictionary of keyword arguments that can be used in the external sampler

    Raises
    ------
    TypeError:
        If external_sampler is neither a string nor an instance of this class
        If not all likelihood.parameters have been defined
    ImportError:
        If the external_sampler string does not refer to a sampler that is
        installed on this system
    AttributeError:
        If some of the priors can't be sampled

    """
    default_kwargs = dict()

    def __init__(
            self, likelihood, priors, outdir='outdir', label='label',
            use_ratio=False, plot=False, skip_import_verification=False,
            injection_parameters=None, meta_data=None, result_class=None,
            likelihood_benchmark=False,
            **kwargs):
        self.likelihood = likelihood
        if isinstance(priors, PriorDict):
            self.priors = priors
        else:
            self.priors = PriorDict(priors)
        self.label = label
        self.outdir = outdir
        self.injection_parameters = injection_parameters
        self.meta_data = meta_data
        self.use_ratio = use_ratio
        if not skip_import_verification:
            self._verify_external_sampler()
        self.external_sampler_function = None
        self.plot = plot
        self.likelihood_benchmark = likelihood_benchmark

        self._search_parameter_keys = list()
        self._fixed_parameter_keys = list()
        self._constraint_parameter_keys = list()
        self._initialise_parameters()
        self._verify_parameters()
        self._time_likelihood()
        self._verify_use_ratio()
        self.kwargs = kwargs

        self._check_cached_result()

        self._log_summary_for_sampler()

        self.result = self._initialise_result(result_class)
        self.likelihood_count = None
        if self.likelihood_benchmark:
            self.likelihood_count = Counter()

    @property
    def search_parameter_keys(self):
        """list: List of parameter keys that are being sampled"""
        return self._search_parameter_keys

    @property
    def fixed_parameter_keys(self):
        """list: List of parameter keys that are not being sampled"""
        return self._fixed_parameter_keys

    @property
    def constraint_parameter_keys(self):
        """list: List of parameters providing prior constraints"""
        return self._constraint_parameter_keys

    @property
    def ndim(self):
        """int: Number of dimensions of the search parameter space"""
        return len(self._search_parameter_keys)

    @property
    def kwargs(self):
        """dict: Container for the kwargs. Has more sophisticated logic in subclasses """
        return self._kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self._kwargs = self.default_kwargs.copy()
        self._translate_kwargs(kwargs)
        self._kwargs.update(kwargs)
        self._verify_kwargs_against_default_kwargs()

    def _translate_kwargs(self, kwargs):
        """ Template for child classes """
        pass

    @property
    def external_sampler_name(self):
        return self.__class__.__name__.lower()

    def _verify_external_sampler(self):
        external_sampler_name = self.external_sampler_name
        try:
            self.external_sampler = __import__(external_sampler_name)
        except (ImportError, SystemExit):
            raise SamplerNotInstalledError(
                "Sampler {} is not installed on this system".format(external_sampler_name))

    def _verify_kwargs_against_default_kwargs(self):
        """
        Check if the kwargs are contained in the list of available arguments
        of the external sampler.
        """
        args = self.default_kwargs
        bad_keys = []
        for user_input in self.kwargs.keys():
            if user_input not in args:
                logger.warning(
                    "Supplied argument '{}' not an argument of '{}', removing."
                    .format(user_input, self.__class__.__name__))
                bad_keys.append(user_input)
        for key in bad_keys:
            self.kwargs.pop(key)

    def _initialise_parameters(self):
        """
        Go through the list of priors and add keys to the fixed and search
        parameter key list depending on whether
        the respective parameter is fixed.
        """
        for key in self.priors:
            if isinstance(self.priors[key], Prior) \
                    and self.priors[key].is_fixed is False:
                self._search_parameter_keys.append(key)
            elif isinstance(self.priors[key], Constraint):
                self._constraint_parameter_keys.append(key)
            elif isinstance(self.priors[key], DeltaFunction):
                self.likelihood.parameters[key] = self.priors[key].sample()
                self._fixed_parameter_keys.append(key)

        logger.info("Search parameters:")
        for key in self._search_parameter_keys + self._constraint_parameter_keys:
            logger.info('  {} = {}'.format(key, self.priors[key]))
        for key in self._fixed_parameter_keys:
            logger.info('  {} = {}'.format(key, self.priors[key].peak))

    def _initialise_result(self, result_class):
        """
        Returns
        -------
        bilby.core.result.Result: An initial template for the result

        """
        result_kwargs = dict(
            label=self.label, outdir=self.outdir,
            sampler=self.__class__.__name__.lower(),
            search_parameter_keys=self._search_parameter_keys,
            fixed_parameter_keys=self._fixed_parameter_keys,
            constraint_parameter_keys=self._constraint_parameter_keys,
            priors=self.priors, meta_data=self.meta_data,
            injection_parameters=self.injection_parameters,
            sampler_kwargs=self.kwargs, use_ratio=self.use_ratio)

        if result_class is None:
            result = Result(**result_kwargs)
        elif issubclass(result_class, Result):
            result = result_class(**result_kwargs)
        else:
            raise ValueError(
                "Input result_class={} not understood".format(result_class))

        return result

    def _check_if_priors_can_be_sampled(self):
        """Check if all priors can be sampled properly.

        Raises
        ------
        AttributeError
            prior can't be sampled.
        """
        for key in self.priors:
            if isinstance(self.priors[key], Constraint):
                continue
            try:
                self.likelihood.parameters[key] = self.priors[key].sample()
            except AttributeError as e:
                logger.warning('Cannot sample from {}, {}'.format(key, e))

    def _verify_parameters(self):
        """ Evaluate a set of parameters drawn from the prior

        Tests if the likelihood evaluation passes

        Raises
        ------
        TypeError
            Likelihood can't be evaluated.

        """

        if self.priors.test_has_redundant_keys():
            raise IllegalSamplingSetError(
                "Your sampling set contains redundant parameters.")

        self._check_if_priors_can_be_sampled()
        try:
            theta = [self.priors[key].sample()
                     for key in self._search_parameter_keys]
            self.log_likelihood(theta)
        except TypeError as e:
            raise TypeError(
                "Likelihood evaluation failed with message: \n'{}'\n"
                "Have you specified all the parameters:\n{}"
                .format(e, self.likelihood.parameters))

    def _time_likelihood(self, n_evaluations=100):
        """ Times the likelihood evaluation and print an info message

        Parameters
        ----------
        n_evaluations: int
            The number of evaluations to estimate the evaluation time from

        """

        t1 = datetime.datetime.now()
        for _ in range(n_evaluations):
            theta = [self.priors[key].sample()
                     for key in self._search_parameter_keys]
            self.log_likelihood(theta)
        total_time = (datetime.datetime.now() - t1).total_seconds()
        self._log_likelihood_eval_time = total_time / n_evaluations

        if self._log_likelihood_eval_time == 0:
            self._log_likelihood_eval_time = np.nan
            logger.info("Unable to measure single likelihood time")
        else:
            logger.info("Single likelihood evaluation took {:.3e} s"
                        .format(self._log_likelihood_eval_time))

    def _verify_use_ratio(self):
        """
        Checks if use_ratio is set. Prints a warning if use_ratio is set but
        not properly implemented.
        """
        self._check_if_priors_can_be_sampled()
        if self.use_ratio is False:
            logger.debug("use_ratio set to False")
            return

        ratio_is_nan = np.isnan(self.likelihood.log_likelihood_ratio())

        if self.use_ratio is True and ratio_is_nan:
            logger.warning(
                "You have requested to use the loglikelihood_ratio, but it "
                " returns a NaN")
        elif self.use_ratio is None and not ratio_is_nan:
            logger.debug(
                "use_ratio not spec. but gives valid answer, setting True")
            self.use_ratio = True

    def prior_transform(self, theta):
        """ Prior transform method that is passed into the external sampler.

        Parameters
        ----------
        theta: list
            List of sampled values on a unit interval

        Returns
        -------
        list: Properly rescaled sampled values
        """
        return self.priors.rescale(self._search_parameter_keys, theta)

    def log_prior(self, theta):
        """

        Parameters
        ----------
        theta: list
            List of sampled values on a unit interval

        Returns
        -------
        float: Joint ln prior probability of theta

        """
        params = {
            key: t for key, t in zip(self._search_parameter_keys, theta)}
        return self.priors.ln_prob(params)

    def log_likelihood(self, theta):
        """

        Parameters
        ----------
        theta: list
            List of values for the likelihood parameters

        Returns
        -------
        float: Log-likelihood or log-likelihood-ratio given the current
            likelihood.parameter values

        """
        if self.likelihood_benchmark:
            try:
                self.likelihood_count.increment()
            except AttributeError:
                pass
        params = {
            key: t for key, t in zip(self._search_parameter_keys, theta)}
        self.likelihood.parameters.update(params)
        if self.use_ratio:
            return self.likelihood.log_likelihood_ratio()
        else:
            return self.likelihood.log_likelihood()

    def get_random_draw_from_prior(self):
        """ Get a random draw from the prior distribution

        Returns
        -------
        draw: array_like
            An ndim-length array of values drawn from the prior. Parameters
            with delta-function (or fixed) priors are not returned

        """
        new_sample = self.priors.sample()
        draw = np.array(list(new_sample[key]
                             for key in self._search_parameter_keys))
        self.check_draw(draw)
        return draw

    def get_initial_points_from_prior(self, npoints=1):
        """ Method to draw a set of live points from the prior

        This iterates over draws from the prior until all the samples have a
        finite prior and likelihood (relevant for constrained priors).

        Parameters
        ----------
        npoints: int
            The number of values to return

        Returns
        -------
        unit_cube, parameters, likelihood: tuple of array_like
            unit_cube (nlive, ndim) is an array of the prior samples from the
            unit cube, parameters (nlive, ndim) is the unit_cube array
            transformed to the target space, while likelihood (nlive) are the
            likelihood evaluations.

        """
        unit_cube = []
        parameters = []
        likelihood = []
        while len(unit_cube) < npoints:
            unit = np.random.rand(self.ndim)
            theta = self.prior_transform(unit)
            if self.check_draw(theta, warning=False):
                unit_cube.append(unit)
                parameters.append(theta)
                likelihood.append(self.log_likelihood(theta))

        return np.array(unit_cube), np.array(parameters), np.array(likelihood)

    def check_draw(self, theta, warning=True):
        """ Checks if the draw will generate an infinite prior or likelihood

        Parameters
        ----------
        theta: array_like
            Parameter values at which to evaluate likelihood

        Returns
        -------
        bool, cube (nlive,
            True if the likelihood and prior are finite, false otherwise

        """
        if np.isinf(self.log_prior(theta)):
            if warning:
                logger.warning('Prior draw {} has inf prior'.format(theta))
            return False
        if np.isinf(self.log_likelihood(theta)):
            if warning:
                logger.warning('Prior draw {} has inf likelihood'.format(theta))
            return False
        return True

    def run_sampler(self):
        """A template method to run in subclasses"""
        pass

    def _run_test(self):
        """
        TODO: Implement this method
        Raises
        -------
        ValueError: in any case
        """
        raise ValueError("Method not yet implemented")

    def _check_cached_result(self):
        """ Check if the cached data file exists and can be used """

        if command_line_args.clean:
            logger.debug("Command line argument clean given, forcing rerun")
            self.cached_result = None
            return

        try:
            self.cached_result = read_in_result(
                outdir=self.outdir, label=self.label)
        except IOError:
            self.cached_result = None

        if command_line_args.use_cached:
            logger.debug(
                "Command line argument cached given, no cache check performed")
            return

        logger.debug("Checking cached data")
        if self.cached_result:
            check_keys = ['search_parameter_keys', 'fixed_parameter_keys',
                          'kwargs']
            use_cache = True
            for key in check_keys:
                if self.cached_result._check_attribute_match_to_other_object(
                        key, self) is False:
                    logger.debug("Cached value {} is unmatched".format(key))
                    use_cache = False
            if use_cache is False:
                self.cached_result = None

    def _log_summary_for_sampler(self):
        """Print a summary of the sampler used and its kwargs"""
        if self.cached_result is None:
            kwargs_print = self.kwargs.copy()
            for k in kwargs_print:
                if type(kwargs_print[k]) in (list, np.ndarray):
                    array_repr = np.array(kwargs_print[k])
                    if array_repr.size > 10:
                        kwargs_print[k] = ('array_like, shape={}'
                                           .format(array_repr.shape))
                elif type(kwargs_print[k]) == DataFrame:
                    kwargs_print[k] = ('DataFrame, shape={}'
                                       .format(kwargs_print[k].shape))
            logger.info("Using sampler {} with kwargs {}".format(
                self.__class__.__name__, kwargs_print))

    def calc_likelihood_count(self):
        if self.likelihood_benchmark:
            self.result.num_likelihood_evaluations = self.likelihood_count.value
        else:
            return None


class NestedSampler(Sampler):
    npoints_equiv_kwargs = ['nlive', 'nlives', 'n_live_points', 'npoints', 'npoint', 'Nlive']
    walks_equiv_kwargs = ['walks', 'steps', 'nmcmc']

    def reorder_loglikelihoods(self, unsorted_loglikelihoods, unsorted_samples,
                               sorted_samples):
        """ Reorders the stored log-likelihood after they have been reweighted

        This creates a sorting index by matching the reweights `result.samples`
        against the raw samples, then uses this index to sort the
        loglikelihoods

        Parameters
        ----------
        sorted_samples, unsorted_samples: array-like
            Sorted and unsorted values of the samples. These should be of the
            same shape and contain the same sample values, but in different
            orders
        unsorted_loglikelihoods: array-like
            The loglikelihoods corresponding to the unsorted_samples

        Returns
        -------
        sorted_loglikelihoods: array-like
            The loglikelihoods reordered to match that of the sorted_samples


        """

        idxs = []
        for ii in range(len(unsorted_loglikelihoods)):
            idx = np.where(np.all(sorted_samples[ii] == unsorted_samples,
                                  axis=1))[0]
            if len(idx) > 1:
                logger.warning(
                    "Multiple likelihood matches found between sorted and "
                    "unsorted samples. Taking the first match.")
            idxs.append(idx[0])
        return unsorted_loglikelihoods[idxs]

    def log_likelihood(self, theta):
        """
        Since some nested samplers don't call the log_prior method, evaluate
        the prior constraint here.

        Parameters
        theta: array_like
            Parameter values at which to evaluate likelihood

        Returns
        -------
        float: log_likelihood
        """
        if self.priors.evaluate_constraints({
                key: theta[ii] for ii, key in
                enumerate(self.search_parameter_keys)}):
            return Sampler.log_likelihood(self, theta)
        else:
            return np.nan_to_num(-np.inf)


class MCMCSampler(Sampler):
    nwalkers_equiv_kwargs = ['nwalker', 'nwalkers', 'draws', 'Niter']
    nburn_equiv_kwargs = ['burn', 'nburn']

    def print_nburn_logging_info(self):
        """ Prints logging info as to how nburn was calculated """
        if type(self.nburn) in [float, int]:
            logger.info("Discarding {} steps for burn-in".format(self.nburn))
        elif self.result.max_autocorrelation_time is None:
            logger.info("Autocorrelation time not calculated, discarding {} "
                        " steps for burn-in".format(self.nburn))
        else:
            logger.info("Discarding {} steps for burn-in, estimated from "
                        "autocorr".format(self.nburn))

    def calculate_autocorrelation(self, samples, c=3):
        """ Uses the `emcee.autocorr` module to estimate the autocorrelation

        Parameters
        ----------
        samples: array_like
            A chain of samples.
        c: float
            The minimum number of autocorrelation times needed to trust the
            estimate (default: `3`). See `emcee.autocorr.integrated_time`.
        """
        import emcee
        try:
            self.result.max_autocorrelation_time = int(np.max(
                emcee.autocorr.integrated_time(samples, c=c)))
            logger.info("Max autocorr time = {}".format(
                self.result.max_autocorrelation_time))
        except emcee.autocorr.AutocorrError as e:
            self.result.max_autocorrelation_time = None
            logger.info("Unable to calculate autocorr time: {}".format(e))


class Error(Exception):
    """ Base class for all exceptions raised by this module """


class SamplerError(Error):
    """ Base class for Error related to samplers in this module """


class SamplerNotInstalledError(SamplerError):
    """ Base class for Error raised by not installed samplers """


class IllegalSamplingSetError(Error):
    """ Class for illegal sets of sampling parameters """
