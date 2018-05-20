from __future__ import print_function, division, absolute_import

import inspect
import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from .result import Result, read_in_result
from .prior import Prior, fill_priors
from . import utils
from . import prior
import tupak


class Sampler(object):
    """ A sampler object to aid in setting up an inference run

    Parameters
    ----------
    likelihood: likelihood.GravitationalWaveTransient
        A  object with a log_l method
    prior: dict
        The prior to be used in the search. Elements can either be floats
        (indicating a fixed value or delta function prior) or they can be
        of type parameter.Parameter with an associated prior
    sampler_string: str
        A string containing the module name of the sampler


    Returns
    -------
    results:
        A dictionary of the results

    """

    def __init__(self, likelihood, priors, external_sampler='nestle', outdir='outdir', label='label', use_ratio=False,
                 **kwargs):
        self.likelihood = likelihood
        self.priors = priors
        self.label = label
        self.outdir = outdir
        self.use_ratio = use_ratio
        self.external_sampler = external_sampler
        self.external_sampler_function = None

        self.__search_parameter_keys = []
        self.__fixed_parameter_keys = []
        self.initialise_parameters()
        self.verify_parameters()
        self.kwargs = kwargs

        self.check_cached_result()

        self.log_summary_for_sampler()

        if os.path.isdir(outdir) is False:
            os.makedirs(outdir)

        self.result = self.initialise_result()

    @property
    def search_parameter_keys(self):
        return self.__search_parameter_keys

    @property
    def fixed_parameter_keys(self):
        return self.__fixed_parameter_keys

    @property
    def ndim(self):
        return len(self.__search_parameter_keys)

    @property
    def kwargs(self):
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self.__kwargs = kwargs

    @property
    def external_sampler(self):
        return self.__external_sampler

    @external_sampler.setter
    def external_sampler(self, sampler):
        if type(sampler) is str:
            try:
                self.__external_sampler = __import__(sampler)
            except ImportError:
                raise ImportError(
                    "Sampler {} not installed on this system".format(sampler))
        elif isinstance(sampler, Sampler):
            self.__external_sampler = sampler
        else:
            raise TypeError('sampler must either be a string referring to built in sampler or a custom made class that '
                            'inherits from sampler')

    def verify_kwargs_against_external_sampler_function(self):
        args = inspect.getargspec(self.external_sampler_function).args
        bad_keys = []
        for user_input in self.kwargs.keys():
            if user_input not in args:
                logging.warning(
                    "Supplied argument '{}' not an argument of '{}', removing."
                    .format(user_input, self.external_sampler_function))
                bad_keys.append(user_input)
        for key in bad_keys:
            self.kwargs.pop(key)

    def initialise_parameters(self):

        for key in self.priors:
            if isinstance(self.priors[key], Prior) is True \
                    and self.priors[key].is_fixed is False:
                self.__search_parameter_keys.append(key)
            elif isinstance(self.priors[key], Prior) \
                    and self.priors[key].is_fixed is True:
                self.likelihood.parameters[key] = \
                    self.priors[key].sample()
                self.__fixed_parameter_keys.append(key)

        logging.info("Search parameters:")
        for key in self.__search_parameter_keys:
            logging.info('  {} ~ {}'.format(key, self.priors[key]))
        for key in self.__fixed_parameter_keys:
            logging.info('  {} = {}'.format(key, self.priors[key].peak))

    def initialise_result(self):
        result = Result()
        result.search_parameter_keys = self.__search_parameter_keys
        result.fixed_parameter_keys = self.__fixed_parameter_keys
        result.parameter_labels = [
            self.priors[k].latex_label for k in
            self.__search_parameter_keys]
        result.label = self.label
        result.outdir = self.outdir
        result.kwargs = self.kwargs
        return result

    def verify_parameters(self):
        for key in self.priors:
            try:
                self.likelihood.parameters[key] = self.priors[key].sample()
            except AttributeError as e:
                logging.warning('Cannot sample from {}, {}'.format(key, e))
        try:
            self.likelihood.log_likelihood_ratio()
        except TypeError:
            raise TypeError('GravitationalWaveTransient evaluation failed. Have you definitely specified all the parameters?\n{}'.format(
                self.likelihood.parameters))

    def prior_transform(self, theta):
        return [self.priors[key].rescale(t) for key, t in zip(self.__search_parameter_keys, theta)]

    def log_prior(self, theta):
        return np.sum(
            [np.log(self.priors[key].prob(t)) for key, t in
                zip(self.__search_parameter_keys, theta)])

    def log_likelihood(self, theta):
        for i, k in enumerate(self.__search_parameter_keys):
            self.likelihood.parameters[k] = theta[i]
        if self.use_ratio:
            return self.likelihood.log_likelihood_ratio()
        else:
            return self.likelihood.log_likelihood()

    def get_random_draw_from_prior(self):
        """ Get a random draw from the prior distribution

        Returns
        draw: array_like
            An ndim-length array of values drawn from the prior. Parameters
            with delta-function (or fixed) priors are not returned

        """

        draw = np.array([self.priors[key].sample()
                        for key in self.__search_parameter_keys])
        if np.isinf(self.log_likelihood(draw)):
            logging.info('Prior draw {} has inf likelihood'.format(draw))
        if np.isinf(self.log_prior(draw)):
            logging.info('Prior draw {} has inf prior'.format(draw))
        return draw

    def run_sampler(self):
        pass

    def check_cached_result(self):
        """ Check if the cached data file exists and can be used """

        if utils.command_line_args.clean:
            logging.debug("Command line argument clean given, forcing rerun")
            self.cached_result = None
            return

        try:
            self.cached_result = read_in_result(self.outdir, self.label)
        except ValueError:
            self.cached_result = None

        if utils.command_line_args.use_cached:
            logging.debug("Command line argument cached given, no cache check performed")
            return

        logging.debug("Checking cached data")
        if self.cached_result:
            check_keys = ['search_parameter_keys', 'fixed_parameter_keys',
                          'kwargs']
            use_cache = True
            for key in check_keys:
                if self.cached_result.check_attribute_match_to_other_object(
                        key, self) is False:
                    logging.debug("Cached value {} is unmatched".format(key))
                    use_cache = False
            if use_cache is False:
                self.cached_result = None

    def log_summary_for_sampler(self):
        if self.cached_result is None:
            logging.info("Using sampler {} with kwargs {}".format(
                self.__class__.__name__, self.kwargs))


class Nestle(Sampler):

    @property
    def kwargs(self):
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self.__kwargs = dict(verbose=True, method='multi')
        self.__kwargs.update(kwargs)

        if 'npoints' not in self.__kwargs:
            for equiv in ['nlive', 'nlives', 'n_live_points']:
                if equiv in self.__kwargs:
                    self.__kwargs['npoints'] = self.__kwargs.pop(equiv)

    def run_sampler(self):
        nestle = self.external_sampler
        self.external_sampler_function = nestle.sample
        if self.kwargs.get('verbose', True):
            self.kwargs['callback'] = nestle.print_progress
            self.kwargs.pop('verbose')
        self.verify_kwargs_against_external_sampler_function()

        out = self.external_sampler_function(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.kwargs)
        print("")

        self.result.sampler_output = out
        self.result.samples = nestle.resample_equal(out.samples, out.weights)
        self.result.logz = out.logz
        self.result.logzerr = out.logzerr
        return self.result


class Dynesty(Sampler):

    @property
    def kwargs(self):
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self.__kwargs = dict(dlogz=0.1, bound='multi', sample='rwalk',
                             walks=self.ndim * 5, verbose=True)
        self.__kwargs.update(kwargs)
        if 'nlive' not in self.__kwargs:
            for equiv in ['nlives', 'n_live_points', 'npoint', 'npoints']:
                if equiv in self.__kwargs:
                    self.__kwargs['nlive'] = self.__kwargs.pop(equiv)
        if 'nlive' not in self.__kwargs:
            self.__kwargs['nlive'] = 250
        if 'update_interval' not in self.__kwargs:
            self.__kwargs['update_interval'] = int(0.6 * self.__kwargs['nlive'])

    def run_sampler(self):
        dynesty = self.external_sampler

        if self.kwargs.get('dynamic', False) is False:
            nested_sampler = dynesty.NestedSampler(
                loglikelihood=self.log_likelihood,
                prior_transform=self.prior_transform,
                ndim=self.ndim, **self.kwargs)
            nested_sampler.run_nested(
                dlogz=self.kwargs['dlogz'],
                print_progress=self.kwargs['verbose'])
        else:
            nested_sampler = dynesty.DynamicNestedSampler(
                loglikelihood=self.log_likelihood,
                prior_transform=self.prior_transform,
                ndim=self.ndim, **self.kwargs)
            nested_sampler.run_nested(print_progress=self.kwargs['verbose'])
        print("")
        out = nested_sampler.results

        # self.result.sampler_output = out
        weights = np.exp(out['logwt'] - out['logz'][-1])
        self.result.samples = dynesty.utils.resample_equal(
            out.samples, weights)
        self.result.logz = out.logz[-1]
        self.result.logzerr = out.logzerr[-1]
        return self.result


class Pymultinest(Sampler):

    @property
    def kwargs(self):
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        outputfiles_basename = self.outdir + '/pymultinest_{}/'.format(self.label)
        utils.check_directory_exists_and_if_not_mkdir(outputfiles_basename)
        self.__kwargs = dict(importance_nested_sampling=False, resume=True,
                             verbose=True, sampling_efficiency='parameter',
                             outputfiles_basename=outputfiles_basename)
        self.__kwargs.update(kwargs)
        if self.__kwargs['outputfiles_basename'].endswith('/') is False:
            self.__kwargs['outputfiles_basename'] = '{}/'.format(
                self.__kwargs['outputfiles_basename'])
        if 'n_live_points' not in self.__kwargs:
            for equiv in ['nlive', 'nlives', 'npoints', 'npoint']:
                if equiv in self.__kwargs:
                    self.__kwargs['n_live_points'] = self.__kwargs.pop(equiv)

    def run_sampler(self):
        pymultinest = self.external_sampler
        self.external_sampler_function = pymultinest.run
        self.verify_kwargs_against_external_sampler_function()
        # Note: pymultinest.solve adds some extra steps, but underneath
        # we are calling pymultinest.run - hence why it is used in checking
        # the arguments.
        out = pymultinest.solve(
            LogLikelihood=self.log_likelihood, Prior=self.prior_transform,
            n_dims=self.ndim, **self.kwargs)

        self.result.sampler_output = out
        self.result.samples = out['samples']
        self.result.logz = out['logZ']
        self.result.logzerr = out['logZerr']
        self.result.outputfiles_basename = self.kwargs['outputfiles_basename']
        return self.result


class Ptemcee(Sampler):

    def run_sampler(self):
        ntemps = self.kwargs.pop('ntemps', 2)
        nwalkers = self.kwargs.pop('nwalkers', 100)
        nsteps = self.kwargs.pop('nsteps', 100)
        nburn = self.kwargs.pop('nburn', 50)
        ptemcee = self.external_sampler
        tqdm = utils.get_progress_bar(self.kwargs.pop('tqdm', 'tqdm'))

        sampler = ptemcee.Sampler(
            ntemps=ntemps, nwalkers=nwalkers, dim=self.ndim,
            logl=self.log_likelihood, logp=self.log_prior,
            **self.kwargs)
        pos0 = [[self.get_random_draw_from_prior()
                 for i in range(nwalkers)]
                for j in range(ntemps)]

        for result in tqdm(
                sampler.sample(pos0, iterations=nsteps, adapt=True), total=nsteps):
            pass

        self.result.sampler_output = np.nan
        self.result.samples = sampler.chain[0, :, nburn:, :].reshape(
            (-1, self.ndim))
        self.result.walkers = sampler.chain[0, :, :, :]
        self.result.logz = np.nan
        self.result.logzerr = np.nan
        self.plot_walkers()
        logging.info("Max autocorr time = {}".format(np.max(sampler.get_autocorr_time())))
        logging.info("Tswap frac = {}".format(sampler.tswap_acceptance_fraction))
        return self.result

    def plot_walkers(self, save=True, **kwargs):
        nwalkers, nsteps, ndim = self.result.walkers.shape
        idxs = np.arange(nsteps)
        fig, axes = plt.subplots(nrows=ndim, figsize=(6, 3*self.ndim))
        for i, ax in enumerate(axes):
            ax.plot(idxs, self.result.walkers[:, :, i].T, lw=0.1, color='k')
            ax.set_ylabel(self.result.parameter_labels[i])

        fig.tight_layout()
        filename = '{}/{}_walkers.png'.format(self.outdir, self.label)
        logging.info('Saving walkers plot to {}'.format('filename'))
        fig.savefig(filename)


def run_sampler(likelihood, priors=None, label='label', outdir='outdir',
                sampler='nestle', use_ratio=True, injection_parameters=None,
                conversion_function=None, **kwargs):
    """
    The primary interface to easy parameter estimation

    Parameters
    ----------
    likelihood: `tupak.likelihood.GravitationalWaveTransient`
        A `GravitationalWaveTransient` instance
    priors: dict
        A dictionary of the priors for each parameter - missing parameters will
        use default priors, if None, all priors will be default
    label: str
        Name for the run, used in output files
    outdir: str
        A string used in defining output files
    sampler: str
        The name of the sampler to use - see
        `tupak.sampler.get_implemented_samplers()` for a list of available
        samplers
    use_ratio: bool (False)
        If True, use the likelihood's loglikelihood_ratio, rather than just
        the log likelhood.
    injection_parameters: dict
        A dictionary of injection parameters used in creating the data (if
        using simulated data). Appended to the result object and saved.

    conversion_function: function, optional
        Function to apply to posterior to generate additional parameters.
    **kwargs:
        All kwargs are passed directly to the samplers `run` function

    Returns
    ------
    result
        An object containing the results
    """

    utils.check_directory_exists_and_if_not_mkdir(outdir)
    implemented_samplers = get_implemented_samplers()

    if priors is None:
        priors = dict()
    priors = fill_priors(priors, likelihood)
    tupak.prior.write_priors_to_file(priors, outdir)

    if implemented_samplers.__contains__(sampler.title()):
        sampler_class = globals()[sampler.title()]
        sampler = sampler_class(likelihood, priors, sampler, outdir=outdir,
                                label=label, use_ratio=use_ratio,
                                **kwargs)

        if sampler.cached_result:
            logging.info("Using cached result")
            return sampler.cached_result

        result = sampler.run_sampler()
        result.noise_logz = likelihood.noise_log_likelihood()
        if use_ratio:
            result.log_bayes_factor = result.logz
            result.logz = result.log_bayes_factor + result.noise_logz
        else:
            result.log_bayes_factor = result.logz - result.noise_logz
        if injection_parameters is not None:
            result.injection_parameters = injection_parameters
            if conversion_function is not None:
                conversion_function(result.injection_parameters)
        result.fixed_parameter_keys = sampler.fixed_parameter_keys
        # result.prior = prior  # Removed as this breaks the saving of the data
        result.samples_to_data_frame(likelihood=likelihood, priors=priors, conversion_function=conversion_function)
        result.kwargs = sampler.kwargs
        result.save_to_file(outdir=outdir, label=label)
        return result
    else:
        raise ValueError(
            "Sampler {} not yet implemented".format(sampler))


def get_implemented_samplers():
    implemented_samplers = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            implemented_samplers.append(obj.__name__)
    return implemented_samplers
