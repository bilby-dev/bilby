from __future__ import print_function, division, absolute_import

import inspect
import logging
import os
import sys
import numpy as np
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt

from .result import Result
from .prior import Prior, fill_priors
from . import utils
from . import prior
import tupak


class Sampler(object):
    """ A sampler object to aid in setting up an inference run

    Parameters
    ----------
    likelihood: likelihood.Likelihood
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

    def __init__(self, likelihood, priors, external_sampler='nestle',
                 outdir='outdir', label='label', use_ratio=False, result=None,
                 **kwargs):
        self.likelihood = likelihood
        self.priors = priors
        self.label = label
        self.outdir = outdir
        self.use_ratio = use_ratio
        self.external_sampler = external_sampler

        self.__search_parameter_keys = []
        self.__fixed_parameter_keys = []
        self.initialise_parameters()
        self.verify_parameters()
        self.ndim = len(self.__search_parameter_keys)
        self.kwargs = kwargs

        self.result = result

        self.log_summary_for_sampler()

        if os.path.isdir(outdir) is False:
            os.makedirs(outdir)

    @property
    def result(self):
        return self.__result

    @result.setter
    def result(self, result):
        if result is None:
            self.__result = Result()
            self.__result.search_parameter_keys = self.__search_parameter_keys
            self.__result.parameter_labels = [
                self.priors[k].latex_label for k in
                self.__search_parameter_keys]
            self.__result.label = self.label
            self.__result.outdir = self.outdir
        elif type(result) is Result:
            self.__result = result
        else:
            raise TypeError('result must either be a Result or None')

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

    @property
    def kwargs(self):
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self.__kwargs = kwargs

    def verify_kwargs_against_external_sampler_function(self):
        args = inspect.getargspec(self.external_sampler_function).args
        for user_input in self.kwargs.keys():
            bad_keys = []
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
                self.likelihood.waveform_generator.parameters[key] = \
                    self.priors[key].sample()
                self.__fixed_parameter_keys.append(key)

        logging.info("Search parameters:")
        for key in self.__search_parameter_keys:
            logging.info('  {} ~ {}'.format(key, self.priors[key]))
        for key in self.__fixed_parameter_keys:
            logging.info('  {} = {}'.format(key, self.priors[key].peak))

    def verify_parameters(self):
        required_keys = self.priors
        unmatched_keys = [r for r in required_keys if r not in self.likelihood.waveform_generator.parameters]
        if len(unmatched_keys) > 0:
            raise KeyError(
                "Source model does not contain keys {}".format(unmatched_keys))

    def prior_transform(self, theta):
        return [self.priors[key].rescale(t) for key, t in zip(self.__search_parameter_keys, theta)]

    def log_prior(self, theta):
        return np.sum(
            [np.log(self.priors[key].prob(t)) for key, t in
                zip(self.__search_parameter_keys, theta)])

    def log_likelihood(self, theta):
        for i, k in enumerate(self.__search_parameter_keys):
            self.likelihood.waveform_generator.parameters[k] = theta[i]
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

    def log_summary_for_sampler(self):
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
        nested_sampler = dynesty.NestedSampler(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.kwargs)
        nested_sampler.run_nested(
            dlogz=self.kwargs['dlogz'], print_progress=self.kwargs['verbose'])
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
                **sampler_kwargs):
    """
    The primary interface to easy parameter estimation

    Parameters
    ----------
    likelihood: `tupak.likelihood.Likelihood`
        A `Likelihood` instance
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
        the loglikelhood.
    injection_parameters: dict
        A dictionary of injection parameters used in creating the data (if
        using simulated data). Appended to the result object and saved.
    **sampler_kwargs:
        All kwargs are passed directly to the samplers `run` functino

    Returns
    ------
    result
        An object containing the results
    """

    utils.check_directory_exists_and_if_not_mkdir(outdir)
    implemented_samplers = get_implemented_samplers()

    if priors is None:
        priors = dict()
    fill_priors(priors, likelihood.waveform_generator)
    tupak.prior.write_priors_to_file(priors, outdir)

    if implemented_samplers.__contains__(sampler.title()):
        sampler_class = globals()[sampler.title()]
        sampler = sampler_class(likelihood, priors, sampler, outdir=outdir,
                                label=label, use_ratio=use_ratio,
                                **sampler_kwargs)
        result = sampler.run_sampler()
        result.noise_logz = likelihood.noise_log_likelihood()
        if use_ratio:
            result.log_bayes_factor = result.logz
            result.logz = result.log_bayes_factor + result.noise_logz
        else:
            result.log_bayes_factor = result.logz - result.noise_logz
        result.injection_parameters = injection_parameters
        result.fixed_parameter_keys = [key for key in priors if isinstance(key, prior.DeltaFunction)]
        # result.prior = prior  # Removed as this breaks the saving of the data
        result.samples_to_data_frame()
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
