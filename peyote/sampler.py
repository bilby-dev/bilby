from __future__ import print_function, division, absolute_import

import inspect
import logging
import os
import sys

import numpy as np

from .result import Result
from .parameter import Parameter


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
                 outdir='outdir', label='label', result=None, **kwargs):
        self.likelihood = likelihood
        self.priors = priors
        self.label = label
        self.outdir = outdir
        self.kwargs = kwargs
        self.external_sampler = external_sampler

        self.__search_parameter_keys = []
        self.initialise_parameters()
        self.verify_parameters()
        self.ndim = len(self.__search_parameter_keys)

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
            self.__result.labels = [self.priors[k].latex_label for k in self.__search_parameter_keys]
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

    def initialise_parameters(self):

        for key in self.likelihood.waveform_generator.parameters:
            if key not in self.priors:
                self.priors[key] = Parameter(key)
                if self.priors[key].prior is None:
                    raise AttributeError(
                        "No default prior known for parameter {}".format(key))

            if isinstance(self.priors[key], Parameter) is False \
                    or self.priors[key].is_fixed is True:
                continue

            self.__search_parameter_keys.append(key)

        logging.info("Search parameters:")
        for key in self.__search_parameter_keys:
            logging.info('  {} ~ {}'.format(key, self.priors[key].prior))

    def verify_parameters(self):
        required_keys = self.likelihood.waveform_generator.parameters
        unmatched_keys = [r for r in required_keys if r not in self.priors]
        if len(unmatched_keys) > 0:
            raise ValueError(
                "Input parameters are missing keys {}".format(unmatched_keys))

    def prior_transform(self, theta):
        return [self.priors[key].prior.rescale(t) for key, t in zip(self.__search_parameter_keys, theta)]

    def log_likelihood(self, theta):
        for i, k in enumerate(self.__search_parameter_keys):
            self.likelihood.waveform_generator.parameters[k] = theta[i]
        return self.likelihood.log_likelihood()

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
        self.__kwargs = kwargs

    def run_sampler(self):
        nestle = self.external_sampler
        if self.kwargs.get('verbose', True):
            self.kwargs['callback'] = nestle.print_progress

        out = nestle.sample(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.kwargs)

        self.result.sampler_output = out
        self.result.samples = nestle.resample_equal(out.samples, out.weights)
        self.result.logz = out.logz
        self.result.logzerr = out.logzerr
        return self.result


class Dynesty(Sampler):
    def run_sampler(self):
        dynesty = self.external_sampler
        nested_sampler = dynesty.NestedSampler(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.kwargs)
        nested_sampler.run_nested()
        out = nested_sampler.results

        self.result.sampler_output = out
        weights = np.exp(out['logwt'] - out['logz'][-1])
        self.result.samples = dynesty.utils.resample_equal(
            out.samples, weights)
        self.result.logz = out.logz
        self.result.logzerr = out.logzerr
        return self.result


class Pymultinest(Sampler):

    @property
    def kwargs(self):
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self.__kwargs = dict(importance_nested_sampling=False, resume=True, verbose=True,
                             sampling_efficiency='parameter', outputfiles_basename=self.outdir)
        self.__kwargs.update(kwargs)
        if self.__kwargs['outputfiles_basename'].endswith('/') is False:
            self.__kwargs['outputfiles_basename'] = '{}/'.format(
                self.__kwargs['outputfiles_basename'])

    def run_sampler(self):
        pymultinest = self.external_sampler
        out = pymultinest.solve(
            LogLikelihood=self.log_likelihood, Prior=self.prior_transform,
            n_dims=self.ndim, **self.kwargs)

        self.result.sampler_output = out
        self.result.samples = out['samples']
        self.result.logz = out['logZ']
        self.result.logzerr = out['logZerr']
        self.result.outputfiles_basename = self.kwargs['outputfiles_basename']
        return self.result


def run_sampler(likelihood, priors, label='label', outdir='outdir',
                sampler='nestle', **sampler_kwargs):
    implemented_samplers = get_implemented_samplers()

    if implemented_samplers.__contains__(sampler.title()):
        sampler_class = globals()[sampler.title()]
        sampler = sampler_class(likelihood, priors, sampler, outdir=outdir,
                                label=label, **sampler_kwargs)
        result = sampler.run_sampler()
        result.noise_logz = likelihood.noise_log_likelihood
        result.log_bayes_factor = result.logz - result.noise_logz
        print("")
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
