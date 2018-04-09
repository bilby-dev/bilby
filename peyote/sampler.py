from __future__ import print_function, division
import numpy as np
import logging
import numbers
import pickle
import os

import peyote


class Result(dict):
    def __init__(self):
        pass

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        """Print a summary """
        return ("nsamples: {:d}\n"
                "logz: {:6.3f} +/- {:6.3f}\n"
                .format(len(self.samples), self.logz, self.logzerr))

    def save_to_file(self, outdir, label):
        file_name = '{}/{}_results.p'.format(outdir, label)
        if os.path.isdir(outdir) is False:
            os.makedirs(outdir)
        if os.path.isfile(file_name):
            logging.info(
                'Renaming existing file {} to {}.old'
                .format(file_name, file_name))
            os.rename(file_name, file_name+'.old')

        logging.info("Saving result to {}".format(file_name))
        with open(file_name, 'w+') as f:
            pickle.dump(self, f)


class Sampler:
    """ A sampler object to aid in setting up an inference run

    Parameters
    ----------
    likelihood: peyote.likelihood.likelihood
        A  object with a log_l method
    prior: dict
        The prior to be used in the search. Elements can either be floats
        (indicating a fixed value or delta function prior) or they can be
        of type peyote.parameter.Parameter with an associated prior
    sampler_string: str
        A string containing the module name of the sampler


    Returns
    -------
    results:
        A dictionary of the results

    """

    def __init__(self, likelihood, prior, sampler_string, outdir='outdir',
                 label='label', **kwargs):
        self.likelihood = likelihood
        self.prior = prior
        self.label = label
        self.outdir = outdir
        self.kwargs = kwargs

        self.sampler_string = sampler_string
        self.import_sampler()

        self.initialise_parameters()
        self.verify_prior()
        self.add_initial_data_to_results()
        self.set_kwargs()

        self.log_summary_for_sampler()

        # Initialize some variables that are going to be used later
        self.fixed_parameters = []
        self.result = 0
        self.search_parameter_keys = []
        self.ndim = 0
        self.sampler = self

        if os.path.isdir(outdir) is False:
            os.makedirs(outdir)

    def set_kwargs(self):
        pass

    def add_initial_data_to_results(self):
        self.result = Result()
        self.result.search_parameter_keys = self.search_parameter_keys
        self.result.labels = [
            self.prior[k].latex_label for k in self.search_parameter_keys]

    def initialise_parameters(self):
        self.fixed_parameters = self.prior.copy()
        self.search_parameter_keys = []
        for key in self.likelihood.parameter_keys:
            if key in self.prior:
                p = self.prior[key]
                CA = isinstance(p, numbers.Real)
                CB = hasattr(p, 'prior')
                CC = getattr(p, 'is_fixed', False) is True
                if CA is False and CB and CC is False:
                    self.search_parameter_keys.append(key)
                    self.fixed_parameters[key] = np.nan
                elif CC:
                    self.fixed_parameters[key] = p.value
            else:
                try:
                    self.prior[key] = getattr(peyote.parameter, key)
                    self.search_parameter_keys.append(key)
                except AttributeError:
                    raise AttributeError(
                        "No default prior known for parameter {}".format(key))
        self.ndim = len(self.search_parameter_keys)

        logging.info("Search parameters:")
        for key in self.search_parameter_keys:
            logging.info('  {} ~ {}'.format(key, self.prior[key].prior))

    def verify_prior(self):
        required_keys = self.likelihood.parameter_keys
        unmatched_keys = [
            r for r in required_keys if r not in self.prior]
        if len(unmatched_keys) > 0:
            raise ValueError(
                "Input prior is missing keys {}".format(unmatched_keys))

    def prior_transform(self, theta):
        return [self.prior[k].prior.rescale(t)
                for k, t in zip(self.search_parameter_keys, theta)]

    def loglikelihood(self, theta):
        for i, k in enumerate(self.search_parameter_keys):
            self.fixed_parameters[k] = theta[i]
        return self.likelihood.loglikelihood(self.fixed_parameters)

    def run_sampler(self):
        pass

    def import_sampler(self):
        try:
            self.sampler = __import__(self.sampler_string)
        except ImportError:
            raise ImportError(
                "Sampler {} not installed on this system".format(
                    self.sampler_string))

    def log_summary_for_sampler(self):
        logging.info("Using sampler {} with kwargs {}".format(
            self.__class__.__name__, self.kwargs))


class Nestle(Sampler):

    def set_kwargs(self):
        self.kwargs_defaults = dict(verbose=True)
        self.kwargs_defaults.update(self.kwargs)
        self.kwargs = self.kwargs_defaults

    def run_sampler(self):
        nestle = self.sampler
        if self.kwargs.get('verbose', True):
            self.kwargs['callback'] = nestle.print_progress

        out = nestle.sample(
            loglikelihood=self.loglikelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.kwargs)

        self.result.sampler_output = out
        self.result.samples = nestle.resample_equal(out.samples, out.weights)
        self.result.logz = out.logz
        self.result.logzerr = out.logzerr
        return self.result


class Dynesty(Sampler):
    def run_sampler(self):
        dynesty = self.sampler
        sampler = dynesty.NestedSampler(
            loglikelihood=self.loglikelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.kwargs)
        sampler.run_nested()
        out = sampler.results

        self.result.sampler_output = out
        weights = np.exp(out['logwt'] - out['logz'][-1])
        self.result.samples = dynesty.utils.resample_equal(
            out.samples, weights)
        self.result.logz = out.logz
        self.result.logzerr = out.logzerr
        return self.result


class Pymultinest(Sampler):

    def set_kwargs(self):
        self.kwargs_defaults = dict(
            importance_nested_sampling=False, resume=True, verbose=True,
            sampling_efficiency='parameter', outputfiles_basename=self.outdir)
        self.kwargs_defaults.update(self.kwargs)
        self.kwargs = self.kwargs_defaults
        if self.kwargs['outputfiles_basename'].endswith('/') is False:
            self.kwargs['outputfiles_basename'] = '{}/'.format(
                self.kwargs['outputfiles_basename'])

    def run_sampler(self):
        pymultinest = self.sampler
        out = pymultinest.solve(
            LogLikelihood=self.loglikelihood, Prior=self.prior_transform,
            n_dims=self.ndim, **self.kwargs)

        self.result.sampler_output = out
        self.result.samples = out['samples']
        self.result.logz = out['logZ']
        self.result.logzerr = out['logZerr']
        self.result.outputfiles_basename = self.kwargs['outputfiles_basename']
        return self.result


def run_sampler(likelihood, prior, label='label', outdir='outdir',
                sampler='nestle', **sampler_kwargs):
    if hasattr(peyote.sampler, sampler.title()):
        _Sampler = getattr(peyote.sampler, sampler.title())
        sampler = _Sampler(likelihood, prior, sampler, outdir=outdir,
                           label=label, **sampler_kwargs)
        result = sampler.run_sampler()
        result.save_to_file(outdir=outdir, label=label)
        return result
    else:
        raise ValueError(
            "Sampler {} not yet implemented".format(sampler))

