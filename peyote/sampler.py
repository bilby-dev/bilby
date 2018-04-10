from __future__ import print_function, division
import numpy as np
import logging
import numbers
import pickle
import os

import peyote


class Result(dict):

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
            os.rename(file_name, file_name + '.old')

        logging.info("Saving result to {}".format(file_name))
        with open(file_name, 'w+') as f:
            pickle.dump(self, f)


class Sampler:

    def __init__(self, likelihood, prior, sampler_string, outdir='outdir',
                 label='label', **kwargs):
        self.likelihood = likelihood
        self.prior = prior
        self.label = label
        self.outdir = outdir
        self.kwargs = kwargs

        self.ndim = 0

        self.sampler_string = sampler_string

        self.fixed_parameters = self.prior.copy()
        self.search_parameter_keys = []
        self.initialise_parameters()

        self.result = Result()
        self.add_initial_data_to_results()

        if os.path.isdir(outdir) is False:
            os.makedirs(outdir)

    def add_initial_data_to_results(self):
        self.result.search_parameter_keys = self.search_parameter_keys
        self.result.labels = [self.prior[k].latex_label for k in self.search_parameter_keys]

    def initialise_parameters(self):
        for key in dir(self.likelihood.source):
            if key.startswith('__'):
                continue
            if key == 'copy':
                continue

            if key in dir(self.prior):
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

    def prior_transform(self, theta):
        return [self.prior[k].prior.rescale(t)
                for k, t in zip(self.search_parameter_keys, theta)]

    def log_likelihood(self, theta):
        for i, k in enumerate(self.search_parameter_keys):
            self.fixed_parameters[k] = theta[i]
        return self.likelihood.log_likelihood(self.fixed_parameters)

    def run_sampler(self):
        pass


class Nestle(Sampler):

    def __init__(self, likelihood, prior, outdir='outdir',
                 label='label', **kwargs):
        Sampler.__init__(self, likelihood, prior, 'nestle', outdir,
                         label, **kwargs)

    def set_kwargs(self):
        self.kwargs_defaults = dict(verbose=True)
        self.kwargs_defaults.update(self.kwargs)
        self.kwargs = self.kwargs_defaults

    def run_sampler(self):
        # nestle = self.extenal_sampler
        import nestle
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

    def __init__(self, likelihood, prior, outdir='outdir',
                 label='label', **kwargs):
        Sampler.__init__(self, likelihood, prior, 'dynesty', outdir,
                         label, **kwargs)

    def run_sampler(self):
        import dynesty
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
        pymultinest = self.extenal_sampler
        out = pymultinest.solve(
            LogLikelihood=self.log_likelihood, Prior=self.prior_transform,
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
        sampler_class = getattr(peyote.sampler, sampler.title())
        sampler = sampler_class(likelihood, prior, sampler, outdir=outdir,
                                label=label, **sampler_kwargs)
        result = sampler.run_sampler()
        result.save_to_file(outdir=outdir, label=label)
        return result
    else:
        raise ValueError(
            "Sampler {} not yet implemented".format(sampler))
