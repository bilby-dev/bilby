from __future__ import division, print_function

import logging
import numpy as np
from ..core.likelihood import Likelihood
from .model import Model


class HyperparameterLikelihood(Likelihood):
    """ A likelihood for infering hyperparameter posterior distributions

    See Eq. (1) of https://arxiv.org/abs/1801.02699 for a definition.

    Parameters
    ----------
    posteriors: list
        An list of pandas data frames of samples sets of samples. Each set may have
        a different size.
    hyper_prior: `tupak.hyper.model.Model`
        The population model, this can alternatively be a function.
    sampling_prior: `tupak.hyper.model.Model`
        The sampling prior, this can alternatively be a function.
    max_samples: int, optional
        Maximum number of samples to use from each set.

    """

    def __init__(self, posteriors, hyper_prior, sampling_prior, max_samples=1e100):
        if not isinstance(hyper_prior, Model):
            hyper_prior = Model([hyper_prior])
        if not isinstance(sampling_prior, Model):
            sampling_prior = Model([sampling_prior])
        self.posteriors = posteriors
        self.hyper_prior = hyper_prior
        self.sampling_prior = sampling_prior
        self.max_samples = max_samples
        Likelihood.__init__(self, hyper_prior.parameters)

        self.data = self.resample_posteriors()
        self.n_posteriors = len(self.posteriors)
        self.samples_per_posterior = self.max_samples
        self.log_factor = - self.n_posteriors * np.log(self.samples_per_posterior)

    def log_likelihood(self):
        self.hyper_prior.parameters.update(self.parameters)
        log_l = np.sum(np.log(np.sum(self.hyper_prior.prob(self.data)
                                     / self.sampling_prior.prob(self.data), axis=-1))) + self.log_factor
        return np.nan_to_num(log_l)

    def resample_posteriors(self, max_samples=None):
        if max_samples is not None:
            self.max_samples = max_samples
        for posterior in self.posteriors:
            self.max_samples = min(len(posterior), self.max_samples)
        data = {key: [] for key in self.posteriors[0]}
        logging.debug('Downsampling to {} samples per posterior.'.format(self.max_samples))
        for posterior in self.posteriors:
            temp = posterior.sample(self.max_samples)
            for key in data:
                data[key].append(temp[key])
        for key in data:
            data[key] = np.array(data[key])
        return data
