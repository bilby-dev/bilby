
import logging

import numpy as np

from ..core.likelihood import Likelihood
from .model import Model
from ..core.prior import PriorDict


class HyperparameterLikelihood(Likelihood):
    """ A likelihood for inferring hyperparameter posterior distributions

    See Eq. (34) of https://arxiv.org/abs/1809.02293 for a definition.

    Parameters
    ==========
    posteriors: list
        An list of pandas data frames of samples sets of samples.
        Each set may have a different size.
    hyper_prior: `bilby.hyper.model.Model`
        The population model, this can alternatively be a function.
    sampling_prior: `bilby.hyper.model.Model`
        The sampling prior, this can alternatively be a function.
    log_evidences: list, optional
        Log evidences for single runs to ensure proper normalisation
        of the hyperparameter likelihood. If not provided, the original
        evidences will be set to 0. This produces a Bayes factor between
        the sampling prior and the hyperparameterised model.
    max_samples: int, optional
        Maximum number of samples to use from each set.

    """

    def __init__(self, posteriors, hyper_prior, sampling_prior=None,
                 log_evidences=None, max_samples=1e100):
        if not isinstance(hyper_prior, Model):
            hyper_prior = Model([hyper_prior])
        if sampling_prior is None:
            if ('log_prior' not in posteriors[0].keys()) and ('prior' not in posteriors[0].keys()):
                raise ValueError('Missing both sampling prior function and prior or log_prior '
                                 'column in posterior dictionary. Must pass one or the other.')
        else:
            if not (isinstance(sampling_prior, Model) or isinstance(sampling_prior, PriorDict)):
                sampling_prior = Model([sampling_prior])
        if log_evidences is not None:
            self.evidence_factor = np.sum(log_evidences)
        else:
            self.evidence_factor = np.nan
        self.posteriors = posteriors
        self.hyper_prior = hyper_prior
        self.sampling_prior = sampling_prior
        self.max_samples = max_samples
        super(HyperparameterLikelihood, self).__init__(hyper_prior.parameters)

        self.data = self.resample_posteriors()
        self.n_posteriors = len(self.posteriors)
        self.samples_per_posterior = self.max_samples
        self.samples_factor =\
            - self.n_posteriors * np.log(self.samples_per_posterior)

    def log_likelihood_ratio(self):
        self.hyper_prior.parameters.update(self.parameters)
        log_l = np.sum(np.log(np.sum(self.hyper_prior.prob(self.data) /
                       self.data['prior'], axis=-1)))
        log_l += self.samples_factor
        return np.nan_to_num(log_l)

    def noise_log_likelihood(self):
        return self.evidence_factor

    def log_likelihood(self):
        return self.noise_log_likelihood() + self.log_likelihood_ratio()

    def resample_posteriors(self, max_samples=None):
        """
        Convert list of pandas DataFrame object to dict of arrays.

        Parameters
        ==========
        max_samples: int, opt
            Maximum number of samples to take from each posterior,
            default is length of shortest posterior chain.
        Returns
        =======
        data: dict
            Dictionary containing arrays of size (n_posteriors, max_samples)
            There is a key for each shared key in self.posteriors.
        """
        if max_samples is not None:
            self.max_samples = max_samples
        for posterior in self.posteriors:
            self.max_samples = min(len(posterior), self.max_samples)
        data = {key: [] for key in self.posteriors[0]}
        if 'log_prior' in data.keys():
            data.pop('log_prior')
        if 'prior' not in data.keys():
            data['prior'] = []
        logging.debug('Downsampling to {} samples per posterior.'.format(
            self.max_samples))
        for posterior in self.posteriors:
            temp = posterior.sample(self.max_samples)
            if self.sampling_prior is not None:
                temp['prior'] = self.sampling_prior.prob(temp, axis=0)
            elif 'log_prior' in temp.keys():
                temp['prior'] = np.exp(temp['log_prior'])
            for key in data:
                data[key].append(temp[key])
        for key in data:
            data[key] = np.array(data[key])
        return data
