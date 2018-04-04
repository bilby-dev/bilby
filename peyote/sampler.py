from __future__ import print_function, division
import numpy as np

import peyote


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

    def __init__(self, likelihood, prior, sampler_string, **kwargs):
        self.sampler_string = sampler_string
        self.import_sampler()
        self.likelihood = likelihood
        self.prior = prior
        self.fixed_parameters = prior.copy()
        self.kwargs = kwargs
        self.parameter_keys = []
        for p in prior:
            if hasattr(prior[p], 'prior'):
                self.parameter_keys.append(prior[p].name)
                self.fixed_parameters[p] = np.nan
        self.ndim = len(self.parameter_keys)
        print('Search parameters = {}'.format(self.parameter_keys))

    def prior_transform(self, theta):
        return [self.prior[k].prior.rescale(t)
                for k, t in zip(self.parameter_keys, theta)]

    def loglikelihood(self, theta):
        for i, k in enumerate(self.parameter_keys):
            self.fixed_parameters[k] = theta[i]
        return self.likelihood.logl(self.fixed_parameters)

    def run_sampler(self):
        pass

    def import_sampler(self):
        try:
            self.sampler = __import__(self.sampler_string)
        except ImportError:
            raise ImportError(
                "Sampler {} not installed on this system".format(
                    self.sampler_string))


class Nestle(Sampler):
    def run_sampler(self):
        nestle = self.sampler
        res = nestle.sample(
            loglikelihood=self.loglikelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.kwargs)
        return res


class Dynesty(Sampler):
    def run_sampler(self):
        dynesty = self.sampler
        sampler = dynesty.NestedSampler(
            loglikelihood=self.loglikelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.kwargs)
        sampler.run_nested()
        return sampler.results


def run_sampler(likelihood, prior, sampler='nestle', **sampler_kwargs):
    if hasattr(peyote.sampler, sampler.title()):
        _Sampler = getattr(peyote.sampler, sampler.title())
        sampler = _Sampler(likelihood, prior, sampler, **sampler_kwargs)
        return sampler.run_sampler()
    else:
        raise ValueError(
            "Sampler {} not yet implemented".format(sampler))

