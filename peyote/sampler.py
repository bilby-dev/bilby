from __future__ import print_function, division
import numpy as np


class Sampler:
    def __init__(self, likelihood, parameters, sampler='nestle', **kwargs):
        self.likelihood = likelihood
        self.sampler = sampler
        self.parameters = parameters
        self.fixed_parameters = parameters.copy()
        self.kwargs = kwargs
        self.parameter_keys = []
        for p in parameters:
            if hasattr(parameters[p], 'prior'):
                self.parameter_keys.append(parameters[p].name)
                self.fixed_parameters[p] = np.nan
        self.ndim = len(self.parameter_keys)
        print('Search parameters = {}'.format(self.parameter_keys))

    def prior_transform(self, theta):
        return [self.parameters[k].prior.rescale(t)
                for k, t in zip(self.parameter_keys, theta)]

    def loglikelihood(self, theta):
        for i, k in enumerate(self.parameter_keys):
            self.fixed_parameters[k] = theta[i]
        return self.likelihood.logl(self.fixed_parameters)

    def run(self):

        if self.sampler == 'nestle':
            try:
                import nestle
            except ImportError:
                raise ImportError(
                    "Sampler nestle not installed on this system")
            res = nestle.sample(
                loglikelihood=self.loglikelihood,
                prior_transform=self.prior_transform,
                ndim=self.ndim, **self.kwargs)
        elif self.sampler == 'dynesty':
            try:
                import dynesty
            except ImportError:
                raise ImportError(
                    "Sampler dynesty not installed on this system")
            sampler = dynesty.NestedSampler(
                loglikelihood=self.loglikelihood,
                prior_transform=self.prior_transform,
                ndim=self.ndim, **self.kwargs)
            sampler.run_nested()
            res = sampler.results

        else:
            raise ValueError(
                "Sampler {} not yet implemented".format(self.sampler))

        return res
