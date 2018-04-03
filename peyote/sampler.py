from __future__ import print_function, division


class Sampler:
    def __init__(self, likelihood, parameters, sampler='nestle', **kwargs):
        self.likelihood = likelihood
        self.sampler = sampler
        self.parameters = parameters
        self.ndim = len(self.parameters)
        self.kwargs = kwargs

    def prior_transform(self, theta):
        return [self.parameters[k].prior.rescale(t)
                for k, t in zip(self.likelihood.parameter_keys, theta)]

    def run(self):

        if self.sampler == 'nestle':
            try:
                import nestle
            except ImportError:
                raise ImportError(
                    "Sampler nestle not installed on this system")
            res = nestle.sample(
                loglikelihood=self.likelihood.logl,
                prior_transform=self.prior_transform,
                ndim=self.ndim, **self.kwargs)
        else:
            raise ValueError(
                "Sampler {} not yet implemented".format(self.sampler))

        return res
