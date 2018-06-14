from __future__ import division, print_function

import inspect
import logging
import numpy as np

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp


class Likelihood(object):
    """ Empty likelihood class to be subclassed by other likelihoods """

    def __init__(self, parameters=None):
        self.parameters = parameters

    def log_likelihood(self):
        return np.nan

    def noise_log_likelihood(self):
        return np.nan

    def log_likelihood_ratio(self):
        return self.log_likelihood() - self.noise_log_likelihood()


class GaussianLikelihood(Likelihood):
    def __init__(self, x, y, function, sigma=None):
        """
        A general Gaussian likelihood for known or unknown noise - the model
        parameters are inferred from the arguments of function

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        function:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments are
            will require a prior and will be sampled over (unless a fixed
            value is given).
        sigma: None, float, array_like
            If None, the standard deviation of the noise is unknown and will be
            estimated (note: this requires a prior to be given for sigma). If
            not None, this defined the standard-deviation of the data points.
            This can either be a single float, or an array with length equal
            to that for `x` and `y`.
        """
        self.x = x
        self.y = y
        self.N = len(x)
        self.sigma = sigma
        self.function = function

        # These lines of code infer the parameters from the provided function
        parameters = inspect.getargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        self.function_keys = self.parameters.keys()
        if self.sigma is None:
            self.parameters['sigma'] = None

    def log_likelihood(self):
        sigma = self.parameters.get('sigma', self.sigma)
        model_parameters = {k: self.parameters[k] for k in self.function_keys}
        res = self.y - self.function(self.x, **model_parameters)
        return -0.5 * (np.sum((res / sigma)**2)
                       + self.N*np.log(2*np.pi*sigma**2))


class HyperparameterLikelihood(Likelihood):
    """ A likelihood for infering hyperparameter posterior distributions

    See Eq. (1) of https://arxiv.org/abs/1801.02699 for a definition.

    Parameters
    ----------
    samples: list
        An N-dimensional list of individual sets of samples. Each set may have
        a different size.
    hyper_prior: `tupak.prior.Prior`
        A prior distribution with a `parameters` argument pointing to the
        hyperparameters to infer from the samples. These may need to be
        initialized to any arbitrary value, but this will not effect the
        result.
    run_prior: `tupak.prior.Prior`
        The prior distribution used in the inidivudal inferences which resulted
        in the set of samples.

    """

    def __init__(self, samples, hyper_prior, run_prior):
        Likelihood.__init__(self, parameters=hyper_prior.__dict__)
        self.samples = samples
        self.hyper_prior = hyper_prior
        self.run_prior = run_prior
        if hasattr(hyper_prior, 'lnprob') and hasattr(run_prior, 'lnprob'):
            logging.info("Using log-probabilities in likelihood")
            self.log_likelihood = self.log_likelihood_using_lnprob
        else:
            logging.info("Using probabilities in likelihood")
            self.log_likelihood = self.log_likelihood_using_prob

    def log_likelihood_using_lnprob(self):
        L = []
        self.hyper_prior.__dict__.update(self.parameters)
        for samp in self.samples:
            f = self.hyper_prior.lnprob(samp) - self.run_prior.lnprob(samp)
            L.append(logsumexp(f))
        return np.sum(L)

    def log_likelihood_using_prob(self):
        L = []
        self.hyper_prior.__dict__.update(self.parameters)
        for samp in self.samples:
            L.append(
                np.sum(self.hyper_prior.prob(samp) /
                       self.run_prior.prob(samp)))
        return np.sum(np.log(L))
