import array_api_compat as aac
import numpy as np

from .utils import BackendNotImplementedError


def erfinv_import(xp):
    if aac.is_numpy_namespace(xp):
        from scipy.special import erfinv
    elif aac.is_jax_namespace(xp):
        from jax.scipy.special import erfinv
    elif aac.is_torch_namespace(xp):
        from torch.special import erfinv
    elif aac.is_cupy_namespace(xp):
        from cupyx.scipy.special import erfinv
    else:
        raise BackendNotImplementedError
    return erfinv


def multivariate_logpdf(xp, mean, cov):
    if aac.is_numpy_namespace(xp):
        from scipy.stats import multivariate_normal

        logpdf = multivariate_normal(mean=mean, cov=cov).logpdf
    elif aac.is_jax_namespace(xp):
        from functools import partial
        from jax.scipy.stats.multivariate_normal import logpdf

        logpdf = partial(logpdf, mean=mean, cov=cov)
    elif aac.is_torch_namespace(xp):
        from torch.distributions.multivariate_normal import MultivariateNormal

        mvn = MultivariateNormal(loc=mean, covariance_matrix=xp.array(cov))
        logpdf = mvn.log_prob
    else:
        raise BackendNotImplementedError
    return logpdf


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False, *, xp=None):
    if xp is None:
        xp = a.__array_namespace__()

    if "jax" in xp.__name__:
        # the scipy version of logsumexp cannot be vmapped
        from jax.scipy.special import logsumexp as lse
    else:
        from scipy.special import logsumexp as lse

    return lse(a=a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign)
