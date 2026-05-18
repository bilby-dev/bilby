import array_api_compat as aac

from .utils import BackendNotImplementedError, BILBY_ARRAY_API


def multivariate_logpdf(xp, mean, cov):
    """
    Return a function to evaluate the log probability density of a multivariate
    Gaussian with given mean vector and covariance matrix for the provided
    array backend.

    Parameters
    ==========
    xp: numpy, torch, jax.numpy
        A module that will resolve to :code:`numpy`, :code:`torch`, or
        :code:`jax.numpy` in :code:`array_api_compat.is_..._namespace`.
    mean: array-like
        A one-dimensional array providing the mean of the distribution.
    cov: array-like
        A two-dimensional array providing the covariance matrix of the
        distribution.

    Returns
    =======
    logpdf: callable
        A callable that provides the log probaility density provided an array
        of points to evaluate at.
    """
    from scipy.stats import multivariate_normal

    if not BILBY_ARRAY_API or aac.is_numpy_namespace(xp):
        logpdf = multivariate_normal(mean=mean, cov=cov).logpdf
    elif aac.is_jax_namespace(xp):
        from functools import partial
        from jax.scipy.stats.multivariate_normal import logpdf

        logpdf = partial(logpdf, mean=mean, cov=cov)
    elif aac.is_torch_namespace(xp):
        from torch.distributions.multivariate_normal import MultivariateNormal

        mvn = MultivariateNormal(loc=mean, covariance_matrix=xp.asarray(cov))
        logpdf = mvn.log_prob
    else:
        raise BackendNotImplementedError(
            f"Unable to import multivariate_logpdf for {xp}"
        )
    return logpdf
