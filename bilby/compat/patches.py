import array_api_compat as aac

from .utils import xp_wrap, BackendNotImplementedError, BILBY_ARRAY_API


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


@xp_wrap
def interp(x, xs, fs, /, left=None, right=None, period=None, *, xp=None):
    """
    A simple implementation of numpy-style linear interpolation

    The logic is copied from
    https://github.com/pytorch/pytorch/issues/50334#issuecomment-1000917964

    Parameters
    ==========
    x: array-like
        The values to evaluate the interpolant at.
    xs: array-like
        The x-values for setting up the interpolant.
    ys: array-like
        The values of the function for setting up the interpolant.
    left: float
        The value to use for x < xs[0]. Default is fs[0]
    right: float
        The value to use for x > xs[-1]. Default is fs[-1].
    period: float
        The period of the interpolant.
        Parameters left and right are ignored if period is specified.

    Notes
    =====
    To avoid overlap with the ``xp`` variable, the second and third variable
    names from differ from numpy.
    These arguments are enforced to be positional only.
    """
    if not BILBY_ARRAY_API or hasattr(xp, "interp"):
        return xp.interp(x, xs, fs, left=left, right=right, period=period)

    if period is not None:
        x = x % period
    if left is None:
        left = fs[0]
    if right is None:
        right = fs[-1]
    
    x = xp.atleast_1d(x)

    m = (fs[1:] - fs[:-1]) / (xs[1:] - xs[:-1])
    b = fs[:-1] - (m * xs[:-1])

    indices = xp.sum(xp.ge(x[:, None], xs[None, :]), axis=1) - 1
    indices = xp.clip(indices, 0, len(m) - 1)

    ret = m[indices] * x + b[indices]

    if period is None:
        ret = xp.where(x < xs[0], xp.asarray(left), ret)
        ret = xp.where(x > xs[-1], xp.asarray(right), ret)

    return ret.squeeze()
