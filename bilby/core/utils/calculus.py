import math

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d as _interp1d
from scipy.special import logsumexp

from .log import logger
from ...compat.utils import array_module


def derivatives(
    vals,
    func,
    releps=1e-3,
    abseps=None,
    mineps=1e-9,
    reltol=1e-3,
    epsscale=0.5,
    nonfixedidx=None,
):
    """
    Calculate the partial derivatives of a function at a set of values. The
    derivatives are calculated using the central difference, using an iterative
    method to check that the values converge as step size decreases.

    Parameters
    ==========
    vals: array_like
        A set of values, that are passed to a function, at which to calculate
        the gradient of that function
    func:
        A function that takes in an array of values.
    releps: float, array_like, 1e-3
        The initial relative step size for calculating the derivative.
    abseps: float, array_like, None
        The initial absolute step size for calculating the derivative.
        This overrides `releps` if set.
        `releps` is set then that is used.
    mineps: float, 1e-9
        The minimum relative step size at which to stop iterations if no
        convergence is achieved.
    epsscale: float, 0.5
        The factor by which releps if scaled in each iteration.
    nonfixedidx: array_like, None
        An array of indices in `vals` that are _not_ fixed values and therefore
        can have derivatives taken. If `None` then derivatives of all values
        are calculated.

    Returns
    =======
    grads: array_like
        An array of gradients for each non-fixed value.
    """

    if nonfixedidx is None:
        nonfixedidx = range(len(vals))

    if len(nonfixedidx) > len(vals):
        raise ValueError("To many non-fixed values")

    if max(nonfixedidx) >= len(vals) or min(nonfixedidx) < 0:
        raise ValueError("Non-fixed indexes contain non-existent indices")

    grads = np.zeros(len(nonfixedidx))

    # maximum number of times the gradient can change sign
    flipflopmax = 10.0

    # set steps
    if abseps is None:
        if isinstance(releps, float):
            eps = np.abs(vals) * releps
            eps[eps == 0.0] = releps  # if any values are zero set eps to releps
            teps = releps * np.ones(len(vals))
        elif isinstance(releps, (list, np.ndarray)):
            if len(releps) != len(vals):
                raise ValueError("Problem with input relative step sizes")
            eps = np.multiply(np.abs(vals), releps)
            eps[eps == 0.0] = np.array(releps)[eps == 0.0]
            teps = releps
        else:
            raise RuntimeError("Relative step sizes are not a recognised type!")
    else:
        if isinstance(abseps, float):
            eps = abseps * np.ones(len(vals))
        elif isinstance(abseps, (list, np.ndarray)):
            if len(abseps) != len(vals):
                raise ValueError("Problem with input absolute step sizes")
            eps = np.array(abseps)
        else:
            raise RuntimeError("Absolute step sizes are not a recognised type!")
        teps = eps

    # for each value in vals calculate the gradient
    count = 0
    for i in nonfixedidx:
        # initial parameter diffs
        leps = eps[i]
        cureps = teps[i]

        flipflop = 0

        # get central finite difference
        fvals = np.copy(vals)
        bvals = np.copy(vals)

        # central difference
        fvals[i] += 0.5 * leps  # change forwards distance to half eps
        bvals[i] -= 0.5 * leps  # change backwards distance to half eps
        cdiff = (func(fvals) - func(bvals)) / leps

        while 1:
            fvals[i] -= 0.5 * leps  # remove old step
            bvals[i] += 0.5 * leps

            # change the difference by a factor of two
            cureps *= epsscale
            if cureps < mineps or flipflop > flipflopmax:
                # if no convergence set flat derivative (TODO: check if there is a better thing to do instead)
                logger.warning(
                    "Derivative calculation did not converge: setting flat derivative."
                )
                grads[count] = 0.0
                break
            leps *= epsscale

            # central difference
            fvals[i] += 0.5 * leps  # change forwards distance to half eps
            bvals[i] -= 0.5 * leps  # change backwards distance to half eps
            cdiffnew = (func(fvals) - func(bvals)) / leps

            if cdiffnew == cdiff:
                grads[count] = cdiff
                break

            # check whether previous diff and current diff are the same within reltol
            rat = cdiff / cdiffnew
            if np.isfinite(rat) and rat > 0.0:
                # gradient has not changed sign
                if np.abs(1.0 - rat) < reltol:
                    grads[count] = cdiffnew
                    break
                else:
                    cdiff = cdiffnew
                    continue
            else:
                cdiff = cdiffnew
                flipflop += 1
                continue

        count += 1

    return grads


def logtrapzexp(lnf, dx):
    """
    Perform trapezium rule integration for the logarithm of a function on a grid.

    Parameters
    ==========
    lnf: array_like
        A :class:`numpy.ndarray` of values that are the natural logarithm of a function
    dx: Union[array_like, float]
        A :class:`numpy.ndarray` of steps sizes between values in the function, or a
        single step size value.

    Returns
    =======
    The natural logarithm of the area under the function.
    """

    lnfdx1 = lnf[:-1]
    lnfdx2 = lnf[1:]
    if isinstance(dx, (int, float)):
        C = np.log(dx / 2.0)
    elif isinstance(dx, (list, np.ndarray)):
        if len(dx) != len(lnf) - 1:
            raise ValueError(
                "Step size array must have length one less than the function length"
            )

        lndx = np.log(dx)
        lnfdx1 = lnfdx1.copy() + lndx
        lnfdx2 = lnfdx2.copy() + lndx
        C = -np.log(2.0)
    else:
        raise TypeError("Step size must be a single value or array-like")

    return C + logsumexp([logsumexp(lnfdx1), logsumexp(lnfdx2)])


class interp1d(_interp1d):
        
    def __call__(self, x):
        from array_api_compat import is_numpy_namespace

        xp = array_module(x)
        if is_numpy_namespace(xp):
            return super().__call__(x)
        else:
            return self._call_alt(x, xp=xp)
    
    def _call_alt(self, x, *, xp=np):
        if isinstance(self.fill_value, tuple):
            left, right = self.fill_value
        else:
            left = right = self.fill_value
        return xp.interp(
            x,
            xp.asarray(self.x),
            xp.asarray(self.y),
            left=left,
            right=right,
        )


class BoundedRectBivariateSpline(RectBivariateSpline):

    def __init__(self, x, y, z, bbox=[None] * 4, kx=3, ky=3, s=0, fill_value=None):
        self.x_min, self.x_max, self.y_min, self.y_max = bbox
        if self.x_min is None:
            self.x_min = min(x)
        if self.x_max is None:
            self.x_max = max(x)
        if self.y_min is None:
            self.y_min = min(y)
        if self.y_max is None:
            self.y_max = max(y)
        self.fill_value = fill_value
        self.x = x
        self.y = y
        self.z = z
        super().__init__(x=x, y=y, z=z, bbox=bbox, kx=kx, ky=ky, s=s)

    def __call__(self, x, y, dx=0, dy=0, grid=False):
        from array_api_compat import is_jax_namespace
        xp = array_module(x)
        if is_jax_namespace(xp):
            return self._call_jax(x, y)
        result = super().__call__(x=x, y=y, dx=dx, dy=dy, grid=grid)
        out_of_bounds_x = (x < self.x_min) | (x > self.x_max)
        out_of_bounds_y = (y < self.y_min) | (y > self.y_max)
        bad = out_of_bounds_x | out_of_bounds_y
        result[bad] = self.fill_value
        if result.size == 1:
            if bad:
                return self.fill_value
            else:
                return result.item()
        else:
            return result
    
    def _call_jax(self, x, y):
        import jax.numpy as jnp
        from interpax import interp2d

        return interp2d(
            x,
            y,
            jnp.asarray(self.x),
            jnp.asarray(self.y),
            jnp.asarray(self.z),
            extrap=self.fill_value,
            method="cubic2",
        )


class WrappedInterp1d(interp1d):
    """
    A wrapper around scipy interp1d which sets equality-by-instantiation and
    makes sure that the output is a float if the input is a float or int.
    """
    def __call__(self, x):
        output = super().__call__(x)
        if isinstance(x, (float, int)):
            output = output.item()
        return output

    def __eq__(self, other):
        for key in self.__dict__:
            if type(self.__dict__[key]) is np.ndarray:
                if not np.array_equal(self.__dict__[key], other.__dict__[key]):
                    return False
            elif key == "_spline":
                pass
            elif getattr(self, key) != getattr(other, key):
                return False
        return True


def round_up_to_power_of_two(x):
    """Round up to the next power of two

    Parameters
    ----------
    x: float

    Returns
    -------
    float: next power of two

    """
    return 2**math.ceil(np.log2(x))
