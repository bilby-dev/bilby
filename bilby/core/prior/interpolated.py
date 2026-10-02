import array_api_compat as aac
import numpy as np
from scipy.integrate import trapezoid

from .base import Prior
from ..utils import logger
from ..utils.calculus import interp1d
from ...compat.utils import array_module, xp_wrap


class Interped(Prior):

    def __init__(self, xx, yy, minimum=np.nan, maximum=np.nan, name=None,
                 latex_label=None, unit=None, boundary=None):
        """Creates an interpolated prior function from arrays of xx and yy=p(xx)

        Parameters
        ==========
        xx: array_like
            x values for the to be interpolated prior function
        yy: array_like
            p(xx) values for the to be interpolated prior function
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass

        Attributes
        ==========
        probability_density: scipy.interpolate.interp1d
            Interpolated prior probability distribution
        cumulative_distribution: callable
            Cumulative prior probability distribution
        inverse_cumulative_distribution: callable
            Inverted cumulative prior probability distribution
        YY: array_like
            Cumulative prior probability distribution

        """
        self.xx = xx
        self.min_limit = min(xx)
        self.max_limit = max(xx)
        self._yy = yy
        self.YY = None
        self.probability_density = None
        self.cumulative_distribution = None
        self.inverse_cumulative_distribution = None
        self.__all_interpolated = interp1d(x=xx, y=yy, bounds_error=False, fill_value=0)
        minimum = float(np.nanmax(np.array((min(xx), minimum))))
        maximum = float(np.nanmin(np.array((max(xx), maximum))))
        super(Interped, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                       minimum=minimum, maximum=maximum, boundary=boundary)
        self._update_instance()

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if np.array_equal(self.xx, other.xx) and np.array_equal(self.yy, other.yy):
            return True
        return False

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val.

        Parameters
        ==========
        val:  Union[float, int, array_like]

        Returns
        =======
         Union[float, array_like]: Prior probability of val
        """
        return self.probability_density(val)[()]

    @xp_wrap
    def cdf(self, val, *, xp=None):
        return self.cumulative_distribution(val)[()]

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the prior.

        This maps to the inverse CDF. This is done using interpolation.
        """
        return self.inverse_cumulative_distribution(val)[()]

    @property
    def minimum(self):
        """Return minimum of the prior distribution.

        Updates the prior distribution if minimum is set to a different value.

        Yields an error if value is set below instantiated x-array minimum.

        Returns
        =======
        float: Minimum of the prior distribution

        """
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        if minimum < self.min_limit:
            raise ValueError('Minimum cannot be set below {}.'.format(round(self.min_limit, 2)))
        self._minimum = minimum
        if '_maximum' in self.__dict__ and self._maximum < np.inf:
            self._update_instance()

    @property
    def maximum(self):
        """Return maximum of the prior distribution.

        Updates the prior distribution if maximum is set to a different value.

        Yields an error if value is set above instantiated x-array maximum.

        Returns
        =======
        float: Maximum of the prior distribution

        """
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        if maximum > self.max_limit:
            raise ValueError('Maximum cannot be set above {}.'.format(round(self.max_limit, 2)))
        self._maximum = maximum
        if '_minimum' in self.__dict__ and self._minimum < np.inf:
            self._update_instance()

    @property
    def yy(self):
        """Return p(xx) values of the interpolated prior function.

        Updates the prior distribution if it is changed

        Returns
        =======
        array_like: p(xx) values

        """
        return self._yy

    @yy.setter
    def yy(self, yy):
        self._yy = yy
        self.__all_interpolated = interp1d(x=self.xx, y=self._yy, bounds_error=False, fill_value=0)
        self._update_instance()

    def _update_instance(self):
        self.xx = np.linspace(self.minimum, self.maximum, len(self.xx))
        self._yy = self.__all_interpolated(self.xx)
        self._initialize_attributes()

    def _initialize_attributes(self):
        from scipy.integrate import cumulative_trapezoid
        if trapezoid(self._yy, self.xx) != 1:
            logger.debug('Supplied PDF for {} is not normalised, normalising.'.format(self.name))
        self._yy /= trapezoid(self._yy, self.xx)
        self.YY = cumulative_trapezoid(self._yy, self.xx, initial=0)
        # Need last element of cumulative distribution to be exactly one.
        self.YY[-1] = 1
        self.probability_density = interp1d(x=self.xx, y=self._yy, bounds_error=False, fill_value=0)
        self.cumulative_distribution = _PiecewiseLinearCDF(self.xx, self._yy, self.YY)
        self.inverse_cumulative_distribution = self.cumulative_distribution.inverse


class _PiecewiseLinearCDF:
    """
    The CDF of the piecewise-linear density through (xx, yy), and its inverse.

    The density is linear inside each cell, so the CDF is quadratic there.
    Interpolating the CDF linearly instead would describe a cell-mean density.
    `xx` must increase and `YY` be the trapezoid cumulative of `yy`, as
    `Interped._initialize_attributes` builds them.
    """

    def __init__(self, xx, yy, YY):
        self.xx = xx
        self.yy = yy
        self.YY = YY  # the CDF at the grid points
        self.widths = np.diff(xx)
        self.slopes = np.diff(yy)  # the rise of the density across a cell, not per unit x

    def _grid(self, xp):
        """
        The grid in the namespace of the input, so that the methods below return the
        kind of array they are given, as `interp1d` also converts it.
        """
        arrays = (self.xx, self.yy, self.YY, self.widths, self.slopes)
        return tuple(xp.asarray(arr) for arr in arrays)

    @staticmethod
    def _cell_index(xp, grid, val):
        """The index of the cell of `grid` each value falls in, clipped to the end cells."""
        return xp.clip(xp.searchsorted(grid, val, side="right") - 1, 0, grid.shape[0] - 2)

    def __call__(self, val):
        """
        The CDF at `val`. With u = (x - xx[i]) / widths[i] from 0 to 1 across cell i,

            F(x) = YY[i] + widths[i] (yy[i] u + slopes[i] u^2 / 2).
        """
        xp = array_module(val)
        val = xp.asarray(val)
        xx, yy, YY, widths, slopes = self._grid(xp)
        i = self._cell_index(xp, xx, val)
        u = xp.clip((val - xx[i]) / widths[i], 0, 1)
        out = YY[i] + widths[i] * u * (yy[i] + slopes[i] * u / 2)
        # Pin the ends: recomputing the last cell can land on 1 - eps.
        out = xp.where(val >= xx[-1], 1.0, out)
        return xp.where(val <= xx[0], 0.0, out)

    def inverse(self, val):
        """The x with F(x) = `val`, solving yy[i] u + slopes[i] u^2 / 2 = t for u."""
        xp = array_module(val)
        val = xp.asarray(val)
        xx, yy, YY, widths, slopes = self._grid(xp)
        # As the interpolation this replaces did, reject quantiles outside the unit
        # interval, where comparing leaves any NaN to propagate instead.
        if aac.is_numpy_namespace(xp) and xp.any((val < 0) | (val > 1)):
            raise ValueError("A value in val is outside the unit interval [0, 1].")
        i = self._cell_index(xp, YY, val)  # the cell is found in the CDF, not in x
        y_left = yy[i]
        # The probability still to cover inside the cell, in units of its width.
        t = (val - YY[i]) / widths[i]
        # Of the two roots, this one keeps its accuracy as a cell flattens (u -> t / y).
        denominator = y_left + xp.sqrt(xp.maximum(y_left ** 2 + 2 * slopes[i] * t, 0))
        # Zero density makes this 0 / 0: the CDF is flat across the cell, so every point
        # in it has the same quantile and u = 0 will do. Dividing first, rather than
        # masking the division, leaves a NaN quantile as NaN.
        flat = denominator == 0
        u = xp.where(flat, 0.0, 2 * t / xp.where(flat, 1.0, denominator))
        out = xx[i] + widths[i] * xp.clip(u, 0, 1)
        # A quantile of one sits at the top of the support, even if the last cells hold
        # no probability and the search lands in one of them.
        return xp.where(val >= 1, xx[-1], out)


class FromFile(Interped):

    def __init__(self, file_name, minimum=None, maximum=None, name=None,
                 latex_label=None, unit=None, boundary=None):
        """Creates an interpolated prior function from arrays of xx and yy=p(xx) extracted from a file

        Parameters
        ==========
        file_name: str
            Name of the file containing the xx and yy arrays
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass

        """
        try:
            self.file_name = file_name
            xx, yy = np.genfromtxt(self.file_name).T
            super(FromFile, self).__init__(xx=xx, yy=yy, minimum=minimum,
                                           maximum=maximum, name=name, latex_label=latex_label,
                                           unit=unit, boundary=boundary)
        except IOError:
            logger.warning("Can't load {}.".format(self.file_name))
            logger.warning("Format should be:")
            logger.warning(r"x\tp(x)")
