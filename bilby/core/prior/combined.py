import numpy as np
from scipy.integrate import cumulative_simpson

from .base import Prior
from .dict import PriorDict


class Combined(Prior):
    def __init__(
        self,
        priors,
        weights=None,
        name=None,
        latex_label=None,
        unit=None,
        boundary=None,
        minimum=None,
        maximum=None,
    ):
        """
        Creates a combined prior from a list of priors and optionally
        corresponding weights. The individual priors are superposed as a
        weighted sum.

        Parameters
        ==========
        priors : Union[list, PriorDict, str]
            The priors to be combined.
        weights : array_like
            The weights for each prior. If None, all priors are given equal
            weight.
        name : str
            See superclass.
        latex_label : str
            See superclass.
        unit : str
            See superclass.
        boundary : str
            See superclass.
        minimum : float
            Set minimum for all priors. If None, the most restrictive 
            minimum of the priors is used.
        maximum : float
            Set maximum for all priors. If None, the most restrictive 
            maximum of the priors is used.
        """
        self.priors = priors
        if minimum is None:
            minimum = np.max([prior.minimum for prior in priors])
        if maximum is None:
            maximum = np.min([prior.maximum for prior in priors])

        if np.any(np.isinf([minimum, maximum])):
            raise ValueError(
                "Unable to use CombinedPrior with infinite bounds. "
                "Please set identical and finite bounds for all priors."
            )

        super().__init__(
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
            minimum=minimum,
            maximum=maximum,
        )

        if weights is None:
            self.weights = np.ones_like(priors) / len(priors)
        else:
            if len(weights) != len(priors):
                raise ValueError("Weights must have the same length as priors")
            self.weights = np.array(weights) / np.sum(weights)
        self.reset_interpolation()

    def reset_interpolation(self):
        if not hasattr(self, "weights"):
            # ignore when setup is not yet complete
            return
        self.support = np.linspace(self.minimum, self.maximum, 1000)
        pdf = self.prob(self.support)
        self.interp_cdf = cumulative_simpson(y=pdf, x=self.support, initial=0)

    def rescale(self, val):
        """
        Rescale a sample from the unit line element to the prior.

        Parameters
        ==========
        val : Union[float, int, array_like]
        """
        return np.interp(
            val,
            self.interp_cdf,
            self.support,
            left=self._minimum,
            right=self._maximum,
        )

    def prob(self, val):
        """
        Return the prior probability of val.

        Parameters
        ==========
        val : Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]
            Prior probability of val.
        """
        prob = 0
        for weight, prior in zip(self.weights, self._priors):
            prob += weight * prior.prob(val)
        return prob

    def cdf(self, val):
        """
        Return the cumulative distribution function of val.

        Parameters
        ==========
        val : Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]
            CDF of val.
        """
        return np.interp(val, self.support, self.interp_cdf, left=0.0, right=1.0)

    @property
    def priors(self):
        return self._priors

    @priors.setter
    def priors(self, priors):
        if isinstance(priors, str):
            priors = PriorDict(priors)
        if isinstance(priors, dict):
            priors = list(priors.values())
        self._priors = priors
        self.reset_interpolation()

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        self._minimum = minimum
        for prior in self._priors:
            prior.minimum = minimum
        self.reset_interpolation()

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        self._maximum = maximum
        for prior in self._priors:
            prior.maximum = maximum
        self.reset_interpolation()