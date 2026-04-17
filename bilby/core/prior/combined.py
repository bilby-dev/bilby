from .base import Prior
import numpy as np
from scipy.integrate import cumulative_trapezoid
class Combined(Prior):
    def __init__(self, priors, weights=None, name=None, latex_label=None, unit=None, boundary=None):
        """
        Creates a combined prior from a list of priors and optionally corresponding weights. The individual priors are superposed as a weighted sum.
        Parameters
        ==========
        priors: list
            The priors to be combined. 
        weights: array_like
            The weights for each prior. If None, all priors are given equal weight.
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        
        prior_mins = [prior.minimum for prior in priors]
        prior_maxs = [prior.maximum for prior in priors]
        assert all(prior_min == prior_mins[0] for prior_min in prior_mins), "All priors must have the same minimum"
        assert all(prior_max == prior_maxs[0] for prior_max in prior_maxs), "All priors must have the same maximum"
        if np.any(np.isinf([self.minimum, self.maximum])):
            raise ValueError(
                "Unable to use CombinedPrior with infinite bounds. Please set identical and finite bounds for all priors.")
        
        super().__init__(name=name, latex_label=latex_label, unit=unit, boundary=boundary, minimum=prior_mins[0], maximum=prior_maxs[0])

        self.priors = priors
        if weights is None:
            self.weights = np.ones_like(priors) / len(priors)
        else:
            self.weights = np.array(weights)/ np.sum(weights)
            assert len(weights) == len(priors), "Weights must have the same length as priors"
        
        self.support = np.linspace(self.minimum, self.maximum, 1000)
        pdf = self.prob(self.support)
        self.interp_cdf = cumulative_trapezoid(pdf, self.support, initial=0)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior. This maps to the inverse CDF.

        Parameters
        ==========
        val: Union[float, int, array_like]

        """
        return np.interp(val, self.interp_cdf, self.support)

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        prob = 0
        for weight, prior in zip(self.weights, self.priors):
            prob += weight * prior.prob(val)
        return prob
    
    def cdf(self, val):
        """Return the cumulative distribution function of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: CDF of val
        """
        return np.interp(val, self.support, self.interp_cdf)
    
