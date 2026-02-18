from inspect import isclass

import numpy as np

from ..prior import Uniform
from ..utils import random


class Sample(dict):
    def __init__(self, dictionary=None):
        if dictionary is None:
            dictionary = dict()
        super(Sample, self).__init__(dictionary)

    def __add__(self, other):
        return Sample({key: self[key] + other[key] for key in self.keys()})

    def __sub__(self, other):
        return Sample({key: self[key] - other[key] for key in self.keys()})

    def __mul__(self, other):
        return Sample({key: self[key] * other for key in self.keys()})

    @classmethod
    def from_cpnest_live_point(cls, cpnest_live_point):
        res = cls(dict())
        for i, key in enumerate(cpnest_live_point.names):
            res[key] = cpnest_live_point.values[i]
        return res

    @classmethod
    def from_external_type(cls, external_sample, sampler_name):
        if sampler_name == "cpnest":
            return cls.from_cpnest_live_point(external_sample)
        return external_sample


class JumpProposal(object):
    def __init__(self, priors=None):
        """A generic class for jump proposals

        Parameters
        ==========
        priors: bilby.core.prior.PriorDict
            Dictionary of priors used in this sampling run

        Attributes
        ==========
        log_j: float
            Log Jacobian of the proposal. Characterises whether or not detailed balance
            is preserved. If not, log_j needs to be adjusted accordingly.
        """
        self.priors = priors
        self.log_j = 0.0

    def __call__(self, sample, **kwargs):
        """A generic wrapper for the jump proposal function

        Parameters
        ==========
        args: Arguments that are going to be passed into the proposal function
        kwargs: Keyword arguments that are going to be passed into the proposal function

        Returns
        =======
        dict: A dictionary with the new samples. Boundary conditions are applied.

        """
        return self._apply_boundaries(sample)

    def _move_reflecting_keys(self, sample):
        keys = [
            key for key in sample.keys() if self.priors[key].boundary == "reflective"
        ]
        for key in keys:
            if (
                sample[key] > self.priors[key].maximum
                or sample[key] < self.priors[key].minimum
            ):
                r = self.priors[key].maximum - self.priors[key].minimum
                delta = (sample[key] - self.priors[key].minimum) % (2 * r)
                if delta > r:
                    sample[key] = (
                        2 * self.priors[key].maximum - self.priors[key].minimum - delta
                    )
                elif delta < r:
                    sample[key] = self.priors[key].minimum + delta
        return sample

    def _move_periodic_keys(self, sample):
        keys = [key for key in sample.keys() if self.priors[key].boundary == "periodic"]
        for key in keys:
            if (
                sample[key] > self.priors[key].maximum
                or sample[key] < self.priors[key].minimum
            ):
                sample[key] = self.priors[key].minimum + (
                    (sample[key] - self.priors[key].minimum)
                    % (self.priors[key].maximum - self.priors[key].minimum)
                )
        return sample

    def _apply_boundaries(self, sample):
        sample = self._move_periodic_keys(sample)
        sample = self._move_reflecting_keys(sample)
        return sample


class JumpProposalCycle(object):
    def __init__(self, proposal_functions, weights, cycle_length=100):
        """A generic wrapper class for proposal cycles

        Parameters
        ==========
        proposal_functions: list
            A list of callable proposal functions/objects
        weights: list
            A list of integer weights for the respective proposal functions
        cycle_length: int, optional
            Length of the proposal cycle
        """
        self.proposal_functions = proposal_functions
        self.weights = weights
        self.cycle_length = cycle_length
        self._index = 0
        self._cycle = np.array([])
        self.update_cycle()

    def __call__(self, **kwargs):
        proposal = self._cycle[self.index]
        self._index = (self.index + 1) % self.cycle_length
        return proposal(**kwargs)

    def __len__(self):
        return len(self.proposal_functions)

    def update_cycle(self):
        self._cycle = random.rng.choice(
            self.proposal_functions,
            size=self.cycle_length,
            p=self.weights,
            replace=True,
        )

    @property
    def proposal_functions(self):
        return self._proposal_functions

    @proposal_functions.setter
    def proposal_functions(self, proposal_functions):
        for i, proposal in enumerate(proposal_functions):
            if isclass(proposal):
                proposal_functions[i] = proposal()
        self._proposal_functions = proposal_functions

    @property
    def index(self):
        return self._index

    @property
    def weights(self):
        """

        Returns
        =======
        Normalised proposal weights

        """
        return np.array(self._weights) / np.sum(np.array(self._weights))

    @weights.setter
    def weights(self, weights):
        assert len(weights) == len(self.proposal_functions)
        self._weights = weights

    @property
    def unnormalised_weights(self):
        return self._weights


class NormJump(JumpProposal):
    def __init__(self, step_size, priors=None):
        """
        A normal distributed step centered around the old sample

        Parameters
        ==========
        step_size: float
            The scalable step size
        priors:
            See superclass
        """
        super(NormJump, self).__init__(priors)
        self.step_size = step_size

    def __call__(self, sample, **kwargs):
        for key in sample.keys():
            sample[key] = random.rng.normal(sample[key], self.step_size)
        return super(NormJump, self).__call__(sample)


class EnsembleWalk(JumpProposal):
    def __init__(
        self,
        random_number_generator=random.rng.uniform,
        n_points=3,
        priors=None,
        **random_number_generator_args
    ):
        """
        An ensemble walk

        Parameters
        ==========
        random_number_generator: func, optional
            A random number generator. Default is random.random
        n_points: int, optional
            Number of points in the ensemble to average over. Default is 3.
        priors:
            See superclass
        random_number_generator_args:
            Additional keyword arguments for the random number generator
        """
        super(EnsembleWalk, self).__init__(priors)
        self.random_number_generator = random_number_generator
        self.n_points = n_points
        self.random_number_generator_args = random_number_generator_args

    def __call__(self, sample, **kwargs):
        subset = random.rng.choice(kwargs["coordinates"], self.n_points, replace=False)
        for i in range(len(subset)):
            subset[i] = Sample.from_external_type(
                subset[i], kwargs.get("sampler_name", None)
            )
        center_of_mass = self.get_center_of_mass(subset)
        for x in subset:
            sample += (x - center_of_mass) * self.random_number_generator(
                **self.random_number_generator_args
            )
        return super(EnsembleWalk, self).__call__(sample)

    @staticmethod
    def get_center_of_mass(subset):
        return {key: np.mean([c[key] for c in subset]) for key in subset[0].keys()}


class EnsembleStretch(JumpProposal):
    def __init__(self, scale=2.0, priors=None):
        """
        Stretch move. Calculates the log Jacobian which can be used in cpnest to bias future moves.

        Parameters
        ==========
        scale: float, optional
            Stretching scale. Default is 2.0.
        """
        super(EnsembleStretch, self).__init__(priors)
        self.scale = scale

    def __call__(self, sample, **kwargs):
        second_sample = random.rng.choice(kwargs["coordinates"])
        second_sample = Sample.from_external_type(
            second_sample, kwargs.get("sampler_name", None)
        )
        step = random.rng.uniform(-1, 1) * np.log(self.scale)
        sample = second_sample + (sample - second_sample) * np.exp(step)
        self.log_j = len(sample) * step
        return super(EnsembleStretch, self).__call__(sample)


class DifferentialEvolution(JumpProposal):
    def __init__(self, sigma=1e-4, mu=1.0, priors=None):
        """
        Differential evolution step. Takes two elements from the existing coordinates and differentially evolves the
        old sample based on them using some Gaussian randomisation in the step.

        Parameters
        ==========
        sigma: float, optional
            Random spread in the evolution step. Default is 1e-4
        mu: float, optional
            Scale of the randomization. Default is 1.0
        """
        super(DifferentialEvolution, self).__init__(priors)
        self.sigma = sigma
        self.mu = mu

    def __call__(self, sample, **kwargs):
        a, b = random.rng.choice(kwargs["coordinates"], 2, replace=False)
        sample = sample + (b - a) * random.rng.normal(self.mu, self.sigma)
        return super(DifferentialEvolution, self).__call__(sample)


class EnsembleEigenVector(JumpProposal):
    def __init__(self, priors=None):
        """
        Ensemble step based on the ensemble eigenvectors.

        Parameters
        ==========
        priors:
            See superclass
        """
        super(EnsembleEigenVector, self).__init__(priors)
        self.eigen_values = None
        self.eigen_vectors = None
        self.covariance = None

    def update_eigenvectors(self, coordinates):
        if coordinates is None:
            return
        elif len(coordinates[0]) == 1:
            self._set_1_d_eigenvectors(coordinates)
        else:
            self._set_n_d_eigenvectors(coordinates)

    def _set_1_d_eigenvectors(self, coordinates):
        n_samples = len(coordinates)
        key = list(coordinates[0].keys())[0]
        variance = np.var([coordinates[j][key] for j in range(n_samples)])
        self.eigen_values = np.atleast_1d(variance)
        self.covariance = self.eigen_values
        self.eigen_vectors = np.eye(1)

    def _set_n_d_eigenvectors(self, coordinates):
        n_samples = len(coordinates)
        dim = len(coordinates[0])
        cov_array = np.zeros((dim, n_samples))
        for i, key in enumerate(coordinates[0].keys()):
            for j in range(n_samples):
                cov_array[i, j] = coordinates[j][key]
        self.covariance = np.cov(cov_array)
        self.eigen_values, self.eigen_vectors = np.linalg.eigh(self.covariance)

    def __call__(self, sample, **kwargs):
        self.update_eigenvectors(kwargs["coordinates"])
        i = random.rng.integers(len(sample))
        jump_size = np.sqrt(np.fabs(self.eigen_values[i])) * random.rng.normal(0, 1)
        for j, key in enumerate(sample.keys()):
            sample[key] += jump_size * self.eigen_vectors[j, i]
        return super(EnsembleEigenVector, self).__call__(sample)


class DrawFlatPrior(JumpProposal):
    """
    Draws a proposal from the flattened prior distribution.
    """

    def __call__(self, sample, **kwargs):
        sample = _draw_from_flat_priors(sample, self.priors)
        return super(DrawFlatPrior, self).__call__(sample)


def _draw_from_flat_priors(sample, priors):
    for key in sample.keys():
        flat_prior = Uniform(priors[key].minimum, priors[key].maximum, priors[key].name)
        sample[key] = flat_prior.sample()
    return sample
