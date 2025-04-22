from packaging import version

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.optimize import minimize
import tqdm

from .utils import random, logger
from .prior import PriorDict


def array_to_dict(keys, array):
    return dict(zip(keys, array))


class FisherMatrixPosteriorEstimator(object):
    def __init__(self, likelihood, priors, parameters=None, minimization_method="Nelder-Mead",
                 fd_eps=1e-6, n_prior_samples=100):
        """ A class to estimate posteriors using a Fisher Information Matrix approach

        Parameters
        ----------
        likelihood: bilby.core.likelihood.Likelihood
            A bilby likelihood object
        priors: bilby.core.prior.PriorDict
            A bilby prior object
        parameters: list
            Names of parameters to sample in
        minimization_method: str (Nelder-Mead)
            The method to use in scipy.optimize.minimize
        fd_eps: float
            A parameter to control the size of perturbation used when finite
            differencing the likelihood
        n_prior_samples: int
            The number of prior samples to draw and use to attempt estimatation
            of the maximum likelihood sample.
        """
        self.likelihood = likelihood

        if isinstance(priors, PriorDict) is False:
            priors = PriorDict(priors)

        if parameters is None:
            self.parameter_names = priors.non_fixed_keys
        else:
            self.parameter_names = parameters
        self.minimization_method = minimization_method
        self.fd_eps = fd_eps
        self.n_prior_samples = n_prior_samples
        self.N = len(self.parameter_names)

        # Construct prior samples at initialisation so that the prior is not stored
        self.prior_samples = [
            priors.sample_subset(self.parameter_names) for _ in range(n_prior_samples)
        ]
        self.prior_bounds_min = np.array([priors[key].minimum for key in self.parameter_names])
        self.prior_bounds_max = np.array([priors[key].maximum for key in self.parameter_names])
        self.prior_bounds = list(zip(self.prior_bounds_min, self.prior_bounds_max))

        self.prior_width_dict = {}
        for key in self.parameter_names:
            width = priors[key].width
            if np.isnan(width):
                raise ValueError(f"Prior width is ill-formed for {key}")
            self.prior_width_dict[key] = width

    def log_likelihood(self, sample):
        if isinstance(sample, dict) is False:
            if isinstance(sample, pd.DataFrame) and len(sample) == 1:
                sample = sample.to_dict()
            else:
                raise ValueError()
        self.likelihood.parameters.update(sample)
        return self.likelihood.log_likelihood()

    def calculate_iFIM(self, sample):
        FIM = self.calculate_FIM(sample)

        # Force the FIM to be symmetric by averaging off-diagonal estimates
        upper_off_diagonal_average = .5 * (np.triu(FIM, 1) + np.triu(FIM.T, 1))
        FIM = np.diag(np.diag(FIM)) + upper_off_diagonal_average + upper_off_diagonal_average.T

        iFIM = scipy.linalg.inv(FIM)

        # Ensure iFIM is positive definite
        min_eig = np.min(np.real(np.linalg.eigvals(iFIM)))
        if min_eig < 0:
            logger.warning("Scaling the iFIM to ensure it is positive definite")
            iFIM -= 10 * min_eig * np.eye(*iFIM.shape)

        return iFIM

    def sample_array(self, sample, n=1):
        if sample == "maxL":
            sample = self.get_maximum_likelihood_sample()

        self.mean = np.array(list(sample.values()))
        self.iFIM = self.calculate_iFIM(sample)
        return random.rng.multivariate_normal(self.mean, self.iFIM, n)

    def sample_dataframe(self, sample, n=1):
        samples = self.sample_array(sample, n)
        return pd.DataFrame(samples, columns=self.parameter_names)

    def calculate_FIM(self, sample):
        if version.parse(scipy.__version__) < version.parse("1.15"):
            logger.info("Scipy version < 1.15, using fallback")
            FIM = np.zeros((self.N, self.N))
            for ii, ii_key in enumerate(self.parameter_names):
                for jj, jj_key in enumerate(self.parameter_names):
                    FIM[ii, jj] = -self.get_second_order_derivative(sample, ii_key, jj_key)
            return FIM
        else:
            import scipy.differentiate as sd
            logger.info("Using Scipy differentiate to estimate the Fisher information matrix (FIM)")
            point = np.array([sample[key] for key in self.parameter_names])
            res = sd.hessian(self.log_likelihood_from_array, point, initial_step=0.5)
            FIM = - res.ddf
            logger.debug(f"Estimated FIM:\n{FIM}")

        return FIM

    def log_likelihood_from_array(self, x_array):
        def wrapped_logl(x_array):
            # Map points outside the bounds to the bounds
            idxs = x_array < self.prior_bounds_min
            x_array[idxs] = self.prior_bounds_min[idxs]

            idxs = x_array > self.prior_bounds_max
            x_array[idxs] = self.prior_bounds_max[idxs]

            return self.log_likelihood(array_to_dict(self.parameter_names, x_array))

        def wrapped_logl_arb(x_array):
            return np.apply_along_axis(wrapped_logl, 0, x_array)

        return wrapped_logl_arb(x_array)

    def get_second_order_derivative(self, sample, ii, jj):
        if ii == jj:
            return self.get_finite_difference_xx(sample, ii)
        else:
            return self.get_finite_difference_xy(sample, ii, jj)

    def get_finite_difference_xx(self, sample, ii):
        # Sample grid
        p = self._shift_sample_x(sample, ii, 1)
        m = self._shift_sample_x(sample, ii, -1)

        dx = .5 * (p[ii] - m[ii])

        loglp = self.log_likelihood(p)
        logl = self.log_likelihood(sample)
        loglm = self.log_likelihood(m)

        return (loglp - 2 * logl + loglm) / dx ** 2

    def get_finite_difference_xy(self, sample, ii, jj):
        # Sample grid
        pp = self._shift_sample_xy(sample, ii, 1, jj, 1)
        pm = self._shift_sample_xy(sample, ii, 1, jj, -1)
        mp = self._shift_sample_xy(sample, ii, -1, jj, 1)
        mm = self._shift_sample_xy(sample, ii, -1, jj, -1)

        dx = .5 * (pp[ii] - mm[ii])
        dy = .5 * (pp[jj] - mm[jj])

        loglpp = self.log_likelihood(pp)
        loglpm = self.log_likelihood(pm)
        loglmp = self.log_likelihood(mp)
        loglmm = self.log_likelihood(mm)

        return (loglpp - loglpm - loglmp + loglmm) / (4 * dx * dy)

    def _shift_sample_x(self, sample, x_key, x_coef):

        vx = sample[x_key]
        dvx = self.fd_eps * self.prior_width_dict[x_key]

        shift_sample = sample.copy()
        shift_sample[x_key] = vx + x_coef * dvx

        return shift_sample

    def _shift_sample_xy(self, sample, x_key, x_coef, y_key, y_coef):

        vx = sample[x_key]
        vy = sample[y_key]

        dvx = self.fd_eps * self.prior_width_dict[x_key]
        dvy = self.fd_eps * self.prior_width_dict[y_key]

        shift_sample = sample.copy()
        shift_sample[x_key] = vx + x_coef * dvx
        shift_sample[y_key] = vy + y_coef * dvy
        return shift_sample

    def _maximize_likelihood_from_initial_sample(self, initial_sample):
        x0 = list(initial_sample.values())

        def neg_log_like(x):
            return - self.log_likelihood_from_array(x)

        out = minimize(
            neg_log_like,
            x0,
            bounds=self.prior_bounds,
            method=self.minimization_method,
        )
        return out

    def get_maximum_likelihood_sample(self, initial_sample=None):
        """ A method to attempt optimization of the maximum likelihood

        This uses a simple scipy optimization approach, starting from a number
        of draws from the prior to avoid problems with local optimization.

        Note: this approach works well in small numbers of dimensions when the
        posterior is narrow relative to the prior. But, if the number of dimensions
        is large or the posterior is wide relative to the prior, the method fails
        to find the global maximum in high dimensional problems.
        """

        if initial_sample:
            out = self._maximize_likelihood_from_initial_sample(initial_sample)
        else:
            logger.info(f"Maximising the likelihood using {self.n_prior_samples} prior samples")
            max_logL = -np.inf
            logL_list = []
            successes = 0
            for sample in tqdm.tqdm(self.prior_samples):
                out = self._maximize_likelihood_from_initial_sample(sample)
                logL = -out.fun
                logL_list.append(logL)
                if out.success:
                    successes += 1
                if logL > max_logL:
                    max_logL = logL
                    minout = out

            if np.isinf(max_logL):
                raise ValueError("Maxisation of the likelihood failed")

            logger.info(
                f"Finished with {100 * successes / self.n_prior_samples}% success rate| "
                f"Maximum log-likelihood {max_logL}| "
                f"(max-mu)/sigma= {(max_logL - np.mean(logL_list)) / np.std(logL_list)} "
            )

        self.minimization_metadata = minout
        logger.info(f"Maximum likelihood estimation: {minout.message}")
        return {key: val for key, val in zip(self.parameter_names, minout.x)}
