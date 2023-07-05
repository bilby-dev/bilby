import numpy as np
import pandas as pd
import scipy.linalg
from scipy.optimize import minimize


class FisherMatrixPosteriorEstimator(object):
    def __init__(self, likelihood, priors, parameters=None, fd_eps=1e-6, n_prior_samples=100):
        """ A class to estimate posteriors using the Fisher Matrix approach

        Parameters
        ----------
        likelihood: bilby.core.likelihood.Likelihood
            A bilby likelihood object
        priors: bilby.core.prior.PriorDict
            A bilby prior object
        parameters: list
            Names of parameters to sample in
        fd_eps: float
            A parameter to control the size of perturbation used when finite
            differencing the likelihood
        n_prior_samples: int
            The number of prior samples to draw and use to attempt estimatation
            of the maximum likelihood sample.
        """
        self.likelihood = likelihood
        if parameters is None:
            self.parameter_names = priors.non_fixed_keys
        else:
            self.parameter_names = parameters
        self.fd_eps = fd_eps
        self.n_prior_samples = n_prior_samples
        self.N = len(self.parameter_names)

        self.prior_samples = [
            priors.sample_subset(self.parameter_names) for _ in range(n_prior_samples)
        ]
        self.prior_bounds = [(priors[key].minimum, priors[key].maximum) for key in self.parameter_names]

        self.prior_width_dict = {}
        for key in self.parameter_names:
            width = priors[key].width
            if np.isnan(width):
                raise ValueError(f"Prior width is ill-formed for {key}")
            self.prior_width_dict[key] = width

    def log_likelihood(self, sample):
        self.likelihood.parameters.update(sample)
        return self.likelihood.log_likelihood()

    def calculate_iFIM(self, sample):
        FIM = self.calculate_FIM(sample)
        iFIM = scipy.linalg.inv(FIM)

        # Ensure iFIM is positive definite
        min_eig = np.min(np.real(np.linalg.eigvals(iFIM)))
        if min_eig < 0:
            iFIM -= 10 * min_eig * np.eye(*iFIM.shape)

        return iFIM

    def sample_array(self, sample, n=1):
        from .utils.random import rng

        if sample == "maxL":
            sample = self.get_maximum_likelihood_sample()

        self.mean = np.array(list(sample.values()))
        self.iFIM = self.calculate_iFIM(sample)
        return rng.multivariate_normal(self.mean, self.iFIM, n)

    def sample_dataframe(self, sample, n=1):
        samples = self.sample_array(sample, n)
        return pd.DataFrame(samples, columns=self.parameter_names)

    def calculate_FIM(self, sample):
        FIM = np.zeros((self.N, self.N))
        for ii, ii_key in enumerate(self.parameter_names):
            for jj, jj_key in enumerate(self.parameter_names):
                FIM[ii, jj] = -self.get_second_order_derivative(sample, ii_key, jj_key)

        return FIM

    def get_second_order_derivative(self, sample, ii, jj):
        if ii == jj:
            return self.get_finite_difference_xx(sample, ii)
        else:
            return self.get_finite_difference_xy(sample, ii, jj)

    def get_finite_difference_xx(self, sample, ii):
        # Sample grid
        p = self.shift_sample_x(sample, ii, 1)
        m = self.shift_sample_x(sample, ii, -1)

        dx = .5 * (p[ii] - m[ii])

        loglp = self.log_likelihood(p)
        logl = self.log_likelihood(sample)
        loglm = self.log_likelihood(m)

        return (loglp - 2 * logl + loglm) / dx ** 2

    def get_finite_difference_xy(self, sample, ii, jj):
        # Sample grid
        pp = self.shift_sample_xy(sample, ii, 1, jj, 1)
        pm = self.shift_sample_xy(sample, ii, 1, jj, -1)
        mp = self.shift_sample_xy(sample, ii, -1, jj, 1)
        mm = self.shift_sample_xy(sample, ii, -1, jj, -1)

        dx = .5 * (pp[ii] - mm[ii])
        dy = .5 * (pp[jj] - mm[jj])

        loglpp = self.log_likelihood(pp)
        loglpm = self.log_likelihood(pm)
        loglmp = self.log_likelihood(mp)
        loglmm = self.log_likelihood(mm)

        return (loglpp - loglpm - loglmp + loglmm) / (4 * dx * dy)

    def shift_sample_x(self, sample, x_key, x_coef):

        vx = sample[x_key]
        dvx = self.fd_eps * self.prior_width_dict[x_key]

        shift_sample = sample.copy()
        shift_sample[x_key] = vx + x_coef * dvx

        return shift_sample

    def shift_sample_xy(self, sample, x_key, x_coef, y_key, y_coef):

        vx = sample[x_key]
        vy = sample[y_key]

        dvx = self.fd_eps * self.prior_width_dict[x_key]
        dvy = self.fd_eps * self.prior_width_dict[y_key]

        shift_sample = sample.copy()
        shift_sample[x_key] = vx + x_coef * dvx
        shift_sample[y_key] = vy + y_coef * dvy
        return shift_sample

    def get_maximum_likelihood_sample(self, initial_sample=None):
        """ A method to attempt optimization of the maximum likelihood

        This uses a simple scipy optimization approach, starting from a number
        of draws from the prior to avoid problems with local optimization.

        Note: this approach works well in small numbers of dimensions when the
        posterior is narrow relative to the prior. But, if the number of dimensions
        is large or the posterior is wide relative to the prior, the method fails
        to find the global maximum in high dimensional problems.
        """
        minlogL = np.inf
        for i in range(self.n_prior_samples):
            initial_sample = self.prior_samples[i]

            x0 = list(initial_sample.values())

            def neg_log_like(x, self, T=1):
                sample = {key: val for key, val in zip(self.parameter_names, x)}
                return - 1 / T * self.log_likelihood(sample)

            out = minimize(
                neg_log_like,
                x0,
                args=(self, 1),
                bounds=self.prior_bounds,
                method="L-BFGS-B",
            )
            if out.fun < minlogL:
                minout = out

        return {key: val for key, val in zip(self.parameter_names, minout.x)}
