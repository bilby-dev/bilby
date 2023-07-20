import numpy as np
import pandas as pd
from packaging import version

from ..core.sampler.base_sampler import SamplerError
from ..core.utils import logger
from .utils import LOGLKEY, LOGLLATEXKEY, LOGPKEY, LOGPLATEXKEY


class Chain(object):
    def __init__(
        self,
        initial_sample,
        burn_in_nact=1,
        thin_by_nact=1,
        fixed_discard=0,
        autocorr_c=5,
        min_tau=1,
        fixed_tau=None,
        tau_window=None,
        block_length=100000,
    ):
        """Object to store a single mcmc chain

        Parameters
        ----------
        initial_sample: bilby.bilby_mcmc.chain.Sample
            The starting point of the chain
        burn_in_nact, thin_by_nact : int (1, 1)
            The number of autocorrelation times (tau) to discard for burn-in
            and the multiplicative factor to thin by (thin_by_nact < 1). I.e
            burn_in_nact=10 and thin_by_nact=1 will discard 10*tau samples from
            the start of the chain, then thin the final chain by a factor
            of 1*tau (resulting in independent samples).
        fixed_discard: int (0)
            A fixed minimum number of samples to discard (can be used to
            override the burn_in_nact if it is too small).
        autocorr_c: float (5)
            The step size of the window search used by emcee.autocorr when
            estimating the autocorrelation time.
        min_tau: int (1)
            A minimum value for the autocorrelation time.
        fixed_tau: int (None)
            A fixed value for the autocorrelation (overrides the automated
            autocorrelation time estimation). Used in testing.
        tau_window: int (None)
            Only calculate the autocorrelation time in a trailing window. If
            None (default) this method is not used.
        block_length: int
            The incremental size to extend the array by when it runs out of
            space.
        """
        self.autocorr_c = autocorr_c
        self.min_tau = min_tau
        self.burn_in_nact = burn_in_nact
        self.thin_by_nact = thin_by_nact
        self.block_length = block_length
        self.fixed_discard = int(fixed_discard)
        self.fixed_tau = fixed_tau
        self.tau_window = tau_window

        self.ndim = initial_sample.ndim
        self.current_sample = initial_sample
        self.keys = self.current_sample.keys
        self.parameter_keys = self.current_sample.parameter_keys

        # Initialize chain
        self._chain_array = self._get_zero_chain_array()
        self._chain_array_length = block_length
        self.position = -1
        self.max_log_likelihood = -np.inf
        self.max_tau_dict = {}
        self.converged = False
        self.cached_tau_count = 0
        self._minimum_index_proposal = 0
        self._minimum_index_adapt = 0
        self._last_minimum_index = (0, 0, "I")
        self.last_full_tau_dict = {key: np.inf for key in self.parameter_keys}

        # Append the initial sample
        self.append(self.current_sample)

    def _get_zero_chain_array(self):
        return np.zeros((self.block_length, self.ndim + 2), dtype=np.float64)

    def _extend_chain_array(self):
        self._chain_array = np.concatenate(
            (self._chain_array, self._get_zero_chain_array()), axis=0
        )
        self._chain_array_length = len(self._chain_array)

    @property
    def current_sample(self):
        return self._current_sample.copy()

    @current_sample.setter
    def current_sample(self, current_sample):
        self._current_sample = current_sample

    def append(self, sample):
        self.position += 1

        # Extend the array if needed
        if self.position >= self._chain_array_length:
            self._extend_chain_array()

        # Store the current sample and append to the array
        self.current_sample = sample
        self._chain_array[self.position] = sample.list

        # Update the maximum log_likelihood
        if sample[LOGLKEY] > self.max_log_likelihood:
            self.max_log_likelihood = sample[LOGLKEY]

    def __getitem__(self, index):
        if index < 0:
            index = index + self.position + 1

        if index <= self.position:
            values = self._chain_array[index]
            return Sample({k: v for k, v in zip(self.keys, values)})
        else:
            raise SamplerError(f"Requested index {index} out of bounds")

    def __setitem__(self, index, sample):
        if index < 0:
            index = index + self.position + 1

        self._chain_array[index] = sample.list

    def key_to_idx(self, key):
        return self.keys.index(key)

    def get_1d_array(self, key):
        return self._chain_array[: 1 + self.position, self.key_to_idx(key)]

    @property
    def _random_idx(self):
        from ..core.utils.random import rng

        mindex = self._last_minimum_index[1]
        # Check if mindex exceeds current position by 10 ACT: if so use a random sample
        # otherwise we draw only from the chain past the minimum_index
        if np.isinf(self.tau_last) or self.position - mindex < 10 * self.tau_last:
            mindex = 0
        return rng.integers(mindex, self.position + 1)

    @property
    def random_sample(self):
        return self[self._random_idx]

    @property
    def fixed_discard(self):
        return self._fixed_discard

    @fixed_discard.setter
    def fixed_discard(self, fixed_discard):
        self._fixed_discard = int(fixed_discard)

    @property
    def minimum_index(self):
        """This calculates a minimum index from which to discard samples

        A number of methods are provided for the calculation. A subset are
        switched off (by `if False` statements) for future development

        """
        position = self.position

        # Return cached minimum index
        last_minimum_index = self._last_minimum_index
        if position == last_minimum_index[0]:
            return int(last_minimum_index[1])

        # If fixed discard is not yet reached, just return that
        if position < self.fixed_discard:
            self.minimum_index_method = "FD"
            return self.fixed_discard

        # Initialize list of minimum index methods with the fixed discard (FD)
        minimum_index_list = [self.fixed_discard]
        minimum_index_method_list = ["FD"]

        # Calculate minimum index from tau
        if self.tau_last < np.inf:
            tau = self.tau_last
        elif len(self.max_tau_dict) == 0:
            # Bootstrap calculating tau when minimum index has not yet been calculated
            tau = self._tau_for_full_chain
        else:
            tau = np.inf

        if tau < np.inf:
            minimum_index_list.append(self.burn_in_nact * tau)
            minimum_index_method_list.append(f"{self.burn_in_nact}tau")

        # Calculate points when log-posterior is within z std of the mean
        if True:
            zfactor = 1
            N = 100
            delta_lnP = zfactor * self.ndim / 2
            logl = self.get_1d_array(LOGLKEY)
            log_prior = self.get_1d_array(LOGPKEY)
            log_posterior = logl + log_prior
            max_posterior = np.max(log_posterior)

            ave = pd.Series(log_posterior).rolling(window=N).mean().iloc[N - 1 :]
            delta = max_posterior - ave
            passes = ave[delta < delta_lnP]
            if len(passes) > 0:
                minimum_index_list.append(passes.index[0] + 1)
                minimum_index_method_list.append(f"z{zfactor}")

        # Add last minimum_index_method
        if False:
            minimum_index_list.append(last_minimum_index[1])
            minimum_index_method_list.append(last_minimum_index[2])

        # Minimum index set by proposals
        minimum_index_list.append(self.minimum_index_proposal)
        minimum_index_method_list.append("PR")

        # Minimum index set by temperature adaptation
        minimum_index_list.append(self.minimum_index_adapt)
        minimum_index_method_list.append("AD")

        # Calculate the maximum minimum index and associated method (reporting)
        minimum_index = int(np.max(minimum_index_list))
        minimum_index_method = minimum_index_method_list[np.argmax(minimum_index_list)]

        # Cache the method
        self._last_minimum_index = (position, minimum_index, minimum_index_method)
        self.minimum_index_method = minimum_index_method

        return minimum_index

    @property
    def minimum_index_proposal(self):
        return self._minimum_index_proposal

    @minimum_index_proposal.setter
    def minimum_index_proposal(self, minimum_index_proposal):
        if minimum_index_proposal > self._minimum_index_proposal:
            self._minimum_index_proposal = minimum_index_proposal

    @property
    def minimum_index_adapt(self):
        return self._minimum_index_adapt

    @minimum_index_adapt.setter
    def minimum_index_adapt(self, minimum_index_adapt):
        if minimum_index_adapt > self._minimum_index_adapt:
            self._minimum_index_adapt = minimum_index_adapt

    @property
    def tau(self):
        """The maximum ACT over all parameters"""

        if self.position in self.max_tau_dict:
            # If we have the ACT at the current position, return it
            return self.max_tau_dict[self.position]
        elif (
            self.tau_last < np.inf
            and self.cached_tau_count < 50
            and self.nsamples_last > 50
        ):
            # If we have a recent ACT return it
            self.cached_tau_count += 1
            return self.tau_last
        else:
            # Calculate the ACT
            return self.tau_nocache

    @property
    def tau_nocache(self):
        """Calculate tau forcing a recalculation (no cached tau)"""
        tau = max(self.tau_dict.values())
        self.max_tau_dict[self.position] = tau
        self.cached_tau_count = 0
        return tau

    @property
    def tau_last(self):
        """Return the last-calculated tau if it exists, else inf"""
        if len(self.max_tau_dict) > 0:
            return list(self.max_tau_dict.values())[-1]
        else:
            return np.inf

    @property
    def _tau_for_full_chain(self):
        """The maximum ACT over all parameters"""
        return max(self._tau_dict_for_full_chain.values())

    @property
    def _tau_dict_for_full_chain(self):
        return self._calculate_tau_dict(minimum_index=0)

    @property
    def tau_dict(self):
        """Calculate a dictionary of tau (ACT) for every parameter"""
        return self._calculate_tau_dict(self.minimum_index)

    def _calculate_tau_dict(self, minimum_index):
        """Calculate a dictionary of tau (ACT) for every parameter"""
        logger.debug(f"Calculating tau_dict {self}")

        # If there are too few samples to calculate tau
        if (self.position - minimum_index) < 2 * self.autocorr_c:
            return {key: np.inf for key in self.parameter_keys}

        # Choose minimimum index for the ACT calculation
        last_tau = self.tau_last
        if self.tau_window is not None and last_tau < np.inf:
            minimum_index_for_act = max(
                minimum_index, int(self.position - self.tau_window * last_tau)
            )
        else:
            minimum_index_for_act = minimum_index

        # Calculate a dictionary of tau's for each parameter
        taus = {}
        for key in self.parameter_keys:
            if self.fixed_tau is None:
                x = self.get_1d_array(key)[minimum_index_for_act:]
                tau = calculate_tau(x, self.autocorr_c)
                taux = round(tau, 1)
            else:
                taux = self.fixed_tau
            taus[key] = max(taux, self.min_tau)

        # Cache the last tau dictionary for future use
        self.last_full_tau_dict = taus

        return taus

    @property
    def thin(self):
        if np.isfinite(self.tau):
            return np.max([1, int(self.thin_by_nact * self.tau)])
        else:
            return 1

    @property
    def nsamples(self):
        nuseable_steps = self.position - self.minimum_index
        n_independent_samples = nuseable_steps / self.tau
        nsamples = int(n_independent_samples / self.thin_by_nact)
        if nuseable_steps >= nsamples:
            return nsamples
        else:
            return 0

    @property
    def nsamples_last(self):
        nuseable_steps = self.position - self.minimum_index
        return int(nuseable_steps / (self.thin_by_nact * self.tau_last))

    @property
    def samples(self):
        samples = self._chain_array[self.minimum_index : self.position : self.thin]
        return pd.DataFrame(samples, columns=self.keys)

    def plot(self, outdir=".", label="label", priors=None, all_samples=None):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            nrows=self.ndim + 3, ncols=2, figsize=(8, 9 + 3 * (self.ndim))
        )
        scatter_kwargs = dict(
            lw=0,
            marker="o",
        )
        K = 1000

        nburn = self.minimum_index
        plot_setups = zip(
            [0, nburn, nburn],
            [nburn, self.position, self.position],
            [1, 1, self.thin],  # Thin-by factor
            ["tab:red", "tab:grey", "tab:blue"],  # Color
            [0.5, 0.05, 0.5],  # Alpha
            [1, 1, 1],  # Marker size
        )

        position_indexes = np.arange(self.position + 1)

        # Plot the traceplots
        for (start, stop, thin, color, alpha, ms) in plot_setups:
            for ax, key in zip(axes[:, 0], self.keys):
                xx = position_indexes[start:stop:thin] / K
                yy = self.get_1d_array(key)[start:stop:thin]

                # Downsample plots to max_pts: avoid memory issues
                max_pts = 10000
                while len(xx) > max_pts:
                    xx = xx[::2]
                    yy = yy[::2]

                ax.plot(
                    xx,
                    yy,
                    color=color,
                    alpha=alpha,
                    ms=ms,
                    **scatter_kwargs,
                )
                ax.set_ylabel(self._get_plot_label_by_key(key, priors))
                if key not in [LOGLKEY, LOGPKEY]:
                    msg = r"$\tau=$" + f"{self.last_full_tau_dict[key]:0.1f}"
                    ax.set_title(msg)

        # Plot the histograms
        for ax, key in zip(axes[:, 1], self.keys):
            if all_samples is not None:
                yy_all = all_samples[key]
                if np.any(np.isinf(yy_all)):
                    logger.warning(
                        f"Could not plot histogram for parameter {key} due to infinite values"
                    )
                else:
                    ax.hist(yy_all, bins=50, alpha=0.6, density=True, color="k")
            yy = self.get_1d_array(key)[nburn : self.position : self.thin]
            if np.any(np.isinf(yy)):
                logger.warning(
                    f"Could not plot histogram for parameter {key} due to infinite values"
                )
            else:
                ax.hist(yy, bins=50, alpha=0.8, density=True)
                ax.set_xlabel(self._get_plot_label_by_key(key, priors))

        # Add x-axes labels to the traceplots
        axes[-1, 0].set_xlabel(r"Iteration $[\times 10^{3}]$")

        # Plot the calculated ACT
        ax = axes[-1, 0]
        tausit = np.array(list(self.max_tau_dict.keys()) + [self.position]) / K
        taus = list(self.max_tau_dict.values()) + [self.tau_last]
        ax.plot(tausit, taus, color="C3")
        ax.set(ylabel=r"Maximum $\tau$")

        axes[-1, 1].set_axis_off()

        filename = "{}/{}_checkpoint_trace.png".format(outdir, label)
        msg = [
            r"Maximum $\tau$" + f"={self.tau:0.1f} ",
            r"$n_{\rm samples}=$" + f"{self.nsamples} ",
        ]
        if self.thin_by_nact != 1:
            msg += [
                r"$n_{\rm samples}^{\rm eff}=$"
                + f"{int(self.nsamples * self.thin_by_nact)} "
            ]
        fig.suptitle(
            "| ".join(msg),
            y=1,
        )
        fig.tight_layout()
        fig.savefig(filename, dpi=200)
        plt.close(fig)

    @staticmethod
    def _get_plot_label_by_key(key, priors=None):
        if priors is not None and key in priors:
            return priors[key].latex_label
        elif key == LOGLKEY:
            return LOGLLATEXKEY
        elif key == LOGPKEY:
            return LOGPLATEXKEY
        else:
            return key


class Sample(object):
    def __init__(self, sample_dict):
        """A single sample

        Parameters
        ----------
        sample_dict: dict
            A dictionary of the sample
        """

        self.sample_dict = sample_dict
        self.keys = list(sample_dict.keys())
        self.parameter_keys = [k for k in self.keys if k not in [LOGPKEY, LOGLKEY]]
        self.ndim = len(self.parameter_keys)

    def __getitem__(self, key):
        return self.sample_dict[key]

    def __setitem__(self, key, value):
        self.sample_dict[key] = value
        if key not in self.keys:
            self.keys = list(self.sample_dict.keys())

    @property
    def list(self):
        return list(self.sample_dict.values())

    def __repr__(self):
        return str(self.sample_dict)

    @property
    def parameter_only_dict(self):
        return {key: self.sample_dict[key] for key in self.parameter_keys}

    @property
    def dict(self):
        return {key: self.sample_dict[key] for key in self.keys}

    def as_dict(self, keys=None):
        sdict = self.dict
        if keys is None:
            return sdict
        else:
            return {key: sdict[key] for key in keys}

    def __eq__(self, other_sample):
        return self.list == other_sample.list

    def copy(self):
        return Sample(self.sample_dict.copy())


def calculate_tau(x, autocorr_c=5):
    import emcee

    if version.parse(emcee.__version__) < version.parse("3"):
        raise SamplerError("bilby-mcmc requires emcee > 3.0 for autocorr analysis")

    if np.all(np.diff(x) == 0):
        return np.inf
    try:
        # Hard code tol=1: we perform this check internally
        tau = emcee.autocorr.integrated_time(x, c=autocorr_c, tol=1)[0]
        if np.isnan(tau):
            tau = np.inf
        return tau
    except emcee.autocorr.AutocorrError:
        return np.inf
