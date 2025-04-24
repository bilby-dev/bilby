import datetime

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import tqdm

from .fisher_matrix import FisherMatrixPosteriorEstimator
from .sampler.base_sampler import Sampler, signal_wrapper, SamplerError
from .result import rejection_sample
from .utils import logger, kish_log_effective_sample_size, safe_save_figure, random


class Fisher(Sampler):
    """
    A sampler class that estimates the maximum likelihood using scipy, then draws
    posterior samples using the Fisher information matrix.

    Parameters
    ==========
    likelihood: likelihood.Likelihood
        A  object with a log_l method
    priors: bilby.core.prior.PriorDict, dict
        Priors to be used in the search.
        This has attributes for each parameter to be sampled.
    outdir: str, optional
        Name of the output directory
    label: str, optional
        Naming scheme of the output files
    rejection_sampling: bool
        If true, utilise rejection sampling to reweight from the Fisher matrix
        Gaussian posterior approximation to the true posterior. Note: if true,
        the number of samples is determined by the efficiency which can be
        low if the maximum likelihood is not found or the posterior is highly
        non-Gaussian.
    nsamples: int
        The target number of samples to draw in the posterior
    minimization_method: str (Nelder-Mead)
        The method to use in scipy.optimize.minimize
    fd_eps: float
        A parameter to control the size of perturbation used when finite
        differencing the likelihood
    n_prior_samples: int
        The number of prior samples to draw and use to attempt estimatation
        of the maximum likelihood sample.
    """

    sampler_name = "fisher"
    sampling_seed_key = "seed"
    default_kwargs = dict(
        resample="importance",
        target_nsamples=10000,
        batch_nsamples=1000,
        prior_nsamples=100,
        minimization_method="Nelder-Mead",
        fd_eps=1e-6,
        plot_diagnostic=False,
        mirror_diagnostic_plot=False,
        cov_scaling=1,
        use_injection_for_maxL=True,
        fail_on_error=False
    )

    def __init__(
        self,
        likelihood,
        priors,
        outdir="outdir",
        label="label",
        use_ratio=False,
        plot=False,
        exit_code=77,
        skip_import_verification=True,
        **kwargs,
    ):
        super(Fisher, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=use_ratio,
            plot=plot,
            skip_import_verification=skip_import_verification,
            exit_code=exit_code,
            **kwargs,
        )

    @classmethod
    def get_expected_outputs(cls, outdir=None, label=None):
        """Get lists of the expected outputs directories and files.

        These are used by :code:`bilby_pipe` when transferring files via HTCondor.

        Parameters
        ----------
        outdir : str
            The output directory.
        label : str
            The label for the run.

        Returns
        -------
        list
            List of file names: empty (resuming not yet implemented)
        list
            List of directory names. Will always be empty for fisher.
        """
        return [], []

        raise NotImplementedError()

    @signal_wrapper
    def run_sampler(self):

        self.start_time = datetime.datetime.now()

        fisher_mpe = FisherMatrixPosteriorEstimator(
            likelihood=self.likelihood,
            priors=self.priors,
            minimization_method=self.kwargs["minimization_method"],
            n_prior_samples=self.kwargs["prior_nsamples"],
            fd_eps=self.kwargs["fd_eps"],
        )

        if self.injection_parameters and "use_injection_for_maxL" in self.kwargs:
            sample = {
                key: self.injection_parameters[key]
                for key in fisher_mpe.parameter_names
            }
        else:
            sample = None
        maxL_sample_dict = fisher_mpe.get_maximum_likelihood_sample(sample)
        mean = np.array(list(maxL_sample_dict.values()))
        iFIM = fisher_mpe.calculate_iFIM(maxL_sample_dict)
        cov = self.kwargs["cov_scaling"] * iFIM

        msg = "Generation-distribution:\n " + "\n ".join(
            [
                f"{key}: {val:0.5f} +/- {np.sqrt(var):.5f}"
                for ((key, val), var) in zip(maxL_sample_dict.items(), np.diag(cov))
            ]
        )
        logger.info(msg)

        nsamples = 0
        target_nsamples = self.kwargs["target_nsamples"]
        batch_nsamples = self.kwargs["batch_nsamples"]
        all_g_samples = []
        all_samples = []
        all_logl = []
        all_weights = []
        resample = self.kwargs["resample"]
        logger.info(
            f"Starting sampling in batches of {batch_nsamples} to produce {target_nsamples} samples"
        )
        pbar = tqdm.tqdm(
            total=target_nsamples, desc=f"{resample.capitalize()} sampling"
        )
        while nsamples < target_nsamples:
            g_samples, g_logl, g_logpi = (
                self._draw_samples_from_generating_distribution(
                    mean, cov, fisher_mpe, batch_nsamples
                )
            )

            _methods = dict(
                rejection=self._rejection_sample, importance=self._importance_sample
            )

            if resample in _methods:
                weights = self._calculate_weights(g_samples, g_logl, g_logpi, mean, cov)
                samples, logl = _methods[resample](g_samples, g_logl, weights)
                efficiency = 100 * len(samples) / len(g_samples)
            else:
                logger.info("No resampling applied")
                samples = g_samples
                logl = g_logl
                weights = np.ones_like(logl)

            nsamples += len(samples)
            pbar.set_postfix(
                {
                    "eff": f"{efficiency:.3f}%",
                }
            )
            pbar.update(len(samples))
            all_g_samples.append(g_samples)
            all_samples.append(samples)
            all_logl.append(logl)
            all_weights.append(weights)

        pbar.close()

        g_samples = pd.concat(all_g_samples)
        samples = pd.concat(all_samples)
        logl = np.concat(all_logl)
        weights = np.concat(all_weights)
        efficiency = 100 * len(samples) / len(g_samples)

        logger.info(f"Finished sampling: total efficiency is {efficiency:0.3f}%")

        if self.kwargs["plot_diagnostic"]:
            self.create_resample_diagnostic(
                samples, g_samples, mean, weights, method=self.kwargs["resample"]
            )

        end_time = datetime.datetime.now()
        self.sampling_time = end_time - self.start_time

        if self.use_ratio:
            logl -= self.likelihood.noise_log_likelihood()

        self._generate_result(
            samples, logl, efficiency=efficiency, nlikelihood=len(g_samples)
        )

        return self.result

    def create_resample_diagnostic(self, samples, raw_samples, mean, weights, method):
        import corner
        import matplotlib.pyplot as plt
        import matplotlib.lines as mpllines

        kwargs = dict(
            bins=50,
            smooth=0.7,
            max_n_ticks=5,
            truths=np.concat((mean, [1])),
            truth_color="C3",
        )

        kwargs["labels"] = [
            label.replace("_", " ") for label in self.search_parameter_keys
        ]
        kwargs["labels"].append("weights")

        # Create the data array to plot and pass everything to corner
        xs = samples[self.search_parameter_keys].values
        xs = np.concat((xs, np.random.uniform(0, 1, len(xs)).reshape(-1, 1)), axis=1)
        rxs = raw_samples[self.search_parameter_keys].values
        rxs = np.concat((rxs, weights.reshape((-1, 1))), axis=1)

        # Sort by weight (only for plotting)
        idxs = np.argsort(weights)
        rxs = rxs[idxs]
        weights = weights[idxs]

        g_color = "k"
        g_ls = "--"
        f_color = "C0"
        f_ls = "-"

        lines = []
        fig = corner.corner(
            rxs,
            color=g_color,
            contour_kwargs={"linestyles": g_ls, "alpha": 0.8},
            hist_kwargs={"density": True, "ls": g_ls, "alpha": 0.8},
            data_kwargs={"alpha": 1},
            no_fill_contours=True,
            alpha=0.8,
            plot_density=False,
            plot_datapoints=False,
            fill_contours=False,
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
            **kwargs,
        )
        lines.append(mpllines.Line2D([0], [0], color=g_color, linestyle=g_ls))

        if len(xs) > len(samples.keys()):
            fig = corner.corner(
                xs,
                color=f_color,
                contour_kwargs={"linestyles": f_ls, "alpha": 0.8},
                contourf_kwargs={"alpha": 0.8},
                hist_kwargs={"density": True, "ls": f_ls, "alpha": 0.8},
                no_fill_contours=True,
                fig=fig,
                alpha=0.1,
                plot_density=True,
                plot_datapoints=False,
                fill_contours=False,
                levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
                range=[1] * self.ndim + [(0, 1)],
                **kwargs,
            )

            # Remove the weights from the re-weighted samples
            axes = fig.get_axes()
            axes[-1].patches[-1].remove()
            for ii in range(2, self.ndim + 2):
                axes[-ii].collections[-1].remove()
                axes[-ii].collections[-1].remove()

            lines.append(mpllines.Line2D([0], [0], color=f_color, linestyle=f_ls))
        else:
            logger.info("Too few samples to add to the diagnostic plot")

        axes = fig.get_axes()
        axes[0].legend(lines, ["$g(x)$", "$f(x)$"])

        axes = np.array(axes).reshape((self.ndim + 1, self.ndim + 1))

        base_alpha = 0.1
        alphas = base_alpha + (1 - base_alpha) * weights
        for ii in range(1, self.ndim + 1):
            for jj in range(0, self.ndim):
                if ii <= jj:
                    continue
                if self.kwargs["mirror_diagnostic_plot"]:
                    ax = axes[jj, ii]
                    xsamples = rxs[:, ii]
                    ysamples = rxs[:, jj]
                else:
                    ax = axes[ii, jj]
                    xsamples = rxs[:, jj]
                    ysamples = rxs[:, ii]
                sc = ax.scatter(
                    xsamples,
                    ysamples,
                    c=np.log(weights),
                    alpha=alphas,
                    edgecolor="none",
                    vmin=-5,
                    vmax=0,
                )
        cbar = fig.colorbar(sc, ax=axes[0, -1])
        cbar.set_label("Log-weight")

        fig.suptitle(f"Resampling method: {method}")

        filename = f"{self.outdir}/{self.label}_resample_{method}.png"
        safe_save_figure(fig=fig, filename=filename, dpi=400)
        plt.close(fig)

        return fig

    def _generate_result(self, samples, log_likelihood_evaluations, **run_stats):
        """
        Extract the information we need from the output.

        Parameters
        ==========
        out: dynesty.result.Result
            The dynesty output.
        """

        self.result.samples = samples
        self.result.log_likelihood_evaluations = log_likelihood_evaluations

        run_stats["sampling_time_s"] = self.sampling_time.total_seconds()
        self.result.meta_data["run_statistics"] = run_stats

    def _draw_samples_from_generating_distribution(
        self, mean, cov, fisher_mpe, nsamples
    ):
        samples_array = random.rng.multivariate_normal(mean, cov, nsamples)
        samples = pd.DataFrame(samples_array, columns=fisher_mpe.parameter_names)

        logpi = []
        logl = []
        logger.debug("Calculating the likelihood and priors")

        logpi = self.priors.ln_prob(samples, axis=0)
        outside_prior_count = np.sum(np.isinf(logpi))
        if outside_prior_count == 0:
            logl = fisher_mpe.log_likelihood_from_array(samples.values.T)
        else:
            for ii, rs in tqdm.tqdm(samples.iterrows(), total=len(samples)):
                if np.isinf(logpi[ii]):
                    logl.append(-np.inf)
                else:
                    logl.append(fisher_mpe.log_likelihood(rs.to_dict()))

        if outside_prior_count == len(samples):
            msg = "Sampling has failed: no viable samples left"
            if self.kwargs["fail_on_error"]:
                raise SamplerError(msg)
            else:
                logger.info(msg)
        elif outside_prior_count > 0:
            logger.info(
                f"Discarding {100 * outside_prior_count / len(samples):0.3f}% of samples that"
                " fall outside the prior"
            )
        else:
            logger.debug("No samples outside the prior bounds")

        logpi = np.real(np.array(logpi))
        logl = np.array(logl)
        return samples, logl, logpi

    def _calculate_weights(self, g_samples, g_logl, g_logpi, mean, cov):
        g_logl_norm = multivariate_normal.logpdf(g_samples, mean=mean, cov=cov)

        ln_weights = g_logl + g_logpi - g_logl_norm

        # Remove impossible samples
        idxs = ~np.isinf(ln_weights)
        ln_weights_viable = ln_weights[idxs]

        # Scale
        ln_weights -= np.max(ln_weights_viable)

        self.ess = int(np.floor(np.exp(kish_log_effective_sample_size(ln_weights_viable))))
        logger.debug(f"Calculated weights have an effective sample size {self.ess}")

        weights = np.exp(ln_weights)

        return weights

    def _importance_sample(self, g_samples, g_logl, weights):
        logger.debug(f"Importance sampling the posterior from {len(g_samples)} samples")

        normalized_weights = weights / np.sum(weights)
        idxs = np.random.choice(len(g_samples), size=self.ess, p=normalized_weights)
        samples = g_samples.iloc[idxs]
        logl = g_logl[idxs]

        if self.ess < self.ndim:
            msg = "Effective sample size less than ndim: sampling has failed"
            if self.kwargs["fail_on_error"]:
                raise SamplerError(msg)
            else:
                logger.info(msg)

        return samples, logl

    def _rejection_sample(self, g_samples, g_logl, weights):
        logger.debug(f"Rejection sampling the posterior from {len(g_samples)} samples")

        samples, idxs = rejection_sample(g_samples, weights, return_idxs=True)
        logl = g_logl[idxs]

        nsamples = len(samples)

        if nsamples < self.ndim:
            msg = "Number of samples less than ndim: sampling has failed"
            if self.kwargs["fail_on_error"]:
                raise SamplerError(msg)
            else:
                logger.info(msg)

        return samples, logl
