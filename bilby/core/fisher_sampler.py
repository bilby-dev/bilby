import datetime

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import tqdm

from .fisher_matrix import FisherMatrixPosteriorEstimator
from .sampler.base_sampler import Sampler, signal_wrapper
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
        rejection_sampling=True,
        nsamples=1000,
        minimization_method="Nelder-Mead",
        n_prior_samples=100,
        fd_eps=1e-6,
        mirror_diagnostic_plot=False,
        cov_scaling=1,
        use_injection_for_maxL=True,
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
            List of file names.
        list
            List of directory names. Will always be empty for dynesty.
        """
        raise NotImplementedError()

    @signal_wrapper
    def run_sampler(self):

        self.start_time = datetime.datetime.now()

        fisher_mpe = FisherMatrixPosteriorEstimator(
            likelihood=self.likelihood,
            priors=self.priors,
            minimization_method=self.kwargs["minimization_method"],
            n_prior_samples=self.kwargs["n_prior_samples"],
            fd_eps=self.kwargs["fd_eps"],
        )

        if self.injection_parameters and "use_injection_for_maxL" in self.kwargs:
            sample = {key: self.injection_parameters[key] for key in fisher_mpe.parameter_names}
        else:
            sample = None
        maxL_sample_dict = fisher_mpe.get_maximum_likelihood_sample(sample)
        maxL_sample_array = np.array(list(maxL_sample_dict.values()))
        iFIM = fisher_mpe.calculate_iFIM(maxL_sample_dict)
        cov = self.kwargs["cov_scaling"] * iFIM

        msg = "Generation-distribution: " + "| ".join(
            [f"{key}: {val:0.5f} +/- {np.sqrt(var):.5f}" for ((key, val), var)
             in zip(maxL_sample_dict.items(), np.diag(cov))]
        )
        logger.info(msg)

        g_samples, g_logl, g_logpi = self._draw_samples_from_generating_distribution(
            maxL_sample_array, cov, fisher_mpe)

        if self.use_ratio:
            g_logl -= self.likelihood.noise_log_likelihood()

        if self.kwargs["rejection_sampling"]:
            samples, logl = self._rejection_sample(g_samples, g_logl, g_logpi, maxL_sample_array, cov)
        else:
            samples = g_samples
            logl = g_logl

        end_time = datetime.datetime.now()
        self.sampling_time = end_time - self.start_time

        self._generate_result(samples, logl)

        return self.result

    def create_rejection_sample_diagnostic(self, samples, raw_samples, maxL, weights):
        import corner
        import matplotlib.pyplot as plt
        import matplotlib.lines as mpllines

        kwargs = dict(
            bins=50,
            smooth=0.7,
            max_n_ticks=5,
            truths=maxL,
            truth_color="C3",
        )

        kwargs["labels"] = [
            label.replace("_", " ") for label in self.search_parameter_keys
        ]

        # Create the data array to plot and pass everything to corner
        xs = samples[self.search_parameter_keys].values
        rxs = raw_samples[self.search_parameter_keys].values

        g_color = "k"
        g_ls = "--"
        f_color = "C0"
        f_ls = "-"

        if len(self.search_parameter_keys) > 1:
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
                quantiles=[0.16, 0.84],
                levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
                **kwargs
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
                    quantiles=[0.16, 0.84],
                    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
                    **kwargs,
                )
                lines.append(mpllines.Line2D([0], [0], color=f_color, linestyle=f_ls))
            else:
                logger.info("Too few samples to add to the diagnostic plot")

            axes = fig.get_axes()
            ndim = int(np.sqrt(len(axes)))
            axes[0].legend(lines, ["$g(x)$", "$f(x)$"])

            axes = np.array(axes).reshape((ndim, ndim))

            base_alpha = 0.1
            alphas = base_alpha + (1 - base_alpha) * weights
            for ii in range(1, ndim):
                for jj in range(0, ndim - 1):
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
                        edgecolor='none',
                        vmin=-5,
                        vmax=0
                    )
            cbar = fig.colorbar(sc, ax=axes[0, -1])
            cbar.set_label("Log-weight")

        else:
            fig, ax = plt.subplots()
            ax.hist(
                xs,
                bins=kwargs["bins"],
                color=f_color,
                histtype="step",
                ls=f_ls,
                density=True,
            )
            ax.hist(
                rxs,
                bins=kwargs["bins"],
                color=g_color,
                histtype="step",
                ls=g_ls,
                density=True,
            )
            ax.set_xlabel(kwargs["labels"][0])

        filename = "{}/{}_rejection_sample.png".format(self.outdir, self.label)
        logger.debug("Saving rejection-sample diagnopstic plot to {}".format(filename))
        safe_save_figure(fig=fig, filename=filename, dpi=400)
        plt.close(fig)

        return fig

    def _generate_result(self, samples, log_likelihood_evaluations):
        """
        Extract the information we need from the output.

        Parameters
        ==========
        out: dynesty.result.Result
            The dynesty output.
        """

        self.result.samples = samples
        self.result.log_likelihood_evaluations = log_likelihood_evaluations

        self.result.meta_data["run_statistics"] = dict(
            nlikelihood=None,
            neffsamples=None,
            sampling_time_s=self.sampling_time.total_seconds(),
        )

    def _draw_samples_from_generating_distribution(self, sample_array, cov, fisher_mpe):
        samples_array = random.rng.multivariate_normal(sample_array, cov, self.kwargs["nsamples"])
        samples = pd.DataFrame(samples_array, columns=fisher_mpe.parameter_names)

        logpi = []
        logl = []
        logger.info("Calculating the likelihood and priors")
        outside_prior_count = 0
        for _, rs in tqdm.tqdm(samples.iterrows(), total=len(samples)):
            logpi.append(self.priors.ln_prob(rs.to_dict(), axis=0))
            if np.isinf(logpi[-1]):
                outside_prior_count += 1
                logl.append(-np.inf)
            else:
                logl.append(fisher_mpe.log_likelihood(rs.to_dict()))

        if outside_prior_count < len(samples):
            logger.info(
                f"Discarding {100 * outside_prior_count / len(samples):0.3f}% of samples that"
                " fall outside the prior"
            )
        else:
            raise ValueError("Sampling has failed: no viable samples left")

        logpi = np.real(np.array(logpi))
        logl = np.array(logl)
        return samples, logl, logpi

    def _rejection_sample(self, g_samples, g_logl, g_logpi, mean, cov):
        logger.info(f"Rejection sampling the posterior from {len(g_samples)} samples")

        g_logl_norm = multivariate_normal.logpdf(
            g_samples, mean=mean, cov=cov
        )

        ln_weights = g_logl + g_logpi - g_logl_norm

        # Remove impossible samples
        idxs = ~np.isinf(ln_weights)
        ln_weights = ln_weights[idxs]
        g_samples = g_samples[idxs]
        g_logl = g_logl[idxs]
        g_logpi = g_logpi[idxs]

        # Scale
        ln_weights -= np.max(ln_weights)

        # Sort by weight
        idxs = np.argsort(ln_weights)
        g_samples = g_samples.iloc[idxs]
        g_logl = g_logl[idxs]
        g_logpi = g_logpi[idxs]
        ln_weights = ln_weights[idxs]

        weights = np.exp(ln_weights)

        samples, idxs = rejection_sample(g_samples, weights, return_idxs=True)
        logl = g_logl[idxs]

        nsamples = len(samples)

        if self.plot:
            self.create_rejection_sample_diagnostic(samples, g_samples, mean, weights)

        if nsamples == 1:
            raise ValueError(
                "Rejection sampling has produced a single sample and therefore failed."
            )

        efficiency = 100 * nsamples / len(g_samples)
        ess = int(np.floor(np.exp(kish_log_effective_sample_size(ln_weights))))
        logger.info(
            f"Rejection sampling Fisher posterior produced {nsamples} samples"
            f" with an efficiency of {efficiency:0.3f}% and effective sample"
            f" size {ess}"
        )

        return samples, logl
