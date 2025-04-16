import datetime

import numpy as np
from scipy.stats import multivariate_normal

from .fisher_matrix import FisherMatrixPosteriorEstimator
from .sampler.base_sampler import Sampler, signal_wrapper
from .result import rejection_sample
from .utils import logger, kish_log_effective_sample_size, safe_save_figure


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

        raw_samples = fisher_mpe.sample_dataframe("maxL", self.kwargs["nsamples"])
        raw_logl = np.array(
            [
                fisher_mpe.log_likelihood(sample.to_dict())
                for _, sample in raw_samples.iterrows()
            ]
        )
        raw_logpi = self.priors.ln_prob(raw_samples, axis=0)

        if self.use_ratio:
            raw_logl -= self.likelihood.noise_log_likelihood()

        if self.kwargs["rejection_sampling"]:
            logger.info(f"Rejection sampling the posterior from {self.kwargs['nsamples']} generated samples")

            logl_norm = multivariate_normal.logpdf(
                raw_samples, mean=fisher_mpe.mean, cov=fisher_mpe.iFIM
            )

            ln_weights = raw_logl + raw_logpi - logl_norm
            ln_weights -= np.mean(ln_weights)
            weights = np.exp(ln_weights)

            samples, idxs = rejection_sample(raw_samples, weights, return_idxs=True)
            logl = raw_logl[idxs]

            nsamples = len(samples)
            efficiency = 100 * nsamples / len(raw_samples)
            ess = int(np.floor(np.exp(kish_log_effective_sample_size(ln_weights))))
            logger.info(
                f"Rejection sampling Fisher posterior produced {nsamples} samples"
                f" with an efficiency of {efficiency}% and effective sample"
                f" size {ess}"
            )
            if self.plot:
                self.create_rejection_sample_diagnostic(samples, raw_samples, fisher_mpe.mean)
        else:
            samples = raw_samples
            logl = raw_logl

        end_time = datetime.datetime.now()
        self.sampling_time = end_time - self.start_time

        self._generate_result(samples, logl)

        return self.result

    def create_rejection_sample_diagnostic(self, samples, raw_samples, maxL):
        import corner
        import matplotlib.pyplot as plt
        import matplotlib.lines as mpllines

        kwargs = dict(
            bins=50,
            smooth=0.9,
            max_n_ticks=3,
            truths=maxL,
            truth_color="C3",
        )

        kwargs["labels"] = [
            label.replace("_", " ") for label in self.search_parameter_keys
        ]

        # Create the data array to plot and pass everything to corner
        xs = samples[self.search_parameter_keys].values
        rxs = raw_samples[self.search_parameter_keys].values
        if len(self.search_parameter_keys) > 1:
            lines = []
            ls = "-"
            c = "C0"
            fig = corner.corner(
                rxs,
                color=c,
                contour_kwargs={"linestyles": ls, "alpha": 0.8},
                hist_kwargs={"density": True, "ls": ls, "alpha": 0.8},
                data_kwargs={"alpha": 1},
                no_fill_contours=True,
                alpha=0.8,
                plot_density=False,
                plot_datapoints=True,
                fill_contours=False,
                quantiles=[0.16, 0.84],
                levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
                **kwargs
            )
            lines.append(mpllines.Line2D([0], [0], color=c, linestyle=ls))

            ls = "--"
            c = "C1"
            fig = corner.corner(
                xs,
                color=c,
                contour_kwargs={"linestyles": ls, "alpha": 0.8},
                contourf_kwargs={"alpha": 0.8},
                hist_kwargs={"density": True, "ls": ls, "alpha": 0.8},
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
            lines.append(mpllines.Line2D([0], [0], color=c, linestyle=ls))

            axes = fig.get_axes()
            ndim = int(np.sqrt(len(axes)))
            axes[ndim - 1].legend(lines, ["$g(x)$", "$f(x)$"])

        else:
            fig, ax = plt.subplots()
            ax.hist(
                xs,
                bins=kwargs["bins"],
                color="C0",
                histtype="step",
                ls="-",
                density=True,
            )
            ax.hist(
                rxs,
                bins=kwargs["bins"],
                color="C1",
                histtype="step",
                ls="--",
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
