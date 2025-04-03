import datetime

from ..fisher import FisherMatrixPosteriorEstimator
from .base_sampler import Sampler, signal_wrapper


class Fisher(Sampler):
    """
    TBD

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
    use_ratio: bool, optional
        Switch to set whether or not you want to use the log-likelihood ratio
        or just the log-likelihood

    """

    sampler_name = "fisher"
    sampling_seed_key = "seed"

    def __init__(
        self,
        likelihood,
        priors,
        outdir="outdir",
        label="label",
        nsamples=1000,
        minimization_method="Nelder-Mead",
        n_prior_samples=100,
        fd_eps=1e-6,
        **kwargs,
    ):
        super(Fisher, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            **kwargs,
        )
        self.nsamples = nsamples
        self.minimization_method = minimization_method
        self.n_prior_samples = n_prior_samples
        self.fd_eps = fd_eps

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

    @property
    def sampler_init(self):
        raise NotImplementedError()

    @signal_wrapper
    def run_sampler(self):

        self.start_time = datetime.datetime.now()

        fisher = FisherMatrixPosteriorEstimator(
            likelihood=self.likelihood,
            priors=self.priors,
            minimization_method=self.minimization_method,
            n_prior_samples=self.n_prior_samples,
            fd_eps=self.fd_eps,
        )

        samples = fisher.sample_dataframe("maxL", self.nsamples)
        logl = [
            fisher.log_likelihood(sample.to_dict()) for _, sample in samples.iterrows()
        ]

        end_time = datetime.datetime.now()
        self.sampling_time = end_time - self.start_time

        self._generate_result(samples, logl)

        return self.result

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
        self.result.sampling_time = self.sampling_time.total_seconds()

        self.result.meta_data["run_statistics"] = dict(
            nlikelihood=None,
            neffsamples=None,
            sampling_time_s=self.sampling_time.total_seconds(),
        )
