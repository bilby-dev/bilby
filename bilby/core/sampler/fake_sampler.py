import numpy as np

from ..result import read_in_result
from .base_sampler import Sampler


class FakeSampler(Sampler):
    """
    A "fake" sampler that evaluates the likelihood at a list of
    configurations read from a posterior data file.

    See base class for parameters. Added parameters are described below.

    Parameters
    ==========
    sample_file: str
        A string pointing to the posterior data file to be loaded.
    """

    sampler_name = "fake_sampler"

    default_kwargs = dict(
        verbose=True, logl_args=None, logl_kwargs=None, print_progress=True
    )

    def __init__(
        self,
        likelihood,
        priors,
        sample_file,
        outdir="outdir",
        label="label",
        use_ratio=False,
        plot=False,
        injection_parameters=None,
        meta_data=None,
        result_class=None,
        **kwargs
    ):
        super(FakeSampler, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=False,
            plot=False,
            skip_import_verification=True,
            injection_parameters=None,
            meta_data=None,
            result_class=None,
            **kwargs
        )
        self._read_parameter_list_from_file(sample_file)
        self.result.outdir = outdir
        self.result.label = label

    def _read_parameter_list_from_file(self, sample_file):
        """Read a list of sampling parameters from file.

        The sample_file should be in bilby posterior HDF5 format.
        """
        self.result = read_in_result(filename=sample_file)

    def run_sampler(self):
        """Compute the likelihood for the list of parameter space points."""
        self.sampler = "fake_sampler"

        # Flushes the output to force a line break
        if self.kwargs["verbose"]:
            print("")

        likelihood_ratios = []
        posterior = self.result.posterior

        for i in np.arange(posterior.shape[0]):
            sample = posterior.iloc[i]

            self.likelihood.parameters = sample.to_dict()
            logl = self.likelihood.log_likelihood_ratio()
            sample.log_likelihood = logl
            likelihood_ratios.append(logl)

            if self.kwargs["verbose"]:
                print(
                    self.likelihood.parameters["log_likelihood"],
                    likelihood_ratios[-1],
                    self.likelihood.parameters["log_likelihood"]
                    - likelihood_ratios[-1],
                )

        self.result.log_likelihood_evaluations = np.array(likelihood_ratios)

        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan

        return self.result
