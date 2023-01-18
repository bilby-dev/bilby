import datetime
import inspect
import sys

import bilby
from bilby.bilby_mcmc import Bilby_MCMC

from ..prior import DeltaFunction, PriorDict
from ..utils import command_line_args, env_package_list, loaded_modules_dict, logger
from . import proposal
from .base_sampler import Sampler, SamplingMarginalisedParameterError
from .cpnest import Cpnest
from .dnest4 import DNest4
from .dynamic_dynesty import DynamicDynesty
from .dynesty import Dynesty
from .emcee import Emcee
from .fake_sampler import FakeSampler
from .kombine import Kombine
from .nessai import Nessai
from .nestle import Nestle
from .polychord import PyPolyChord
from .ptemcee import Ptemcee
from .ptmcmc import PTMCMCSampler
from .pymc import Pymc
from .pymultinest import Pymultinest
from .ultranest import Ultranest
from .zeus import Zeus

IMPLEMENTED_SAMPLERS = {
    "bilby_mcmc": Bilby_MCMC,
    "cpnest": Cpnest,
    "dnest4": DNest4,
    "dynamic_dynesty": DynamicDynesty,
    "dynesty": Dynesty,
    "emcee": Emcee,
    "kombine": Kombine,
    "nessai": Nessai,
    "nestle": Nestle,
    "ptemcee": Ptemcee,
    "ptmcmcsampler": PTMCMCSampler,
    "pymc": Pymc,
    "pymultinest": Pymultinest,
    "pypolychord": PyPolyChord,
    "ultranest": Ultranest,
    "zeus": Zeus,
    "fake_sampler": FakeSampler,
}

if command_line_args.sampler_help:
    sampler = command_line_args.sampler_help
    if sampler in IMPLEMENTED_SAMPLERS:
        sampler_class = IMPLEMENTED_SAMPLERS[sampler]
        print(f'Help for sampler "{sampler}":')
        print(sampler_class.__doc__)
    else:
        if sampler == "None":
            print(
                "For help with a specific sampler, call sampler-help with "
                "the name of the sampler"
            )
        else:
            print(f"Requested sampler {sampler} not implemented")
        print(f"Available samplers = {IMPLEMENTED_SAMPLERS}")

    sys.exit()


def run_sampler(
    likelihood,
    priors=None,
    label="label",
    outdir="outdir",
    sampler="dynesty",
    use_ratio=None,
    injection_parameters=None,
    conversion_function=None,
    plot=False,
    default_priors_file=None,
    clean=None,
    meta_data=None,
    save=True,
    gzip=False,
    result_class=None,
    npool=1,
    **kwargs,
):
    """
    The primary interface to easy parameter estimation

    Parameters
    ==========
    likelihood: `bilby.Likelihood`
        A `Likelihood` instance
    priors: `bilby.PriorDict`
        A PriorDict/dictionary of the priors for each parameter - missing
        parameters will use default priors, if None, all priors will be default
    label: str
        Name for the run, used in output files
    outdir: str
        A string used in defining output files
    sampler: str, Sampler
        The name of the sampler to use - see
        `bilby.sampler.get_implemented_samplers()` for a list of available
        samplers.
        Alternatively a Sampler object can be passed
    use_ratio: bool (False)
        If True, use the likelihood's log_likelihood_ratio, rather than just
        the log_likelihood.
    injection_parameters: dict
        A dictionary of injection parameters used in creating the data (if
        using simulated data). Appended to the result object and saved.
    plot: bool
        If true, generate a corner plot and, if applicable diagnostic plots
    conversion_function: function, optional
        Function to apply to posterior to generate additional parameters.
    default_priors_file: str
        If given, a file containing the default priors; otherwise defaults to
        the bilby defaults for a binary black hole.
    clean: bool
        If given, override the command line interface `clean` option.
    meta_data: dict
        If given, adds the key-value pairs to the 'results' object before
        saving. For example, if `meta_data={dtype: 'signal'}`. Warning: in case
        of conflict with keys saved by bilby, the meta_data keys will be
        overwritten.
    save: bool, str
        If true, save the priors and results to disk.
        If hdf5, save as an hdf5 file instead of json.
        If pickle or pkl, save as an pickle file instead of json.
    gzip: bool
        If true, and save is true, gzip the saved results file.
    result_class: bilby.core.result.Result, or child of
        The result class to use. By default, `bilby.core.result.Result` is used,
        but objects which inherit from this class can be given providing
        additional methods.
    npool: int
        An integer specifying the available CPUs to create pool objects for
        parallelization.
    **kwargs:
        All kwargs are passed directly to the samplers `run` function

    Returns
    =======
    result: bilby.core.result.Result
        An object containing the results
    """

    logger.info(f"Running for label '{label}', output will be saved to '{outdir}'")

    if clean:
        command_line_args.clean = clean
    if command_line_args.clean:
        kwargs["resume"] = False

    from . import IMPLEMENTED_SAMPLERS

    if priors is None:
        priors = dict()

    _check_marginalized_parameters_not_sampled(likelihood, priors)

    if type(priors) == dict:
        priors = PriorDict(priors)
    elif isinstance(priors, PriorDict):
        pass
    else:
        raise ValueError("Input priors not understood should be dict or PriorDict")

    priors.fill_priors(likelihood, default_priors_file=default_priors_file)

    # Generate the meta-data if not given and append the likelihood meta_data
    if meta_data is None:
        meta_data = dict()
    likelihood.label = label
    likelihood.outdir = outdir
    meta_data["likelihood"] = likelihood.meta_data
    meta_data["loaded_modules"] = loaded_modules_dict()
    meta_data["environment_packages"] = env_package_list(as_dataframe=True)

    if command_line_args.bilby_zero_likelihood_mode:
        from bilby.core.likelihood import ZeroLikelihood

        likelihood = ZeroLikelihood(likelihood)

    if isinstance(sampler, Sampler):
        pass
    elif isinstance(sampler, str):
        if sampler.lower() in IMPLEMENTED_SAMPLERS:
            sampler_class = IMPLEMENTED_SAMPLERS[sampler.lower()]
            sampler = sampler_class(
                likelihood,
                priors=priors,
                outdir=outdir,
                label=label,
                injection_parameters=injection_parameters,
                meta_data=meta_data,
                use_ratio=use_ratio,
                plot=plot,
                result_class=result_class,
                npool=npool,
                **kwargs,
            )
        else:
            print(IMPLEMENTED_SAMPLERS)
            raise ValueError(f"Sampler {sampler} not yet implemented")
    elif inspect.isclass(sampler):
        sampler = sampler.__init__(
            likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=use_ratio,
            plot=plot,
            injection_parameters=injection_parameters,
            meta_data=meta_data,
            npool=npool,
            **kwargs,
        )
    else:
        raise ValueError(
            "Provided sampler should be a Sampler object or name of a known "
            f"sampler: {', '.join(IMPLEMENTED_SAMPLERS.keys())}."
        )

    if sampler.cached_result:
        logger.warning("Using cached result")
        result = sampler.cached_result
    else:
        # Run the sampler
        start_time = datetime.datetime.now()
        if command_line_args.bilby_test_mode:
            result = sampler._run_test()
        else:
            result = sampler.run_sampler()
        end_time = datetime.datetime.now()

        # Some samplers calculate the sampling time internally
        if result.sampling_time is None:
            result.sampling_time = end_time - start_time
        elif isinstance(result.sampling_time, (float, int)):
            result.sampling_time = datetime.timedelta(result.sampling_time)

        logger.info(f"Sampling time: {result.sampling_time}")
        # Convert sampling time into seconds
        result.sampling_time = result.sampling_time.total_seconds()

        if sampler.use_ratio:
            result.log_noise_evidence = likelihood.noise_log_likelihood()
            result.log_bayes_factor = result.log_evidence
            result.log_evidence = result.log_bayes_factor + result.log_noise_evidence
        else:
            result.log_noise_evidence = likelihood.noise_log_likelihood()
            result.log_bayes_factor = result.log_evidence - result.log_noise_evidence

        if None not in [result.injection_parameters, conversion_function]:
            result.injection_parameters = conversion_function(
                result.injection_parameters
            )

        # Initial save of the sampler in case of failure in samples_to_posterior
        if save:
            result.save_to_file(extension=save, gzip=gzip, outdir=outdir)

    if None not in [result.injection_parameters, conversion_function]:
        result.injection_parameters = conversion_function(result.injection_parameters)

    # Check if the posterior has already been created
    if getattr(result, "_posterior", None) is None:
        result.samples_to_posterior(
            likelihood=likelihood,
            priors=result.priors,
            conversion_function=conversion_function,
            npool=npool,
        )

    if save:
        # The overwrite here ensures we overwrite the initially stored data
        result.save_to_file(overwrite=True, extension=save, gzip=gzip, outdir=outdir)

    if plot:
        result.plot_corner()
    logger.info(f"Summary of results:\n{result}")
    return result


def _check_marginalized_parameters_not_sampled(likelihood, priors):
    for key in likelihood.marginalized_parameters:
        if key in priors:
            if not isinstance(priors[key], (float, DeltaFunction)):
                raise SamplingMarginalisedParameterError(
                    f"Likelihood is {key} marginalized but you are trying to sample in {key}. "
                )
