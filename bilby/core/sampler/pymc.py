import numpy as np

from ...gw.likelihood import BasicGravitationalWaveTransient, GravitationalWaveTransient
from ..likelihood import (
    ExponentialLikelihood,
    GaussianLikelihood,
    PoissonLikelihood,
    StudentTLikelihood,
)
from ..prior import Cosine, DeltaFunction, MultivariateGaussian, PowerLaw, Sine
from ..utils import derivatives, infer_args_from_method
from .base_sampler import MCMCSampler


class Pymc(MCMCSampler):
    """bilby wrapper of the PyMC sampler (https://www.pymc.io/)

    All keyword arguments (i.e., the kwargs) passed to `run_sampler` will be
    propapated to `pymc.sample` where appropriate, see documentation for that
    class for further help. Under Other Parameters, we list commonly used
    kwargs and the bilby, or where appropriate, PyMC defaults.

    Parameters
    ==========
    draws: int, (1000)
        The number of sample draws from the posterior per chain.
    chains: int, (2)
        The number of independent MCMC chains to run.
    cores: int, (1)
        The number of CPU cores to use.
    tune: int, (500)
        The number of tuning (or burn-in) samples per chain.
    discard_tuned_samples: bool, True
        Set whether to automatically discard the tuning samples from the final
        chains.
    step: str, dict
        Provide a step method name, or dictionary of step method names keyed to
        particular variable names (these are case insensitive). If passing a
        dictionary of methods, the values keyed on particular variables can be
        lists of methods to form compound steps. If no method is provided for
        any particular variable then PyMC will automatically decide upon a
        default, with the first option being the NUTS sampler. The currently
        allowed methods are 'NUTS', 'HamiltonianMC', 'Metropolis',
        'BinaryMetropolis', 'BinaryGibbsMetropolis', 'Slice', and
        'CategoricalGibbsMetropolis'. Note: you cannot provide a PyMC step
        method function itself here as it is outside of the model context
        manager.
    step_kwargs: dict
        Options for steps methods other than NUTS. The dictionary is keyed on
        lowercase step method names with values being dictionaries of keywords
        for the given step method.

    """

    sampler_name = "pymc"
    default_kwargs = dict(
        draws=500,
        step=None,
        init="auto",
        n_init=200000,
        initvals=None,
        trace=None,
        chains=2,
        cores=1,
        tune=500,
        progressbar=True,
        model=None,
        random_seed=None,
        discard_tuned_samples=True,
        compute_convergence_checks=True,
        nuts_kwargs=None,
        step_kwargs=None,
    )

    default_nuts_kwargs = dict(
        target_accept=None,
        max_treedepth=None,
        step_scale=None,
        Emax=None,
        gamma=None,
        k=None,
        t0=None,
        adapt_step_size=None,
        early_max_treedepth=None,
        scaling=None,
        is_cov=None,
        potential=None,
    )

    default_kwargs.update(default_nuts_kwargs)

    sampling_seed_key = "random_seed"

    def __init__(
        self,
        likelihood,
        priors,
        outdir="outdir",
        label="label",
        use_ratio=False,
        plot=False,
        skip_import_verification=False,
        **kwargs,
    ):
        # add default step kwargs
        _, STEP_METHODS, _ = self._import_external_sampler()
        self.default_step_kwargs = {m.__name__.lower(): None for m in STEP_METHODS}
        self.default_kwargs.update(self.default_step_kwargs)

        super(Pymc, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=use_ratio,
            plot=plot,
            skip_import_verification=skip_import_verification,
            **kwargs,
        )
        self.draws = self._kwargs["draws"]
        self.chains = self._kwargs["chains"]

    @staticmethod
    def _import_external_sampler():
        import pymc
        from pymc import floatX
        from pymc.step_methods import STEP_METHODS

        return pymc, STEP_METHODS, floatX

    @staticmethod
    def _import_tensor():
        try:
            import pytensor as tensor  # noqa
            import pytensor.tensor as tt
            from pytensor.compile.ops import as_op  # noqa
        except ImportError:
            import aesara as tensor  # noqa
            import aesara.tensor as tt
            from aesara.compile.ops import as_op  # noqa

        return tensor, tt, as_op

    def _verify_parameters(self):
        """
        Change `_verify_parameters()` to just pass, i.e., don't try and
        evaluate the likelihood for PyMC.
        """
        pass

    def _verify_use_ratio(self):
        """
        Change `_verify_use_ratio() to just pass.
        """
        pass

    def setup_prior_mapping(self):
        """
        Set the mapping between predefined bilby priors and the equivalent
        PyMC distributions.
        """

        prior_map = {}
        self.prior_map = prior_map

        # predefined PyMC distributions
        prior_map["Gaussian"] = {
            "pymc": "Normal",
            "argmap": {"mu": "mu", "sigma": "sigma"},
        }
        prior_map["TruncatedGaussian"] = {
            "pymc": "TruncatedNormal",
            "argmap": {
                "mu": "mu",
                "sigma": "sigma",
                "minimum": "lower",
                "maximum": "upper",
            },
        }
        prior_map["HalfGaussian"] = {"pymc": "HalfNormal", "argmap": {"sigma": "sigma"}}
        prior_map["Uniform"] = {
            "pymc": "Uniform",
            "argmap": {"minimum": "lower", "maximum": "upper"},
        }
        prior_map["LogNormal"] = {
            "pymc": "Lognormal",
            "argmap": {"mu": "mu", "sigma": "sigma"},
        }
        prior_map["Exponential"] = {
            "pymc": "Exponential",
            "argmap": {"mu": "lam"},
            "argtransform": {"mu": lambda mu: 1.0 / mu},
        }
        prior_map["StudentT"] = {
            "pymc": "StudentT",
            "argmap": {"df": "nu", "mu": "mu", "scale": "sigma"},
        }
        prior_map["Beta"] = {
            "pymc": "Beta",
            "argmap": {"alpha": "alpha", "beta": "beta"},
        }
        prior_map["Logistic"] = {
            "pymc": "Logistic",
            "argmap": {"mu": "mu", "scale": "s"},
        }
        prior_map["Cauchy"] = {
            "pymc": "Cauchy",
            "argmap": {"alpha": "alpha", "beta": "beta"},
        }
        prior_map["Gamma"] = {
            "pymc": "Gamma",
            "argmap": {"k": "alpha", "theta": "beta"},
            "argtransform": {"theta": lambda theta: 1.0 / theta},
        }
        prior_map["ChiSquared"] = {"pymc": "ChiSquared", "argmap": {"nu": "nu"}}
        prior_map["Interped"] = {
            "pymc": "Interpolated",
            "argmap": {"xx": "x_points", "yy": "pdf_points"},
        }
        prior_map["Normal"] = prior_map["Gaussian"]
        prior_map["TruncatedNormal"] = prior_map["TruncatedGaussian"]
        prior_map["HalfNormal"] = prior_map["HalfGaussian"]
        prior_map["LogGaussian"] = prior_map["LogNormal"]
        prior_map["Lorentzian"] = prior_map["Cauchy"]
        prior_map["FromFile"] = prior_map["Interped"]

        # GW specific priors
        prior_map["UniformComovingVolume"] = prior_map["Interped"]

        # internally defined mappings for bilby priors
        prior_map["DeltaFunction"] = {"internal": self._deltafunction_prior}
        prior_map["Sine"] = {"internal": self._sine_prior}
        prior_map["Cosine"] = {"internal": self._cosine_prior}
        prior_map["PowerLaw"] = {"internal": self._powerlaw_prior}
        prior_map["LogUniform"] = {"internal": self._powerlaw_prior}
        prior_map["MultivariateGaussian"] = {
            "internal": self._multivariate_normal_prior
        }
        prior_map["MultivariateNormal"] = {"internal": self._multivariate_normal_prior}

    def _deltafunction_prior(self, key, **kwargs):
        """
        Map the bilby delta function prior to a single value for PyMC.
        """

        # check prior is a DeltaFunction
        if isinstance(self.priors[key], DeltaFunction):
            return self.priors[key].peak
        else:
            raise ValueError(f"Prior for '{key}' is not a DeltaFunction")

    def _sine_prior(self, key):
        """
        Map the bilby Sine prior to a PyMC style function
        """

        # check prior is a Sine
        pymc, _, floatX = self._import_external_sampler()
        _, tt, _ = self._import_tensor()
        if isinstance(self.priors[key], Sine):

            class PymcSine(pymc.Continuous):
                def __init__(self, lower=0.0, upper=np.pi):
                    if lower >= upper:
                        raise ValueError("Lower bound is above upper bound!")

                    # set the mode
                    self.lower = lower = tt.as_tensor_variable(floatX(lower))
                    self.upper = upper = tt.as_tensor_variable(floatX(upper))
                    self.norm = tt.cos(lower) - tt.cos(upper)
                    self.mean = (
                        tt.sin(upper)
                        + lower * tt.cos(lower)
                        - tt.sin(lower)
                        - upper * tt.cos(upper)
                    ) / self.norm

                    transform = pymc.distributions.transforms.interval(lower, upper)

                    super(PymcSine, self).__init__(transform=transform)

                def logp(self, value):
                    upper = self.upper
                    lower = self.lower
                    return pymc.distributions.dist_math.bound(
                        tt.log(tt.sin(value) / self.norm),
                        lower <= value,
                        value <= upper,
                    )

            return PymcSine(
                key, lower=self.priors[key].minimum, upper=self.priors[key].maximum
            )
        else:
            raise ValueError(f"Prior for '{key}' is not a Sine")

    def _cosine_prior(self, key):
        """
        Map the bilby Cosine prior to a PyMC style function
        """

        # check prior is a Cosine
        pymc, _, floatX = self._import_external_sampler()
        _, tt, _ = self._import_tensor()
        if isinstance(self.priors[key], Cosine):

            class PymcCosine(pymc.Continuous):
                def __init__(self, lower=-np.pi / 2.0, upper=np.pi / 2.0):
                    if lower >= upper:
                        raise ValueError("Lower bound is above upper bound!")

                    self.lower = lower = tt.as_tensor_variable(floatX(lower))
                    self.upper = upper = tt.as_tensor_variable(floatX(upper))
                    self.norm = tt.sin(upper) - tt.sin(lower)
                    self.mean = (
                        upper * tt.sin(upper)
                        + tt.cos(upper)
                        - lower * tt.sin(lower)
                        - tt.cos(lower)
                    ) / self.norm

                    transform = pymc.distributions.transforms.interval(lower, upper)

                    super(PymcCosine, self).__init__(transform=transform)

                def logp(self, value):
                    upper = self.upper
                    lower = self.lower
                    return pymc.distributions.dist_math.bound(
                        tt.log(tt.cos(value) / self.norm),
                        lower <= value,
                        value <= upper,
                    )

            return PymcCosine(
                key, lower=self.priors[key].minimum, upper=self.priors[key].maximum
            )
        else:
            raise ValueError(f"Prior for '{key}' is not a Cosine")

    def _powerlaw_prior(self, key):
        """
        Map the bilby PowerLaw prior to a PyMC style function
        """

        # check prior is a PowerLaw
        pymc, _, floatX = self._import_external_sampler()
        _, tt, _ = self._import_tensor()
        if isinstance(self.priors[key], PowerLaw):

            # check power law is set
            if not hasattr(self.priors[key], "alpha"):
                raise AttributeError("No 'alpha' attribute set for PowerLaw prior")

            if self.priors[key].alpha < -1.0:
                # use Pareto distribution
                palpha = -(1.0 + self.priors[key].alpha)

                return pymc.Bound(pymc.Pareto, upper=self.priors[key].minimum)(
                    key, alpha=palpha, m=self.priors[key].maximum
                )
            else:

                class PymcPowerLaw(pymc.Continuous):
                    def __init__(self, lower, upper, alpha, testval=1):
                        falpha = alpha
                        self.lower = lower = tt.as_tensor_variable(floatX(lower))
                        self.upper = upper = tt.as_tensor_variable(floatX(upper))
                        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))

                        if falpha == -1:
                            self.norm = 1.0 / (tt.log(self.upper / self.lower))
                        else:
                            beta = 1.0 + self.alpha
                            self.norm = 1.0 / (
                                beta
                                * (tt.pow(self.upper, beta) - tt.pow(self.lower, beta))
                            )

                        transform = pymc.distributions.transforms.interval(lower, upper)

                        super(PymcPowerLaw, self).__init__(
                            transform=transform, testval=testval
                        )

                    def logp(self, value):
                        upper = self.upper
                        lower = self.lower
                        alpha = self.alpha

                        return pymc.distributions.dist_math.bound(
                            alpha * tt.log(value) + tt.log(self.norm),
                            lower <= value,
                            value <= upper,
                        )

                return PymcPowerLaw(
                    key,
                    lower=self.priors[key].minimum,
                    upper=self.priors[key].maximum,
                    alpha=self.priors[key].alpha,
                )
        else:
            raise ValueError(f"Prior for '{key}' is not a Power Law")

    def _multivariate_normal_prior(self, key):
        """
        Map the bilby MultivariateNormal prior to a PyMC style function.
        """

        # check prior is a PowerLaw
        pymc, _, _ = self._import_external_sampler()
        if isinstance(self.priors[key], MultivariateGaussian):
            # get names of multivariate Gaussian parameters
            mvpars = self.priors[key].mvg.names

            # set the prior on multiple parameters if not present yet
            if not np.all([p in self.multivariate_normal_sets for p in mvpars]):
                mvg = self.priors[key].mvg

                # get bounds
                lower = [bound[0] for bound in mvg.bounds.values()]
                upper = [bound[1] for bound in mvg.bounds.values()]

                # test values required for mixture
                testvals = []
                for bound in mvg.bounds.values():
                    if np.isinf(bound[0]) and np.isinf(bound[1]):
                        testvals.append(0.0)
                    elif np.isinf(bound[0]):
                        testvals.append(bound[1] - 1.0)
                    elif np.isinf(bound[1]):
                        testvals.append(bound[0] + 1.0)
                    else:
                        # half-way between the two bounds
                        testvals.append(bound[0] + (bound[1] - bound[0]) / 2.0)

                # if bounds are at +/-infinity set to 100 sigmas as infinities
                # cause problems for the Bound class
                maxmu = np.max(mvg.mus, axis=0)
                minmu = np.min(mvg.mus, axis=0)
                maxsigma = np.max(mvg.sigmas, axis=0)
                for i in range(len(mvpars)):
                    if np.isinf(lower[i]):
                        lower[i] = minmu[i] - 100.0 * maxsigma[i]
                    if np.isinf(upper[i]):
                        upper[i] = maxmu[i] + 100.0 * maxsigma[i]

                # create a bounded MultivariateNormal distribution
                BoundedMvN = pymc.Bound(pymc.MvNormal, lower=lower, upper=upper)

                comp_dists = []  # list of any component modes
                for i in range(mvg.nmodes):
                    comp_dists.append(
                        BoundedMvN(
                            f"comp{i}",
                            mu=mvg.mus[i],
                            cov=mvg.covs[i],
                            shape=len(mvpars),
                        ).distribution
                    )

                # create a Mixture model
                setname = f"mixture{self.multivariate_normal_num_sets}"
                mix = pymc.Mixture(
                    setname,
                    w=mvg.weights,
                    comp_dists=comp_dists,
                    shape=len(mvpars),
                    testval=testvals,
                )

                for i, p in enumerate(mvpars):
                    self.multivariate_normal_sets[p] = {}
                    self.multivariate_normal_sets[p]["prior"] = mix[i]
                    self.multivariate_normal_sets[p]["set"] = setname
                    self.multivariate_normal_sets[p]["index"] = i

                self.multivariate_normal_num_sets += 1

            # return required parameter
            return self.multivariate_normal_sets[key]["prior"]

        else:
            raise ValueError(f"Prior for '{key}' is not a MultivariateGaussian")

    def run_sampler(self):
        # set the step method
        pymc, STEP_METHODS, floatX = self._import_external_sampler()
        step_methods = {m.__name__.lower(): m.__name__ for m in STEP_METHODS}
        if "step" in self._kwargs:
            self.step_method = self._kwargs.pop("step")

            # 'step' could be a dictionary of methods for different parameters,
            # so check for this
            if self.step_method is None:
                pass
            elif isinstance(self.step_method, dict):
                for key in self.step_method:
                    if key not in self._search_parameter_keys:
                        raise ValueError(
                            f"Setting a step method for an unknown parameter '{key}'"
                        )
                    else:
                        # check if using a compound step (a list of step
                        # methods for a particular parameter)
                        if isinstance(self.step_method[key], list):
                            sms = self.step_method[key]
                        else:
                            sms = [self.step_method[key]]
                        for sm in sms:
                            if sm.lower() not in step_methods:
                                raise ValueError(
                                    f"Using invalid step method '{self.step_method[key]}'"
                                )
            else:
                # check if using a compound step (a list of step
                # methods for a particular parameter)
                if isinstance(self.step_method, list):
                    sms = self.step_method
                else:
                    sms = [self.step_method]

                for i in range(len(sms)):
                    if sms[i].lower() not in step_methods:
                        raise ValueError(f"Using invalid step method '{sms[i]}'")
        else:
            self.step_method = None

        # initialise the PyMC model
        self.pymc_model = pymc.Model()

        # set the prior
        self.set_prior()

        # if a custom log_likelihood function requires a `sampler` argument
        # then use that log_likelihood function, with the assumption that it
        # takes in a Pymc Sampler, with a pymc_model attribute, and defines
        # the likelihood within that context manager
        likeargs = infer_args_from_method(self.likelihood.log_likelihood)
        if "sampler" in likeargs:
            self.likelihood.log_likelihood(sampler=self)
        else:
            # set the likelihood function from predefined functions
            self.set_likelihood()

        # get the step method keyword arguments
        step_kwargs = self.kwargs.pop("step_kwargs")
        if step_kwargs is not None:
            # remove all individual default step kwargs if passed together using
            # step_kwargs keywords
            for key in self.default_step_kwargs:
                self.kwargs.pop(key)
        else:
            # remove any None default step keywords and place others in step_kwargs
            step_kwargs = {}
            for key in self.default_step_kwargs:
                if self.kwargs[key] is None:
                    self.kwargs.pop(key)
                else:
                    step_kwargs[key] = self.kwargs.pop(key)

        nuts_kwargs = self.kwargs.pop("nuts_kwargs")
        if nuts_kwargs is not None:
            # remove all individual default nuts kwargs if passed together using
            # nuts_kwargs keywords
            for key in self.default_nuts_kwargs:
                self.kwargs.pop(key)
        else:
            # remove any None default nuts keywords and place others in nut_kwargs
            nuts_kwargs = {}
            for key in self.default_nuts_kwargs:
                if self.kwargs[key] is None:
                    self.kwargs.pop(key)
                else:
                    nuts_kwargs[key] = self.kwargs.pop(key)
        methodslist = []

        # set the step method
        if isinstance(self.step_method, dict):
            # create list of step methods (any not given will default to NUTS)
            self.kwargs["step"] = []
            with self.pymc_model:
                for key in self.step_method:
                    # check for a compound step list
                    if isinstance(self.step_method[key], list):
                        for sms in self.step_method[key]:
                            curmethod = sms.lower()
                            methodslist.append(curmethod)
                            nuts_kwargs = self._create_nuts_kwargs(
                                curmethod,
                                key,
                                nuts_kwargs,
                                pymc,
                                step_kwargs,
                                step_methods,
                            )
                    else:
                        curmethod = self.step_method[key].lower()
                        methodslist.append(curmethod)
                        nuts_kwargs = self._create_nuts_kwargs(
                            curmethod,
                            key,
                            nuts_kwargs,
                            pymc,
                            step_kwargs,
                            step_methods,
                        )
        else:
            with self.pymc_model:
                # check for a compound step list
                if isinstance(self.step_method, list):
                    compound = []
                    for sms in self.step_method:
                        curmethod = sms.lower()
                        methodslist.append(curmethod)
                        args, nuts_kwargs = self._create_args_and_nuts_kwargs(
                            curmethod, nuts_kwargs, step_kwargs
                        )
                        compound.append(pymc.__dict__[step_methods[curmethod]](**args))
                        self.kwargs["step"] = compound
                else:
                    self.kwargs["step"] = None
                    if self.step_method is not None:
                        curmethod = self.step_method.lower()
                        methodslist.append(curmethod)
                        args, nuts_kwargs = self._create_args_and_nuts_kwargs(
                            curmethod, nuts_kwargs, step_kwargs
                        )
                        self.kwargs["step"] = pymc.__dict__[step_methods[curmethod]](
                            **args
                        )

        # check whether only NUTS step method has been assigned
        if np.all([sm.lower() == "nuts" for sm in methodslist]):
            # in this case we can let PyMC autoinitialise NUTS, so remove the step methods and re-add nuts_kwargs
            self.kwargs["step"] = None

            if len(nuts_kwargs) > 0:
                # add NUTS kwargs to standard kwargs
                self.kwargs.update(nuts_kwargs)

        with self.pymc_model:
            # perform the sampling and then convert to inference data
            trace = pymc.sample(**self.kwargs, return_inferencedata=False)
            ikwargs = dict(
                model=self.pymc_model,
                save_warmup=not self.kwargs["discard_tuned_samples"],
                log_likelihood=True,
            )
            trace = pymc.to_inference_data(trace, **ikwargs)

        posterior = trace.posterior.to_dataframe().reset_index()
        self.result.samples = posterior[self.search_parameter_keys]
        self.result.log_likelihood_evaluations = np.sum(
            trace.log_likelihood.likelihood.values, axis=-1
        ).flatten()
        self.result.sampler_output = np.nan
        self.calculate_autocorrelation(self.result.samples)
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
        self.calc_likelihood_count()
        return self.result

    def _create_args_and_nuts_kwargs(self, curmethod, nuts_kwargs, step_kwargs):
        if curmethod == "nuts":
            args, nuts_kwargs = self._get_nuts_args(nuts_kwargs, step_kwargs)
        else:
            args = step_kwargs.get(curmethod, {})
        return args, nuts_kwargs

    def _create_nuts_kwargs(
        self, curmethod, key, nuts_kwargs, pymc, step_kwargs, step_methods
    ):
        if curmethod == "nuts":
            args, nuts_kwargs = self._get_nuts_args(nuts_kwargs, step_kwargs)
        else:
            if step_kwargs is not None:
                args = step_kwargs.get(curmethod, {})
            else:
                args = {}
        self.kwargs["step"].append(
            pymc.__dict__[step_methods[curmethod]](vars=[self.pymc_priors[key]], **args)
        )
        return nuts_kwargs

    @staticmethod
    def _get_nuts_args(nuts_kwargs, step_kwargs):
        if nuts_kwargs is not None:
            args = nuts_kwargs
        elif step_kwargs is not None:
            args = step_kwargs.pop("nuts", {})
            # add values into nuts_kwargs
            nuts_kwargs = args
        else:
            args = {}
        return args, nuts_kwargs

    def set_prior(self):
        """
        Set the PyMC prior distributions.
        """

        self.setup_prior_mapping()

        self.pymc_priors = dict()
        pymc, _, _ = self._import_external_sampler()

        # initialise a dictionary of multivariate Gaussian parameters
        self.multivariate_normal_sets = {}
        self.multivariate_normal_num_sets = 0

        # set the parameter prior distributions (in the model context manager)
        with self.pymc_model:
            for key in self.priors:
                # if the prior contains ln_prob method that takes a 'sampler' argument
                # then try using that
                lnprobargs = infer_args_from_method(self.priors[key].ln_prob)
                if "sampler" in lnprobargs:
                    try:
                        self.pymc_priors[key] = self.priors[key].ln_prob(sampler=self)
                    except RuntimeError:
                        raise RuntimeError((f"Problem setting PyMC prior for '{key}'"))
                else:
                    # use Prior distribution name
                    distname = self.priors[key].__class__.__name__

                    if distname in self.prior_map:
                        # check if we have a predefined PyMC distribution
                        if (
                            "pymc" in self.prior_map[distname]
                            and "argmap" in self.prior_map[distname]
                        ):
                            # check the required arguments for the PyMC distribution
                            pymcdistname = self.prior_map[distname]["pymc"]

                            if pymcdistname not in pymc.__dict__:
                                raise ValueError(
                                    f"Prior '{pymcdistname}' is not a known PyMC distribution."
                                )

                            reqargs = infer_args_from_method(
                                pymc.__dict__[pymcdistname].dist
                            )

                            # set keyword arguments
                            priorkwargs = {}
                            for (targ, parg) in self.prior_map[distname][
                                "argmap"
                            ].items():
                                if hasattr(self.priors[key], targ):
                                    if parg in reqargs:
                                        if "argtransform" in self.prior_map[distname]:
                                            if (
                                                targ
                                                in self.prior_map[distname][
                                                    "argtransform"
                                                ]
                                            ):
                                                tfunc = self.prior_map[distname][
                                                    "argtransform"
                                                ][targ]
                                            else:

                                                def tfunc(x):
                                                    return x

                                        else:

                                            def tfunc(x):
                                                return x

                                        priorkwargs[parg] = tfunc(
                                            getattr(self.priors[key], targ)
                                        )
                                    else:
                                        raise ValueError(f"Unknown argument {parg}")
                                else:
                                    if parg in reqargs:
                                        priorkwargs[parg] = None
                            self.pymc_priors[key] = pymc.__dict__[pymcdistname](
                                key, **priorkwargs
                            )
                        elif "internal" in self.prior_map[distname]:
                            self.pymc_priors[key] = self.prior_map[distname][
                                "internal"
                            ](key)
                        else:
                            raise ValueError(
                                f"Prior '{distname}' is not a known distribution."
                            )
                    else:
                        raise ValueError(
                            f"Prior '{distname}' is not a known distribution."
                        )

    def set_likelihood(self):
        """
        Convert any bilby likelihoods to PyMC distributions.
        """

        # create Op for the log likelihood if not using a predefined model
        pymc, _, _ = self._import_external_sampler()
        _, tt, _ = self._import_tensor()

        class LogLike(tt.Op):

            itypes = [tt.dvector]
            otypes = [tt.dscalar]

            def __init__(self, parameters, loglike, priors):
                self.parameters = parameters
                self.likelihood = loglike
                self.priors = priors

                # set the fixed parameters
                for key in self.priors.keys():
                    if isinstance(self.priors[key], float):
                        self.likelihood.parameters[key] = self.priors[key]

                self.logpgrad = LogLikeGrad(
                    self.parameters, self.likelihood, self.priors
                )

            def perform(self, node, inputs, outputs):
                (theta,) = inputs
                for i, key in enumerate(self.parameters):
                    self.likelihood.parameters[key] = theta[i]

                outputs[0][0] = np.array(self.likelihood.log_likelihood())

            def grad(self, inputs, g):
                (theta,) = inputs
                return [g[0] * self.logpgrad(theta)]

        # create Op for calculating the gradient of the log likelihood
        class LogLikeGrad(tt.Op):

            itypes = [tt.dvector]
            otypes = [tt.dvector]

            def __init__(self, parameters, loglike, priors):
                self.parameters = parameters
                self.Nparams = len(parameters)
                self.likelihood = loglike
                self.priors = priors

                # set the fixed parameters
                for key in self.priors.keys():
                    if isinstance(self.priors[key], float):
                        self.likelihood.parameters[key] = self.priors[key]

            def perform(self, node, inputs, outputs):
                (theta,) = inputs

                # define version of likelihood function to pass to derivative function
                def lnlike(values):
                    for i, key in enumerate(self.parameters):
                        self.likelihood.parameters[key] = values[i]
                    return self.likelihood.log_likelihood()

                # calculate gradients
                grads = derivatives(
                    theta, lnlike, abseps=1e-5, mineps=1e-12, reltol=1e-2
                )

                outputs[0][0] = grads

        with self.pymc_model:
            #  check if it is a predefined likelhood function
            if isinstance(self.likelihood, GaussianLikelihood):
                # check required attributes exist
                if (
                    not hasattr(self.likelihood, "sigma")
                    or not hasattr(self.likelihood, "x")
                    or not hasattr(self.likelihood, "y")
                ):
                    raise ValueError(
                        "Gaussian Likelihood does not have all the correct attributes!"
                    )

                if "sigma" in self.pymc_priors:
                    # if sigma is suppled use that value
                    if self.likelihood.sigma is None:
                        self.likelihood.sigma = self.pymc_priors.pop("sigma")
                    else:
                        del self.pymc_priors["sigma"]

                for key in self.pymc_priors:
                    if key not in self.likelihood.function_keys:
                        raise ValueError(f"Prior key '{key}' is not a function key!")

                model = self.likelihood.func(self.likelihood.x, **self.pymc_priors)

                # set the distribution
                pymc.Normal(
                    "likelihood",
                    mu=model,
                    sigma=self.likelihood.sigma,
                    observed=self.likelihood.y,
                )
            elif isinstance(self.likelihood, PoissonLikelihood):
                # check required attributes exist
                if not hasattr(self.likelihood, "x") or not hasattr(
                    self.likelihood, "y"
                ):
                    raise ValueError(
                        "Poisson Likelihood does not have all the correct attributes!"
                    )

                for key in self.pymc_priors:
                    if key not in self.likelihood.function_keys:
                        raise ValueError(f"Prior key '{key}' is not a function key!")

                # get rate function
                model = self.likelihood.func(self.likelihood.x, **self.pymc_priors)

                # set the distribution
                pymc.Poisson("likelihood", mu=model, observed=self.likelihood.y)
            elif isinstance(self.likelihood, ExponentialLikelihood):
                # check required attributes exist
                if not hasattr(self.likelihood, "x") or not hasattr(
                    self.likelihood, "y"
                ):
                    raise ValueError(
                        "Exponential Likelihood does not have all the correct attributes!"
                    )

                for key in self.pymc_priors:
                    if key not in self.likelihood.function_keys:
                        raise ValueError(f"Prior key '{key}' is not a function key!")

                # get mean function
                model = self.likelihood.func(self.likelihood.x, **self.pymc_priors)

                # set the distribution
                pymc.Exponential(
                    "likelihood", lam=1.0 / model, observed=self.likelihood.y
                )
            elif isinstance(self.likelihood, StudentTLikelihood):
                # check required attributes exist
                if (
                    not hasattr(self.likelihood, "x")
                    or not hasattr(self.likelihood, "y")
                    or not hasattr(self.likelihood, "nu")
                    or not hasattr(self.likelihood, "sigma")
                ):
                    raise ValueError(
                        "StudentT Likelihood does not have all the correct attributes!"
                    )

                if "nu" in self.pymc_priors:
                    # if nu is suppled use that value
                    if self.likelihood.nu is None:
                        self.likelihood.nu = self.pymc_priors.pop("nu")
                    else:
                        del self.pymc_priors["nu"]

                for key in self.pymc_priors:
                    if key not in self.likelihood.function_keys:
                        raise ValueError(f"Prior key '{key}' is not a function key!")

                model = self.likelihood.func(self.likelihood.x, **self.pymc_priors)

                # set the distribution
                pymc.StudentT(
                    "likelihood",
                    nu=self.likelihood.nu,
                    mu=model,
                    sigma=self.likelihood.sigma,
                    observed=self.likelihood.y,
                )
            elif isinstance(
                self.likelihood,
                (GravitationalWaveTransient, BasicGravitationalWaveTransient),
            ):
                # set theano Op - pass _search_parameter_keys, which only contains non-fixed variables
                logl = LogLike(
                    self._search_parameter_keys, self.likelihood, self.pymc_priors
                )

                parameters = dict()
                for key in self._search_parameter_keys:
                    try:
                        parameters[key] = self.pymc_priors[key]
                    except KeyError:
                        raise KeyError(
                            f"Unknown key '{key}' when setting GravitationalWaveTransient likelihood"
                        )

                # convert to tensor variable
                values = tt.as_tensor_variable(list(parameters.values()))

                pymc.DensityDist(
                    "likelihood", lambda v: logl(v), observed={"v": values}
                )
            else:
                raise ValueError("Unknown likelihood has been provided")
