import numpy as np
import os
from pandas import DataFrame

from .base_sampler import NestedSampler
from ..utils import logger, check_directory_exists_and_if_not_mkdir, load_json


class Nessai(NestedSampler):
    """bilby wrapper of nessai (https://github.com/mj-will/nessai)

    All positional and keyword arguments passed to `run_sampler` are propagated
    to `nessai.flowsampler.FlowSampler`

    See the documentation for an explanation of the different kwargs.

    Documentation: https://nessai.readthedocs.io/
    """
    _default_kwargs = None
    seed_equiv_kwargs = ['sampling_seed']

    @property
    def default_kwargs(self):
        """Default kwargs for nessai.

        Retrieves default values from nessai directly and then includes any
        bilby specific defaults. This avoids the need to update bilby when the
        defaults change or new kwargs are added to nessai.
        """
        if not self._default_kwargs:
            from inspect import signature
            from nessai.flowsampler import FlowSampler
            from nessai.nestedsampler import NestedSampler
            from nessai.proposal import AugmentedFlowProposal, FlowProposal

            kwargs = {}
            classes = [
                AugmentedFlowProposal,
                FlowProposal,
                NestedSampler,
                FlowSampler,
            ]
            for c in classes:
                kwargs.update(
                    {k: v.default for k, v in signature(c).parameters.items() if v.default is not v.empty}
                )
            # Defaults for bilby that will override nessai defaults
            bilby_defaults = dict(
                output=None,
            )
            kwargs.update(bilby_defaults)
            self._default_kwargs = kwargs
        return self._default_kwargs

    def log_prior(self, theta):
        """

        Parameters
        ----------
        theta: list
            List of sampled values on a unit interval

        Returns
        -------
        float: Joint ln prior probability of theta

        """
        return self.priors.ln_prob(theta, axis=0)

    def run_sampler(self):
        from nessai.flowsampler import FlowSampler
        from nessai.model import Model as BaseModel
        from nessai.livepoint import dict_to_live_points, live_points_to_array
        from nessai.posterior import compute_weights
        from nessai.utils import setup_logger

        class Model(BaseModel):
            """A wrapper class to pass our log_likelihood and priors into nessai

            Parameters
            ----------
            names : list of str
                List of parameters to sample
            priors : :obj:`bilby.core.prior.PriorDict`
                Priors to use for sampling. Needed for the bounds and the
                `sample` method.
            """
            def __init__(self, names, priors):
                self.names = names
                self.priors = priors
                self._update_bounds()

            @staticmethod
            def log_likelihood(x, **kwargs):
                """Compute the log likelihood"""
                theta = [x[n].item() for n in self.search_parameter_keys]
                return self.log_likelihood(theta)

            @staticmethod
            def log_prior(x, **kwargs):
                """Compute the log prior"""
                theta = {n: x[n] for n in self._search_parameter_keys}
                return self.log_prior(theta)

            def _update_bounds(self):
                self.bounds = {key: [self.priors[key].minimum, self.priors[key].maximum]
                               for key in self.names}

            def new_point(self, N=1):
                """Draw a point from the prior"""
                prior_samples = self.priors.sample(size=N)
                samples = {n: prior_samples[n] for n in self.names}
                return dict_to_live_points(samples)

            def new_point_log_prob(self, x):
                """Proposal probability for new the point"""
                return self.log_prior(x)

        # Setup the logger for nessai using the same settings as the bilby logger
        setup_logger(self.outdir, label=self.label,
                     log_level=logger.getEffectiveLevel())
        model = Model(self.search_parameter_keys, self.priors)
        out = None
        while out is None:
            try:
                out = FlowSampler(model, **self.kwargs)
            except TypeError as e:
                raise TypeError("Unable to initialise nessai sampler with error: {}".format(e))
        try:
            out.run(save=True, plot=self.plot)
        except SystemExit as e:
            import sys
            logger.info("Caught exit code {}, exiting with signal {}".format(e.args[0], self.exit_code))
            sys.exit(self.exit_code)

        # Manually set likelihood evaluations because parallelisation breaks the counter
        self.result.num_likelihood_evaluations = out.ns.likelihood_evaluations[-1]

        self.result.samples = live_points_to_array(
            out.posterior_samples, self.search_parameter_keys
        )
        self.result.log_likelihood_evaluations = out.posterior_samples['logL']
        self.result.nested_samples = DataFrame(out.nested_samples)
        self.result.nested_samples.rename(
            columns=dict(logL='log_likelihood', logP='log_prior'), inplace=True)
        _, log_weights = compute_weights(np.array(self.result.nested_samples.log_likelihood),
                                         np.array(out.ns.state.nlive))
        self.result.nested_samples['weights'] = np.exp(log_weights)
        self.result.log_evidence = out.ns.log_evidence
        self.result.log_evidence_err = np.sqrt(out.ns.information / out.ns.nlive)

        return self.result

    def _translate_kwargs(self, kwargs):
        if 'nlive' not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['nlive'] = kwargs.pop(equiv)
        if 'n_pool' not in kwargs:
            for equiv in self.npool_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['n_pool'] = kwargs.pop(equiv)
        if 'seed' not in kwargs:
            for equiv in self.seed_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['seed'] = kwargs.pop(equiv)

    def _verify_kwargs_against_default_kwargs(self):
        """
        Set the directory where the output will be written
        and check resume and checkpoint status.
        """
        if 'config_file' in self.kwargs:
            d = load_json(self.kwargs['config_file'], None)
            self.kwargs.update(d)
            self.kwargs.pop('config_file')

        if not self.kwargs['plot']:
            self.kwargs['plot'] = self.plot

        if self.kwargs['n_pool'] == 1 and self.kwargs['max_threads'] == 1:
            logger.warning('Setting pool to None (n_pool=1 & max_threads=1)')
            self.kwargs['n_pool'] = None

        if not self.kwargs['output']:
            self.kwargs['output'] = os.path.join(
                self.outdir, '{}_nessai'.format(self.label), ''
            )

        check_directory_exists_and_if_not_mkdir(self.kwargs['output'])
        NestedSampler._verify_kwargs_against_default_kwargs(self)
