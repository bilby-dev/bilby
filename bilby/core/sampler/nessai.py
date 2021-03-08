import numpy as np
from pandas import DataFrame

from .base_sampler import NestedSampler
from ..utils import logger, check_directory_exists_and_if_not_mkdir, load_json


class Nessai(NestedSampler):
    """bilby wrapper of nessai (https://github.com/mj-will/nessai)

    All positional and keyword arguments passed to `run_sampler` are propogated
    to `nessai.flowsampler.FlowSampler`

    See the documentation for an explanation of the different kwargs.

    Documentation: https://nessai.readthedocs.io/
    """
    default_kwargs = dict(
        output=None,
        nlive=1000,
        stopping=0.1,
        resume=True,
        max_iteration=None,
        checkpointing=True,
        seed=1234,
        acceptance_threshold=0.01,
        analytic_priors=False,
        maximum_uninformed=1000,
        uninformed_proposal=None,
        uninformed_proposal_kwargs=None,
        flow_class=None,
        flow_config=None,
        training_frequency=None,
        reset_weights=False,
        reset_permutations=False,
        reset_acceptance=False,
        train_on_empty=True,
        cooldown=100,
        memory=False,
        poolsize=None,
        drawsize=None,
        max_poolsize_scale=10,
        update_poolsize=False,
        latent_prior='truncated_gaussian',
        draw_latent_kwargs=None,
        compute_radius_with_all=False,
        min_radius=False,
        max_radius=50,
        check_acceptance=False,
        fuzz=1.0,
        expansion_fraction=1.0,
        rescale_parameters=True,
        rescale_bounds=[-1, 1],
        update_bounds=False,
        boundary_inversion=False,
        inversion_type='split', detect_edges=False,
        detect_edges_kwargs=None,
        reparameterisations=None,
        n_pool=None,
        max_threads=1,
        pytorch_threads=None,
        plot=None,
        proposal_plots=False
    )
    seed_equiv_kwargs = ['sampling_seed']

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
        from nessai.livepoint import dict_to_live_points
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

        self.result.posterior = DataFrame(out.posterior_samples)
        self.result.nested_samples = DataFrame(out.nested_samples)
        self.result.nested_samples.rename(
            columns=dict(logL='log_likelihood', logP='log_prior'), inplace=True)
        self.result.posterior.rename(
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
            self.kwargs['output'] = self.outdir + '/{}_nessai/'.format(self.label)
        if self.kwargs['output'].endswith('/') is False:
            self.kwargs['output'] = '{}/'.format(self.kwargs['output'])

        check_directory_exists_and_if_not_mkdir(self.kwargs['output'])
        NestedSampler._verify_kwargs_against_default_kwargs(self)
