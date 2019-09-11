from __future__ import absolute_import

import copy

import numpy as np
from pandas import DataFrame

from .base_sampler import NestedSampler
from .proposal import Sample, JumpProposalCycle
from ..utils import logger, check_directory_exists_and_if_not_mkdir


class Cpnest(NestedSampler):
    """ bilby wrapper of cpnest (https://github.com/johnveitch/cpnest)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `cpnest.CPNest`, see documentation
    for that class for further help. Under Other Parameters, we list commonly
    used kwargs and the bilby defaults.

    Other Parameters
    ----------------
    nlive: int
        The number of live points, note this can also equivalently be given as
        one of [npoints, nlives, n_live_points]
    seed: int (1234)
        Initialised random seed
    nthreads: int, (1)
        Number of threads to use
    maxmcmc: int (1000)
        The maximum number of MCMC steps to take
    verbose: Bool (True)
        If true, print information information about the convergence during
    resume: Bool (True)
        Whether or not to resume from a previous run
    output: str
        Where to write the CPNest, by default this is
        {self.outdir}/cpnest_{self.label}/

    """
    default_kwargs = dict(verbose=1, nthreads=1, nlive=500, maxmcmc=1000,
                          seed=None, poolsize=100, nhamiltonian=0, resume=True,
                          output=None, proposals=None)

    def _translate_kwargs(self, kwargs):
        if 'nlive' not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['nlive'] = kwargs.pop(equiv)
        if 'seed' not in kwargs:
            logger.warning('No seed provided, cpnest will use 1234.')

    def run_sampler(self):
        from cpnest import model as cpmodel, CPNest
        from cpnest.parameter import LivePoint
        from cpnest.nest2pos import compute_weights

        class Model(cpmodel.Model):
            """ A wrapper class to pass our log_likelihood into cpnest """

            def __init__(self, names, priors):
                self.names = names
                self.priors = priors
                self.bounds = [
                    [self.priors[key].minimum, self.priors[key].maximum]
                    for key in self.names]

            @staticmethod
            def log_likelihood(x, **kwargs):
                theta = [x[n] for n in self.search_parameter_keys]
                return self.log_likelihood(theta)

            @staticmethod
            def log_prior(x, **kwargs):
                theta = [x[n] for n in self.search_parameter_keys]
                return self.log_prior(theta)

            def new_point(self):
                """Draw a point from the prior"""
                point = LivePoint(
                    self.names, [self.priors[name].sample()
                                 for name in self.names])
                return point

        self._resolve_proposal_functions()
        model = Model(self.search_parameter_keys, self.priors)
        try:
            out = CPNest(model, **self.kwargs)
        except TypeError as e:
            if 'proposals' in self.kwargs.keys():
                logger.warning('YOU ARE TRYING TO USE PROPOSALS IN A VERSION OF CPNEST THAT DOES'
                               'NOT ACCEPT CUSTOM PROPOSALS. SAMPLING WILL COMMENCE WITH THE DEFAULT'
                               'PROPOSALS.')
                del self.kwargs['proposals']
                out = CPNest(model, **self.kwargs)
            else:
                raise TypeError(e)
        out.run()

        if self.plot:
            out.plot()

        self.calc_likelihood_count()
        self.result.posterior = DataFrame(out.posterior_samples)
        self.result.nested_samples = DataFrame(out.get_nested_samples(filename=''))
        self.result.nested_samples.rename(columns=dict(logL='log_likelihood'), inplace=True)
        self.result.posterior.rename(columns=dict(logL='log_likelihood', logPrior='log_prior'),
                                     inplace=True)
        _, log_weights = compute_weights(np.array(self.result.nested_samples.log_likelihood),
                                         np.array(out.NS.state.nlive))
        self.result.nested_samples.weights = np.exp(log_weights)
        self.result.log_evidence = out.NS.state.logZ
        self.result.log_evidence_err = np.sqrt(out.NS.state.info / out.NS.state.nlive)
        return self.result

    def _verify_kwargs_against_default_kwargs(self):
        """
        Set the directory where the output will be written.
        """
        if not self.kwargs['output']:
            self.kwargs['output'] = \
                '{}/cpnest_{}/'.format(self.outdir, self.label)
        if self.kwargs['output'].endswith('/') is False:
            self.kwargs['output'] = '{}/'.format(self.kwargs['output'])
        check_directory_exists_and_if_not_mkdir(self.kwargs['output'])
        NestedSampler._verify_kwargs_against_default_kwargs(self)

    def _resolve_proposal_functions(self):
        from cpnest.proposal import ProposalCycle
        if 'proposals' in self.kwargs:
            if self.kwargs['proposals'] is None:
                return
            if type(self.kwargs['proposals']) == JumpProposalCycle:
                self.kwargs['proposals'] = dict(mhs=self.kwargs['proposals'], hmc=self.kwargs['proposals'])
            for key, proposal in self.kwargs['proposals'].items():
                if isinstance(proposal, JumpProposalCycle):
                    self.kwargs['proposals'][key] = cpnest_proposal_cycle_factory(proposal)
                elif isinstance(proposal, ProposalCycle):
                    pass
                else:
                    raise TypeError("Unknown proposal type")


def cpnest_proposal_factory(jump_proposal):
    import cpnest.proposal

    class CPNestEnsembleProposal(cpnest.proposal.EnsembleProposal):

        def __init__(self, jp):
            self.jump_proposal = jp
            self.ensemble = None

        def __call__(self, sample, **kwargs):
            return self.get_sample(sample, **kwargs)

        def get_sample(self, cpnest_sample, **kwargs):
            sample = Sample.from_cpnest_live_point(cpnest_sample)
            self.ensemble = kwargs.get('coordinates', self.ensemble)
            sample = self.jump_proposal(sample=sample, sampler_name='cpnest', **kwargs)
            self.log_J = self.jump_proposal.log_j
            return self._update_cpnest_sample(cpnest_sample, sample)

        @staticmethod
        def _update_cpnest_sample(cpnest_sample, sample):
            cpnest_sample.names = list(sample.keys())
            for i, value in enumerate(sample.values()):
                cpnest_sample.values[i] = value
            return cpnest_sample

    return CPNestEnsembleProposal(jump_proposal)


def cpnest_proposal_cycle_factory(jump_proposals):
    import cpnest.proposal

    class CPNestProposalCycle(cpnest.proposal.ProposalCycle):
        def __init__(self):
            self.jump_proposals = copy.deepcopy(jump_proposals)
            for i, prop in enumerate(self.jump_proposals.proposal_functions):
                self.jump_proposals.proposal_functions[i] = cpnest_proposal_factory(prop)
            self.jump_proposals.update_cycle()
            super(CPNestProposalCycle, self).__init__(proposals=self.jump_proposals.proposal_functions,
                                                      weights=self.jump_proposals.weights,
                                                      cyclelength=self.jump_proposals.cycle_length)

        def get_sample(self, old, **kwargs):
            return self.jump_proposals(sample=old, coordinates=self.ensemble, **kwargs)

        def set_ensemble(self, ensemble):
            self.ensemble = ensemble

    return CPNestProposalCycle
