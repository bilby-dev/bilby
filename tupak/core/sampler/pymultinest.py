from ..utils import check_directory_exists_and_if_not_mkdir
from .base_sampler import Sampler


class Pymultinest(Sampler):
    """
    tupak wrapper of pymultinest
    (https://github.com/JohannesBuchner/PyMultiNest)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `pymultinest.run`, see documentation
    for that class for further help. Under Keyword Arguments, we list commonly
    used kwargs and the tupak defaults.

    Keyword Arguments
    ------------------
    npoints: int
        The number of live points, note this can also equivalently be given as
        one of [nlive, nlives, n_live_points]
    importance_nested_sampling: bool, (False)
        If true, use importance nested sampling
    sampling_efficiency: float or {'parameter', 'model'}, ('parameter')
        Defines the sampling efficiency
    verbose: Bool
        If true, print information information about the convergence during
    resume: bool
        If true, resume run from checkpoint (if available)

    """

    @property
    def kwargs(self):
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        outputfiles_basename =\
            self.outdir + '/pymultinest_{}/'.format(self.label)
        check_directory_exists_and_if_not_mkdir(outputfiles_basename)
        self.__kwargs = dict(importance_nested_sampling=False, resume=True,
                             verbose=True, sampling_efficiency='parameter',
                             outputfiles_basename=outputfiles_basename)
        self.__kwargs.update(kwargs)
        if self.__kwargs['outputfiles_basename'].endswith('/') is False:
            self.__kwargs['outputfiles_basename'] = '{}/'.format(
                self.__kwargs['outputfiles_basename'])
        if 'n_live_points' not in self.__kwargs:
            for equiv in ['nlive', 'nlives', 'npoints', 'npoint']:
                if equiv in self.__kwargs:
                    self.__kwargs['n_live_points'] = self.__kwargs.pop(equiv)

    def _run_external_sampler(self):
        pymultinest = self.external_sampler
        self.external_sampler_function = pymultinest.run
        self._verify_kwargs_against_external_sampler_function()
        # Note: pymultinest.solve adds some extra steps, but underneath
        # we are calling pymultinest.run - hence why it is used in checking
        # the arguments.
        out = pymultinest.solve(
            LogLikelihood=self.log_likelihood, Prior=self.prior_transform,
            n_dims=self.ndim, **self.kwargs)

        self.result.sampler_output = out
        self.result.samples = out['samples']
        self.result.log_evidence = out['logZ']
        self.result.log_evidence_err = out['logZerr']
        self.result.outputfiles_basename = self.kwargs['outputfiles_basename']
        return self.result
