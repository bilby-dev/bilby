import numpy as np
from .base_sampler import Sampler


class Nestle(Sampler):
    """tupak wrapper `nestle.Sampler` (http://kylebarbary.com/nestle/)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `nestle.sample`, see documentation for
    that function for further help. Under Keyword Arguments, we list commonly
    used kwargs and the tupak defaults

    Keyword Arguments
   ------------------
    npoints: int
        The number of live points, note this can also equivalently be given as
        one of [nlive, nlives, n_live_points]
    method: {'classic', 'single', 'multi'} ('multi')
        Method used to select new points
    verbose: Bool
        If true, print information information about the convergence during
        sampling

    """

    @property
    def kwargs(self):
        """
        Ensures that proper keyword arguments are used for the Nestle sampler.

        Returns
        -------
        dict: Keyword arguments used for the Nestle Sampler

        """
        return self.__kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self.__kwargs = dict(verbose=True, method='multi')
        self.__kwargs.update(kwargs)

        if 'npoints' not in self.__kwargs:
            for equiv in ['nlive', 'nlives', 'n_live_points']:
                if equiv in self.__kwargs:
                    self.__kwargs['npoints'] = self.__kwargs.pop(equiv)

    def _run_external_sampler(self):
        """ Runs Nestle sampler with given kwargs and returns the result

        Returns
        -------
        tupak.core.result.Result: Packaged information about the result

        """
        nestle = self.external_sampler
        self.external_sampler_function = nestle.sample
        if 'verbose' in self.kwargs:
            if self.kwargs['verbose']:
                self.kwargs['callback'] = nestle.print_progress
            self.kwargs.pop('verbose')
        self._verify_kwargs_against_external_sampler_function()

        out = self.external_sampler_function(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, **self.kwargs)
        print("")

        self.result.sampler_output = out
        self.result.samples = nestle.resample_equal(out.samples, out.weights)
        self.result.log_likelihood_evaluations = out.logl
        self.result.log_evidence = out.logz
        self.result.log_evidence_err = out.logzerr
        return self.result

    def _run_test(self):
        """
        Runs to test whether the sampler is properly running with the given
        kwargs without actually running to the end

        Returns
        -------
        tupak.core.result.Result: Dummy container for sampling results.

        """
        nestle = self.external_sampler
        self.external_sampler_function = nestle.sample
        self.external_sampler_function(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim, maxiter=2, **self.kwargs)
        self.result.samples = np.random.uniform(0, 1, (100, self.ndim))
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan
        return self.result
