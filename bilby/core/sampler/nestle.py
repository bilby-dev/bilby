from pandas import DataFrame

from .base_sampler import NestedSampler, signal_wrapper


class Nestle(NestedSampler):
    """bilby wrapper `nestle.Sampler` (http://kylebarbary.com/nestle/)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `nestle.sample`, see documentation for
    that function for further help. Under Other Parameters, we list commonly
    used kwargs and the bilby defaults

    Parameters
    ==========
    npoints: int
        The number of live points, note this can also equivalently be given as
        one of [nlive, nlives, n_live_points]
    method: {'classic', 'single', 'multi'} ('multi')
        Method used to select new points
    verbose: Bool
        If true, print information information about the convergence during
        sampling

    """

    sampler_name = "nestle"
    default_kwargs = dict(
        verbose=True,
        method="multi",
        npoints=500,
        update_interval=None,
        npdim=None,
        maxiter=None,
        maxcall=None,
        dlogz=None,
        decline_factor=None,
        rstate=None,
        callback=None,
        steps=20,
        enlarge=1.2,
    )

    def _translate_kwargs(self, kwargs):
        kwargs = super()._translate_kwargs(kwargs)
        if "npoints" not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["npoints"] = kwargs.pop(equiv)
        if "steps" not in kwargs:
            for equiv in self.walks_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["steps"] = kwargs.pop(equiv)

    def _verify_kwargs_against_default_kwargs(self):
        if self.kwargs["verbose"]:
            import nestle

            self.kwargs["callback"] = nestle.print_progress
            self.kwargs.pop("verbose")
        NestedSampler._verify_kwargs_against_default_kwargs(self)

    @signal_wrapper
    def run_sampler(self):
        """Runs Nestle sampler with given kwargs and returns the result

        Returns
        =======
        bilby.core.result.Result: Packaged information about the result

        """
        import nestle

        if nestle.__version__ == "0.2.0":
            # This is a very ugly hack to support numpy>=1.24
            nestle.np.float = float
            nestle.np.int = int

        out = nestle.sample(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim,
            **self.kwargs
        )
        print("")

        self.result.sampler_output = out
        self.result.samples = nestle.resample_equal(out.samples, out.weights)
        self.result.nested_samples = DataFrame(
            out.samples, columns=self.search_parameter_keys
        )
        self.result.nested_samples["weights"] = out.weights
        self.result.nested_samples["log_likelihood"] = out.logl
        self.result.log_likelihood_evaluations = self.reorder_loglikelihoods(
            unsorted_loglikelihoods=out.logl,
            unsorted_samples=out.samples,
            sorted_samples=self.result.samples,
        )
        self.result.log_evidence = out.logz
        self.result.log_evidence_err = out.logzerr
        self.result.information_gain = out.h
        self.calc_likelihood_count()
        return self.result

    def _run_test(self):
        """
        Runs to test whether the sampler is properly running with the given
        kwargs without actually running to the end

        Returns
        =======
        bilby.core.result.Result: Dummy container for sampling results.

        """
        self.kwargs["maxiter"] = 2
        return self.run_sampler()

    def write_current_state(self):
        """
        Nestle doesn't support checkpointing so no current state will be
        written on interrupt.
        """
        pass
