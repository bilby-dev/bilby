from .dynesty import Dynesty


class DynamicDynesty(Dynesty):
    """
    bilby wrapper of `dynesty.DynamicNestedSampler`
    (https://dynesty.readthedocs.io/en/latest/)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `dynesty.DynamicNestedSampler`, see
    documentation for that class for further help.

    For additional documentation see bilby.core.sampler.Dynesty.
    """

    external_sampler_name = "dynesty"
    sampler_name = "dynamic_dynesty"

    @property
    def nlive(self):
        """
        Users can either specify :code:`nlive_init` or :code:`nlive` (with
        that precedence) or specify no value, in which case 500 is used.
        """
        if self.kwargs["nlive_init"] is not None:
            return self.kwargs["nlive_init"]
        elif self.kwargs["nlive"] is not None:
            return self.kwargs["nlive"]
        else:
            return 500

    @property
    def sampler_init(self):
        from dynesty import DynamicNestedSampler

        return DynamicNestedSampler

    @property
    def sampler_class(self):
        from dynesty.dynamicsampler import DynamicSampler

        return DynamicSampler

    def finalize_sampler_kwargs(self, sampler_kwargs):
        sampler_kwargs["maxcall"] = self.sampler.ncall + self.n_check_point

    def _add_live(self):
        pass

    def _remove_live(self):
        pass

    def read_saved_state(self, continuing=False):
        resume = super(DynamicDynesty, self).read_saved_state(continuing=continuing)
        if not resume:
            return resume
        else:
            self.sampler.loglikelihood.pool = self.pool
            return resume
