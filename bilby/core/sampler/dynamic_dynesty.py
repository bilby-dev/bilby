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

    @property
    def nlive(self):
        return self.kwargs["nlive_init"]

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

    def read_saved_state(self, continuing=False):
        resume = super(DynamicDynesty, self).read_saved_state(continuing=continuing)
        if not resume:
            return resume
        else:
            self.sampler.loglikelihood.pool = self.pool
            return resume
