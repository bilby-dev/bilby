"""
Tests that all of the implemented samplers can be initialized.

The :code:`FakeSampler` is omitted as that doesn't require importing
any package.
"""
import bilby

bilby.core.utils.logger.setLevel("ERROR")
IMPLEMENTED_SAMPLERS = bilby.core.sampler.IMPLEMENTED_SAMPLERS
likelihood = bilby.core.likelihood.Likelihood(dict())
priors = bilby.core.prior.PriorDict(dict(a=bilby.core.prior.Uniform(0, 1)))
for sampler in IMPLEMENTED_SAMPLERS:
    if sampler == "fake_sampler":
        continue
    sampler_class = IMPLEMENTED_SAMPLERS[sampler]
    sampler = sampler_class(likelihood=likelihood, priors=priors)
