import bilby
import pytest


@pytest.mark.parametrize(
    "sampler_name", bilby.core.sampler.IMPLEMENTED_SAMPLERS.keys()
)
def test_sampler_import(sampler_name):
    """
    Tests that all of the implemented samplers can be initialized.

    Do not test :code:`FakeSampler` since it requires an additional argument.
    """
    if sampler_name in ["fake_sampler", "pypolychord"]:
        pytest.skip(f"Skipping import test for {sampler_name}")
    bilby.core.utils.logger.setLevel("ERROR")
    likelihood = bilby.core.likelihood.Likelihood(dict())
    priors = bilby.core.prior.PriorDict(dict(a=bilby.core.prior.Uniform(0, 1)))
    sampler_class = bilby.core.sampler.IMPLEMENTED_SAMPLERS[sampler_name].load()
    sampler = sampler_class(likelihood=likelihood, priors=priors)
    assert sampler is not None
