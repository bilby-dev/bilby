from bilby.core.sampler import IMPLEMENTED_SAMPLERS, ImplementedSamplers
import pytest


def test_singleton():
    assert ImplementedSamplers() is IMPLEMENTED_SAMPLERS


def test_keys():
    # The fake sampler should never have a plugin, so this should always work
    assert "fake_sampler" in IMPLEMENTED_SAMPLERS.keys()
    assert "bilby.fake_sampler" not in IMPLEMENTED_SAMPLERS.keys()


def test_allowed_keys():
    # The fake sampler should never have a plugin, so this should always work
    assert "fake_sampler" in IMPLEMENTED_SAMPLERS.valid_keys()
    assert "bilby.fake_sampler" in IMPLEMENTED_SAMPLERS.valid_keys()


def test_values():
    # Values and keys should have the same lengths
    assert len(list(IMPLEMENTED_SAMPLERS.values())) \
        == len(list(IMPLEMENTED_SAMPLERS.keys()))
    assert len(list(IMPLEMENTED_SAMPLERS.values())) \
        == len(list(IMPLEMENTED_SAMPLERS._samplers.values()))


def test_items():
    keys, values = list(zip(*IMPLEMENTED_SAMPLERS.items()))
    assert len(keys) == len(values)
    # Keys and values should be the same as the individual methods
    assert list(keys) == list(IMPLEMENTED_SAMPLERS.keys())
    assert list(values) == list(IMPLEMENTED_SAMPLERS.values())


@pytest.mark.parametrize("sampler", ["fake_sampler", "bilby.fake_sampler"])
def test_in_operator(sampler):
    assert sampler in IMPLEMENTED_SAMPLERS
