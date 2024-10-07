from bilby.core.sampler import (
    get_implemented_samplers,
    get_sampler_class,
)
import pytest


def test_get_implemented_samplers():
    """Assert the function returns a list of the correct length"""
    from bilby.core.sampler import IMPLEMENTED_SAMPLERS

    out = get_implemented_samplers()
    assert isinstance(out, list)
    assert len(out) == len(IMPLEMENTED_SAMPLERS)
    assert "dynesty" in out


def test_get_sampler_class():
    """Assert the function returns the correct class"""
    from bilby.core.sampler.dynesty import Dynesty

    sampler_class = get_sampler_class("dynesty")
    assert sampler_class is Dynesty


def test_get_sampler_class_not_implemented():
    """Assert an error is raised if the sampler is not recognized"""
    with pytest.raises(
        ValueError,
        match=r"Sampler not_a_valid_sampler not yet implemented"
    ):
        get_sampler_class("not_a_valid_sampler")
