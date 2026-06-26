import importlib

import array_api_compat as aac
import pytest

from bilby.compat.utils import BILBY_DEVICE


def pytest_addoption(parser):
    parser.addoption(
        "--skip-roqs", action="store_true", default=False, help="Skip all tests that require ROQs"
    )
    parser.addoption(
        "--array-backend",
        default=None,
        help="Which array to use for testing",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_roqs: mark a test that requires ROQs")
    config.addinivalue_line("markers", "array_backend: mark that a test uses all array backends")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-roqs"):
        skip_roqs = pytest.mark.skip(reason="Skipping tests that require ROQs")
    else:
        skip_roqs = None
    if config.getoption("--array-backend") is not None:
        array_only = pytest.mark.skip(reason="Only running backend dependent tests")
    else:
        array_only = None
    for item in items:
        if "requires_roqs" in item.keywords and config.getoption("--skip-roqs"):
            item.add_marker(skip_roqs)
        elif "array_backend" not in item.keywords and array_only is not None:
            item.add_marker(array_only)


def _xp(request):
    # The configuration here loosely follows scipy
    # https://github.com/scipy/scipy/blob/b167cae18888a34fc43a439e729383b50f4d373e/scipy/conftest.py#L186
    backend = request.config.getoption("--array-backend")
    match backend:
        case None | "numpy":
            import numpy as xp
        case "jax" | "jax.numpy":
            import jax

            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_default_device", jax.devices(BILBY_DEVICE)[0])
            xp = jax.numpy
        case "torch":
            import torch
            # torch starts a lot of threads, so disable this on the first import
            # to avoid segfaults
            try:
                torch.set_default_device(BILBY_DEVICE)
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
                torch.set_default_dtype(torch.float64)
            except RuntimeError:
                pass
            xp = torch
        case _:
            try:

                xp = importlib.import_module(backend)
            except ImportError:
                raise ValueError(f"Unknown backend for testing: {backend}")
    return aac.get_namespace(xp.ones(1))


def _rng(xp):
    import array_api_compat as aac
    from bilby.core.utils.random import resolve_random_state

    if aac.is_numpy_namespace(xp):
        return resolve_random_state(12345)
    elif aac.is_jax_namespace(xp):
        import jax.random
        return resolve_random_state(jax.random.key(12345))
    elif aac.is_torch_namespace(xp):
        import torch
        return resolve_random_state(torch.Tensor([12345]))
    else:
        raise ValueError(f"Unknown array namespace {xp} for RNG")


@pytest.fixture
def xp(request):
    return _xp(request)


@pytest.fixture(scope="class")
def xp_class(request):
    request.cls.xp = _xp(request)
    request.cls.rng = _rng(request.cls.xp)
