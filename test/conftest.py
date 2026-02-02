import importlib
import pytest


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
    backend = request.config.getoption("--array-backend")
    match backend:
        case None | "numpy":
            import numpy
            return numpy
        case "jax" | "jax.numpy":
            import os
            import jax

            os.environ["SCIPY_ARRAY_API"] = "1"
            jax.config.update("jax_enable_x64", True)
            return jax.numpy
        case _:
            try:
                importlib.import_module(backend)
            except ImportError:
                raise ValueError(f"Unknown backend for testing: {backend}")


@pytest.fixture
def xp(request):
    return _xp(request)


@pytest.fixture(scope="class")
def xp_class(request):
    request.cls.xp = _xp(request)
