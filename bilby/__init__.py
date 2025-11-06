"""
Bilby
=====

Bilby: a user-friendly Bayesian inference library.

The aim of bilby is to provide a user-friendly interface to perform parameter
estimation. It is primarily designed and built for inference of compact
binary coalescence events in interferometric data, but it can also be used for
more general problems.

The code, and many examples are hosted at https://github.com/bilby-dev/bilby.
For installation instructions see
https://bilby-dev.github.io/bilby/installation.html.

"""

from . import core, gw, hyper
from .core import likelihood, prior, result, sampler, utils
from .core.likelihood import Likelihood
from .core.result import read_in_result, read_in_result_list
from .core.sampler import run_sampler

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = "unknown"

__all__ = [
    core,
    gw,
    hyper,
    likelihood,
    prior,
    result,
    sampler,
    utils,
    Likelihood,
    read_in_result,
    read_in_result_list,
    run_sampler,
    __version__,
]
