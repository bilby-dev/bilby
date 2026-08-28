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
import logging

# Avoid configuring logging for applications that import bilby.
logging.getLogger(__name__).addHandler(logging.NullHandler())

from . import core, gw, hyper

from .core import utils, likelihood, prior, result, sampler
from .core.sampler import run_sampler
from .core.likelihood import Likelihood
from .core.result import read_in_result, read_in_result_list

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = 'unknown'
