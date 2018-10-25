"""
Bilby
=====

Bilby: a user-friendly Bayesian inference library.

The aim of bilby is to provide user friendly interface to perform parameter
estimation. It is primarily designed and built for inference of compact
binary coalescence events in interferometric data, but it can also be used for
more general problems.

The code, and many examples are hosted at https://git.ligo.org/lscsoft/bilby.
For installation instructions see
https://lscsoft.docs.ligo.org/bilby/installation.html.

"""


from __future__ import absolute_import

from . import core, gw, hyper

from .core import utils, likelihood, prior, result, sampler
from .core.sampler import run_sampler
from .core.likelihood import Likelihood

__version__ = utils.get_version_information()
