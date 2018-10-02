"""
tupak
=====

Tupak is The User friendly Parameter estimAtion Kode.

The aim of tupak is to provide user friendly interface to perform parameter
estimation. It is primarily designed and built for inference of compact
binary coalescence events in interferometric data, but it can also be used for
more general problems.

The code, and many examples are hosted at https://git.ligo.org/Monash/tupak.
For installation instructions see
https://monash.docs.ligo.org/tupak/installation.html.

"""


from __future__ import absolute_import

from . import core, gw, hyper

from .core import utils, likelihood, prior, result, sampler
from .core.sampler import run_sampler
from .core.likelihood import Likelihood
