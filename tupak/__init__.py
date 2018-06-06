"""
tupak
=====

Tupak is The User friendly Parameter estimAtion Kode

The aim of tupak is to provide user friendly interface to perform parameter
estimation. It is primarily designed and built for inference of compact
binary coalesence events in interferometric data, but it can also be used for
more general problems.

For installation instructions see https://git.ligo.org/Monash/tupak

"""


from __future__ import print_function, division, absolute_import

# import local files, utils should be imported first
from . import utils
from . import detector
from . import prior
from . import source
from . import likelihood
from . import waveform_generator
from . import result
from . import sampler
from . import conversion

# import a few often-used functions and classes to simplify scripts
from .likelihood import Likelihood, GravitationalWaveTransient
from .waveform_generator import WaveformGenerator
from .sampler import run_sampler
