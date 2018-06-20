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


from __future__ import print_function, division, absolute_import

# import local files, core utils should be imported first
from tupak.core import utils, likelihood, prior, result, sampler
from tupak.gw import detector, conversion, source, waveform_generator

# import a few often-used functions and classes to simplify scripts
from tupak.core.likelihood import Likelihood
from tupak.gw.likelihood import GravitationalWaveTransient
from tupak.gw.waveform_generator import WaveformGenerator
from tupak.core.sampler import run_sampler
