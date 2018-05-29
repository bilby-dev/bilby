"""
tupak
=====

Tupak is The User friendly Parameter estimAtion Kode

FILL IN THE REST

"""


from __future__ import print_function, division

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

# import a few oft-used functions and classes to simplify scripts
from likelihood import Likelihood, GravitationalWaveTransient
from waveform_generator import WaveformGenerator
from sampler import run_sampler
