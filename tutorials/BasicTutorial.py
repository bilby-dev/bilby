"""
Basic tutorial to get PEYOte running
"""
import matplotlib
matplotlib.use('agg')
import numpy as np
import pylab as plt

import peyote.source as src
import peyote.parameter as par
import peyote.detector as det
from peyote.utils import sampling_frequency


time_duration = 1
time = np.linspace(0, time_duration, 10000)
fs = sampling_frequency(time)


params = dict(A=10, f=0.1, geocent_time=0)

foo = src.SimpleSinusoidSource('foo')
ht = foo.model(params)



"""
Create a noise realisation with a default power spectral density
"""
PSD = det.PowerSpectralDensity()  # instantiate a detector psd
PSD.import_power_spectral_density()  # import default psd
#PSD.import_power_spectral_density(spectral_density_file="CE_psd.txt")  # import cosmic explorer
hf , ff = PSD.noise_realisation(fs, time_duration)

plt.loglog(ff, np.abs(hf))
plt.xlabel(r'frequency [Hz]')
plt.show()
