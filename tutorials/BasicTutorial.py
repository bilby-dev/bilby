"""
Basic tutorial to get PEYOte running
"""
import numpy as np
import pylab as plt

import peyote.source as src
import peyote.parameter as par
import peyote.detector as det
import peyote.utils as utils


time_duration = 1
fs = 4096.
time = np.arange(0, time_duration, 1./fs)

signal_amplitude = 1e-21
signal_frequency = 100

params = dict(A=signal_amplitude, f=2.*np.pi*signal_frequency, geocent_time=time)

foo = src.SimpleSinusoidSource('foo')
ht_signal = foo.model(params)

hf_signal, ff = utils.nfft(ht_signal, fs)

"""
Create a noise realisation with a default power spectral density
"""
PSD = det.PowerSpectralDensity()  # instantiate a detector psd
PSD.import_power_spectral_density()  # import default psd
#PSD.import_power_spectral_density(spectral_density_file="CE_psd.txt")  # import cosmic explorer
hf_noise , _ = PSD.noise_realisation(fs, time_duration)

plt.loglog(ff, np.abs(hf_noise))

plt.loglog(ff, np.abs(hf_signal))

plt.loglog(ff, np.abs(hf_signal + hf_noise))
plt.xlabel(r'frequency [Hz]')

plt.tight_layout()
plt.show()
