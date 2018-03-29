"""
Basic tutorial to get PEYOte running
"""
import numpy as np
import pylab as plt

import peyote
import peyote.source as src
import peyote.parameter as par
import peyote.detector as det
import peyote.utils as utils


time_duration = 1.
sampling_frequency = 4096.
time = utils.create_time_series(sampling_frequency, time_duration)

signal_amplitude = 1e-21
signal_frequency = 100

params = dict(A=signal_amplitude, f=2.*np.pi*signal_frequency, geocent_time=1,
              modes=['plus'], ra=1, dec=2, psi=0, deltaF=sampling_frequency)

foo = src.SimpleSinusoidSource('foo')
ht_signal = foo.model(time, params)['plus']

hf_signal, ff = utils.nfft(ht_signal, sampling_frequency)

"""
Create a noise realisation with a default power spectral density
"""

PSD = det.PowerSpectralDensity()  # instantiate a detector psd
PSD.import_power_spectral_density()  # import default psd
#PSD.import_power_spectral_density(spectral_density_file="CE_psd.txt")  # import cosmic explorer
hf_noise , _ = PSD.noise_realisation(sampling_frequency, time_duration)


plt.clf()
plt.loglog(ff, np.abs(hf_signal + hf_noise), label='signal+noise')

plt.loglog(ff, np.abs(hf_noise), label='noise')

plt.loglog(ff, np.abs(hf_signal), label='signal')
plt.xlabel(r'frequency [Hz]')

plt.legend(loc='best')

plt.tight_layout()
plt.show()

hf_noise[0] = np.max(hf_noise)
hf_noise[-1] = np.max(hf_noise)
IFO = peyote.detector.H1
IFO.data = hf_signal
IFO.psd = hf_noise
likelihood = peyote.likelihood.likelihood([IFO], foo)
