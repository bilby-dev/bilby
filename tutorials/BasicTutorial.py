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

params = dict(A=signal_amplitude, f=signal_frequency, geocent_time=1,
              ra=1, dec=2, psi=0)

foo = src.SimpleSinusoidSource('foo', sampling_frequency, time_duration)

ht_signal = foo.time_domain_strain(params)['plus']
hf_signal, ff = utils.nfft(ht_signal, sampling_frequency)
print hf_signal
hf_signal = foo.frequency_domain_strain(params)['plus']
print hf_signal

"""
Create a noise realisation with a default power spectral density
"""

IFO_1 = peyote.detector.H1
IFOs = [IFO_1]
for IFO in IFOs:
    hf_noise, ff = IFO.power_spectral_density.get_noise_realisation(sampling_frequency, time_duration)
    IFO.set_data(frequency_domain_strain=hf_noise)
    IFO.inject_signal(foo, params)
    IFO.set_spectral_densities(ff)
    IFO.whiten_data()

plt.clf()
plt.loglog(ff, np.abs(hf_noise), label='noise')
plt.loglog(ff, np.abs(hf_signal), label='signal')
plt.loglog(ff, np.abs(hf_signal + hf_noise), '--', label='signal+noise')
plt.xlabel(r'frequency [Hz]')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

logLs = []
search_frequencies = np.linspace(params['f']-10, params['f'] + 10, 500)
likelihood = peyote.likelihood.likelihood(IFOs, foo, params)
for f in search_frequencies:
    logLs.append(likelihood.logl([f]))

plt.plot(search_frequencies, logLs)
plt.show()

