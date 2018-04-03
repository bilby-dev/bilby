"""
Tutorial to show signal injection using new features of detector.py
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

import peyote

time_duration = 400
sampling_frequency = 4096.
times = peyote.utils.create_time_series(sampling_frequency, time_duration)

signal_amplitude = 1e-30
signal_frequency = 100

params = dict(A=signal_amplitude, f=signal_frequency, geocent_time=1, ra=1, dec=2, psi=0, deltaF=sampling_frequency)

foo = peyote.source.SimpleSinusoidSource('foo')
hf_signal = foo.frequency_domain_strain(sampling_frequency, time_duration, params)

IFO_1 = peyote.detector.H1
IFO_2 = peyote.detector.L1
IFO_3 = peyote.detector.V1
IFOs = [IFO_1, IFO_2, IFO_3]
for IFO in IFOs:
    hf_noise, ff = IFO.power_spectral_density.get_noise_realisation(sampling_frequency, time_duration)
    IFO.set_data(frequency_domain_strain=hf_noise)
    IFO.inject_signal(foo, params, sampling_frequency, time_duration)
    IFO.set_spectral_densities(ff)
    IFO.whiten_data()

likelihood = peyote.likelihood.logl_gravitational_wave(sampling_frequency, time_duration, params, foo, IFOs)

noise_params = params.copy()
noise_params['A'] = 0
noise_likelihood = peyote.likelihood.logl_gravitational_wave(sampling_frequency, time_duration, noise_params, foo, IFOs)

print('delta_log_l = {}'.format(likelihood-noise_likelihood))

plt.clf()
for IFO in IFOs:
    plt.loglog(ff, np.abs(IFO.data), label=IFO.name)

plt.loglog(ff, np.abs(hf_signal['cross']), label='signal', linestyle='--')

plt.xlim(10, 1e3)
plt.ylim(1e-30, 1e-18)

plt.xlabel(r'frequency [Hz]')
plt.legend(loc='best')

plt.tight_layout()
plt.show()
