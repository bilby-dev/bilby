import tupak

import matplotlib.pyplot as plt
import numpy as np


def frequency_domain_sine_gaussian(f, A, f0, tau, phi0, geocent_time, ra, dec, psi):
    arg = -(np.pi * tau * (f-f0))**2 + 1j * phi0
    plus = np.sqrt(np.pi) * A * tau * np.exp(arg) / 2.
    cross = plus * np.exp(1j*np.pi/2)
    return {'plus': plus, 'cross': cross}


def time_domain_sine_gaussian(t, A, t0, f0, tau, phi0, geocent_time, ra, dec, psi):
    arg = -(-(t-t0)/tau)**2
    plus = A * np.exp(arg) *np.cos(2*np.pi*f0*t + phi0)
    cross = plus * np.exp(1j*np.pi/2)
    return {'plus': plus, 'cross': cross}



parameters = dict()
parameters['A'] = 10000
parameters['f0'] = 5
parameters['t0'] = 10
parameters['tau'] = 3
parameters['geocent_time'] = 0
parameters['phi0'] = 0
parameters['ra'] = 0
parameters['dec'] = 0
parameters['psi'] = 0

wg = tupak.waveform_generator.WaveformGenerator(time_domain_source_model=time_domain_sine_gaussian, time_duration=2000, sampling_frequency=1000)
wg.parameters = parameters
plt.plot(wg.frequency_array, wg.frequency_domain_strain()['plus'])
plt.xlim(4, 6)
plt.show()
plt.plot(wg.frequency_array, wg.frequency_domain_strain()['cross'])
plt.xlim(4, 6)
plt.show()
plt.plot(wg.time_array, wg.time_domain_strain()['plus'])
plt.xlim(0, 20)
plt.show()
plt.plot(wg.time_array, wg.time_domain_strain()['cross'])
plt.xlim(0, 20)
plt.show()
