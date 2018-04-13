"""
WIP
"""

import numpy as np
import pylab as plt

import dynesty.plotting as dyplot
import corner
import peyote

peyote.setup_logger()

time_duration = 1.
sampling_frequency = 4096.


def delta_function_frequency_domain_strain(frequency_array, amplitude, peak_time, phase, ra, dec, geocent_time, psi):
    ht = {'plus': amplitude * np.sin(2 * np.pi * peak_time * frequency_array + phase) + 0j,
          'cross': amplitude * np.cos(2 * np.pi * peak_time * frequency_array + phase) + 0j}
    return ht


def gaussian_frequency_domain_strain(frequency_array, amplitude, sigma, ra, dec, geocent_time, psi):
    ht = {'plus': amplitude*np.exp(-frequency_array**2 * sigma**2/2 + 0j),
          'cross': amplitude*np.exp(-frequency_array**2 * sigma**2/2) + 0j}
    return ht


simulation_parameters = dict(amplitude=10, sigma=0.1,
                             ra=1.375,
                             dec=-1.2108,
                             geocent_time=1126259642.413,
                             psi=2.659)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = peyote.waveform_generator.WaveformGenerator('BBH', sampling_frequency, time_duration,
                                                                 gaussian_frequency_domain_strain)
waveform_generator.set_values(simulation_parameters)
hf_signal = waveform_generator.frequency_domain_strain()

# Simulate the data in H1
H1 = peyote.detector.H1
H1_hf_noise, frequencies = H1.power_spectral_density.get_noise_realisation(sampling_frequency, time_duration)
H1.set_data(sampling_frequency, time_duration, frequency_domain_strain=H1_hf_noise)
H1.inject_signal(waveform_generator)
H1.set_spectral_densities()

# Simulate the data in L1
L1 = peyote.detector.L1
L1_hf_noise, frequencies = L1.power_spectral_density.get_noise_realisation(sampling_frequency, time_duration)
L1.set_data(sampling_frequency, time_duration, frequency_domain_strain=L1_hf_noise)
L1.inject_signal(waveform_generator)
L1.set_spectral_densities()

IFOs = [H1, L1]

# Plot the noise and signal
fig, ax = plt.subplots()
plt.loglog(frequencies, np.abs(H1_hf_noise), lw=1.5, label='H1 noise+signal')
plt.loglog(frequencies, np.abs(L1_hf_noise), lw=1.5, label='L1 noise+signal')
plt.loglog(frequencies, np.abs(hf_signal['plus']), lw=0.8, label='signal')
plt.xlim(10, 1000)
plt.legend()
plt.xlabel(r'frequency')
plt.ylabel(r'strain')
fig.savefig('data')

likelihood = peyote.likelihood.Likelihood(IFOs, waveform_generator)

prior = simulation_parameters.copy()
prior['amplitude'] = peyote.parameter.Parameter('amplitude', prior=peyote.prior.Uniform(lower=9, upper=11),
                                                latex_label='$A$')
prior['sigma'] = peyote.parameter.Parameter('sigma', prior=peyote.prior.Uniform(lower=0.05, upper=0.2),
                                                latex_label='$f$')

result = peyote.run_sampler(likelihood, prior, sampler='nestle', verbose=True)

truths = [simulation_parameters[x] for x in result.search_parameter_keys]
fig = corner.corner(result.samples, truths=truths, labels=result.search_parameter_keys)
fig.savefig('corner')

fig, axes = dyplot.traceplot(result['sampler_output'])
fig.savefig('trace')
