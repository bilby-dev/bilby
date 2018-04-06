import numpy as np
import pylab as plt

import peyote

import logging
logging.getLogger().addHandler(logging.StreamHandler())
logging.getLogger().setLevel('DEBUG')


time_duration = 1.
sampling_frequency = 4096.

simulation_parameters = dict(
    mass_1 = 36.,
    mass_2 = 29.,
    spin_1 = [0, 0, 0], 
    spin_2 = [0, 0, 0],
    luminosity_distance = 410.,
    iota = 0., 
    phase = 0., 
    waveform_approximant = 'IMRPhenomPv2',
    reference_frequency = 50.,
    ra = 0,
    dec = 1,
    geocent_time = 0,
    psi=1
    )

source = peyote.source.BinaryBlackHole('BBH', sampling_frequency, time_duration)
hf_signal = source.frequency_domain_strain(simulation_parameters)

plt.loglog(source.frequency_array, np.abs(hf_signal['plus']))
plt.show()


# In[3]:


# Simulate the data in H1
IFO = peyote.detector.H1
hf_noise, frequencies = IFO.power_spectral_density.get_noise_realisation(
    sampling_frequency, time_duration)
IFO.set_data(frequency_domain_strain=hf_noise)
IFO.inject_signal(source, simulation_parameters)
IFO.set_spectral_densities(frequencies)
IFO.whiten_data()

# Plot the noise and signal
plt.loglog(frequencies, np.abs(hf_noise), lw=1.5, label='noise+signal')
plt.loglog(frequencies, np.abs(hf_signal['plus']), lw=0.8, label='signal')
plt.xlim(10, 1000)
plt.legend()
plt.xlabel(r'frequency')
plt.ylabel(r'strain')
plt.show()


likelihood = peyote.likelihood.likelihood([IFO], source)

prior = simulation_parameters.copy()
prior['mass_1'] = peyote.parameter.Parameter(
    'mass_1', prior=peyote.prior.Uniform(lower=35, upper=37),
    latex_label='$m_1$')
prior.pop('mass_2')

result = peyote.run_sampler(likelihood, prior, sampler='pymultinest',
                            verbose=True)
