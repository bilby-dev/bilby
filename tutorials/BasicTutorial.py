import numpy as np
import pylab as plt

#%load_ext autoreload
#%autoreload 2
import peyote
import corner

import logging
logging.getLogger().addHandler(logging.StreamHandler())
logging.getLogger().setLevel('DEBUG')

"""
Define a source and simulate data
"""

print("creating signal injection")
time_duration = 1.
sampling_frequency = 4096.

simulation_parameters = dict(mass_1 = 36., mass_2 = 29., spin_1 = [0, 0, 0], spin_2 = [0, 0, 0],
                            luminosity_distance = 410., iota = 0., phase = 0., waveform_approximant = 'IMRPhenomPv2',
                            reference_frequency = 50., ra = 0, dec = 1, geocent_time = 0, psi=1)

source = peyote.source.BinaryBlackHole('BBH', sampling_frequency, time_duration)
hf_signal = source.frequency_domain_strain(simulation_parameters)


print("plotting signal")
plt.loglog(source.frequency_array, np.abs(hf_signal['plus']))
plt.show()

"""
# Simulate the data in H1
"""
print("Simulating data in H1")

IFO = peyote.detector.H1
hf_noise, frequencies = IFO.power_spectral_density.get_noise_realisation(
    sampling_frequency, time_duration)
IFO.set_data(frequency_domain_strain=hf_noise)
IFO.inject_signal(source, simulation_parameters)
IFO.set_spectral_densities(frequencies)
IFO.whiten_data()

"""
# Plot the noise and signal
"""
print("Plotting noise")

plt.loglog(frequencies, np.abs(hf_noise), lw=1.5, label='noise+signal')
plt.loglog(frequencies, np.abs(hf_signal['plus']), lw=0.8, label='signal')
plt.xlim(10, 1000)
plt.legend()
plt.xlabel(r'frequency')
plt.ylabel(r'strain')
plt.show()

"""
Search the data
"""
# %%time
print("Setting up likelihood")
likelihood = peyote.likelihood.likelihood([IFO], source)

print("Setting up priors")
prior = simulation_parameters.copy()
prior['mass_1'] = peyote.parameter.Parameter(
    'mass_1', prior=peyote.prior.Uniform(lower=35, upper=37),
    latex_label='$m_1$')
prior['mass_2'] = peyote.parameter.Parameter(
    'mass_2', prior=peyote.prior.Uniform(lower=28, upper=30),
    latex_label='$m_2$')

print("Running sampler")
result = peyote.run_sampler(likelihood, prior, 'nestle', npoints=100)

print("making corner plot")
truths = [simulation_parameters[k] for k in result.search_parameter_keys]
fig = corner.corner(result.samples, labels=result.labels,
                    truths=truths)
fig.show()

print("Done; you're welcome!")
