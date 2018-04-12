# coding: utf-8

# # GW150914 analysis

# Analyse GW150914 data using TUPAK

# In[1]:


import numpy as np
import pylab as plt

import peyote
import corner

import logging

logging.getLogger().addHandler(logging.StreamHandler())
logging.getLogger().setLevel('DEBUG')

import matplotlib.mlab as mlab

data_file = 'tutorial_data/GW150914_strain_data.npy'
time_series, strain_H1, strain_L1 = np.load(data_file)
time_duration = time_series[-1] - time_series[0]

time_of_event = 1126259462.44

sampling_frequency = np.int(peyote.utils.sampling_frequency(time_series))
NFFT = 4 * sampling_frequency
power_spectral_density_H1, frequency_series = mlab.psd(strain_H1, Fs=sampling_frequency, NFFT=NFFT)
power_spectral_density_L1, frequency_series = mlab.psd(strain_L1, Fs=sampling_frequency, NFFT=NFFT)

with open('150914_PSD_H1.txt', 'w+') as file:
    for f, p in zip(frequency_series, power_spectral_density_H1):
        file.write('{} {}\n'.format(f, p))
with open('150914_PSD_L1.txt', 'w+') as file:
    for f, p in zip(frequency_series, power_spectral_density_L1):
        file.write('{} {}\n'.format(f, p))

search_idxs = (time_series > time_of_event - 0.5) * (time_series < time_of_event + 0.5)
time_series = time_series[search_idxs]
strain_H1 = strain_H1[search_idxs]
strain_L1 = strain_L1[search_idxs]
time_duration = time_series[-1] - time_series[0]
H1 = peyote.detector.H1
H1.power_spectral_density = peyote.detector.PowerSpectralDensity(psd_file='./150914_PSD_H1.txt')
H1.set_data(sampling_frequency, time_duration,
            frequency_domain_strain=peyote.utils.nfft(strain_H1, sampling_frequency)[0])

L1 = peyote.detector.L1
L1.power_spectral_density = peyote.detector.PowerSpectralDensity(psd_file='./150914_PSD_L1.txt')
L1.set_data(sampling_frequency, time_duration,
            frequency_domain_strain=peyote.utils.nfft(strain_L1, sampling_frequency)[0])

IFOs = [H1, L1]

simulation_parameters = dict(
    spin_1=[0, 0, 0],
    spin_2=[0, 0, 0],
    luminosity_distance=410.,
    iota=2.97305,
    phase=1.145,
    waveform_approximant='IMRPhenomPv2',
    reference_frequency=50.,
    ra=1.375,
    dec=-1.2108,
    geocent_time=1126259642.413,
    psi=2.659)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = peyote.waveform_generator.WaveformGenerator(
    'BBH', sampling_frequency, time_duration, peyote.source.lal_binary_black_hole)
waveform_generator.set_values(simulation_parameters)
hf_signal = waveform_generator.frequency_domain_strain()

likelihood = peyote.likelihood.Likelihood(IFOs, waveform_generator)

prior = simulation_parameters.copy()

prior['mass_1'] = peyote.parameter.Parameter(
    'mass_1', prior=peyote.prior.Uniform(lower=32, upper=41),
    latex_label='$m_1$')
prior['mass_2'] = peyote.parameter.Parameter(
    'mass_2', prior=peyote.prior.Uniform(lower=25, upper=33),
    latex_label='$m_2$')

result = peyote.run_sampler(likelihood, prior, sampler='pymultinest',
                            n_live_points=400, verbose=True)

fig = corner.corner(result.samples)
fig.savefig('test')
