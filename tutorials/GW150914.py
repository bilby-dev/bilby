from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import peyote
import corner

peyote.setup_logger()

event = 'GW150914'
outdir = 'GW150914_results'
if os.path.isdir(outdir) is False:
    os.mkdir(outdir)

# Load in the data
data_file = 'tutorial_data/GW150914_strain_data.npy'
if os.path.isfile(data_file) is False:
    os.system('python get_LOSC_event_data.py -e GW150914 -o tutorial_data')
time_series, strain_H1, strain_L1 = np.load(data_file)
time_duration = time_series[-1] - time_series[0]
time_of_event = 1126259462.44

# Create and save PSDs
sampling_frequency = np.int(peyote.utils.get_sampling_frequency(time_series))
NFFT = 4 * sampling_frequency
psd_H1, psd_frequencies = mlab.psd(strain_H1, Fs=sampling_frequency, NFFT=NFFT)
psd_L1, psd_frequencies = mlab.psd(strain_L1, Fs=sampling_frequency, NFFT=NFFT)
with open('GW150914_PSD_H1.txt', 'w+') as file:
    for f, p in zip(psd_frequencies, psd_H1):
        file.write('{} {}\n'.format(f, p))
with open('GW150914_PSD_L1.txt', 'w+') as file:
    for f, p in zip(psd_frequencies, psd_L1):
        file.write('{} {}\n'.format(f, p))

# Cut out 1 second period around the data and make IFOs with this data
search_idxs = (time_series > time_of_event - 0.5) * (time_series < time_of_event + 0.5)
time_series = time_series[search_idxs]
strain_H1 = strain_H1[search_idxs]
strain_L1 = strain_L1[search_idxs]
time_duration = time_series[-1] - time_series[0]
H1 = peyote.detector.H1
H1.power_spectral_density = peyote.detector.PowerSpectralDensity(
    psd_file='./150914_PSD_H1.txt')
H1.set_data(sampling_frequency, time_duration,
            frequency_domain_strain=peyote.utils.nfft(
                strain_H1, sampling_frequency)[0])
L1 = peyote.detector.L1
L1.power_spectral_density = peyote.detector.PowerSpectralDensity(
    psd_file='./150914_PSD_L1.txt')
L1.set_data(sampling_frequency, time_duration,
            frequency_domain_strain=peyote.utils.nfft(
                strain_L1, sampling_frequency)[0])
IFOs = [H1, L1]

# Plot the data and PSDs
fig, axes = plt.subplots(nrows=2, figsize=(8, 8))
for ax, IFO in zip(axes, IFOs):
    ax.loglog(IFO.frequency_array, IFO.data, '-C0', label=IFO.name, lw=1.5)
    ax.loglog(IFO.frequency_array,
              np.abs(IFO.amplitude_spectral_density_array), '-C1', lw=0.5,
              label=IFO.name+' PSD')
    ax.grid('on')
    ax.set_ylabel(r'amplitude spectral density [strain/$\sqrt{\rm Hz}$]')
    ax.set_xlabel(r'frequency [Hz]')
    ax.set_ylim(1e-24, 1e-19)
    ax.set_xlim(20, 2000)
    ax.legend(loc='best')
fig.savefig('{}/frequency_domain_data.png'.format(outdir))

# Create the waveformgenerator
waveformgenerator = peyote.source.WaveformGenerator(
    'BBH', sampling_frequency, time_duration, peyote.source.LALBinaryBlackHole)

# Define the prior
prior = dict(spin_1=[0, 0, 0], spin_2=[0, 0, 0], luminosity_distance=410.,
             iota=2.97305, phase=1.145, waveform_approximant='IMRPhenomPv2',
             reference_frequency=50., ra=1.375, dec=-1.2108,
             geocent_time=1126259642.413, psi=2.659, mass_1=32, mass_2=32)
prior['mass_1'] = peyote.parameter.Parameter(
    'mass_1', prior=peyote.prior.Uniform(lower=32, upper=41),
    latex_label='$m_1$')
prior['mass_2'] = peyote.parameter.Parameter(
    'mass_2', prior=peyote.prior.Uniform(lower=25, upper=33),
    latex_label='$m_2$')

# Define a likelihood
likelihood = peyote.likelihood.Likelihood(IFOs, waveformgenerator)

# Run the sampler
result = peyote.run_sampler(likelihood, prior, sampler='pymultinest',
                            n_live_points=400, verbose=True, outdir=outdir)

fig = corner.corner(result.samples)
fig.savefig('test')
