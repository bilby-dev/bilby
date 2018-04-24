from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import peyote
import corner

from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design

from scipy import signal

peyote.utils.setup_logger()

outdir = 'GW150914_results'
time_of_event = 1126259462.422
T = 4
alpha = 0.1 / T  # Tukey window roll off

strain_H1 = TimeSeries.fetch_open_data(
        'H1', time_of_event-1100, time_of_event+5, cache=True, version=1)
strain_L1 = TimeSeries.fetch_open_data(
        'L1', time_of_event-1100, time_of_event+5, cache=True, version=1)

sampling_frequency = int(strain_L1.sample_rate.value)

# Low pass filter
bp = filter_design.lowpass(2048.-1, strain_H1.sample_rate)
strain_H1 = strain_H1.filter(bp, filtfilt=True)
strain_H1 = strain_H1.crop(*strain_H1.span.contract(1))
strain_L1 = strain_L1.filter(bp, filtfilt=True)
strain_L1 = strain_L1.crop(*strain_L1.span.contract(1))

# Create and save PSDs
NFFT = T * sampling_frequency
psd_start = time_of_event - 1024 - T
psd_end = time_of_event - T
window = signal.tukey(NFFT, alpha=alpha)
sides = 'onesided'
psd_H1, psd_frequencies = mlab.psd(strain_H1.crop(psd_start, psd_end).value,
                                   Fs=sampling_frequency, NFFT=NFFT,
                                   window=window)
psd_L1, psd_frequencies = mlab.psd(strain_L1.crop(psd_start, psd_end).value,
                                   Fs=sampling_frequency, NFFT=NFFT,
                                   window=window)
with open('{}/GW150914_PSD_H1.txt'.format(outdir), 'w+') as file:
    for f, p in zip(psd_frequencies, psd_H1):
        file.write('{} {}\n'.format(f, p))
with open('{}/GW150914_PSD_L1.txt'.format(outdir), 'w+') as file:
    for f, p in zip(psd_frequencies, psd_L1):
        file.write('{} {}\n'.format(f, p))

strain_H1_crop = strain_H1.crop(time_of_event-T/2, time_of_event+T/2)
strain_L1_crop = strain_L1.crop(time_of_event-T/2, time_of_event+T/2)
time_series = strain_L1_crop.times.value
time_duration = time_series[-1] - time_series[0]

# Apply Tukey window
N = len(time_series)
strain_H1_crop = strain_H1_crop * signal.tukey(N, alpha=alpha)
strain_L1_crop = strain_L1_crop * signal.tukey(N, alpha=alpha)

# Plot time-domain data (shift and flip L1 to lie on top)
fig, ax = plt.subplots()
ax.plot(strain_H1_crop.times.value-time_of_event,
        strain_H1_crop.value, label='H1')
ax.plot(strain_L1_crop.times.value-time_of_event+0.0069,
        -1*strain_L1_crop.value, label='L1')
axS = ax.twinx()
axS.plot(time_series-time_of_event, window, color='k')
ax.legend()
fig.savefig('{}/time_domain_data.png'.format(outdir))

# Create IFOs with the data
H1 = peyote.detector.H1
H1.power_spectral_density = peyote.detector.PowerSpectralDensity(
    psd_file='{}/GW150914_PSD_H1.txt'.format(outdir))
H1.set_data(sampling_frequency, time_duration,
            frequency_domain_strain=peyote.utils.nfft(
                strain_H1_crop.value, sampling_frequency)[0])

L1 = peyote.detector.L1
L1.power_spectral_density = peyote.detector.PowerSpectralDensity(
    psd_file='{}/GW150914_PSD_L1.txt'.format(outdir))
L1.set_data(sampling_frequency, time_duration,
            frequency_domain_strain=peyote.utils.nfft(
                strain_L1_crop.value, sampling_frequency)[0])
IFOs = [H1, L1]

# Plot the data and PSDs
fig, axes = plt.subplots(nrows=2, figsize=(8, 8))
for ax, IFO in zip(axes, IFOs):
    ax.loglog(IFO.frequency_array, np.abs(IFO.data), '-C0', label=IFO.name, lw=1.5)
    ax.loglog(IFO.frequency_array,
              IFO.amplitude_spectral_density_array, '-C1', lw=0.5,
              label=IFO.name+' PSD')
    ax.grid('on')
    ax.set_ylabel(r'amplitude spectral density [strain/$\sqrt{\rm Hz}$]')
    ax.set_xlabel(r'frequency [Hz]')
    ax.set_xlim(20, 2000)
    ax.legend(loc='best')
fig.savefig('{}/frequency_domain_data.png'.format(outdir))

# Define the prior
prior = dict(spin11=0, spin12=0, spin13=0, spin21=0, spin22=0, spin23=0,
             luminosity_distance=410., iota=2.97305, phase=1.145,
             waveform_approximant='IMRPhenomPv2', reference_frequency=50.,
             ra=1.375, dec=-1.2108, geocent_time=time_of_event, psi=2.659,
             mass_1=36, mass_2=29)
prior = peyote.parameter.PriorFactory.parse_floats_to_parameters(prior)
prior['mass_1'] = peyote.parameter.PriorFactory(
    'mass_1', prior=peyote.prior.Uniform(lower=35, upper=41),
    latex_label='$m_1$')
prior['mass_2'] = peyote.parameter.PriorFactory(
    'mass_2', prior=peyote.prior.Uniform(lower=20, upper=35),
    latex_label='$m_2$')
prior['geocent_time'] = peyote.parameter.PriorFactory(
    'mass_2', prior=peyote.prior.Uniform(
        lower=time_of_event-0.1, upper=time_of_event+0.1))

# Create the waveformgenerator
waveformgenerator = peyote.waveform_generator.WaveformGenerator(
     peyote.source.lal_binary_black_hole, sampling_frequency, time_duration,
     parameters=prior)

# Define a likelihood
likelihood = peyote.likelihood.Likelihood(IFOs, waveformgenerator)

# Run the sampler
result = peyote.sampler.run_sampler(
    likelihood, sampler='pymultinest', n_live_points=200, verbose=True,
    resume=False, outdir=outdir)

fig = corner.corner(result.samples, labels=result.search_parameter_keys)
fig.savefig('{}/corner.png'.format(outdir))
