from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import peyote
import corner

from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design

from scipy import signal

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
t0 = time_series[0]
sampling_frequency = np.int(peyote.utils.get_sampling_frequency(time_series))
time_of_event = 1126259462.413

strain_H1 = TimeSeries(strain_H1, sample_rate=sampling_frequency,
                       t0=t0, name='H1')
strain_L1 = TimeSeries(strain_L1, sample_rate=sampling_frequency,
                       t0=t0, name='L1')

# Create and save PSDs
NFFT = 4 * sampling_frequency
psd_start = time_series[0]
psd_end = time_of_event - 5
psd_H1, psd_frequencies = mlab.psd(strain_H1.crop(psd_start, psd_end).value,
                                   Fs=sampling_frequency, NFFT=NFFT)
psd_L1, psd_frequencies = mlab.psd(strain_L1.crop(psd_start, psd_end).value,
                                   Fs=sampling_frequency, NFFT=NFFT)
with open('{}/GW150914_PSD_H1.txt'.format(outdir), 'w+') as file:
    for f, p in zip(psd_frequencies, psd_H1):
        file.write('{} {}\n'.format(f, p))
with open('{}/GW150914_PSD_L1.txt'.format(outdir), 'w+') as file:
    for f, p in zip(psd_frequencies, psd_L1):
        file.write('{} {}\n'.format(f, p))


# Low pass filter
#bp = filter_design.lowpass(strain_H1.sample_rate/4., strain_H1.sample_rate)
#strain_H1 = strain_H1.filter(bp, filtfilt=True)
#strain_H1 = strain_H1.crop(*strain_H1.span.contract(1))
#strain_L1 = strain_L1.filter(bp, filtfilt=True)
#strain_L1 = strain_L1.crop(*strain_L1.span.contract(1))

# Band pass filter
#bp = filter_design.bandpass(50, 250, strain_H1.sample_rate)
#strain_H1 = strain_H1.filter(bp, filtfilt=True)
#strain_H1 = strain_H1.crop(*strain_H1.span.contract(1))
#strain_L1 = strain_L1.filter(bp, filtfilt=True)
#strain_L1 = strain_L1.crop(*strain_L1.span.contract(1))

# Cut out 4 second period around the data and make IFOs with this data
strain_H1_crop = strain_H1.crop(time_of_event-2, time_of_event+2)
strain_L1_crop = strain_L1.crop(time_of_event-2, time_of_event+2)
time_series = strain_L1_crop.times.value
time_duration = time_series[-1] - time_series[0]

# Apply Tukey window
N = len(time_series)
strain_H1_crop = strain_H1_crop * signal.tukey(N)
strain_L1_crop = strain_L1_crop * signal.tukey(N)


# Plot time-domain data (shift and flip L1 to lie on top)
fig, ax = plt.subplots()
ax.plot(strain_H1_crop.times.value-time_of_event,
        strain_H1_crop.value, label='H1')
ax.plot(strain_L1_crop.times.value-time_of_event+0.0069,
        -1*strain_L1_crop.value, label='L1')
#ax.set_xlim(-0.2, 0.1)
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

# Plot a simulated signal
#simulation_parameters = dict(
#    mass_1=36., mass_2=29., spin_1=[0, 0, 0], spin_2=[0, 0, 0],
#    luminosity_distance=410., iota=0.4, phase=1.3,
#    waveform_approximant='IMRPhenomPv2', reference_frequency=50.,
#    ra=1.375, dec=-1.2108, geocent_time=time_of_event, psi=2.659)
#waveform_generator = peyote.waveform_generator.WaveformGenerator(
#    'BBH', sampling_frequency, time_duration, peyote.source.lal_binary_black_hole)
#waveform_generator.set_values(simulation_parameters)
#hf_signal = waveform_generator.frequency_domain_strain()
#H1.inject_signal(waveform_generator)
#L1.inject_signal(waveform_generator)

# Plot the data and PSDs
fig, axes = plt.subplots(nrows=2, figsize=(8, 8))
for ax, IFO in zip(axes, IFOs):
    #ax.loglog(IFO.frequency_array, np.abs(hf_signal['plus']), '-r')
    ax.loglog(IFO.frequency_array, np.abs(IFO.data), '-C0', label=IFO.name, lw=1.5)
    ax.loglog(IFO.frequency_array,
              IFO.amplitude_spectral_density_array, '-C1', lw=0.5,
              label=IFO.name+' PSD')
    ax.grid('on')
    ax.set_ylabel(r'amplitude spectral density [strain/$\sqrt{\rm Hz}$]')
    ax.set_xlabel(r'frequency [Hz]')
    #ax.set_ylim(-1e-22, 1e-22)
    ax.set_xlim(20, 2000)
    ax.legend(loc='best')
fig.savefig('{}/frequency_domain_data.png'.format(outdir))


# Create the waveformgenerator
waveformgenerator = peyote.waveform_generator.WaveformGenerator(
    'BBH', sampling_frequency, time_duration, peyote.source.lal_binary_black_hole)

# Define the prior
prior = dict(spin_1=[0, 0, 0], spin_2=[0, 0, 0], luminosity_distance=410.,
             iota=2.97305, phase=1.145, waveform_approximant='IMRPhenomPv2',
             reference_frequency=50., ra=1.375, dec=-1.2108,
             geocent_time=time_of_event, psi=2.659, mass_1=36, mass_2=29)
prior['mass_1'] = peyote.parameter.Parameter(
    'mass_1', prior=peyote.prior.Uniform(lower=32, upper=41),
    latex_label='$m_1$')
prior['mass_2'] = peyote.parameter.Parameter(
    'mass_2', prior=peyote.prior.Uniform(lower=20, upper=30),
    latex_label='$m_2$')

# Define a likelihood
likelihood = peyote.likelihood.Likelihood(IFOs, waveformgenerator)

# Run the sampler
result = peyote.run_sampler(likelihood, prior, sampler='pymultinest',
                            n_live_points=100, verbose=True, resume=False,
                            outdir=outdir)

fig = corner.corner(result.samples, labels=result.search_parameter_keys)
fig.savefig('{}/corner.png'.format(outdir))

#import dynesty.plotting as dyplot
#fig, axes = dyplot.traceplot(result['sampler_output'])
#fig.savefig('trace')
