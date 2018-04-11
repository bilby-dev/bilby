import numpy as np
import pylab as plt

import peyote
import dynesty.plotting as dyplot
import corner
import lal
import lalsimulation as lalsim

peyote.setup_logger()


def BBH(frequency_array, mass_1, mass_2, luminosity_distance, spin_1, spin_2,
        iota, phase, waveform_approximant, reference_frequency, ra, dec,
        geocent_time, psi):
    luminosity_distance = luminosity_distance * 1e6 * lal.PC_SI
    mass_1 = mass_1 * lal.MSUN_SI
    mass_2 = mass_2 * lal.MSUN_SI

    longitude_ascending_nodes = 0.0
    eccentricity = 0.0
    meanPerAno = 0.0

    waveform_dictionary = lal.CreateDict()

    approximant = lalsim.GetApproximantFromString(waveform_approximant)

    frequency_minimum = 20
    frequency_maximum = frequency_array[-1]
    delta_frequency = frequency_array[1] - frequency_array[0]

    hplus, hcross = lalsim.SimInspiralChooseFDWaveform(
        mass_1, mass_2, spin_1[0], spin_1[1], spin_1[2], spin_2[0], spin_2[1],
        spin_2[2], luminosity_distance, iota, phase,
        longitude_ascending_nodes, eccentricity, meanPerAno, delta_frequency,
        frequency_minimum, frequency_maximum, reference_frequency,
        waveform_dictionary, approximant)

    h_plus = hplus.data.data
    h_cross = hcross.data.data

    return {'plus': h_plus, 'cross': h_cross}


time_duration = 1.
sampling_frequency = 4096.

simulation_parameters = dict(
    mass_1=36., mass_2=29.,
    spin_1=[0, 0, 0],
    spin_2=[0, 0, 0],
    luminosity_distance=100.,
    iota=0.4,
    phase=1.3,
    waveform_approximant='IMRPhenomPv2',
    reference_frequency=50.,
    ra=1.375,
    dec=-1.2108,
    geocent_time=1126259642.413,
    psi=2.659
    )

waveformgenerator = peyote.source.WaveformGenerator('BBH', sampling_frequency, time_duration, BBH)
hf_signal = waveformgenerator.frequency_domain_strain(simulation_parameters)

# Simulate the data in H1
H1 = peyote.detector.H1
H1_hf_noise, frequencies = H1.power_spectral_density.get_noise_realisation(
    sampling_frequency, time_duration)
H1.set_data(sampling_frequency, time_duration, frequency_domain_strain=H1_hf_noise)
H1.inject_signal(waveformgenerator, simulation_parameters)
H1.set_spectral_densities()

# Simulate the data in L1
L1 = peyote.detector.L1
L1_hf_noise, frequencies = L1.power_spectral_density.get_noise_realisation(
    sampling_frequency, time_duration)
L1.set_data(sampling_frequency, time_duration, frequency_domain_strain=L1_hf_noise)
L1.inject_signal(waveformgenerator, simulation_parameters)
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

likelihood = peyote.likelihood.likelihood(IFOs, waveformgenerator)

prior = simulation_parameters.copy()
prior['mass_1'] = peyote.parameter.Parameter(
    'mass_1', prior=peyote.prior.Uniform(lower=35, upper=37),
    latex_label='$m_1$')
#prior['mass_2'] = peyote.parameter.Parameter(
#    'mass_2', prior=peyote.prior.Uniform(lower=27, upper=31),
#    latex_label='$m_2$')
#prior['iota'] = peyote.parameter.Parameter(
#    'iota', prior=peyote.prior.Uniform(lower=0, upper=np.pi),
#    latex_label='$iota$')
#prior['phase'] = peyote.parameter.Parameter(
#    'phase', prior=peyote.prior.Uniform(lower=0, upper=np.pi))
#prior['geocent_time'] = peyote.parameter.Parameter(
#    'geocent_time', prior=peyote.prior.Uniform(
#        lower=simulation_parameters['geocent_time']-100,
#        upper=simulation_parameters['geocent_time']+100))
prior['luminosity_distance'] = peyote.parameter.Parameter(
    'luminosity_distance', prior=peyote.prior.Uniform(lower=30, upper=200),
    latex_label='$d_L$')

result = peyote.run_sampler(likelihood, prior, sampler='nestle',
                            n_live_points=200, verbose=True)

truths = [simulation_parameters[x] for x in result.search_parameter_keys]
fig = corner.corner(result.samples, truths=truths, labels=result.search_parameter_keys)
fig.savefig('corner')

fig, axes = dyplot.traceplot(result['sampler_output'])
fig.savefig('trace')
