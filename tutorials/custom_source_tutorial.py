"""
WIP
"""

import numpy as np
import pylab as plt

import dynesty.plotting as dyplot
import corner
import peyote

peyote.utils.setup_logger()


def generate_and_plot_data(waveform_generator):
    hf_signal = waveform_generator.frequency_domain_strain()
    # Simulate the data in H1
    H1 = peyote.detector.H1
    H1_hf_noise, frequencies = H1.power_spectral_density. \
        get_noise_realisation(waveform_generator.sampling_frequency,
                              waveform_generator.time_duration)
    H1.set_data(waveform_generator.sampling_frequency,
                waveform_generator.time_duration,
                frequency_domain_strain=H1_hf_noise)
    H1.inject_signal(waveform_generator)
    H1.set_spectral_densities()
    # Simulate the data in L1
    L1 = peyote.detector.L1
    L1_hf_noise, frequencies = L1.power_spectral_density. \
        get_noise_realisation(waveform_generator.sampling_frequency,
                              waveform_generator.time_duration)
    L1.set_data(waveform_generator.sampling_frequency,
                waveform_generator.time_duration,
                frequency_domain_strain=L1_hf_noise)
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
    return IFOs


def delta_function_frequency_domain_strain(frequency_array, amplitude,
                                           peak_time, phase, ra, dec,
                                           geocent_time, psi):
    ht = {'plus': amplitude * np.sin(2 * np.pi * peak_time * frequency_array + phase),
          'cross': amplitude * np.cos(2 * np.pi * peak_time * frequency_array + phase)}
    return ht


def gaussian_frequency_domain_strain(frequency_array, amplitude, mu, sigma,
                                     ra, dec, geocent_time, psi):
    ht = {'plus': amplitude * np.exp(-(mu - frequency_array) ** 2 / sigma ** 2 / 2),
          'cross': amplitude * np.exp(-(mu - frequency_array) ** 2 / sigma ** 2 / 2)}
    return ht


simulation_parameters = dict(amplitude=1e-21,
                             mu=100,
                             sigma=1,
                             ra=1.375,
                             dec=-1.2108,
                             geocent_time=1126259642.413,
                             psi=2.659)
sampling_parameters = peyote.parameter.PriorFactory.parse_floats_to_parameters(simulation_parameters)

wg = peyote.waveform_generator.WaveformGenerator(
     source_model=gaussian_frequency_domain_strain,
     parameters=simulation_parameters)

IFOs = generate_and_plot_data(wg)

likelihood = peyote.likelihood.Likelihood(IFOs, wg)

sampling_parameters['amplitude'].prior = peyote.prior.Uniform(lower=0.9 * 1e-21, upper=1.1 * 1e-21)
sampling_parameters['sigma'].prior = peyote.prior.Uniform(lower=0, upper=10)
sampling_parameters['mu'].prior = peyote.prior.Uniform(lower=50, upper=200)

result = peyote.sampler.run_sampler(likelihood, priors=sampling_parameters, verbose=True)

#
# Make some nice plots
#

truths = [simulation_parameters[x] for x in result.search_parameter_keys]
corner_plot = corner.corner(result.samples, truths=truths, labels=result.search_parameter_keys)
corner_plot.savefig('corner')

trace_plot, axes = dyplot.traceplot(result['sampler_output'])
trace_plot.savefig('trace')
