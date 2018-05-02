import numpy as np
import pylab as plt
import peyote

peyote.utils.setup_logger()

time_duration = 1.
sampling_frequency = 4096.

injection_parameters = dict(mass_1=36., mass_2=29., a_1=0, a_2=0, tilt_1=0, tilt_2=0, phi_1=0, phi_2=0,
                            luminosity_distance=2500., iota=0.4, psi=2.659, phase=1.3, geocent_time=1126259642.413,
                            waveform_approximant='IMRPhenomPv2', reference_frequency=50., ra=1.375, dec=-1.2108)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = peyote.waveform_generator.WaveformGenerator(
    sampling_frequency=sampling_frequency,
    time_duration=time_duration,
    frequency_domain_source_model=peyote.source.lal_binary_black_hole,
    parameters=injection_parameters)
hf_signal = waveform_generator.frequency_domain_strain()

# Set up interferometers.
IFOs = [peyote.detector.get_empty_interferometer(name) for name in ['H1', 'L1', 'V1']]

for IFO in IFOs:
    IFO.set_data(sampling_frequency=sampling_frequency, duration=time_duration, from_power_spectral_density=True)
    IFO.inject_signal(waveform_polarizations=hf_signal, parameters=injection_parameters)

# Plot the noise and signal
fig, ax = plt.subplots()
for IFO in IFOs:
    plt.loglog(IFO.frequency_array, np.abs(IFO.data), lw=1.5, label='{} noise+signal'.format(IFO.name))
    plt.loglog(IFO.frequency_array, IFO.amplitude_spectral_density_array, lw=1.5, label='{} ASD'.format(IFO.name))
plt.xlim(10, 1000)
plt.legend()
plt.xlabel(r'frequency')
plt.ylabel(r'strain')
fig.savefig('data')

# Set up likelihood
likelihood = peyote.likelihood.Likelihood(IFOs, waveform_generator)

# Set up prior
priors = {}
priors['mass_1'] = peyote.prior.Uniform(
    lower=35, upper=37, name='mass_1', latex_label='$m_1$')
priors['luminosity_distance'] = peyote.prior.create_default_prior('luminosity_distance')
for key in ['mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_1', 'phi_2', 'phase', 'psi', 'iota', 'ra',
            'dec', 'geocent_time']:
    priors[key] = injection_parameters[key]

# Run sampler
result = peyote.sampler.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty',
                                    label='BasicTutorial', use_ratio=True, npoints=500, verbose=True,
                                    injection_parameters=injection_parameters)
result.plot_corner()
result.plot_walks()
result.plot_distributions()
print(result)
