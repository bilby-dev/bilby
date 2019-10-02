import matplotlib.pyplot as plt
import numpy as np

import bilby.gw.prior


def condition_function(reference_params, mass_1):
    return dict(minimum=reference_params['minimum'], maximum=mass_1)


mass_1_min = 5
mass_1_max = 50

mass_1 = bilby.core.prior.PowerLaw(alpha=-2.5, minimum=mass_1_min, maximum=mass_1_max, name='mass_1',
                                   latex_label='$m_1$')
mass_2 = bilby.core.prior.ConditionalPowerLaw(alpha=-2.5, minimum=mass_1_min, maximum=mass_1_max, name='mass_2',
                                              latex_label='$m_2$', condition_func=condition_function)

correlated_dict = bilby.core.prior.ConditionalPriorDict(dictionary=dict(mass_1=mass_1, mass_2=mass_2))

res = correlated_dict.sample(100000)

plt.hist(res['mass_1'], bins='fd', alpha=0.6, density=True, label='Sampled')
plt.plot(np.linspace(2, 50, 200), correlated_dict['mass_1'].prob(np.linspace(2, 50, 200)), label='Powerlaw prior')
plt.xlabel('$m_1$')
plt.ylabel('$p(m_1)$')
plt.loglog()
plt.legend()
plt.tight_layout()
plt.show()
plt.clf()


plt.hist(res['mass_2'], bins='fd', alpha=0.6, density=True, label='Sampled')
plt.xlabel('$q$')
plt.ylabel('$p(m_2 | m_1)$')
plt.loglog()
plt.legend()
plt.tight_layout()
plt.show()
plt.clf()


duration = 4.
sampling_frequency = 2048.
outdir = 'outdir'
label = 'conditional_prior'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

np.random.seed(88170235)

injection_parameters = dict(
    mass_1=9., mass_2=9, a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=1800., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=50., minimum_frequency=20.)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments)

ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)

priors = bilby.core.prior.ConditionalPriorDict()
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'psi', 'ra',
            'dec', 'geocent_time', 'phase', 'luminosity_distance', 'theta_jn']:
    priors[key] = injection_parameters[key]
priors['mass_1'] = mass_1
priors['mass_2'] = mass_2

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator)

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=300,
    injection_parameters=injection_parameters, outdir=outdir, label=label)

# Make a corner plot.
result.plot_corner()
