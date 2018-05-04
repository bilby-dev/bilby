import numpy as np
import pylab as plt
import peyote

peyote.utils.setup_logger()

time_duration = 4.
sampling_frequency = 2048.
outdir = 'outdir'

injection_parameters = dict(mass_1=36., mass_2=29., a_1=0, a_2=0, tilt_1=0, tilt_2=0, phi_1=0, phi_2=0,
                            luminosity_distance=2500., iota=0.4, psi=2.659, phase=1.3, geocent_time=1126259642.413,
                            waveform_approximant='IMRPhenomPv2', reference_frequency=50., ra=1.375, dec=-1.2108)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = peyote.waveform_generator.WaveformGenerator(
    sampling_frequency=sampling_frequency, time_duration=time_duration,
    frequency_domain_source_model=peyote.source.lal_binary_black_hole,
    parameters=injection_parameters)
hf_signal = waveform_generator.frequency_domain_strain()

# Set up interferometers.
IFOs = [peyote.detector.get_inteferometer_with_fake_noise_and_injection(
    name, injection_polarizations=hf_signal, injection_parameters=injection_parameters,
    sampling_frequency=sampling_frequency, time_duration=time_duration, outdir=outdir)
    for name in ['H1', 'L1', 'V1']]

# Set up likelihood
likelihood = peyote.likelihood.Likelihood(IFOs, waveform_generator)

# Set up prior
priors = dict()
# These parameters will not be sampled
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_1', 'phi_2', 'phase', 'psi', 'iota', 'ra', 'dec', 'geocent_time']:
    priors[key] = injection_parameters[key]

# Run sampler
result = peyote.sampler.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty',
                                    label='BasicTutorial', use_ratio=True, npoints=500, verbose=True,
                                    injection_parameters=injection_parameters, outdir=outdir)
result.plot_corner()
result.plot_walks()
result.plot_distributions()
print(result)
