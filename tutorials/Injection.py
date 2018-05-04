"""
Tutorial to show signal injection using new features of detector.py
"""
from __future__ import division, print_function
import numpy as np
import peyote

peyote.utils.setup_logger()

outdir = 'outdir'
label = 'injection'

# Create the waveform generator
waveform_generator = peyote.waveform_generator.WaveformGenerator(
    peyote.source.lal_binary_black_hole, sampling_frequency=4096, time_duration=4,
    parameters={'reference_frequency': 50.0, 'waveform_approximant': 'IMRPhenomPv2'})

# Define the prior
prior = dict()
prior['mass_1'] = peyote.prior.Uniform(10, 80, 'mass_1')
prior['mass_2'] = peyote.prior.Uniform(10, 80, 'mass_2')
# Merger time is some time in 2018, shame LIGO will never see it...
time_of_event = np.random.uniform(1198800018, 1230336018)
prior['geocent_time'] = peyote.prior.Uniform(time_of_event-0.01, time_of_event+0.01, name='geocent_time')
peyote.prior.fill_priors(prior, waveform_generator)

# Create signal injection
injection_parameters = {name: prior[name].sample() for name in prior}
print(injection_parameters)
for parameter in injection_parameters:
    waveform_generator.parameters[parameter] = injection_parameters[parameter]
injection_polarizations = waveform_generator.frequency_domain_strain()

# Create interferometers and inject signal
IFOs = [peyote.detector.get_inteferometer_with_fake_noise_and_injection(
    name, injection_polarizations=injection_polarizations, injection_parameters=injection_parameters,
    sampling_frequency=4096, time_duration=4, outdir=outdir) for name in ['H1', 'L1', 'V1', 'GEO600']]

# Define a likelihood
likelihood = peyote.likelihood.Likelihood(IFOs, waveform_generator)

# Run the sampler
result = peyote.sampler.run_sampler(
    likelihood, prior, label='injection', sampler='nestle',
    npoints=200, resume=False, outdir=outdir, use_ratio=True, injection_parameters=injection_parameters)
truth = [injection_parameters[parameter] for parameter in result.search_parameter_keys]
result.plot_corner(truth=truth)
print(result)
