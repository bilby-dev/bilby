#!/bin/python
"""
Tutorial to demonstrate how to specify the prior distributions used for parameter estimation.
"""
from __future__ import division, print_function
import tupak
import numpy as np

import tupak.gw.prior

tupak.core.utils.setup_logger()

time_duration = 4.
sampling_frequency = 2048.
outdir = 'outdir'

np.random.seed(151012)

injection_parameters = dict(mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0, phi_12=1.7, phi_jl=0.3,
                            luminosity_distance=4000., iota=0.4, psi=2.659, phase=1.3, geocent_time=1126259642.413,
                            waveform_approximant='IMRPhenomPv2', reference_frequency=50., ra=1.375, dec=-1.2108)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = tupak.WaveformGenerator(time_duration=time_duration,
                                             sampling_frequency=sampling_frequency,
                                             frequency_domain_source_model=tupak.gw.source.lal_binary_black_hole,
                                             parameters=injection_parameters)
hf_signal = waveform_generator.frequency_domain_strain()

# Set up interferometers.
IFOs = [tupak.gw.detector.get_interferometer_with_fake_noise_and_injection(
    name, injection_polarizations=hf_signal, injection_parameters=injection_parameters, time_duration=time_duration,
    sampling_frequency=sampling_frequency, outdir=outdir) for name in ['H1', 'L1', 'V1']]

# Set up prior
# This loads in a predefined set of priors for BBHs.
priors = tupak.gw.prior.BBHPriorSet()
# These parameters will not be sampled
for key in ['tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'phase', 'iota', 'ra', 'dec', 'geocent_time', 'psi']:
    priors[key] = injection_parameters[key]
# We can make uniform distributions.
priors['mass_2'] = tupak.core.prior.Uniform(name='mass_2', minimum=20, maximum=40)
# We can make a power-law distribution, p(x) ~ x^{alpha}
# Note: alpha=0 is a uniform distribution, alpha=-1 is uniform-in-log
priors['a_1'] = tupak.core.prior.PowerLaw(name='a_1', alpha=-1, minimum=1e-2, maximum=1)
# We can define a prior from an array as follows.
# Note: this doesn't have to be properly normalised.
a_2 = np.linspace(0, 1, 1001)
p_a_2 = a_2 ** 4
priors['a_2'] = tupak.core.prior.Interped(name='a_2', xx=a_2, yy=p_a_2, minimum=0, maximum=0.5)
# Additionally, we have Gaussian, TruncatedGaussian, Sine and Cosine.
# It's also possible to load an interpolate a prior from a file.
# Finally, if you don't specify any necessary parameters it will be filled in from the default when the sampler starts.
# Enjoy.

# Initialise GravitationalWaveTransient
likelihood = tupak.GravitationalWaveTransient(interferometers=IFOs, waveform_generator=waveform_generator)

# Run sampler
result = tupak.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty',
                           injection_parameters=injection_parameters, outdir=outdir, label='specify_prior')
result.plot_corner()
print(result)
