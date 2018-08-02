#!/bin/python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter space for an injected eccentric binary 
black hole signal with masses & distnace similar to GW150914.

This uses the same binary parameters that were used to make Figures 1, 2 & 5 in Lower et al. (2018) -> arXiv:1806.05350.

For a more comprehensive look at what goes on in each step, refer to the "basic_tutorial.py" example.
"""
from __future__ import division, print_function

import numpy as np

import tupak

import matplotlib.pyplot as plt

duration = 64.
sampling_frequency = 2048.

outdir = 'outdir'
label = 'eccentric_GW140914'
tupak.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.
np.random.seed(150914)

injection_parameters = dict(mass_1=35., mass_2=30., eccentricity=0.1, luminosity_distance=440.,
                            iota=0.4, psi=0.1, phase=1.2, geocent_time=1180002601.0, ra=45, dec=5.73)

waveform_arguments = dict(waveform_approximant='EccentricFD', reference_frequency=10., minimum_frequency=10.)

# Create the waveform_generator using the LAL eccentric black hole no spins source function
waveform_generator = tupak.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=tupak.gw.source.lal_eccentric_binary_black_hole_no_spins,
    parameters=injection_parameters, waveform_arguments=waveform_arguments)

hf_signal = waveform_generator.frequency_domain_strain()

# Setting up three interferometers (LIGO-Hanford (H1), LIGO-Livingston (L1), and Virgo (V1)) at their design sensitivities.
# The maximum frequency is set just prior to the point at which the waveform model terminates. This is to avoid any biases 
# introduced from using a sharply terminating waveform model.
minimum_frequency = 10.0
maximum_frequency = 133.0

def get_interferometer(name, injection_polarizations, injection_parameters, duration, sampling_frequency,
                       minimum_frequency, maximum_frequency, outdir):
    """
    Sets up the interferometers & injects the signal into them
    """
    start_time = injection_parameters['geocent_time'] + 2 - duration
    
    ifo = tupak.gw.detector.get_empty_interferometer(name)
    if name == 'V1':
        ifo.power_spectral_density.set_from_power_spectral_density_file('AdV_psd.txt')
    else:
        ifo.power_spectral_density.set_from_power_spectral_density_file('aLIGO_ZERO_DET_high_P_psd.txt')
    
    ifo.set_strain_data_from_power_spectral_density(sampling_frequency=sampling_frequency, 
            duration=duration, start_time=start_time)
    
    injection_polarizations = ifo.inject_signal(parameters=injection_parameters,
                              injection_polarizations=injection_polarizations,
                              waveform_generator=waveform_generator)

    signal = ifo.get_detector_response(injection_polarizations, injection_parameters)

    ifo.minimum_frequency = minimum_frequency
    ifo.maximum_frequency = maximum_frequency
    
    ifo.plot_data(signal=signal, outdir=outdir, label=label)
    ifo.save_data(outdir, label=label)
    
    return ifo

# IFOs = [tupak.gw.detector.get_interferometer_with_fake_noise_and_injection(name, injection_polarizations=hf_signal, 
#     injection_parameters=injection_parameters, duration=duration,
#     sampling_frequency=sampling_frequency, outdir=outdir) for name in ['H1', 'L1', 'V1']]

name = ['H1', 'L1', 'V1']
IFOs = []

for i in range(0,3):
    IFOs.append(get_interferometer(name[i], injection_polarizations=hf_signal, injection_parameters=injection_parameters,
                                   duration=duration, sampling_frequency=sampling_frequency, 
                                   minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency, 
                                   outdir=outdir))

# Now we set up the priors on each of the binary parameters.
priors = dict()
priors["mass_1"] = tupak.core.prior.Uniform(name='mass_1', minimum=5, maximum=60)
priors["mass_2"] = tupak.core.prior.Uniform(name='mass_2', minimum=5, maximum=60)
priors["eccentricity"] = tupak.core.prior.PowerLaw(name='eccentricity', latex_label='$e$', alpha=-1, minimum=1e-4, maximum=0.4)
priors["luminosity_distance"] =  tupak.gw.prior.UniformComovingVolume(name='luminosity_distance', minimum=1e2, maximum=2e3)
priors["dec"] =  tupak.core.prior.Cosine(name='dec')
priors["ra"] =  tupak.core.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi)
priors["iota"] =  tupak.core.prior.Sine(name='iota')
priors["psi"] =  tupak.core.prior.Uniform(name='psi', minimum=0, maximum=2 * np.pi)
priors["phase"] =  tupak.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi)
priors["geocent_time"] = tupak.core.prior.Uniform(1180002600.9, 1180002601.1, name='geocent_time')

# Initialising the likelihood function.
likelihood = tupak.gw.likelihood.GravitationalWaveTransient(interferometers=IFOs, waveform_generator=waveform_generator,
                                              time_marginalization=False, phase_marginalization=False,
                                              distance_marginalization=False, prior=priors)

# Now we run sampler (PyMultiNest in our case).
result = tupak.run_sampler(likelihood=likelihood, priors=priors, sampler='pymultinest', npoints=1000,
                           injection_parameters=injection_parameters, outdir=outdir, label=label)

# And finally we make some plots of the output posteriors.
result.plot_corner()
