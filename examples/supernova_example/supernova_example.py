#!/bin/python
"""

Tutorial to demonstrate running parameter estimation/model selection on an NR
supernova injected signal.  Signal model is made by applying PCA to a set of
supernova waveforms. The first few PCs are then linearly combined with a scale
factor. (See https://arxiv.org/pdf/1202.3256.pdf)

"""
from __future__ import division, print_function
import numpy as np

# Set the duration and sampling frequency of the data segment that we're going to inject the signal into
import tupak.gw.likelihood

time_duration = 3.
sampling_frequency = 4096.

# Specify the output directory and the name of the simulation.
outdir = 'outdir'
label = 'supernova'
tupak.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(170801)

# We are going to inject a supernova waveform.  We first establish a dictionary
# of parameters that includes all of the different waveform parameters. It will
# read in a signal to inject from a txt file.

injection_parameters = dict(file_path='MuellerL15_example_inj.txt',
                            luminosity_distance=10.0, ra=1.375,
                            dec=-1.2108, geocent_time=1126259642.413,
                            psi=2.659)

# Create the waveform_generator using a supernova source function
waveform_generator = tupak.gw.waveform_generator.WaveformGenerator(
    time_duration=time_duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=tupak.gw.source.supernova,
    parameters=injection_parameters)
hf_signal = waveform_generator.frequency_domain_strain()

# Set up interferometers.  In this case we'll use three interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1), and Virgo (V1)).  These default to
# their design sensitivity
IFOs = [tupak.gw.detector.get_interferometer_with_fake_noise_and_injection(
    name, injection_polarizations=hf_signal,
    injection_parameters=injection_parameters, time_duration=time_duration,
    sampling_frequency=sampling_frequency, outdir=outdir)
    for name in ['H1', 'L1', 'V1']]

# read in from a file the PCs used to create the signal model.
realPCs = np.loadtxt('SupernovaRealPCs.txt')
imagPCs = np.loadtxt('SupernovaImagPCs.txt')

# Now we make another waveform_generator because the signal model is
# not the same as the injection in this case.
simulation_parameters = dict(
    realPCs=realPCs, imagPCs=imagPCs)

search_waveform_generator = tupak.gw.waveform_generator.WaveformGenerator(
    time_duration=time_duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=tupak.gw.source.supernova_pca_model,
    waveform_arguments=simulation_parameters)

# Set up prior
priors = dict()
for key in ['psi', 'geocent_time']:
    priors[key] = injection_parameters[key]
priors['luminosity_distance'] = tupak.core.prior.Uniform(
    2, 20, 'luminosity_distance')
priors['pc_coeff1'] = tupak.core.prior.Uniform(-1, 1, 'pc_coeff1')
priors['pc_coeff2'] = tupak.core.prior.Uniform(-1, 1, 'pc_coeff2')
priors['pc_coeff3'] = tupak.core.prior.Uniform(-1, 1, 'pc_coeff3')
priors['pc_coeff4'] = tupak.core.prior.Uniform(-1, 1, 'pc_coeff4')
priors['pc_coeff5'] = tupak.core.prior.Uniform(-1, 1, 'pc_coeff5')
priors['ra'] = tupak.core.prior.Uniform(minimum=0, maximum=2 * np.pi,
                                        name='ra')
priors['dec'] = tupak.core.prior.Sine(name='dec')
priors['geocent_time'] = tupak.core.prior.Uniform(
    injection_parameters['geocent_time'] - 1,
    injection_parameters['geocent_time'] + 1,
    'geocent_time')

# Initialise the likelihood by passing in the interferometer data (IFOs) and
# the waveoform generator
likelihood = tupak.gw.likelihood.GravitationalWaveTransient(
    interferometers=IFOs, waveform_generator=search_waveform_generator)

# Run sampler.
result = tupak.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=500,
    outdir=outdir, label=label)

# make some plots of the outputs
result.plot_corner()
print(result)
