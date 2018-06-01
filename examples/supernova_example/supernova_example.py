#!/bin/python
"""
Tutorial to demonstrate running parameter estimation/model selection on an NR supernova injected signal. 
Signal model is made by applying PCA to a set of supernova waveforms. The first few PCs are then linearly
combined with a scale factor. (See https://arxiv.org/pdf/1202.3256.pdf)

"""
from __future__ import division, print_function
import tupak
import numpy as np

# Set the duration and sampling frequency of the data segment that we're going to inject the signal into
time_duration = 3.
sampling_frequency = 4096.

# Specify the output directory and the name of the simulation.
outdir = 'outdir'
label = 'supernova'
tupak.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(170801)

# We are going to inject a supernova waveform.  We first establish a dictionary of parameters that
# includes all of the different waveform parameters. It will read in a signal to inject from a txt file.
injection_parameters = dict(file_path = 'MuellerL15_example_inj.txt', luminosity_distance = 60.0, ra = 1.375,
                             dec = -1.2108, geocent_time = 1126259642.413, psi= 2.659)

# Create the waveform_generator using a supernova source function
waveform_generator = tupak.waveform_generator.WaveformGenerator(time_duration=time_duration,
                                                                sampling_frequency=sampling_frequency,
                                                                frequency_domain_source_model=tupak.source.supernova,
                                                                parameters=injection_parameters)
hf_signal = waveform_generator.frequency_domain_strain()

# Set up interferometers.  In this case we'll use three interferometers (LIGO-Hanford (H1), LIGO-Livingston (L1),
# and Virgo (V1)).  These default to their design sensitivity
IFOs = [tupak.detector.get_interferometer_with_fake_noise_and_injection(
    name, injection_polarizations=hf_signal, injection_parameters=injection_parameters, time_duration=time_duration,
    sampling_frequency=sampling_frequency, outdir=outdir) for name in ['H1', 'L1', 'V1']]

# read in from a file the PCs used to create the signal model.
realPCs = np.loadtxt('SupernovaRealPCs.txt')
imagPCs = np.loadtxt('SupernovaImagPCs.txt')

# now we have to do the waveform_generator again because the signal model is not the same as the injection in this case.
simulation_parameters = dict(realPCs=realPCs, imagPCs=imagPCs, coeff1 = 0.1, coeff2 = 0.1, 
                            coeff3 = 0.1, coeff4 = 0.1, coeff5 = 0.1, luminosity_distance = 60.0,
                            ra = 1.375, dec = -1.2108, geocent_time = 1126259642.413, psi=2.659)

waveform_generator = tupak.waveform_generator.WaveformGenerator(time_duration=time_duration,
                                                                sampling_frequency=sampling_frequency,
                                                                frequency_domain_source_model=tupak.source.supernova_pca_model,
                                                                parameters=simulation_parameters)

# Set up prior, which is a dictionary
priors = dict()
# By default we will sample all terms in the signal models.  However, this will take a long time for the calculation,
# so for this example we will set almost all of the priors to be equall to their injected values.  This implies the
# prior is a delta function at the true, injected value.  In reality, the sampler implementation is smart enough to
# not sample any parameter that has a delta-function prior.
for key in ['psi', 'geocent_time']:
    priors[key] = injection_parameters[key]

# The above list does *not* include frequency and Q, which means those are the parameters
# that will be included in the sampler.  If we do nothing, then the default priors get used.
priors['luminosity_distance'] = tupak.prior.create_default_prior(name='luminosity_distance')
priors['coeff1'] = tupak.prior.create_default_prior(name='coeff1')
priors['coeff2'] = tupak.prior.create_default_prior(name='coeff2')
priors['coeff3'] = tupak.prior.create_default_prior(name='coeff3')
priors['coeff4'] = tupak.prior.create_default_prior(name='coeff4')
priors['coeff5'] = tupak.prior.create_default_prior(name='coeff5')
priors['ra'] = tupak.prior.create_default_prior(name='ra')
priors['dec'] = tupak.prior.create_default_prior(name='dec')

# Initialise the likelihood by passing in the interferometer data (IFOs) and the waveoform generator
likelihood = tupak.likelihood.GravitationalWaveTransient(interferometers=IFOs, waveform_generator=waveform_generator)

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = tupak.sampler.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
                                   injection_parameters=injection_parameters, outdir=outdir, label=label)

# make some plots of the outputs
#result.plot_corner()
print(result)











