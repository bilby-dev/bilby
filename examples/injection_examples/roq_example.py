#!/usr/bin/env python
"""
Example of how to use the Reduced Order Quadrature method (see Smith et al.,
(2016) Phys. Rev. D 94, 044031) for a Binary Black hole simulated signal in
Gaussian noise.

This requires files specifying the appropriate basis weights.
These aren't shipped with Bilby, but are available on LDG clusters and
from the public repository https://git.ligo.org/lscsoft/ROQ_data.
"""
from __future__ import division, print_function

import numpy as np

import bilby

outdir = 'outdir'
label = 'roq'

# Load in the pieces for the linear part of the ROQ. Note you will need to
# adjust the filenames here to the correct paths on your machine
basis_matrix_linear = np.load("B_linear.npy").T
freq_nodes_linear = np.load("fnodes_linear.npy")

# Load in the pieces for the quadratic part of the ROQ
basic_matrix_quadratic = np.load("B_quadratic.npy").T
freq_nodes_quadratic = np.load("fnodes_quadratic.npy")

np.random.seed(170808)

duration = 4
sampling_frequency = 2048

injection_parameters = dict(
    chirp_mass=36., mass_ratio=0.9, a_1=0.4, a_2=0.3, tilt_1=0.0, tilt_2=0.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=1000., theta_jn=0.4, psi=0.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=20., minimum_frequency=20.)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)

ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)

# make ROQ waveform generator
search_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.roq,
    waveform_arguments=dict(frequency_nodes_linear=freq_nodes_linear,
                            frequency_nodes_quadratic=freq_nodes_quadratic,
                            reference_frequency=20., minimum_frequency=20.,
                            approximant='IMRPhenomPv2'),
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)

priors = bilby.gw.prior.BBHPriorDict()
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn', 'phase', 'psi', 'ra',
            'dec', 'phi_12', 'phi_jl', 'luminosity_distance']:
    priors[key] = injection_parameters[key]
priors.pop('mass_1')
priors.pop('mass_2')
priors['chirp_mass'] = bilby.core.prior.Uniform(
    15, 40, latex_label='$\\mathcal{M}$')
priors['mass_ratio'] = bilby.core.prior.Uniform(0.5, 1, latex_label='$q$')
priors['geocent_time'] = bilby.core.prior.Uniform(
    injection_parameters['geocent_time'] - 0.1,
    injection_parameters['geocent_time'] + 0.1, latex_label='$t_c$', unit='s')

likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient(
    interferometers=ifos, waveform_generator=search_waveform_generator,
    linear_matrix=basis_matrix_linear, quadratic_matrix=basic_matrix_quadratic,
    prior=priors)

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=500,
    injection_parameters=injection_parameters, outdir=outdir, label=label)

# Make a corner plot.
result.plot_corner()
