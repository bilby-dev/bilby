#!/usr/bin/env python
"""
Tutorial for running cpnest with custom jump proposals.
"""
from __future__ import division, print_function

import numpy as np
import bilby.gw.sampler.proposal
from bilby.core.sampler import proposal


# The set up here is the same as in fast_tutorial.py. Look there for descriptive explanations.

duration = 4.
sampling_frequency = 2048.

outdir = 'outdir'
label = 'custom_jump_proposals'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

np.random.seed(88170235)

injection_parameters = dict(
    mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., theta_jn=0.4, psi=2.659,
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
priors = bilby.gw.prior.BBHPriorDict()
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 1,
    maximum=injection_parameters['geocent_time'] + 1,
    name='geocent_time', latex_label='$t_c$', unit='$s$')
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'geocent_time']:
    priors[key] = injection_parameters[key]
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator)

# Definition of the custom jump proposals. Define a JumpProposalCycle. The first argument is a list
# of all allowed jump proposals. The second argument is a list of weights for the respective jump
# proposal.

jump_proposals = proposal.JumpProposalCycle(
    [proposal.EnsembleWalk(priors=priors), proposal.EnsembleStretch(priors=priors),
     proposal.DifferentialEvolution(priors=priors), proposal.EnsembleEigenVector(priors=priors),
     bilby.gw.sampler.proposal.SkyLocationWanderJump(priors=priors), bilby.gw.sampler.proposal.CorrelatedPolarisationPhaseJump(priors=priors),
     bilby.gw.sampler.proposal.PolarisationPhaseJump(priors=priors), proposal.DrawFlatPrior(priors=priors)],
    weights=[2, 2, 5, 1, 1, 1, 1, 1])

# Run cpnest with the proposals kwarg specified.
# Make sure to have a version of cpnest installed that supports custom proposals.
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='cpnest', npoints=1000,
    injection_parameters=injection_parameters, outdir=outdir, label=label,
    proposals=jump_proposals)

# Make a corner plot.
result.plot_corner()
