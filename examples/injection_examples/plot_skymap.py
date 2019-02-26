#!/usr/bin/env python
"""
Example script which produces posterior samples of ra and dec and generates a
skymap
"""
from __future__ import division, print_function
import bilby

duration = 4.
sampling_frequency = 2048.
outdir = 'outdir'
label = 'plot_skymap'
injection_parameters = dict(
    mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=4000., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-0.2108)

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=50.)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameters=injection_parameters, waveform_arguments=waveform_arguments)

ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)

priors = bilby.gw.prior.BBHPriorDict()
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'psi',
            'mass_1', 'mass_2', 'phase', 'geocent_time', 'luminosity_distance',
            'theta_jn']:
    priors[key] = injection_parameters[key]

likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator,
    time_marginalization=True, phase_marginalization=True,
    distance_marginalization=False, prior=priors)

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
    injection_parameters=injection_parameters, outdir=outdir, label=label)

# make some plots of the outputs
result.plot_corner()

bilby.gw.utils.plot_skymap(result, nside=2**6)
