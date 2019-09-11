#!/usr/bin/env python
"""
A script to show how to create your own time domain source model.
A simple damped Gaussian signal is defined in the time domain, injected into
noise in two interferometers (LIGO Livingston and Hanford at design
sensitivity), and then recovered.
"""

import numpy as np
import bilby


# define the time-domain model
def time_domain_damped_sinusoid(
        time, amplitude, damping_time, frequency, phase, t0):
    """
    This example only creates a linearly polarised signal with only plus
    polarisation.
    """
    plus = np.zeros(len(time))
    tidx = time >= t0
    plus[tidx] = amplitude * np.exp(-(time[tidx] - t0) / damping_time) *\
        np.sin(2 * np.pi * frequency * (time[tidx] - t0) + phase)
    cross = np.zeros(len(time))
    return {'plus': plus, 'cross': cross}


# define parameters to inject.
injection_parameters = dict(amplitude=5e-22, damping_time=0.1, frequency=50,
                            phase=0, ra=0, dec=0, psi=0, t0=0., geocent_time=0.)

duration = 1.0
sampling_frequency = 1024
outdir = 'outdir'
label = 'time_domain_source_model'

# call the waveform_generator to create our waveform model.
waveform = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    time_domain_source_model=time_domain_damped_sinusoid,
    start_time=injection_parameters['geocent_time'] - 0.5)

# inject the signal into three interferometers
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 0.5)
ifos.inject_signal(waveform_generator=waveform,
                   parameters=injection_parameters)

#  create the priors
prior = injection_parameters.copy()
prior['amplitude'] = bilby.core.prior.LogUniform(1e-23, 1e-21, r'$h_0$')
prior['damping_time'] = bilby.core.prior.Uniform(
    0.01, 1, r'damping time', unit='$s$')
prior['frequency'] = bilby.core.prior.Uniform(0, 200, r'frequency', unit='Hz')
prior['phase'] = bilby.core.prior.Uniform(-np.pi / 2, np.pi / 2, r'$\phi$')

# define likelihood
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(ifos, waveform)

# launch sampler
result = bilby.core.sampler.run_sampler(
    likelihood, prior, sampler='dynesty', npoints=1000,
    injection_parameters=injection_parameters, outdir=outdir, label=label)

result.plot_corner()
