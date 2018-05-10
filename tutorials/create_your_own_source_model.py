#!/bin/python
"""

"""
from __future__ import division, print_function
import tupak
import numpy as np

tupak.utils.setup_logger()


def sine_gaussian(f, A, f0, tau, phi0, geocent_time, ra, dec, psi):
    arg = -(np.pi * tau * (f-f0))**2 + 1j * phi0
    plus = np.sqrt(np.pi) * A * tau * np.exp(arg) / 2.
    cross = plus * np.exp(1j*np.pi/2)
    return {'plus': plus, 'cross': cross}


outdir = 'outdir'
label = 'GW150914_sine_gaussian'
time_of_event = 1126259462.422

H1 = tupak.detector.get_interferometer('H1', time_of_event, version=1, outdir=outdir)
L1 = tupak.detector.get_interferometer('L1', time_of_event, version=1, outdir=outdir)
interferometers = [H1, L1]

prior = dict()
prior['A'] = tupak.prior.Uniform(0, 1e-20, 'A')
prior['f0'] = tupak.prior.Uniform(0, 10, 'f')
prior['tau'] = tupak.prior.Uniform(0, 10, 'tau')
prior['geocent_time'] = tupak.prior.Uniform(
    time_of_event-0.1, time_of_event+0.1, 'geocent_time')
prior['phi0'] = 0 #tupak.prior.Uniform(0, 2*np.pi, 'phi')
prior['ra'] = 0
prior['dec'] = 0
prior['psi'] = 0

waveform_generator = tupak.waveform_generator.WaveformGenerator(
    sine_gaussian, H1.sampling_frequency, H1.duration)

likelihood = tupak.likelihood.Likelihood(interferometers, waveform_generator)

result = tupak.sampler.run_sampler(
    likelihood, prior, sampler='pymultinest', outdir=outdir, label=label,
    resume=False)
result.plot_walks()
result.plot_distributions()
result.plot_corner()
print(result)
