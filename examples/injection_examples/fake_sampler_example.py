#!/usr/bin/env python
"""
Read ROQ posterior and calculate full likelihood at same parameter space points.
"""
from __future__ import division, print_function

import numpy as np
import deepdish as dd
import bilby
import matplotlib.pyplot as plt


def make_comparison_histograms(file_full, file_roq):
    # Returns a dictionary
    data_full = dd.io.load(file_full)
    data_roq = dd.io.load(file_roq)

    # These are pandas dataframes
    pos_full = data_full['posterior']
    pos_roq = data_roq['posterior']

    plt.figure()
    plt.hist(pos_full['log_likelihood_evaluations'], 50, label='full', histtype='step')
    plt.hist(pos_roq['log_likelihood_evaluations'], 50, label='roq', histtype='step')
    plt.xlabel(r'delta_logl')
    plt.legend(loc=2)
    plt.savefig('delta_logl.pdf')
    plt.close()

    plt.figure()
    delta_dlogl = pos_full['log_likelihood_evaluations'] - pos_roq['log_likelihood_evaluations']
    plt.hist(delta_dlogl, 50)
    plt.xlabel(r'delta_logl_full - delta_logl_roq')
    plt.savefig('delta_delta_logl.pdf')
    plt.close()

    plt.figure()
    delta_dlogl = np.abs(pos_full['log_likelihood_evaluations'] - pos_roq['log_likelihood_evaluations'])
    bins = np.logspace(np.log10(delta_dlogl.min()), np.log10(delta_dlogl.max()), 25)
    plt.hist(delta_dlogl, bins=bins)
    plt.xscale('log')
    plt.xlabel(r'|delta_logl_full - delta_logl_roq|')
    plt.savefig('log_abs_delta_delta_logl.pdf')
    plt.close()


def main():
    outdir = 'outdir_full'
    label = 'full'

    np.random.seed(170808)

    duration = 4
    sampling_frequency = 2048
    noise = 'zero'

    sampler = 'fake_sampler'
    # This example assumes that the following posterior file exists.
    # It comes from a run using the full likelihood using the same
    # injection and sampling parameters, but the ROQ likelihood.
    # See roq_example.py for such an example.
    sample_file = 'outdir_dynesty_zero_noise_SNR22/roq_result.h5'

    injection_parameters = dict(
        chirp_mass=36., mass_ratio=0.9, a_1=0.4, a_2=0.3, tilt_1=0.0, tilt_2=0.0,
        phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., iota=0.4, psi=0.659,
        phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                              reference_frequency=20.0, minimum_frequency=20.0)

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)

    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])

    if noise == 'Gaussian':
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=injection_parameters['geocent_time'] - 3)
    elif noise == 'zero':
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=injection_parameters['geocent_time'] - 3)

    ifos.inject_signal(waveform_generator=waveform_generator,
                       parameters=injection_parameters)

    priors = bilby.gw.prior.BBHPriorDict()
    for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'iota', 'psi', 'ra',
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

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator)

    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler=sampler, sample_file=sample_file,
        injection_parameters=injection_parameters, outdir=outdir, label=label)

    # Make a corner plot.
    result.plot_corner()

    # Compare full and ROQ likelihoods
    make_comparison_histograms(outdir + '/%s_result.h5' % label, sample_file)


if __name__ == '__main__':
    main()
