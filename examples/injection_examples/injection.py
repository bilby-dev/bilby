"""
Tutorial to show signal injection using new features of detector.py
"""
from __future__ import division, print_function
import numpy as np
import tupak
import logging


def main():

    tupak.utils.setup_logger()

    outdir = 'outdir'
    label = 'injection'

    # Create the waveform generator
    waveform_generator = tupak.waveform_generator.WaveformGenerator(time_duration=4, sampling_frequency=2048,
                                                                    frequency_domain_source_model=tupak.source.lal_binary_black_hole,
                                                                    parameters={'reference_frequency': 50.0,
                                                                                'waveform_approximant': 'IMRPhenomPv2'})

    # Define the prior
    # Merger time is some time in 2018, shame LIGO will never see it...
    time_of_event = np.random.uniform(1198800018, 1230336018)
    prior = dict()
    prior['luminosity_distance'] = tupak.prior.PowerLaw(alpha=2, minimum=100, maximum=5000, name='luminosity_distance')
    prior['geocent_time'] = tupak.prior.Uniform(time_of_event - 0.01, time_of_event + 0.01, name='geocent_time')
    prior['mass_1'] = tupak.prior.Gaussian(mu=40, sigma=5, name='mass_1')
    prior['mass_2'] = tupak.prior.Gaussian(mu=40, sigma=5, name='mass_2')
    tupak.prior.fill_priors(prior, waveform_generator)

    # Create signal injection
    injection_parameters = {name: prior[name].sample() for name in prior}
    if injection_parameters['mass_1'] < injection_parameters['mass_2']:
        injection_parameters['mass_1'], injection_parameters['mass_2'] =\
            injection_parameters['mass_2'], injection_parameters['mass_1']
    logging.info("Injection parameters:\n{}".format("\n".join(["{}: {}".format(key, injection_parameters[key])
                                                               for key in injection_parameters])))
    for parameter in injection_parameters:
        waveform_generator.parameters[parameter] = injection_parameters[parameter]
    injection_polarizations = waveform_generator.frequency_domain_strain()

    # Create interferometers and inject signal
    interferometers = [tupak.detector.get_inteferometer_with_fake_noise_and_injection(
            name, injection_polarizations=injection_polarizations, injection_parameters=injection_parameters,
            sampling_frequency=2048, time_duration=4, outdir=outdir) for name in ['H1', 'L1', 'V1']]

    # Define a likelihood
    likelihood = tupak.likelihood.MarginalizedLikelihood(interferometers, waveform_generator, prior=prior,
                                                         distance_marginalization=True, phase_marginalization=True)

    # Run the sampler
    result = tupak.sampler.run_sampler(
        likelihood, prior, label=label, sampler='dynesty', npoints=500, resume=False, outdir=outdir, use_ratio=True,
        injection_parameters=injection_parameters)
    truth = [injection_parameters[parameter] for parameter in result.search_parameter_keys]
    result.plot_corner(truth=truth)
    print(result)


if __name__ == "__main__":
    main()
