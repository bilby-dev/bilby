"""
Tutorial to show signal injection using new features of detector.py
"""
from __future__ import division, print_function
import numpy as np
import peyote
import logging


def main():

    peyote.utils.setup_logger()

    outdir = 'outdir'
    label = 'injection'

    # Create the waveform generator
    waveform_generator = peyote.waveform_generator.WaveformGenerator(
        peyote.source.lal_binary_black_hole, sampling_frequency=2048, time_duration=4,
        parameters={'reference_frequency': 50.0, 'waveform_approximant': 'IMRPhenomPv2'})

    # Define the prior
    # Merger time is some time in 2018, shame LIGO will never see it...
    time_of_event = np.random.uniform(1198800018, 1230336018)
    prior = dict()
    prior['luminosity_distance'] = peyote.prior.PowerLaw(alpha=2, minimum=100, maximum=5000, name='luminosity_distance')
    prior['geocent_time'] = peyote.prior.Uniform(time_of_event-0.01, time_of_event+0.01, name='geocent_time')
    prior['mass_1'] = peyote.prior.Gaussian(mu=40, sigma=5, name='mass_1')
    prior['mass_2'] = peyote.prior.Gaussian(mu=40, sigma=5, name='mass_2')
    peyote.prior.fill_priors(prior, waveform_generator)

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
    interferometers = [peyote.detector.get_inteferometer_with_fake_noise_and_injection(
            name, injection_polarizations=injection_polarizations, injection_parameters=injection_parameters,
            sampling_frequency=2048, time_duration=4, outdir=outdir) for name in ['H1', 'L1', 'V1']]

    # Define a likelihood
    likelihood = peyote.likelihood.MarginalizedLikelihood(interferometers, waveform_generator, prior=prior,
                                                          distance_marginalization=True, phase_marginalization=True)

    # Run the sampler
    result = peyote.sampler.run_sampler(
        likelihood, prior, label=label, sampler='dynesty', npoints=500, resume=False, outdir=outdir, use_ratio=True,
        injection_parameters=injection_parameters)
    truth = [injection_parameters[parameter] for parameter in result.search_parameter_keys]
    result.plot_corner(truth=truth)
    print(result)


if __name__ == "__main__":
    main()
