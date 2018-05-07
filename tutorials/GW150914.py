from __future__ import division, print_function
import peyote

peyote.utils.setup_logger()

outdir = 'outdir'
label = 'GW150914'
time_of_event = 1126259462.422

H1, sampling_frequency, time_duration = peyote.detector.get_inteferometer('H1', time_of_event, version=1, outdir=outdir)
L1, _, _ = peyote.detector.get_inteferometer('L1', time_of_event, version=1, outdir=outdir)
interferometers = [H1, L1]

# Define the prior
prior = dict()
prior['mass_1'] = peyote.prior.Uniform(30, 50, 'mass_1')
prior['mass_2'] = peyote.prior.Uniform(20, 40, 'mass_2')
prior['geocent_time'] = peyote.prior.Uniform(time_of_event-0.1, time_of_event+0.1, name='geocent_time')
prior['luminosity_distance'] = peyote.prior.PowerLaw(alpha=2, minimum=100, maximum=1000)

# Create the waveform generator
waveform_generator = peyote.waveform_generator.WaveformGenerator(
    peyote.source.lal_binary_black_hole, sampling_frequency, time_duration,
    parameters={'waveform_approximant': 'IMRPhenomPv2', 'reference_frequency': 50})

# Define a likelihood
likelihood = peyote.likelihood.Likelihood(interferometers, waveform_generator)

# Run the sampler
result = peyote.sampler.run_sampler(likelihood, prior, sampler='dynesty', outdir=outdir, label='label')
result.plot_corner()
result.plot_walks()
result.plot_distributions()
print(result)
