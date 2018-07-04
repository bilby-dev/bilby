"""
A script to show how to create your own time domain source model.
A simple damped Gaussian signal is defined in the time domain, injected into noise in
two interferometers (LIGO Livingston and Hanford at design sensitivity),
and then recovered.
"""

import tupak
import numpy as np



# define the time-domain model
def time_domain_damped_sinusoid(time, amplitude, damping_time, frequency, phase, ra, dec, psi, geocent_time):
    """
    This example only creates a linearly polarised signal with only plus polarisation.
    """

    plus = amplitude * np.exp(-time / damping_time) * np.sin(2.*np.pi*frequency*time + phase)
    cross = np.zeros(len(time))

    return {'plus': plus, 'cross': cross}

# define parameters to inject.
injection_parameters = dict(amplitude=5e-22, damping_time=0.1, frequency=50,
                            phase=0,
                            ra=0, dec=0, psi=0, geocent_time=0.)

duration = 0.5
sampling_frequency = 2048
outdir='outdir'
label='time_domain_source_model'

# call the waveform_generator to create our waveform model.
waveform = tupak.gw.waveform_generator.WaveformGenerator(duration=duration, sampling_frequency=sampling_frequency,
                                                         time_domain_source_model=time_domain_damped_sinusoid,
                                                         parameters=injection_parameters)

hf_signal = waveform.frequency_domain_strain()
#note we could plot the time domain signal with the following code
# import matplotlib.pyplot as plt
# plt.plot(waveform.time_array, waveform.time_domain_strain()['plus'])

# or the frequency-domain signal:
# plt.loglog(waveform.frequency_array, abs(waveform.frequency_domain_strain()['plus']))



# inject the signal into three interferometers
IFOs = [tupak.gw.detector.get_interferometer_with_fake_noise_and_injection(
        name, injection_polarizations=hf_signal,
        injection_parameters=injection_parameters, duration=duration,
        sampling_frequency=sampling_frequency, outdir=outdir)
        for name in ['H1', 'L1']]


#  create the priors
prior = injection_parameters.copy()
prior['amplitude'] = tupak.core.prior.Uniform(1e-23, 1e-21, r'$h_0$')
prior['damping_time'] = tupak.core.prior.Uniform(0, 1, r'damping time')
prior['frequency'] = tupak.core.prior.Uniform(0, 200, r'frequency')
prior['phase'] = tupak.core.prior.Uniform(-np.pi / 2, np.pi / 2, r'$\phi$')


# define likelihood
likelihood = tupak.gw.likelihood.GravitationalWaveTransient(IFOs, waveform)

# launch sampler
result = tupak.core.sampler.run_sampler(likelihood, prior, sampler='dynesty', npoints=1000,
                                        injection_parameters=injection_parameters,
                                        outdir=outdir, label=label)

result.plot_corner()

