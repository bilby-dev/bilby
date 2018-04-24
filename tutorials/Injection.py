"""
Tutorial to show signal injection using new features of detector.py
"""
from __future__ import print_function, division

import corner

import peyote

time_duration = 1.
sampling_frequency = 4096.

source = peyote.source.BinaryBlackHole('BBH', sampling_frequency, time_duration, mass_1=36., mass_2=29., spin11=0,
                                       spin12=0, spin13=0, spin21=0, spin22=0,
                                       spin23=0, luminosity_distance=5000., iota=0., phase=0.,
                                       waveform_approximant='IMRPhenomPv2', reference_frequency=50., ra=0, dec=1,
                                       geocent_time=0, psi=1)

IFO_1 = peyote.detector.H1
IFO_2 = peyote.detector.L1
IFO_3 = peyote.detector.V1
IFOs = [IFO_1, IFO_2, IFO_3]
for IFO in IFOs:
    IFO.set_data(from_power_spectral_density=True, sampling_frequency=sampling_frequency, duration=time_duration)
    IFO.inject_signal(source)


likelihood = peyote.likelihood.Likelihood(source=source, interferometers=IFOs)

print(likelihood.noise_log_likelihood())
print(likelihood.log_likelihood())
print(likelihood.log_likelihood_ratio())

prior = source.copy()
prior.mass_1 = peyote.parameter.Parameter('mass_1', prior=peyote.prior.Uniform(lower=35, upper=37),
                                          latex_label='$m_1$')
prior.mass_2 = peyote.parameter.Parameter('mass_2', prior=peyote.prior.Uniform(lower=28, upper=30),
                                          latex_label='$m_2$')

# result = peyote.sampler.run_sampler(likelihood, prior, sampler='dynesty', npoints=100, print_progress=True)

# truths = [source.__dict__[x] for x in result.search_parameter_keys]
# fig = corner.corner(result.samples, labels=result.labels, truths=truths)
# fig.savefig('Injection Test')
