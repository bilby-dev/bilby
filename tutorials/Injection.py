"""
Tutorial to show signal injection using new features of detector.py
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

import peyote
import corner

time_duration = 1.
sampling_frequency = 4096.

simulation_parameters = dict(
    mass_1=36.,
    mass_2=29.,
    spin_1=[0, 0, 0],
    spin_2=[0, 0, 0],
    luminosity_distance=5000.,
    iota=0.,
    phase=0.,
    waveform_approximant='IMRPhenomPv2',
    reference_frequency=50.,
    ra=0,
    dec=1,
    geocent_time=0,
    psi=1
    )

source = peyote.source.BinaryBlackHole('BBH', sampling_frequency, time_duration)

IFO_1 = peyote.detector.H1
IFO_2 = peyote.detector.L1
IFO_3 = peyote.detector.V1
IFOs = [IFO_1, IFO_2, IFO_3]
for IFO in IFOs:
    IFO.set_data(from_power_spectral_density=True, sampling_frequency=sampling_frequency, duration=time_duration)
    IFO.inject_signal(source, simulation_parameters)

# ff = peyote.utils.create_fequency_series(sampling_frequency, time_duration)
# for IFO in IFOs:
#     plt.loglog(ff, np.abs(IFO.data), label=IFO.name)
#
# plt.xlim(10, 1e3)
# plt.ylim(1e-30, 1e-18)
#
# plt.xlabel(r'frequency [Hz]')
# plt.legend(loc='best')
#
# plt.tight_layout()
# plt.show()
# plt.close()

likelihood = peyote.likelihood.likelihoodB(source=source, interferometers=IFOs)

prior = simulation_parameters.copy()
prior['mass_1'] = peyote.parameter.Parameter('mass_1', prior=peyote.prior.Uniform(lower=35, upper=37),
                                             latex_label='$m_1$')
# prior['mass_2'] = peyote.parameter.Parameter('mass_2', prior=peyote.prior.Uniform(lower=28, upper=30),
#                                              latex_label='$m_2$')

result = peyote.sampler.run_sampler(likelihood, prior, 'dynesty', npoints=100, print_progress=True)

print(result.samples)

truths = [simulation_parameters[k] for k in result.parameter_keys]
fig = corner.corner(result.samples, labels=result.labels, truths=truths)
fig.show()

