import peyote
import os
import numpy as np
np.random.seed(10)

time_duration = 1.
sampling_frequency = 4096.

simulation_parameters = dict(
    mass_1=36.,
    mass_2=29.,
    spin_1=[0, 0, 0],
    spin_2=[0, 0, 0],
    luminosity_distance=410.,
    iota=0.,
    phase=0.,
    waveform_approximant='IMRPhenomPv2',
    reference_frequency=50.,
    ra=0,
    dec=1,
    geocent_time=0,
    psi=1
    )

source = peyote.source.BinaryBlackHole(
    'BBH', sampling_frequency, time_duration)
hf_signal = source.frequency_domain_strain(simulation_parameters)

IFO = peyote.detector.H1
hf_noise, frequencies = IFO.power_spectral_density.get_noise_realisation(
    sampling_frequency, time_duration)
IFO.set_data(frequency_domain_strain=hf_noise)
IFO.inject_signal(source, simulation_parameters)
IFO.set_spectral_densities(frequencies)
IFO.whiten_data()
hf_signal_and_noise = IFO.data

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/standard_data.txt', 'w+') as f:
        np.savetxt(
            f,
            np.column_stack([frequencies,
                             hf_signal_and_noise.view(float).reshape(-1, 2)]),
            header='frequency hf_real hf_imag')
