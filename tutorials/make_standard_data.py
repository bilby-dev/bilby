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
IFO.set_data(from_power_spectral_density=True, sampling_frequency=sampling_frequency, duration=time_duration)
IFO.inject_signal(source, simulation_parameters)
hf_signal_and_noise = IFO.data
frequencies = peyote.utils.create_fequency_series(sampling_frequency=sampling_frequency, duration=time_duration)

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/standard_data.txt', 'w+') as f:
        np.savetxt(
            f,
            np.column_stack([frequencies,
                             hf_signal_and_noise.view(float).reshape(-1, 2)]),
            header='frequency hf_real hf_imag')
