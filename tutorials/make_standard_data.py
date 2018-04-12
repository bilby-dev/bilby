import peyote
import os
import numpy as np

from waveform_generator import WaveformGenerator

np.random.seed(10)

time_duration = 1.
sampling_frequency = 4096.

simulation_parameters = dict(
    mass_1=36.,
    mass_2=29.,
    spin_1=[0, 0, 0],
    spin_2=[0, 0, 0],
    luminosity_distance=100.,
    iota=0.4,
    phase=1.3,
    waveform_approximant='IMRPhenomPv2',
    reference_frequency=50.,
    ra=1.375,
    dec=-1.2108,
    geocent_time=1126259642.413,
    psi=2.659
    )

waveform_generator = WaveformGenerator(
    'BBH', sampling_frequency, time_duration, peyote.source.lal_binary_black_hole)
waveform_generator.set_values(simulation_parameters)

IFO = peyote.detector.H1
IFO.set_data(
    from_power_spectral_density=True, sampling_frequency=sampling_frequency,
    duration=time_duration)
IFO.inject_signal(waveform_generator)
hf_signal_and_noise = IFO.data
frequencies = peyote.utils.create_fequency_series(
        sampling_frequency=sampling_frequency, duration=time_duration)

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/standard_data.txt', 'w+') as f:
        np.savetxt(
            f,
            np.column_stack([frequencies,
                             hf_signal_and_noise.view(float).reshape(-1, 2)]),
            header='frequency hf_real hf_imag')
