import os

import numpy as np

import bilby
from bilby.gw.waveform_generator import WaveformGenerator

bilby.core.utils.random.seed(10)

time_duration = 4.0
sampling_frequency = 4096.0

simulation_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.0,
    a_2=0.0,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=0.0,
    phi_jl=0.0,
    luminosity_distance=100.0,
    theta_jn=0.4,
    phase=1.3,
    waveform_approximant="IMRPhenomPv2",
    reference_frequency=50.0,
    ra=1.375,
    dec=-1.2108,
    geocent_time=1126259642.413,
    psi=2.659,
)

waveform_generator = WaveformGenerator(
    duration=time_duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameters=simulation_parameters,
)

signal = waveform_generator.frequency_domain_strain()

IFO = bilby.gw.detector.get_interferometer_with_fake_noise_and_injection(
    name="H1",
    injection_polarizations=signal,
    injection_parameters=simulation_parameters,
    duration=time_duration,
    plot=False,
    sampling_frequency=sampling_frequency,
)

hf_signal_and_noise = IFO.strain_data.frequency_domain_strain
frequencies = bilby.core.utils.create_frequency_series(
    sampling_frequency=sampling_frequency, duration=time_duration
)

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + "/standard_data.txt", "w+") as f:
        np.savetxt(
            f,
            np.column_stack(
                [frequencies, hf_signal_and_noise.view(float).reshape(-1, 2)]
            ),
            header="frequency hf_real hf_imag",
        )
