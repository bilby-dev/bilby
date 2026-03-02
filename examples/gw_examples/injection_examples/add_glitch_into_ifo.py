import bilby
import gengli
import matplotlib.pyplot as plt
import numpy as np

# To be installed from https://git.ligo.org/melissa.lopez/gengli
# or use glitch generator of your choice.

duration = 4.0
sampling_frequency = 1024.0
glitch_snr = 12
injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=2000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

bilby.core.utils.random.seed(23)

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant="IMRPhenomXPHM",
    reference_frequency=50.0,
    minimum_frequency=20.0,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
# the generator will convert all the parameters
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)

ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

generator = gengli.glitch_generator("L1")
glitch = generator.get_glitch(seed=200) * 1e-23

# Inject glitch
ifos[0].inject_glitch(
    glitch,
    glitch_injection_time=ifos[0].start_time + 2,
    glitch_sampling_frequency=sampling_frequency,
    glitch_snr=glitch_snr,
)

fig, axes = plt.subplots(2, 1)
ax = axes[0]
ax.plot(ifos[0].time_array, ifos[0].time_domain_strain)

ax = axes[1]
ax.plot(np.arange(len(glitch)) / sampling_frequency, glitch)
fig.savefig("./time_domain_strain.pdf")
