import bilby
import gengli
import matplotlib.pyplot as plt
import numpy as np

# gengli package needs to be installed from ``pip install gengli``

duration = 8.0
sampling_frequency = 2048.0
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
    luminosity_distance=2100.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=205.0,
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
ifos.set_strain_data_from_zero_noise(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 4,
)

ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

glitch_generator = gengli.glitch_generator("L1")

# Produces whitened time domain blip glitch
glitch_time_domain_strain = glitch_generator.get_glitch(srate=sampling_frequency)

# onset_time is the time at which glitch will be injected and snr is the snr of the glitch.
glitch_parameters = dict(onset_time=injection_parameters["geocent_time"] + 2, snr=9)
glitch_sample_times = (
    np.arange(0, len(glitch_time_domain_strain), 1) / sampling_frequency
)

ifos[0].inject_glitch(
    glitch_snr,
    glitch_parameters=glitch_parameters,
    glitch_time_domain_strain=glitch_time_domain_strain,
    glitch_sample_times=glitch_sample_times,
)

fig, axes = plt.subplots(2, 1)
ax = axes[0]

ax.axvline(x=injection_parameters["geocent_time"], color="black", label="Merger time")
ax.axvline(x=glitch_parameters["onset_time"], color="blue", label="glitch onset time")
ax.legend()

ax.plot(ifos[0].time_array, ifos[0].time_domain_strain)
fig.savefig("./time_domain_strain.pdf")
