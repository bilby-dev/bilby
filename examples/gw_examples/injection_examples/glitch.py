import bilby
import matplotlib.pyplot as plt
import numpy as np

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


def sine_gaussian_glitch(
    time_array, amplitude=1e-22, f0=100.0, tau=0.1, t0=0.0, phase=0.0
):
    """Generate a sine-Gaussian waveform.

    Parameters
    ----------
    time_array : array_like
        Time samples.
    amplitude : float
        Peak amplitude.
    f0 : float
        Central frequency in Hz.
    tau : float
        Decay time (Gaussian width) in seconds.
    t0 : float
        Central time of the burst in seconds.
    phase : float
        Phase offset in radians.
    """
    envelope = np.exp(-((time_array - t0) ** 2) / (2 * tau**2))
    return amplitude * envelope * np.sin(2 * np.pi * f0 * (time_array - t0) + phase)


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

inject_glitch_from_time_domain_strain = True

if inject_glitch_from_time_domain_strain:
    glitch_sample_times = np.arange(0, 2, 1 / 256)
    from scipy.signal.windows import tukey

    glitch = sine_gaussian_glitch(glitch_sample_times, tau=0.1, t0=0.5) * tukey(
        len(glitch_sample_times), alpha=0.2
    )

    # Inject glitch
    glitch_parameters = {
        "onset_time": injection_parameters["geocent_time"] + 1,
        "snr": glitch_snr,
    }
    ifos[0].inject_glitch(
        glitch_snr,
        glitch_parameters=glitch_parameters,
        glitch_time_domain_strain=glitch,
        glitch_sample_times=glitch_sample_times,
    )


inject_glitch_from_a_glitch_waveform_generator = False
if inject_glitch_from_a_glitch_waveform_generator:

    glitch_waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        time_domain_source_model=sine_gaussian_glitch,
    )
    glitch_parameters = {
        "onset_time": injection_parameters["geocent_time"] + 1,
        "amplitude": 1e-22,
        "f0": 100.0,
        "tau": 0.1,
        "t0": 1.3,
        "phase": 0.0,
        "snr": glitch_snr,
    }

    ifos[0].inject_glitch(
        glitch_parameters=glitch_parameters,
        glitch_waveform_generator=glitch_waveform_generator,
    )


fig, axes = plt.subplots(2, 1)
ax = axes[0]

ax.axvline(x=injection_parameters["geocent_time"], color="black", label="Merger time")
ax.axvline(x=glitch_parameters["onset_time"], color="blue", label="glitch onset time")
ax.legend()

ax.plot(ifos[0].time_array, ifos[0].time_domain_strain)
fig.savefig("./time_domain_strain.pdf")
