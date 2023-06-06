import bilby
import numpy as np
from bilby.gw.source import lal_binary_black_hole

# Set the parameters of the binary black hole merger
mass_1 = 36.
mass_2 = 29.
a_1 = 0.4
a_2 = 0.3
tilt_1 = 0.5
tilt_2 = 1.0
phi_12 = 1.7
phi_jl = 0.3
luminosity_distance = 2000.
theta_jn = 0.4
psi = 2.659
phase = 1.3
geocent_time = 1126259642.413
ra = 1.375
dec = -1.2108

## Define the waveform generator
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=lal_binary_black_hole,
    parameters={"mass_1": mass_1, "mass_2": mass_2, "a_1": a_1, "a_2": a_2,
                "tilt_1": tilt_1, "tilt_2": tilt_2, "phi_12": phi_12, "phi_jl": phi_jl,
                "luminosity_distance": luminosity_distance, "theta_jn": theta_jn, "psi": psi,
                "phase": phase, "geocent_time": geocent_time, "ra": ra, "dec": dec},
    waveform_arguments={"reference_frequency": 20},
    duration=4,
    sampling_frequency=4096)

# Generate the time-domain waveform
waveform_polarizations = waveform_generator.time_domain_strain()

# Generate the noise
minimum_frequency = 20.
maximum_frequency = 1024.
delta_f = 1.0 / waveform_generator.duration
frequency_array = np.arange(0, waveform_generator.sampling_frequency / 2, delta_f)

asd_file = "aLIGO_O4_high_asd.txt"  # Path to the ASD file
asd = bilby.gw.detector.PowerSpectralDensity.from_amplitude_spectral_density_file(asd_file)
print(dir(asd))
psd = asd.psd_array
# Print the PSD array
print(psd)
import matplotlib.pyplot as plt

# assuming you have the asd object and psd_array attribute
frequency_array = asd.frequency_array
plt.loglog(frequency_array, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.title('Power Spectral Density')

# Plot ASD
asd_array = asd.asd_array
plt.loglog(asd.frequency_array, asd_array)
plt.xlabel('Frequency (Hz)')
plt.ylabel('ASD (strain/rtHz)')
plt.title('Amplitude Spectral Density')

plt.show()

##to thsi part we got psd and asd working
