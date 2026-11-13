import itertools
import unittest
from copy import deepcopy

import bilby
import numpy as np
from parameterized import parameterized

from bilby.gw.likelihood.relative_modes import get_safe_fiducial_parameters

CHIRP_MASSES = [3, 10, 30, 100, 200]
MASS_RATIOS = [1, 0.5, 0.2]
# At q = 1 with in-plane spins, IMRPhenomXPHM on a uniform grid and on a frequency sequence differ by about 2% in
# relative norm, so both binned likelihoods disagree with the unbinned one; at q = 0.9999 the two agree.
PRECESSING_MASS_RATIOS = [0.999, 0.5, 0.2]
MINIMUM_FREQUENCY = 20
SAMPLING_FREQUENCY = 2048
WAVEFORM_ARGUMENTS = dict(
    waveform_approximant="IMRPhenomXPHM",
    reference_frequency=MINIMUM_FREQUENCY,
    minimum_frequency=MINIMUM_FREQUENCY,
    mode_array=[[2, 2], [2, 1], [3, 3], [3, 2], [4, 4]],
    PhenomXPrecVersion=102,
    PhenomXHMThresholdMband=0,
    PhenomXPHMThresholdMband=0,
)
ZERO_SPINS = dict(a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0)
PRECESSING_SPINS = dict(
    a_1=0.5, a_2=0.3, tilt_1=1.0, tilt_2=2.0, phi_12=1.0, phi_jl=0.5
)


def make_likelihoods(chirp_mass, mass_ratio, spins, epsilon=0.05):
    """Inject a signal with network optimal SNR squared of 200 into zero noise
    and return the unbinned, the mode-by-mode and the existing relative-binning
    likelihoods, with the injection as fiducial parameters."""
    parameters = dict(
        chirp_mass=chirp_mass,
        mass_ratio=mass_ratio,
        luminosity_distance=1000.0,
        theta_jn=0.8,
        psi=0.5,
        phase=1.3,
        geocent_time=1187008882.0,
        ra=1.3,
        dec=-1.2,
        **spins,
    )
    mass_1, mass_2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
        chirp_mass, mass_ratio
    )
    time_to_merger = bilby.gw.utils.calculate_time_to_merger(
        MINIMUM_FREQUENCY, mass_1, mass_2
    )
    duration = max(4, 2 ** int(np.ceil(np.log2(time_to_merger + 2))))

    def waveform_generator(source_model):
        return bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=SAMPLING_FREQUENCY,
            frequency_domain_source_model=source_model,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=deepcopy(WAVEFORM_ARGUMENTS),
        )

    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=SAMPLING_FREQUENCY,
        duration=duration,
        start_time=parameters["geocent_time"] - duration + 2,
    )
    for ifo in ifos:
        ifo.minimum_frequency = MINIMUM_FREQUENCY
    polarizations = waveform_generator(
        bilby.gw.source.lal_binary_black_hole
    ).frequency_domain_strain(parameters)
    snr_squared = sum(
        ifo.optimal_snr_squared(
            ifo.get_detector_response(polarizations, parameters)
        ).real
        for ifo in ifos
    )
    parameters["luminosity_distance"] *= (snr_squared / 200) ** 0.5
    ifos.inject_signal(
        waveform_generator=waveform_generator(bilby.gw.source.lal_binary_black_hole),
        parameters=parameters,
        raise_error=False,
    )

    unbinned = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator(bilby.gw.source.lal_binary_black_hole),
    )
    by_mode = bilby.gw.likelihood.ModeByModeRelativeBinningGravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator(
            bilby.gw.source.lal_binary_black_hole_relative_binning_by_mode
        ),
        fiducial_parameters=parameters,
        epsilon=epsilon,
    )
    summed = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator(
            bilby.gw.source.lal_binary_black_hole_relative_binning
        ),
        fiducial_parameters=parameters,
        epsilon=epsilon,
    )
    return parameters, unbinned, by_mode, summed


class TestModeByModeRelativeBinningLikelihood(unittest.TestCase):
    @parameterized.expand(CHIRP_MASSES)
    def test_equal_mass_zero_spin_matches_both(self, chirp_mass):
        parameters, unbinned, by_mode, summed = make_likelihoods(
            chirp_mass, 1, ZERO_SPINS
        )
        reference = unbinned.log_likelihood_ratio(parameters)
        self.assertAlmostEqual(
            by_mode.log_likelihood_ratio(parameters), reference, delta=1e-3
        )
        self.assertAlmostEqual(
            summed.log_likelihood_ratio(parameters), reference, delta=1e-3
        )

    @parameterized.expand(itertools.product(CHIRP_MASSES, PRECESSING_MASS_RATIOS))
    def test_precessing_matches_unbinned(self, chirp_mass, mass_ratio):
        parameters, unbinned, by_mode, _ = make_likelihoods(
            chirp_mass, mass_ratio, PRECESSING_SPINS
        )
        nearby = parameters.copy()
        nearby.update(chirp_mass=chirp_mass * 1.0002, phase=1.5, a_1=0.45, theta_jn=0.9)
        for point in [parameters, nearby]:
            self.assertAlmostEqual(
                by_mode.log_likelihood_ratio(point),
                unbinned.log_likelihood_ratio(point),
                delta=0.1,
            )

    def test_splines_interpolate_bin_edges(self):
        _, _, by_mode, _ = make_likelihoods(30, 0.5, PRECESSING_SPINS, epsilon=0.5)
        frequency_array = by_mode.interferometers[0].frequency_array
        cardinal = np.asarray(
            by_mode._basis_splines.T @ by_mode._spline_coefficients.T
        ).T
        in_band = (frequency_array >= by_mode.bin_freqs[0]) & (
            frequency_array <= by_mode.bin_freqs[-1]
        )
        np.testing.assert_allclose(cardinal[:, in_band].sum(axis=0), 1, atol=1e-10)
        np.testing.assert_allclose(
            cardinal[:, by_mode.bin_inds], np.eye(len(by_mode.bin_freqs)), atol=1e-10
        )

    def test_safe_fiducial_parameters(self):
        parameters, _, by_mode, _ = make_likelihoods(
            30, 1, PRECESSING_SPINS, epsilon=0.5
        )
        safe = get_safe_fiducial_parameters(parameters, by_mode.waveform_generator)
        self.assertLessEqual(
            bilby.gw.conversion.component_masses_to_symmetric_mass_ratio(
                safe["mass_1"], safe["mass_2"]
            ),
            0.24 + 1e-12,
        )
        self.assertAlmostEqual(
            bilby.gw.conversion.component_masses_to_chirp_mass(
                safe["mass_1"], safe["mass_2"]
            ),
            30,
        )
        iota, spin_1x, spin_1y, _, spin_2x, spin_2y, _ = (
            bilby.gw.conversion.bilby_to_lalsimulation_spins(
                theta_jn=safe["theta_jn"],
                phi_jl=safe["phi_jl"],
                tilt_1=safe["tilt_1"],
                tilt_2=safe["tilt_2"],
                phi_12=safe["phi_12"],
                a_1=safe["a_1"],
                a_2=safe["a_2"],
                mass_1=safe["mass_1"] * bilby.core.utils.solar_mass,
                mass_2=safe["mass_2"] * bilby.core.utils.solar_mass,
                reference_frequency=MINIMUM_FREQUENCY,
                phase=safe["phase"],
            )
        )
        np.testing.assert_allclose([spin_1x, spin_1y, spin_2x, spin_2y], 0, atol=1e-12)
        self.assertAlmostEqual(iota, 1)

    def test_time_marginalization_not_supported(self):
        with self.assertRaises(NotImplementedError):
            bilby.gw.likelihood.ModeByModeRelativeBinningGravitationalWaveTransient(
                interferometers=None, waveform_generator=None, time_marginalization=True
            )


if __name__ == "__main__":
    unittest.main()
