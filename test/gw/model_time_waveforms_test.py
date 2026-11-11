import pickle
import unittest

import lalsimulation
import numpy as np
import pytest

import bilby
from bilby.gw.model_time_waveforms import spin_arguments


class TestSpinArguments(unittest.TestCase):
    def test_aligned_spin_models_take_the_z_components(self):
        self.assertEqual(spin_arguments("IMRPhenomT", 0, 0, 0.1, 0, 0, 0.2), (0.1, 0.2))
        self.assertEqual(spin_arguments("IMRPhenomTHM", 0, 0, 0.1, 0, 0, 0.2), (0.1, 0.2))

    def test_precessing_models_take_all_components(self):
        self.assertEqual(spin_arguments("IMRPhenomTPHM", 1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6))

    def test_in_plane_spins_with_an_aligned_spin_model_raise(self):
        with self.assertRaises(ValueError):
            spin_arguments("IMRPhenomT", 0.1, 0, 0.1, 0, 0, 0.2)

    def test_other_approximants_raise(self):
        with self.assertRaises(ValueError):
            spin_arguments("IMRPhenomXPHM", 0, 0, 0, 0, 0, 0)

    def test_missing_lalsimulation_function_raises(self):
        from bilby.gw.model_time_waveforms import _POLARIZATION_FUNCTION, _get_approximant
        with self.assertRaises(ImportError):
            _get_approximant(_POLARIZATION_FUNCTION, "IMRPhenomNotAModel", True)


@pytest.mark.skipif(
    not hasattr(lalsimulation, "SimIMRPhenomTModelTimesTruncated"),
    reason="lalsimulation without the IMRPhenomT model-time functions",
)
class TestModelTimeWaveforms(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            mass_1=36.0,
            mass_2=29.0,
            luminosity_distance=1000.0,
            a_1=0.4,
            tilt_1=0.5,
            phi_12=1.7,
            a_2=0.3,
            tilt_2=1.0,
            phi_jl=0.3,
            theta_jn=0.4,
            phase=1.3,
        )
        self.aligned_spins = dict(tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0)
        self.time_array = bilby.core.utils.create_time_series(2048, 4)
        self.times = np.linspace(-0.5, 0.05, 300)

    def tearDown(self):
        del self.parameters
        del self.time_array
        del self.times

    def _waveform(self, approximant, **changes):
        parameters = dict(self.parameters, **changes)
        if approximant in ("IMRPhenomT", "IMRPhenomTHM"):
            parameters.update(self.aligned_spins)
        return bilby.gw.source.lal_binary_black_hole_time_domain_heterodyning(
            self.time_array, waveform_approximant=approximant, reference_frequency=20.0,
            minimum_frequency=10.0, fiducial=1, **parameters)

    def test_modes_and_harmonics_reproduce_the_polarizations(self):
        """ h_+ - i h_x = sum over modes of Z h_lm + Zbar h_lm^*, with constant (aligned spins)
        or time-dependent (precessing) harmonic factors """
        for approximant in ["IMRPhenomT", "IMRPhenomTHM", "IMRPhenomTP", "IMRPhenomTPHM"]:
            waveform = self._waveform(approximant)
            modes, harmonics = waveform["heterodyning_terms_at"](self.times)
            polarizations = waveform["polarizations_at"](self.times)
            h = sum(harmonics[key][0] * mode + harmonics[key][1] * np.conj(mode) for key, mode in modes.items())
            scale = np.max(abs(polarizations["plus"]))
            np.testing.assert_allclose(h.real, polarizations["plus"], atol=1e-12 * scale)
            np.testing.assert_allclose(-h.imag, polarizations["cross"], atol=1e-12 * scale)

    def test_modes(self):
        modes, _ = self._waveform("IMRPhenomTHM")["heterodyning_terms_at"](self.times)
        self.assertEqual(sorted(modes), [(2, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
        modes, _ = self._waveform("IMRPhenomTPHM")["heterodyning_terms_at"](self.times)
        self.assertEqual(sorted(modes), [(2, 1), (2, 2), (3, 3), (4, 4), (5, 5)])

    def test_distance_scaling(self):
        near = self._waveform("IMRPhenomTPHM")["polarizations_at"](self.times)
        far = self._waveform("IMRPhenomTPHM", luminosity_distance=2000.0)["polarizations_at"](self.times)
        for mode in ["plus", "cross"]:
            np.testing.assert_allclose(far[mode], near[mode] / 2, rtol=1e-10, atol=1e-35)

    def test_times_outside_the_model_give_zeros(self):
        polarizations = self._waveform("IMRPhenomTHM")["polarizations_at"](np.array([-1e3, 1e3]))
        np.testing.assert_array_equal(polarizations["plus"], 0)

    def test_waveform_can_be_pickled(self):
        polarizations_at = self._waveform("IMRPhenomTPHM")["polarizations_at"]
        restored = pickle.loads(pickle.dumps(polarizations_at))
        for mode in ["plus", "cross"]:
            np.testing.assert_array_equal(restored(self.times)[mode], polarizations_at(self.times)[mode])


if __name__ == "__main__":
    unittest.main()
