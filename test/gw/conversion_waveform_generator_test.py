import unittest
from types import SimpleNamespace

from bilby.gw.conversion import _generate_all_cbc_parameters


def _identity_conversion(sample):
    return sample, list()


class TestCbcWaveformGeneratorBind(unittest.TestCase):
    def setUp(self):
        self.defaults = {
            "reference_frequency": 50.0,
            "waveform_approximant": "IMRPhenomPv2",
            "minimum_frequency": 20.0,
        }
        self.sample = {"mass_1": 38.9, "mass_2": 31.6}
        self.generator = SimpleNamespace(
            waveform_arguments={
                "reference_frequency": 20.0,
                "waveform_approximant": "IMRPhenomXPHM",
                "minimum_frequency": 8.0,
            }
        )

    def tearDown(self):
        del self.defaults
        del self.sample
        del self.generator

    def test_supplied_generator_occupies_waveform_arguments(self):
        dest = _generate_all_cbc_parameters(
            self.sample,
            self.defaults,
            _identity_conversion,
            waveform_generator=self.generator,
        )
        self.assertEqual(dest["reference_frequency"], 20.0)
        self.assertEqual(dest["waveform_approximant"], "IMRPhenomXPHM")
        self.assertEqual(dest["minimum_frequency"], 8.0)

    def test_absent_generator_keeps_hardcoded_defaults(self):
        dest = _generate_all_cbc_parameters(
            self.sample, self.defaults, _identity_conversion
        )
        self.assertEqual(dest["reference_frequency"], 50.0)
        self.assertEqual(dest["waveform_approximant"], "IMRPhenomPv2")
        self.assertEqual(dest["minimum_frequency"], 20.0)

    def test_likelihood_waveform_generator_still_binds(self):
        likelihood = SimpleNamespace(waveform_generator=self.generator)
        dest = _generate_all_cbc_parameters(
            self.sample, self.defaults, _identity_conversion, likelihood=likelihood
        )
        self.assertEqual(dest["reference_frequency"], 20.0)
        self.assertEqual(dest["waveform_approximant"], "IMRPhenomXPHM")
        self.assertEqual(dest["minimum_frequency"], 8.0)

    def test_explicit_generator_wins_over_likelihood(self):
        other = SimpleNamespace(
            waveform_arguments={
                "reference_frequency": 15.0,
                "waveform_approximant": "IMRPhenomPv2",
                "minimum_frequency": 10.0,
            }
        )
        likelihood = SimpleNamespace(waveform_generator=other)
        dest = _generate_all_cbc_parameters(
            self.sample,
            self.defaults,
            _identity_conversion,
            likelihood=likelihood,
            waveform_generator=self.generator,
        )
        self.assertEqual(dest["reference_frequency"], 20.0)
        self.assertEqual(dest["minimum_frequency"], 8.0)
