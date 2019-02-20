from __future__ import division, absolute_import
import unittest
import bilby


class TestLalBBH(unittest.TestCase):

    def setUp(self):
        self.parameters = dict(
            mass_1=30.0, mass_2=30.0, luminosity_distance=400.0, a_1=0.0,
            tilt_1=0.0, phi_12=0.0, a_2=0.0, tilt_2=0.0, phi_jl=0.0, iota=0.0,
            phase=0.0)
        self.waveform_kwargs = dict(
            waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
            minimum_frequency=20.0)
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 4)

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs
        del self.frequency_array

    def test_lal_bbh_works_runs_valid_parameters(self):
        self.assertIsInstance(
            bilby.gw.source.lal_binary_black_hole(
                self.frequency_array, **self.parameters,
                **self.waveform_kwargs), dict)

    def test_mass_ratio_greater_one_returns_none(self):
        self.parameters['mass_2'] = 1000.0
        self.assertIsNone(
            bilby.gw.source.lal_binary_black_hole(
                self.frequency_array, **self.parameters,
                **self.waveform_kwargs), dict)

    def test_lal_bbh_works_without_waveform_parameters(self):
        self.assertIsInstance(
            bilby.gw.source.lal_binary_black_hole(
                self.frequency_array, **self.parameters), dict)


class TestLalBNS(unittest.TestCase):

    def setUp(self):
        self.parameters = dict(
            mass_1=1.4, mass_2=1.4, luminosity_distance=400.0, chi_1=0.0,
            chi_2=0.0, iota=0.0, phase=0.0, lambda_1=0.0, lambda_2=0.0)
        self.waveform_kwargs = dict(
            waveform_approximant='TaylorF2', reference_frequency=50.0,
            minimum_frequency=20.0)
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 4)

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs
        del self.frequency_array

    def test_lal_bns_works_runs_valid_parameters(self):
        self.assertIsInstance(
            bilby.gw.source.lal_binary_neutron_star(
                self.frequency_array, **self.parameters,
                **self.waveform_kwargs), dict)

    def test_mass_ratio_greater_one_returns_none(self):
        self.parameters['mass_2'] = 1000.0
        self.assertIsNone(
            bilby.gw.source.lal_binary_neutron_star(
                self.frequency_array, **self.parameters,
                **self.waveform_kwargs), dict)

    def test_lal_bns_works_without_waveform_parameters(self):
        self.assertIsInstance(
            bilby.gw.source.lal_binary_neutron_star(
                self.frequency_array, **self.parameters), dict)

    def test_fails_without_tidal_parameters(self):
        self.parameters.pop('lambda_1')
        self.parameters.pop('lambda_2')
        with self.assertRaises(TypeError):
            bilby.gw.source.lal_binary_neutron_star(
                self.frequency_array, **self.parameters, **self.waveform_kwargs)

    def test_fails_without_aligned_spins(self):
        self.parameters.pop('chi_1')
        self.parameters.pop('chi_2')
        with self.assertRaises(TypeError):
            bilby.gw.source.lal_binary_neutron_star(
                self.frequency_array, **self.parameters, **self.waveform_kwargs)


class TestEccentricLalBBH(unittest.TestCase):

    def setUp(self):
        self.parameters = dict(
            mass_1=30.0, mass_2=30.0, luminosity_distance=400.0, iota=0.0,
            phase=0.0, eccentricity=0.1)
        self.waveform_kwargs = dict(
            waveform_approximant='EccentricFD', reference_frequency=10.0,
            minimum_frequency=10.0)
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 4)

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs
        del self.frequency_array

    def test_lal_ebbh_works_runs_valid_parameters(self):
        self.assertIsInstance(
            bilby.gw.source.lal_eccentric_binary_black_hole_no_spins(
                self.frequency_array, **self.parameters,
                **self.waveform_kwargs), dict)

    def test_mass_ratio_greater_one_returns_none(self):
        self.parameters['mass_2'] = 1000.0
        self.assertIsNone(
            bilby.gw.source.lal_eccentric_binary_black_hole_no_spins(
                self.frequency_array, **self.parameters,
                **self.waveform_kwargs), dict)

    def test_lal_ebbh_works_without_waveform_parameters(self):
        self.assertIsInstance(
            bilby.gw.source.lal_eccentric_binary_black_hole_no_spins(
                self.frequency_array, **self.parameters), dict)

    def test_fails_without_eccentricity(self):
        self.parameters.pop('eccentricity')
        with self.assertRaises(TypeError):
            bilby.gw.source.lal_eccentric_binary_black_hole_no_spins(
                self.frequency_array, **self.parameters, **self.waveform_kwargs)


if __name__ == '__main__':
    unittest.main()
