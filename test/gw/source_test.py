import unittest

import numpy as np
from copy import copy

import bilby


class TestLalBBH(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            mass_1=30.0,
            mass_2=30.0,
            luminosity_distance=400.0,
            a_1=0.4,
            tilt_1=0.2,
            phi_12=1.0,
            a_2=0.8,
            tilt_2=2.7,
            phi_jl=2.9,
            theta_jn=0.3,
            phase=0.0,
        )
        self.waveform_kwargs = dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=50.0,
            minimum_frequency=20.0,
            catch_waveform_errors=True,
        )
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 4)
        self.bad_parameters = copy(self.parameters)
        self.bad_parameters["mass_1"] = -30.0

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs
        del self.frequency_array
        del self.bad_parameters

    def test_lal_bbh_works_runs_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs)
        self.assertIsInstance(
            bilby.gw.source.lal_binary_black_hole(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_waveform_error_catching(self):
        self.bad_parameters.update(self.waveform_kwargs)
        self.assertIsNone(
            bilby.gw.source.lal_binary_black_hole(
                self.frequency_array, **self.bad_parameters
            )
        )

    def test_waveform_error_raising(self):
        raise_error_parameters = copy(self.bad_parameters)
        raise_error_parameters.update(self.waveform_kwargs)
        raise_error_parameters["catch_waveform_errors"] = False
        with self.assertRaises(Exception):
            bilby.gw.source.lal_binary_black_hole(
                self.frequency_array, **raise_error_parameters
            )

    def test_lal_bbh_works_without_waveform_parameters(self):
        self.assertIsInstance(
            bilby.gw.source.lal_binary_black_hole(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    # Removed due to issue with SimInspiralFD - see https://git.ligo.org/lscsoft/lalsuite/issues/153
    # def test_lal_bbh_works_with_time_domain_approximant(self):
    #     self.waveform_kwargs['waveform_approximant'] = 'SEOBNRv3'
    #     self.parameters.update(self.waveform_kwargs)
    #     self.assertIsInstance(
    #         bilby.gw.source.lal_binary_black_hole(
    #             self.frequency_array, **self.parameters), dict)

    def test_lal_bbh_xpprecession_version(self):
        self.parameters.update(self.waveform_kwargs)
        self.parameters["waveform_approximant"] = "IMRPhenomXP"

        # Test that we can modify the XP precession version
        out_v223 = bilby.gw.source.lal_binary_black_hole(
            self.frequency_array, PhenomXPrecVersion=223, **self.parameters
        )
        out_v102 = bilby.gw.source.lal_binary_black_hole(
            self.frequency_array, PhenomXPrecVersion=102, **self.parameters
        )
        self.assertFalse(np.all(out_v223["plus"] == out_v102["plus"]))


class TestLalBNS(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            mass_1=1.4,
            mass_2=1.4,
            luminosity_distance=400.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.2,
            tilt_2=1.7,
            phi_jl=0.2,
            phi_12=0.9,
            theta_jn=1.7,
            phase=0.0,
            lambda_1=100.0,
            lambda_2=100.0,
        )
        self.waveform_kwargs = dict(
            waveform_approximant="IMRPhenomPv2_NRTidal",
            reference_frequency=50.0,
            minimum_frequency=20.0,
        )
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 4)

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs
        del self.frequency_array

    def test_lal_bns_runs_with_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs)
        self.assertIsInstance(
            bilby.gw.source.lal_binary_neutron_star(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_lal_bns_works_without_waveform_parameters(self):
        self.assertIsInstance(
            bilby.gw.source.lal_binary_neutron_star(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_fails_without_tidal_parameters(self):
        self.parameters.pop("lambda_1")
        self.parameters.pop("lambda_2")
        self.parameters.update(self.waveform_kwargs)
        with self.assertRaises(TypeError):
            bilby.gw.source.lal_binary_neutron_star(
                self.frequency_array, **self.parameters
            )


class TestEccentricLalBBH(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            mass_1=30.0,
            mass_2=30.0,
            luminosity_distance=400.0,
            theta_jn=0.0,
            phase=0.0,
            eccentricity=0.1,
        )
        self.waveform_kwargs = dict(
            waveform_approximant="EccentricFD",
            reference_frequency=10.0,
            minimum_frequency=10.0,
        )
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 4)

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs
        del self.frequency_array

    def test_lal_ebbh_works_runs_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs)
        self.assertIsInstance(
            bilby.gw.source.lal_eccentric_binary_black_hole_no_spins(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_lal_ebbh_works_without_waveform_parameters(self):
        self.assertIsInstance(
            bilby.gw.source.lal_eccentric_binary_black_hole_no_spins(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_fails_without_eccentricity(self):
        self.parameters.pop("eccentricity")
        self.parameters.update(self.waveform_kwargs)
        with self.assertRaises(TypeError):
            bilby.gw.source.lal_eccentric_binary_black_hole_no_spins(
                self.frequency_array, **self.parameters
            )


class TestROQBBH(unittest.TestCase):
    def setUp(self):
        roq_dir = "/roq_basis"

        fnodes_linear_file = "{}/fnodes_linear.npy".format(roq_dir)
        fnodes_linear = np.load(fnodes_linear_file).T
        fnodes_quadratic_file = "{}/fnodes_quadratic.npy".format(roq_dir)
        fnodes_quadratic = np.load(fnodes_quadratic_file).T

        self.parameters = dict(
            mass_1=30.0,
            mass_2=30.0,
            luminosity_distance=400.0,
            a_1=0.0,
            tilt_1=0.0,
            phi_12=0.0,
            a_2=0.0,
            tilt_2=0.0,
            phi_jl=0.0,
            theta_jn=0.0,
            phase=0.0,
        )
        self.waveform_kwargs = dict(
            frequency_nodes_linear=fnodes_linear,
            frequency_nodes_quadratic=fnodes_quadratic,
            reference_frequency=50.0,
            minimum_frequency=20.0,
            approximant="IMRPhenomPv2",
        )
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 4)

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs
        del self.frequency_array

    def test_roq_runs_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs)
        self.assertIsInstance(
            bilby.gw.source.roq(self.frequency_array, **self.parameters), dict
        )

    def test_roq_fails_without_frequency_nodes(self):
        self.parameters.update(self.waveform_kwargs)
        del self.parameters["frequency_nodes_linear"]
        del self.parameters["frequency_nodes_quadratic"]
        with self.assertRaises(KeyError):
            bilby.gw.source.roq(self.frequency_array, **self.parameters)


if __name__ == "__main__":
    unittest.main()
