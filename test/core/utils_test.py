import unittest
import os

import dill
import numpy as np
from astropy import constants
import lal
import logging
import matplotlib.pyplot as plt
import h5py
import json
import pytest

import bilby
from bilby.core import utils
from bilby.core.utils import global_meta_data


class TestConstants(unittest.TestCase):
    def test_speed_of_light(self):
        self.assertEqual(utils.speed_of_light, lal.C_SI)
        self.assertLess(
            abs(utils.speed_of_light - constants.c.value) / utils.speed_of_light, 1e-16
        )

    def test_parsec(self):
        self.assertEqual(utils.parsec, lal.PC_SI)
        self.assertLess(abs(utils.parsec - constants.pc.value) / utils.parsec, 1e-11)

    def test_solar_mass(self):
        self.assertEqual(utils.solar_mass, lal.MSUN_SI)
        self.assertLess(
            abs(utils.solar_mass - constants.M_sun.value) / utils.solar_mass, 1e-4
        )

    def test_radius_of_earth(self):
        self.assertEqual(bilby.core.utils.radius_of_earth, lal.REARTH_SI)
        self.assertLess(
            abs(utils.radius_of_earth - constants.R_earth.value)
            / utils.radius_of_earth,
            1e-5,
        )

    def test_gravitational_constant(self):
        self.assertEqual(bilby.core.utils.gravitational_constant, lal.G_SI)


class TestFFT(unittest.TestCase):
    def setUp(self):
        self.sampling_frequency = 10

    def tearDown(self):
        del self.sampling_frequency

    def test_nfft_sine_function(self):
        injected_frequency = 2.7324
        duration = 100
        times = utils.create_time_series(self.sampling_frequency, duration)

        time_domain_strain = np.sin(2 * np.pi * times * injected_frequency + 0.4)

        frequency_domain_strain, frequencies = bilby.core.utils.nfft(
            time_domain_strain, self.sampling_frequency
        )
        frequency_at_peak = frequencies[np.argmax(np.abs(frequency_domain_strain))]
        self.assertAlmostEqual(injected_frequency, frequency_at_peak, places=1)

    def test_nfft_infft(self):
        time_domain_strain = np.random.normal(0, 1, 10)
        frequency_domain_strain, _ = bilby.core.utils.nfft(
            time_domain_strain, self.sampling_frequency
        )
        new_time_domain_strain = bilby.core.utils.infft(
            frequency_domain_strain, self.sampling_frequency
        )
        self.assertTrue(np.allclose(time_domain_strain, new_time_domain_strain))


class TestInferParameters(unittest.TestCase):
    def setUp(self):
        def source_function(freqs, a, b, *args, **kwargs):
            return None

        class TestClass:
            def test_method(self, a, b, *args, **kwargs):
                pass

        class TestClass2:
            def test_method(self, freqs, a, b, *args, **kwargs):
                pass

        self.source1 = source_function
        test_obj = TestClass()
        self.source2 = test_obj.test_method
        test_obj2 = TestClass2()
        self.source3 = test_obj2.test_method

    def tearDown(self):
        del self.source1
        del self.source2

    def test_args_kwargs_handling(self):
        expected = ["a", "b"]
        actual = utils.infer_parameters_from_function(self.source1)
        self.assertListEqual(expected, actual)

    def test_self_handling(self):
        expected = ["a", "b"]
        actual = utils.infer_args_from_method(self.source2)
        self.assertListEqual(expected, actual)

    def test_self_handling_method_as_function(self):
        expected = ["a", "b"]
        actual = utils.infer_parameters_from_function(self.source3)
        self.assertListEqual(expected, actual)


class TestTimeAndFrequencyArrays(unittest.TestCase):
    def setUp(self):
        self.start_time = 1.3
        self.sampling_frequency = 5
        self.duration = 1.6
        self.frequency_array = utils.create_frequency_series(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.time_array = utils.create_time_series(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            starting_time=self.start_time,
        )

    def tearDown(self):
        del self.start_time
        del self.sampling_frequency
        del self.duration
        del self.frequency_array
        del self.time_array

    def test_create_time_array(self):
        expected_time_array = np.array([1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7])
        time_array = utils.create_time_series(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            starting_time=self.start_time,
        )
        self.assertTrue(np.allclose(expected_time_array, time_array))

    def test_create_frequency_array(self):
        expected_frequency_array = np.array([0.0, 0.625, 1.25, 1.875, 2.5])
        frequency_array = utils.create_frequency_series(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.assertTrue(np.allclose(expected_frequency_array, frequency_array))

    def test_get_sampling_frequency_from_time_array(self):
        (
            new_sampling_freq,
            _,
        ) = utils.get_sampling_frequency_and_duration_from_time_array(self.time_array)
        self.assertEqual(self.sampling_frequency, new_sampling_freq)

    def test_get_sampling_frequency_from_time_array_unequally_sampled(self):
        self.time_array[-1] += 0.0001
        with self.assertRaises(ValueError):
            _, _ = utils.get_sampling_frequency_and_duration_from_time_array(
                self.time_array
            )

    def test_get_duration_from_time_array(self):
        _, new_duration = utils.get_sampling_frequency_and_duration_from_time_array(
            self.time_array
        )
        self.assertEqual(self.duration, new_duration)

    def test_get_start_time_from_time_array(self):
        new_start_time = self.time_array[0]
        self.assertEqual(self.start_time, new_start_time)

    def test_get_sampling_frequency_from_frequency_array(self):
        (
            new_sampling_freq,
            _,
        ) = utils.get_sampling_frequency_and_duration_from_frequency_array(
            self.frequency_array
        )
        self.assertEqual(self.sampling_frequency, new_sampling_freq)

    def test_get_sampling_frequency_from_frequency_array_unequally_sampled(self):
        self.frequency_array[-1] += 0.0001
        with self.assertRaises(ValueError):
            _, _ = utils.get_sampling_frequency_and_duration_from_frequency_array(
                self.frequency_array
            )

    def test_get_duration_from_frequency_array(self):
        (
            _,
            new_duration,
        ) = utils.get_sampling_frequency_and_duration_from_frequency_array(
            self.frequency_array
        )
        self.assertEqual(self.duration, new_duration)

    def test_consistency_time_array_to_time_array(self):
        (
            new_sampling_frequency,
            new_duration,
        ) = utils.get_sampling_frequency_and_duration_from_time_array(self.time_array)
        new_start_time = self.time_array[0]
        new_time_array = utils.create_time_series(
            sampling_frequency=new_sampling_frequency,
            duration=new_duration,
            starting_time=new_start_time,
        )
        self.assertTrue(np.allclose(self.time_array, new_time_array))

    def test_consistency_frequency_array_to_frequency_array(self):
        (
            new_sampling_frequency,
            new_duration,
        ) = utils.get_sampling_frequency_and_duration_from_frequency_array(
            self.frequency_array
        )
        new_frequency_array = utils.create_frequency_series(
            sampling_frequency=new_sampling_frequency, duration=new_duration
        )
        self.assertTrue(np.allclose(self.frequency_array, new_frequency_array))

    def test_illegal_sampling_frequency_and_duration(self):
        with self.assertRaises(utils.IllegalDurationAndSamplingFrequencyException):
            _ = utils.create_time_series(
                sampling_frequency=7.7, duration=1.3, starting_time=0
            )


class TestReflect(unittest.TestCase):
    def test_in_range(self):
        xprime = np.array([0.1, 0.5, 0.9])
        x = np.array([0.1, 0.5, 0.9])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)

    def test_in_one_to_two(self):
        xprime = np.array([1.1, 1.5, 1.9])
        x = np.array([0.9, 0.5, 0.1])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)

    def test_in_two_to_three(self):
        xprime = np.array([2.1, 2.5, 2.9])
        x = np.array([0.1, 0.5, 0.9])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)

    def test_in_minus_one_to_zero(self):
        xprime = np.array([-0.9, -0.5, -0.1])
        x = np.array([0.9, 0.5, 0.1])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)

    def test_in_minus_two_to_minus_one(self):
        xprime = np.array([-1.9, -1.5, -1.1])
        x = np.array([0.1, 0.5, 0.9])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)


class TestLatexPlotFormat(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 1)
        self.y = np.sin(self.x)
        self.filename = "test_plot.png"

    def tearDown(self):
        if os.path.isfile(self.filename):
            os.remove(self.filename)

    def test_default(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)

        plot()
        self.assertTrue(os.path.isfile(self.filename))

    def test_mathedefault_one(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)

        plot(BILBY_MATHDEFAULT=1)
        self.assertTrue(os.path.isfile(self.filename))

    def test_mathedefault_zero(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)

        plot(BILBY_MATHDEFAULT=0)
        self.assertTrue(os.path.isfile(self.filename))

    def test_matplotlib_style(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)

        plot(BILBY_STYLE="fivethirtyeight")
        self.assertTrue(os.path.isfile(self.filename))

    def test_user_style(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)

        plot(BILBY_STYLE="test/test.mplstyle")
        self.assertTrue(os.path.isfile(self.filename))


class TestUnsortedInterp2d(unittest.TestCase):
    def setUp(self):
        self.xx = np.linspace(0, 1, 10)
        self.yy = np.linspace(0, 1, 10)
        self.zz = np.random.random((10, 10))
        self.interpolant = bilby.core.utils.BoundedRectBivariateSpline(self.xx, self.yy, self.zz)

    def tearDown(self):
        pass

    def test_returns_float_for_floats(self):
        self.assertIsInstance(self.interpolant(0.5, 0.5), float)

    def test_returns_none_for_floats_outside_range(self):
        self.assertIsNone(self.interpolant(0.5, -0.5))
        self.assertIsNone(self.interpolant(-0.5, 0.5))

    def test_returns_float_for_float_and_array(self):
        self.assertIsInstance(self.interpolant(0.5, np.random.random(10)), np.ndarray)
        self.assertIsInstance(self.interpolant(np.random.random(10), 0.5), np.ndarray)
        self.assertIsInstance(
            self.interpolant(np.random.random(10), np.random.random(10)), np.ndarray
        )

    def test_raises_for_mismatched_arrays(self):
        with self.assertRaises(ValueError):
            self.interpolant(np.random.random(10), np.random.random(20))

    def test_returns_fill_in_correct_place(self):
        x_data = np.random.random(10)
        y_data = np.random.random(10)
        x_data[3] = -1
        self.assertTrue(np.isnan(self.interpolant(x_data, y_data)[3]))


class TestTrapeziumRuleIntegration(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 1, 100)
        self.dxs = np.diff(self.x)
        self.dx = self.dxs[0]
        with np.errstate(divide="ignore"):
            self.lnfunc1 = np.log(self.x)
        self.func1int = (self.x[-1] ** 2 - self.x[0] ** 2) / 2
        with np.errstate(divide="ignore"):
            self.lnfunc2 = np.log(self.x ** 2)
        self.func2int = (self.x[-1] ** 3 - self.x[0] ** 3) / 3

        self.irregularx = np.array(
            [
                self.x[0],
                self.x[12],
                self.x[19],
                self.x[33],
                self.x[49],
                self.x[55],
                self.x[59],
                self.x[61],
                self.x[73],
                self.x[89],
                self.x[93],
                self.x[97],
                self.x[-1],
            ]
        )
        with np.errstate(divide="ignore"):
            self.lnfunc1irregular = np.log(self.irregularx)
            self.lnfunc2irregular = np.log(self.irregularx ** 2)
        self.irregulardxs = np.diff(self.irregularx)

    def test_incorrect_step_type(self):
        with self.assertRaises(TypeError):
            utils.logtrapzexp(self.lnfunc1, "blah")

    def test_inconsistent_step_length(self):
        with self.assertRaises(ValueError):
            utils.logtrapzexp(self.lnfunc1, self.x[0 : len(self.x) // 2])

    def test_integral_func1(self):
        res1 = utils.logtrapzexp(self.lnfunc1, self.dx)
        res2 = utils.logtrapzexp(self.lnfunc1, self.dxs)

        self.assertTrue(np.abs(res1 - res2) < 1e-12)
        self.assertTrue(np.abs((np.exp(res1) - self.func1int) / self.func1int) < 1e-12)

    def test_integral_func2(self):
        res = utils.logtrapzexp(self.lnfunc2, self.dxs)
        self.assertTrue(np.abs((np.exp(res) - self.func2int) / self.func2int) < 1e-4)

    def test_integral_func1_irregular_steps(self):
        res = utils.logtrapzexp(self.lnfunc1irregular, self.irregulardxs)
        self.assertTrue(np.abs((np.exp(res) - self.func1int) / self.func1int) < 1e-12)

    def test_integral_func2_irregular_steps(self):
        res = utils.logtrapzexp(self.lnfunc2irregular, self.irregulardxs)
        self.assertTrue(np.abs((np.exp(res) - self.func2int) / self.func2int) < 1e-2)


class TestSavingNumpyRandomGenerator(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def init_outdir(self, tmp_path):
        # Use pytest's tmp_path fixture to create a temporary directory
        self.outdir = tmp_path / "test"
        self.outdir.mkdir()

    def setUp(self):
        self.filename = "test_random_state.npy"
        self.data = {
            "rng": np.random.default_rng(),
            "seed": 1234,
        }

    def test_hdf5(self):
        with h5py.File(self.outdir / "test.h5", "w") as f:
            bilby.core.utils.recursively_save_dict_contents_to_group(
                f, "/", self.data
            )
        a = self.data["rng"].random()

        with h5py.File(self.outdir / "test.h5", "r") as f:
            data = bilby.core.utils.recursively_load_dict_contents_from_group(f, "/")

        b = data["rng"].random()
        self.assertEqual(a, b)

    def test_json(self):
        with open(self.outdir / "test.json", 'w') as file:
            json.dump(self.data, file, indent=2, cls=bilby.core.utils.BilbyJsonEncoder)

        a = self.data["rng"].random()

        with open(self.outdir / "test.json", 'r') as file:
            data = json.load(file, object_hook=bilby.core.utils.decode_bilby_json)

        b = data["rng"].random()
        self.assertEqual(a, b)

    def test_pickle(self):
        with open(self.outdir / "test.pkl", 'wb') as file:
            dill.dump(self.data, file)
        a = self.data["rng"].random()

        with open(self.outdir / "test.pkl", 'rb') as file:
            data = dill.load(file)
        b = data["rng"].random()
        self.assertEqual(a, b)


class TestGlobalMetaData(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def set_caplog(self, caplog):
        self._caplog = caplog

    def setUp(self):
        global_meta_data.clear()
        global_meta_data["rng"] = bilby.core.utils.random.rng
        bilby.gw.cosmology.DEFAULT_COSMOLOGY = None
        bilby.gw.cosmology.COSMOLOGY = [None, str(None)]

    def tearDown(self):
        global_meta_data.clear()
        global_meta_data["rng"] = bilby.core.utils.random.rng
        bilby.gw.cosmology.DEFAULT_COSMOLOGY = None
        bilby.gw.cosmology.COSMOLOGY = [None, str(None)]

    def test_set_item(self):
        global_meta_data["test"] = 123
        self.assertEqual(global_meta_data["test"], 123)

    def test_set_rng(self):
        bilby.core.utils.random.seed(1234)
        self.assertTrue(global_meta_data["rng"] is bilby.core.utils.random.rng)
        self.assertEqual(global_meta_data["seed"], 1234)

    def test_set_cosmology(self):
        bilby.gw.cosmology.set_cosmology("Planck15_LAL")
        self.assertTrue(global_meta_data["cosmology"] is bilby.gw.cosmology.COSMOLOGY[0])

    def test_update(self):
        bilby.core.utils.meta_data.logger.propagate = True
        with self._caplog.at_level(logging.DEBUG, logger="bilby"):
            global_meta_data.update({"test": 123})
        assert "Setting meta data key test with value 123" in str(self._caplog.text)

    def test_init(self):
        bilby.core.utils.meta_data.logger.propagate = True
        with self._caplog.at_level(logging.DEBUG, logger="bilby"):
            bilby.core.utils.GlobalMetaData({"test": 123})
        assert "Setting meta data key test with value 123" in str(self._caplog.text)
        assert "GlobalMetaData has already been instantiated" in str(self._caplog.text)


if __name__ == "__main__":
    unittest.main()
