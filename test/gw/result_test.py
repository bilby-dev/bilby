import os
import logging
import unittest

import pandas as pd
from parameterized import parameterized, parameterized_class
import pytest

import bilby


class BaseCBCResultTest(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def init_outdir(self, tmp_path):
        # Use pytest's tmp_path fixture to create a temporary directory
        self.outdir = str(tmp_path / "test")

    def setUp(self):
        bilby.utils.command_line_args.bilby_test_mode = False
        priors = bilby.gw.prior.BBHPriorDict()
        priors["geocent_time"] = 2
        injection_parameters = priors.sample()
        self.meta_data = dict(
            likelihood=dict(
                phase_marginalization=True,
                distance_marginalization=False,
                time_marginalization=True,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                waveform_arguments=dict(
                    reference_frequency=20.0, waveform_approximant="IMRPhenomPv2"
                ),
                interferometers=dict(
                    H1=dict(optimal_SNR=1, parameters=injection_parameters),
                    L1=dict(optimal_SNR=1, parameters=injection_parameters),
                ),
                sampling_frequency=4096,
                duration=4,
                start_time=0,
                waveform_generator_class=bilby.gw.waveform_generator.WaveformGenerator,
                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            )
        )
        self.result = bilby.gw.result.CBCResult(
            label="label",
            outdir=self.outdir,
            sampler="nestle",
            search_parameter_keys=list(priors.keys()),
            fixed_parameter_keys=list(),
            priors=priors,
            sampler_kwargs=dict(test="test", func=lambda x: x),
            injection_parameters=injection_parameters,
            meta_data=self.meta_data,
            posterior=pd.DataFrame(priors.sample(100)),
        )
        if not os.path.isdir(self.result.outdir):
            os.mkdir(self.result.outdir)

    def tearDown(self):
        bilby.utils.command_line_args.bilby_test_mode = True
        del self.result


class TestCBCResult(BaseCBCResultTest):

    @pytest.fixture(autouse=True)
    def set_caplog(self, caplog):
        self._caplog = caplog

    def test_phase_marginalization(self):
        self.assertEqual(
            self.result.phase_marginalization,
            self.meta_data["likelihood"]["phase_marginalization"],
        )

    def test_phase_marginalization_unset(self):
        self.result.meta_data["likelihood"].pop("phase_marginalization")
        with self.assertRaises(AttributeError):
            self.result.phase_marginalization

    def test_time_marginalization(self):
        self.assertEqual(
            self.result.time_marginalization,
            self.meta_data["likelihood"]["time_marginalization"],
        )

    def test_time_marginalization_unset(self):
        self.result.meta_data["likelihood"].pop("time_marginalization")
        with self.assertRaises(AttributeError):
            self.result.time_marginalization

    def test_distance_marginalization(self):
        self.assertEqual(
            self.result.distance_marginalization,
            self.meta_data["likelihood"]["distance_marginalization"],
        )

    def test_distance_marginalization_unset(self):
        self.result.meta_data["likelihood"].pop("distance_marginalization")
        with self.assertRaises(AttributeError):
            self.result.distance_marginalization

    def test_reference_frequency(self):
        self.assertEqual(
            self.result.reference_frequency,
            self.meta_data["likelihood"]["waveform_arguments"]["reference_frequency"],
        )

    def test_reference_frequency_unset(self):
        self.result.meta_data["likelihood"]["waveform_arguments"].pop(
            "reference_frequency"
        )
        with self.assertRaises(AttributeError):
            self.result.reference_frequency

    def test_sampling_frequency(self):
        self.assertEqual(
            self.result.sampling_frequency,
            self.meta_data["likelihood"]["sampling_frequency"],
        )

    def test_sampling_frequency_unset(self):
        self.result.meta_data["likelihood"].pop("sampling_frequency")
        with self.assertRaises(AttributeError):
            self.result.sampling_frequency

    def test_duration(self):
        self.assertEqual(self.result.duration, self.meta_data["likelihood"]["duration"])

    def test_duration_unset(self):
        self.result.meta_data["likelihood"].pop("duration")
        with self.assertRaises(AttributeError):
            self.result.duration

    def test_start_time(self):
        self.assertEqual(
            self.result.start_time, self.meta_data["likelihood"]["start_time"]
        )

    def test_start_time_unset(self):
        self.result.meta_data["likelihood"].pop("start_time")
        with self.assertRaises(AttributeError):
            self.result.start_time

    def test_waveform_approximant(self):
        self.assertEqual(
            self.result.waveform_approximant,
            self.meta_data["likelihood"]["waveform_arguments"]["waveform_approximant"],
        )

    def test_waveform_approximant_unset(self):
        self.result.meta_data["likelihood"]["waveform_arguments"].pop(
            "waveform_approximant"
        )
        with self.assertRaises(AttributeError):
            self.result.waveform_approximant

    def test_waveform_arguments(self):
        self.assertEqual(
            self.result.waveform_arguments,
            self.meta_data["likelihood"]["waveform_arguments"],
        )

    def test_frequency_domain_source_model(self):
        self.assertEqual(
            self.result.frequency_domain_source_model,
            self.meta_data["likelihood"]["frequency_domain_source_model"],
        )

    def test_frequency_domain_source_model_unset(self):
        self.result.meta_data["likelihood"].pop("frequency_domain_source_model")
        with self.assertRaises(AttributeError):
            self.result.frequency_domain_source_model

    def test_parameter_conversion(self):
        self.assertEqual(
            self.result.parameter_conversion,
            self.meta_data["likelihood"]["parameter_conversion"],
        )

    def test_parameter_conversion_unset(self):
        self.result.meta_data["likelihood"].pop("parameter_conversion")
        with self.assertRaises(AttributeError):
            self.result.parameter_conversion

    def test_waveform_generator_class(self):
        self.assertEqual(
            self.result.waveform_generator_class,
            self.meta_data["likelihood"]["waveform_generator_class"],
        )

    def test_waveform_generator_class_unset(self):
        self.result.meta_data["likelihood"].pop("waveform_generator_class")
        with self.assertRaises(AttributeError):
            self.result.waveform_generator_class

    def test_interferometer_names(self):
        self.assertEqual(
            self.result.interferometers,
            [name for name in self.meta_data["likelihood"]["interferometers"]],
        )

    def test_detector_injection_properties(self):
        self.assertEqual(
            self.result.detector_injection_properties("H1"),
            self.meta_data["likelihood"]["interferometers"]["H1"],
        )

    def test_detector_injection_properties_no_injection(self):
        self.assertEqual(
            self.result.detector_injection_properties("not_a_detector"), None
        )


@parameterized_class(
    ["include_global_meta_data"], [["True"], ["False"]]
)
class CBCResultsGlobalMetaDataTest(BaseCBCResultTest):

    @pytest.fixture(autouse=True)
    def set_caplog(self, caplog):
        self._caplog = caplog

    def setUp(self):
        # Current default is False
        self.meta_data_env_var = os.getenv("BILBY_INCLUDE_GLOBAL_META_DATA") or "False"
        os.environ["BILBY_INCLUDE_GLOBAL_META_DATA"] = self.include_global_meta_data
        super().setUp()

    def tearDown(self):
        super().tearDown()
        os.environ["BILBY_INCLUDE_GLOBAL_META_DATA"] = self.meta_data_env_var

    def test_global_meta_data(self):
        if self.include_global_meta_data == "True":
            assert "global_meta_data" in self.result.meta_data
        else:
            assert "global_meta_data" not in self.result.meta_data

    def test_cosmology(self):
        bilby.core.utils.meta_data.logger.propagate = True
        with self._caplog.at_level(logging.DEBUG, logger="bilby"):
            cosmology = self.result.cosmology
        if self.include_global_meta_data == "True":
            self.assertEqual(
                self.result.cosmology,
                self.meta_data["global_meta_data"]["cosmology"],
            )
        else:
            self.assertEqual(cosmology, None)
            assert "not containing global meta data" in str(self._caplog.text)


@parameterized_class(
    ["cosmology_name"],
    [["Planck15"], ["Planck15_LAL"]]
)
class TestCBCResultSaveAndLoad(BaseCBCResultTest):

    def setUp(self):
        self.orig_cosmology = bilby.gw.cosmology.get_cosmology()
        bilby.gw.cosmology.set_cosmology(self.cosmology_name)
        return super().setUp()

    def tearDown(self):
        super().tearDown()
        bilby.gw.cosmology.set_cosmology(self.orig_cosmology)

    @parameterized.expand(
        [
            ("json", False),
            ("json", True),
            ("pkl", False),
            ("hdf5", False),
        ]
    )
    def test_save_and_load(self, extension, gzip):
        self.result.save_to_file(extension=extension, gzip=gzip)
        loaded_result = bilby.core.result.read_in_result(
            outdir=self.result.outdir,
            label=self.result.label,
            extension=extension,
            gzip=gzip,
            result_class=bilby.gw.result.CBCResult,
        )
        self.assertEqual(self.result.cosmology, loaded_result.cosmology)
        self.assertEqual(self.result.parameter_conversion, loaded_result.parameter_conversion)
        self.assertEqual(self.result.frequency_domain_source_model, loaded_result.frequency_domain_source_model)
        self.assertEqual(self.result.waveform_generator_class, loaded_result.waveform_generator_class)
        self.assertEqual(self.result.waveform_arguments, loaded_result.waveform_arguments)
        self.assertEqual(self.result.interferometers, loaded_result.interferometers)


if __name__ == "__main__":
    unittest.main()
