import os
import shutil
import unittest

import pandas as pd

import bilby


class TestCBCResult(unittest.TestCase):
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
                time_domain_source_model=None,
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
            outdir="outdir",
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
        try:
            shutil.rmtree(self.result.outdir)
        except OSError:
            pass
        del self.result

    def test_calibration_plot(self):
        calibration_prior = bilby.gw.prior.CalibrationPriorDict.constant_uncertainty_spline(
            amplitude_sigma=0.1,
            phase_sigma=0.1,
            minimum_frequency=20,
            maximum_frequency=2048,
            label="recalib_H1_",
            n_nodes=5,
        )
        calibration_filename = (
            f"{self.result.outdir}/{self.result.label}_calibration.png"
        )
        for key in calibration_prior:
            self.result.posterior[key] = calibration_prior[key].sample(100)
        self.result.plot_calibration_posterior()
        self.assertTrue(os.path.exists(calibration_filename))

    def test_calibration_plot_returns_none_with_no_calibration_parameters(self):
        self.assertIsNone(self.result.plot_calibration_posterior())
        calibration_filename = (
            f"{self.result.outdir}/{self.result.label}_calibration.png"
        )
        self.assertFalse(os.path.exists(calibration_filename))

    def test_calibration_pdf_plot(self):
        calibration_prior = bilby.gw.prior.CalibrationPriorDict.constant_uncertainty_spline(
            amplitude_sigma=0.1,
            phase_sigma=0.1,
            minimum_frequency=20,
            maximum_frequency=2048,
            label="recalib_H1_",
            n_nodes=5,
        )
        calibration_filename = (
            f"{self.result.outdir}/{self.result.label}_calibration.pdf"
        )
        for key in calibration_prior:
            self.result.posterior[key] = calibration_prior[key].sample(100)
        self.result.plot_calibration_posterior(format="pdf")
        self.assertTrue(os.path.exists(calibration_filename))

    def test_calibration_invalid_format_raises_error(self):
        with self.assertRaises(ValueError):
            self.result.plot_calibration_posterior(format="bilby")

    def test_waveform_plotting_png(self):
        self.result.plot_waveform_posterior(n_samples=200)
        for ifo in self.result.interferometers:
            self.assertTrue(
                os.path.exists(
                    f"{self.result.outdir}/{self.result.label}_{ifo}_waveform.png"
                )
            )

    def test_plot_skymap_meta_data(self):
        from ligo.skymap import io

        expected_keys = {
            "HISTORY",
            "creator",
            "distmean",
            "diststd",
            "gps_creation_time",
            "gps_time",
            "nest",
            "objid",
            "origin",
            "vcs_version",
            "instruments",
        }
        self.result.plot_skymap(maxpts=50, geo=False, objid="test", instruments="H1L1")
        fits_filename = f"{self.result.outdir}/{self.result.label}_skymap.fits"
        skymap_filename = f"{self.result.outdir}/{self.result.label}_skymap.png"
        pickle_filename = f"{self.result.outdir}/{self.result.label}_skypost.obj"
        hpmap, meta = io.read_sky_map(fits_filename)
        self.assertEqual(expected_keys, set(meta.keys()))
        self.assertTrue(os.path.exists(skymap_filename))
        self.assertTrue(os.path.exists(pickle_filename))
        self.result.plot_skymap(
            maxpts=50,
            geo=False,
            objid="test",
            instruments="H1L1",
            load_pickle=True,
            colorbar=False,
        )


if __name__ == "__main__":
    unittest.main()
