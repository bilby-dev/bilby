import os
import pickle
import shutil
import tempfile
import unittest

import dill
import lalsimulation
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import trapezoid

import bilby
from bilby.gw.time_domain_utils import gohberg_semencul_product

GPS_START = 1126259640.0
SAMPLING_FREQUENCY = 1024.0
DURATION = 8.0
INJECTION = dict(
    luminosity_distance=200.0,
    frequency=150.0,
    phase=0.3,
    ra=1.375,
    dec=-1.2108,
    psi=2.659,
    geocent_time=GPS_START + 4.3 + 0.37 / SAMPLING_FREQUENCY,
)


def burst(times, luminosity_distance, frequency, phase):
    """A band-limited sine-Gaussian burst peaking at t = 0"""
    envelope = 1e-21 * 100 / luminosity_distance * np.exp(-((times / 0.04) ** 2))
    return dict(
        plus=envelope * np.cos(2 * np.pi * frequency * times + phase),
        cross=envelope * np.sin(2 * np.pi * frequency * times + phase),
    )


def burst_time_domain(time_array, luminosity_distance, frequency, phase):
    """The burst as a time-domain source model: the waveform on its own grid and its epoch"""
    delta_t = time_array[1] - time_array[0]
    times = np.arange(-int(round(0.4 / delta_t)), int(round(0.2 / delta_t))) * delta_t
    return dict(burst(times, luminosity_distance, frequency, phase), epoch=times[0])


def burst_model_time(time_array, luminosity_distance, frequency, phase):
    """The burst as a model-time source model"""
    return dict(
        polarizations_at=lambda times: burst(
            np.asarray(times), luminosity_distance, frequency, phase
        )
    )


def failing_model_time(time_array, luminosity_distance, frequency, phase):
    """A model-time source model whose waveform fails, as a LAL input domain error does"""
    return dict(polarizations_at=lambda times: None)


def waveform_generator(source):
    return bilby.gw.WaveformGenerator(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        start_time=GPS_START,
        time_domain_source_model=source,
    )


def zero_noise_interferometers(
    window_start=GPS_START + 2, window_duration=4.0, inject=True
):
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in ifos:
        ifo.minimum_frequency, ifo.maximum_frequency = 20, SAMPLING_FREQUENCY / 2
    ifos.set_analysis_windows(duration=window_duration, start_time=window_start)
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=SAMPLING_FREQUENCY, duration=DURATION, start_time=GPS_START
    )
    ifos.set_autocovariance_functions_from_power_spectral_densities()
    if inject:
        ifos.inject_signal(
            parameters=INJECTION,
            waveform_generator=waveform_generator(burst_model_time),
            domain="time",
            placement="model_time",
        )
    return ifos


def distance_priors():
    priors = bilby.core.prior.PriorDict(INJECTION.copy())
    priors["luminosity_distance"] = bilby.core.prior.PowerLaw(
        alpha=2, minimum=50, maximum=1000, name="luminosity_distance"
    )
    return priors


LOOKUP_TABLE_DIRECTORY = None


def setUpModule():
    global LOOKUP_TABLE_DIRECTORY
    LOOKUP_TABLE_DIRECTORY = tempfile.mkdtemp()


def tearDownModule():
    shutil.rmtree(LOOKUP_TABLE_DIRECTORY)


def lookup_table_filename():
    """The distance-marginalization lookup table, built once and shared by the tests"""
    return os.path.join(LOOKUP_TABLE_DIRECTORY, "distance_lookup.npz")


class TestTimeDomainGravitationalWaveTransient(unittest.TestCase):
    def setUp(self):
        self.ifos = zero_noise_interferometers()
        self.likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
            self.ifos, waveform_generator(burst_model_time), placement="model_time"
        )

    def tearDown(self):
        del self.ifos
        del self.likelihood

    def test_repr(self):
        self.assertIn("placement=model_time", repr(self.likelihood))

    def test_meta_data(self):
        self.assertEqual(self.likelihood.meta_data["likelihood_domain"], "time")
        self.assertEqual(self.likelihood.meta_data["placement"], "model_time")

    def test_default_placement_is_fd_shift(self):
        likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
            self.ifos, waveform_generator(burst_time_domain)
        )
        self.assertEqual(likelihood.placement, "fd_shift")

    def test_noise_log_likelihood(self):
        expected = -0.5 * sum(
            ifo.analysis_data
            @ gohberg_semencul_product(*ifo.gohberg_semencul_vectors, ifo.analysis_data)
            for ifo in self.ifos
        )
        self.assertAlmostEqual(
            self.likelihood.noise_log_likelihood() / expected, 1, places=12
        )

    def test_log_likelihood_ratio_at_the_injection_is_half_the_snr_squared(self):
        snr_squared = sum(ifo.meta_data["optimal_SNR"] ** 2 for ifo in self.ifos)
        self.assertGreater(snr_squared, 100)
        self.assertAlmostEqual(
            self.likelihood.log_likelihood_ratio(INJECTION) / (snr_squared / 2),
            1,
            places=10,
        )
        self.assertAlmostEqual(
            self.likelihood.log_likelihood(INJECTION),
            self.likelihood.noise_log_likelihood()
            + self.likelihood.log_likelihood_ratio(INJECTION),
            places=6,
        )

    def test_per_detector_log_likelihoods_add_up(self):
        parameters = dict(INJECTION, geocent_time=INJECTION["geocent_time"] + 2e-4)
        per_detector = self.likelihood.compute_per_detector_log_likelihood(
            parameters.copy()
        )
        total = sum(per_detector[f"{ifo.name}_log_likelihood"] for ifo in self.ifos)
        self.assertAlmostEqual(
            total, self.likelihood.log_likelihood_ratio(parameters), places=6
        )

    def test_snrs_match_the_injection(self):
        polarizations = self.likelihood.waveform_polarizations(INJECTION)
        for ifo in self.ifos:
            snrs = self.likelihood.calculate_snrs(
                polarizations, ifo, parameters=INJECTION
            )
            self.assertAlmostEqual(
                snrs.optimal_snr_squared**0.5 / ifo.meta_data["optimal_SNR"],
                1,
                places=10,
            )
            self.assertAlmostEqual(
                snrs.complex_matched_filter_snr / ifo.meta_data["matched_filter_SNR"],
                1,
                places=10,
            )

    def test_compute_snrs(self):
        sample = dict(INJECTION)
        bilby.gw.conversion.compute_snrs(sample, self.likelihood)
        for ifo in self.ifos:
            self.assertAlmostEqual(
                sample[f"{ifo.name}_optimal_snr"],
                ifo.meta_data["optimal_SNR"],
                places=8,
            )
            self.assertAlmostEqual(
                abs(sample[f"{ifo.name}_matched_filter_snr"]),
                ifo.meta_data["matched_filter_SNR"],
                places=8,
            )

    def test_fd_shift_matches_model_time(self):
        likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
            self.ifos, waveform_generator(burst_time_domain), placement="fd_shift"
        )
        for offset in np.linspace(-1, 1, 7) / SAMPLING_FREQUENCY:
            parameters = dict(
                INJECTION, geocent_time=INJECTION["geocent_time"] + offset
            )
            self.assertAlmostEqual(
                likelihood.log_likelihood_ratio(parameters),
                self.likelihood.log_likelihood_ratio(parameters),
                places=4,
            )

    def test_matches_the_frequency_domain_likelihood(self):
        """With a window covering the whole segment, the likelihood agrees with the
        frequency-domain inner products (20 Hz - Nyquist) of the same signals"""
        ifos = zero_noise_interferometers(
            window_start=GPS_START, window_duration=DURATION
        )
        generator = waveform_generator(burst_model_time)
        likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
            ifos, generator, placement="model_time"
        )

        def frequency_domain(parameters):
            total = 0
            for ifo in ifos:
                template = ifo.get_detector_response_time_domain(
                    generator.time_domain_strain(parameters),
                    parameters,
                    placement="model_time",
                    times=ifo.time_array,
                )
                template_fd, frequencies = bilby.core.utils.nfft(
                    template, SAMPLING_FREQUENCY
                )
                data_fd, _ = bilby.core.utils.nfft(
                    ifo.time_domain_strain, SAMPLING_FREQUENCY
                )
                band = frequencies >= 20
                psd = ifo.power_spectral_density.get_power_spectral_density_array(
                    frequencies[band]
                )
                total += (
                    bilby.gw.utils.noise_weighted_inner_product(
                        data_fd[band], template_fd[band], psd, DURATION
                    ).real
                    - bilby.gw.utils.noise_weighted_inner_product(
                        template_fd[band], template_fd[band], psd, DURATION
                    ).real
                    / 2
                )
            return total

        for change in [
            dict(),
            dict(geocent_time=INJECTION["geocent_time"] + 5e-4),
            dict(phase=1.0),
            dict(frequency=152.0),
            dict(luminosity_distance=300.0),
        ]:
            parameters = dict(INJECTION, **change)
            expected = frequency_domain(parameters)
            self.assertAlmostEqual(
                likelihood.log_likelihood_ratio(parameters),
                expected,
                delta=1e-3 * max(abs(expected), 1),
            )


class TestTimeDomainGravitationalWaveTransientSetUp(unittest.TestCase):
    def setUp(self):
        self.ifos = zero_noise_interferometers(inject=False)

    def tearDown(self):
        del self.ifos

    def test_unsupported_marginalizations_raise(self):
        for key in [
            "time_marginalization",
            "phase_marginalization",
            "calibration_marginalization",
        ]:
            with self.assertRaises(NotImplementedError):
                bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
                    self.ifos, waveform_generator(burst_time_domain), **{key: True}
                )

    def test_needs_a_time_domain_source_model(self):
        generator = bilby.gw.WaveformGenerator(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        )
        with self.assertRaises(ValueError):
            bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
                self.ifos, generator
            )

    def test_unknown_placement_raises(self):
        with self.assertRaises(ValueError):
            bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
                self.ifos, waveform_generator(burst_time_domain), placement="linear"
            )

    def test_needs_the_autocovariance_functions(self):
        self.ifos[1].set_analysis_window(
            GPS_START + 2, 4
        )  # removes the autocovariance function
        with self.assertRaises(ValueError):
            bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
                self.ifos, waveform_generator(burst_time_domain)
            )

    def test_placement_different_from_the_injection_warns(self):
        self.ifos.inject_signal(
            parameters=INJECTION,
            waveform_generator=waveform_generator(burst_time_domain),
            domain="time",
            placement="fd_shift",
        )
        with self.assertLogs("bilby", level="WARNING") as logs:
            bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
                self.ifos, waveform_generator(burst_time_domain), placement="nearest"
            )
        self.assertIn("injected with placement", "".join(logs.output))
        with self.assertNoLogs("bilby", level="WARNING"):
            bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
                self.ifos, waveform_generator(burst_time_domain), placement="fd_shift"
            )

    def test_maximum_frequency_below_nyquist_warns(self):
        self.ifos[0].maximum_frequency = 256
        with self.assertLogs("bilby", level="WARNING") as logs:
            bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
                self.ifos, waveform_generator(burst_time_domain)
            )
        self.assertIn(
            "is outside the band and filled in the autocovariance function",
            "".join(logs.output),
        )


class TestTimeDomainWaveformFailure(unittest.TestCase):
    def _likelihood(self, distance_marginalization):
        return bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
            zero_noise_interferometers(),
            waveform_generator(failing_model_time),
            placement="model_time",
            distance_marginalization=distance_marginalization,
            priors=distance_priors(),
            distance_marginalization_lookup_table=lookup_table_filename(),
        )

    def test_log_likelihood_ratio_is_minus_infinity(self):
        self.assertEqual(
            self._likelihood(False).log_likelihood_ratio(dict(INJECTION)),
            np.nan_to_num(-np.inf),
        )

    def test_distance_marginalized_log_likelihood_ratio_is_minus_infinity(self):
        self.assertEqual(
            self._likelihood(True).log_likelihood_ratio(dict(INJECTION)),
            np.nan_to_num(-np.inf),
        )

    def test_per_detector_log_likelihoods_are_minus_infinity(self):
        result = self._likelihood(False).compute_per_detector_log_likelihood(
            dict(INJECTION)
        )
        self.assertEqual(result["H1_log_likelihood"], np.nan_to_num(-np.inf))
        self.assertEqual(result["L1_log_likelihood"], np.nan_to_num(-np.inf))


class TestTimeDomainDistanceMarginalization(unittest.TestCase):
    def setUp(self):
        self.priors = distance_priors()
        self.ifos = zero_noise_interferometers()

    def tearDown(self):
        del self.priors
        del self.ifos

    def _likelihood(
        self,
        source=burst_model_time,
        placement="model_time",
        distance_marginalization=True,
    ):
        return bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
            self.ifos,
            waveform_generator(source),
            placement=placement,
            distance_marginalization=distance_marginalization,
            priors=self.priors.copy(),
            distance_marginalization_lookup_table=lookup_table_filename(),
        )

    def test_matches_numerical_marginalization(self):
        non_marginalized = self._likelihood(distance_marginalization=False)
        prior = self.priors["luminosity_distance"]
        distances = np.linspace(prior.minimum, prior.maximum, 1000)
        log_likelihoods = np.array(
            [
                non_marginalized.log_likelihood_ratio(
                    dict(INJECTION, luminosity_distance=distance)
                )
                for distance in distances
            ]
        )
        expected = (
            np.log(
                trapezoid(
                    np.exp(log_likelihoods - log_likelihoods.max())
                    * prior.prob(distances),
                    distances,
                )
            )
            + log_likelihoods.max()
        )
        self.assertAlmostEqual(
            self._likelihood().log_likelihood_ratio(dict(INJECTION)),
            expected,
            delta=0.5,
        )

    def test_distance_reconstruction(self):
        sample = (
            self._likelihood().generate_posterior_sample_from_marginalized_likelihood(
                dict(INJECTION)
            )
        )
        prior = self.priors["luminosity_distance"]
        self.assertTrue(prior.minimum <= sample["luminosity_distance"] <= prior.maximum)

    def test_distance_reconstruction_rescales_the_signal(self):
        for source, placement in [
            (burst_model_time, "model_time"),
            (burst_time_domain, "fd_shift"),
        ]:
            likelihood = self._likelihood(source, placement)
            parameters = dict(INJECTION, luminosity_distance=likelihood._ref_dist)
            signal = likelihood.waveform_polarizations(dict(parameters))
            distance = likelihood.generate_distance_sample_from_marginalized_likelihood(
                signal, parameters=dict(parameters)
            )
            expected = likelihood.waveform_polarizations(
                dict(parameters, luminosity_distance=distance)
            )
            for ifo in self.ifos:
                np.testing.assert_allclose(
                    ifo.get_detector_response_time_domain(
                        signal, parameters, placement=placement
                    ),
                    ifo.get_detector_response_time_domain(
                        expected, parameters, placement=placement
                    ),
                    rtol=1e-10,
                    atol=1e-35,
                )


class TestTimeDomainNoiseStatistics(unittest.TestCase):
    def test_matched_filter_snr_in_gaussian_noise(self):
        """In Gaussian noise from the PSD the matched-filter SNR of a fixed template is N(0, 1)"""
        bilby.core.utils.random.seed(5)
        ifo = bilby.gw.detector.get_empty_interferometer("H1")
        ifo.minimum_frequency, ifo.maximum_frequency = 20, SAMPLING_FREQUENCY / 2
        ifo.set_analysis_window(0.0, 1.0)
        ifo.set_strain_data_from_zero_noise(
            sampling_frequency=SAMPLING_FREQUENCY, duration=64, start_time=0
        )
        ifo.set_autocovariance_function_from_power_spectral_density()
        times = np.arange(int(SAMPLING_FREQUENCY)) / SAMPLING_FREQUENCY - 0.5
        template = burst(times, 100, 150, 0)["plus"]
        inverse_covariance_template = gohberg_semencul_product(
            *ifo.gohberg_semencul_vectors, template
        )
        norm = np.sqrt(template @ inverse_covariance_template)
        snrs = list()
        for _ in range(4):
            ifo.set_strain_data_from_power_spectral_density(
                sampling_frequency=SAMPLING_FREQUENCY, duration=64, start_time=0
            )
            data = ifo.time_domain_strain
            for start in range(2, 61):
                window = data[
                    int(start * SAMPLING_FREQUENCY) : int(
                        (start + 1) * SAMPLING_FREQUENCY
                    )
                ]
                snrs.append(window @ inverse_covariance_template / norm)
        snrs = np.array(snrs)
        self.assertLess(abs(np.mean(snrs)), 4 / np.sqrt(len(snrs)))
        self.assertLess(abs(np.std(snrs, ddof=1) - 1), 4 / np.sqrt(2 * len(snrs)))


class TestTimeDomainResultPlotting(unittest.TestCase):
    def setUp(self):
        self.outdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.outdir)

    def test_plot_interferometer_waveform_posterior_raises(self):
        ifos = zero_noise_interferometers()
        likelihood = bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
            ifos, waveform_generator(burst_model_time), placement="model_time"
        )
        result = bilby.gw.result.CBCResult(
            label="td",
            outdir=self.outdir,
            posterior=pd.DataFrame([INJECTION, INJECTION]),
            meta_data=dict(likelihood=likelihood.meta_data),
            injection_parameters=INJECTION,
            search_parameter_keys=["luminosity_distance"],
        )
        with self.assertRaises(NotImplementedError):
            result.plot_interferometer_waveform_posterior(
                interferometer=ifos[0], n_samples=2, save=True
            )


@pytest.mark.skipif(
    not hasattr(lalsimulation, "SimIMRPhenomTModelTimesTruncated"),
    reason="lalsimulation without the IMRPhenomT model-time functions",
)
class TestTimeDomainIMRPhenomT(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            mass_1=36.0,
            mass_2=29.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=0.0,
            phi_jl=0.0,
            luminosity_distance=1000.0,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3,
            geocent_time=GPS_START + 6.41337,
            ra=1.375,
            dec=-1.2108,
        )

    def tearDown(self):
        del self.parameters

    @staticmethod
    def _waveform_generator(source):
        return bilby.gw.WaveformGenerator(
            duration=DURATION,
            sampling_frequency=2048.0,
            start_time=GPS_START,
            time_domain_source_model=source,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(
                waveform_approximant="IMRPhenomT",
                reference_frequency=20.0,
                minimum_frequency=10.0,
            ),
        )

    def _likelihood(self, ifos=None):
        if ifos is None:
            ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
            ifos.set_analysis_windows(
                duration=4, start_time=self.parameters["geocent_time"] - 3.5
            )
            ifos.set_strain_data_from_zero_noise(
                sampling_frequency=2048.0, duration=DURATION, start_time=GPS_START
            )
            ifos.set_autocovariance_functions_from_power_spectral_densities()
        return bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
            ifos,
            self._waveform_generator(bilby.gw.source.lal_binary_black_hole_model_time),
            placement="model_time",
        )

    def test_model_time_matches_lalsimulation_waveform_with_fd_shift(self):
        """IMRPhenomT at model times agrees with SimInspiralChooseTDWaveform placed with fd_shift"""
        likelihood = self._likelihood()
        likelihood.interferometers.inject_signal(
            parameters=self.parameters,
            domain="time",
            placement="model_time",
            waveform_generator=self._waveform_generator(
                bilby.gw.source.lal_binary_black_hole_model_time
            ),
        )
        ifos = likelihood.interferometers
        likelihoods = [
            bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
                ifos,
                self._waveform_generator(
                    bilby.gw.source.lal_binary_black_hole_model_time
                ),
                placement="model_time",
            ),
            bilby.gw.likelihood.TimeDomainGravitationalWaveTransient(
                ifos,
                self._waveform_generator(
                    bilby.gw.source.lal_binary_black_hole_time_domain
                ),
                placement="fd_shift",
            ),
        ]
        for offset in [0, 3e-4]:
            parameters = dict(
                self.parameters, geocent_time=self.parameters["geocent_time"] + offset
            )
            values = [
                likelihood.log_likelihood_ratio(parameters)
                for likelihood in likelihoods
            ]
            self.assertAlmostEqual(values[0], values[1], delta=1e-2)

    def test_rescale_signal(self):
        likelihood = self._likelihood()
        parameters = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(
            self.parameters
        )[0]
        times = np.linspace(-0.5, 0.05, 200)
        generator = self._waveform_generator(
            bilby.gw.source.lal_binary_black_hole_model_time
        )
        near = generator.time_domain_strain(parameters)
        far = self._waveform_generator(
            bilby.gw.source.lal_binary_black_hole_model_time
        ).time_domain_strain(
            dict(parameters, luminosity_distance=2 * parameters["luminosity_distance"])
        )
        likelihood._ref_dist = parameters["luminosity_distance"]
        likelihood._rescale_signal(near, 2 * parameters["luminosity_distance"])
        np.testing.assert_allclose(
            near["polarizations_at"](times)["plus"],
            far["polarizations_at"](times)["plus"],
            rtol=1e-10,
            atol=1e-35,
        )

    def test_likelihood_can_be_pickled_after_an_evaluation(self):
        """Needed for npool > 1 when worker processes are spawned, and for MPI pools"""
        likelihood = self._likelihood()
        value = likelihood.log_likelihood_ratio(dict(self.parameters))
        for module in [pickle, dill]:
            restored = module.loads(module.dumps(likelihood))
            self.assertEqual(
                restored.log_likelihood_ratio(dict(self.parameters)), value
            )

    def test_compute_snrs_in_parallel(self):
        likelihood = self._likelihood()
        likelihood.log_likelihood_ratio(dict(self.parameters))
        samples = pd.DataFrame(
            [dict(self.parameters, phase=phase) for phase in [1.3, 0.4]]
        )
        serial, parallel = samples.copy(), samples.copy()
        bilby.gw.conversion.compute_snrs(serial, likelihood, npool=1)
        bilby.gw.conversion.compute_snrs(parallel, likelihood, npool=2)
        for column in ["H1_optimal_snr", "L1_optimal_snr", "H1_matched_filter_snr"]:
            np.testing.assert_array_equal(serial[column], parallel[column])


if __name__ == "__main__":
    unittest.main()
