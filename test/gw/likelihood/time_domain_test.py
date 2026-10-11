import os
import shutil
import tempfile
import unittest

import numpy as np
import scipy.linalg
from scipy.integrate import trapezoid

import bilby
from bilby.gw.detector import AutoCovarianceFunction
from bilby.gw.likelihood import TimeDomainGravitationalWaveTransient
from bilby.gw.time_domain_utils import (
    fractional_time_shift,
    gohberg_semencul_product,
    gohberg_semencul_vectors,
    place_time_domain_signal,
)

INJECTION = dict(
    mass_1=36.0, mass_2=29.0, a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=1000.0, theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108,
)
SAMPLING_FREQUENCY = 1024
DURATION = 4
START_TIME = INJECTION["geocent_time"] - 3


def waveform_generator(source_model=bilby.gw.source.lal_binary_black_hole_time_domain):
    return bilby.gw.WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY, start_time=START_TIME,
        time_domain_source_model=source_model,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant="IMRPhenomXP",
                                reference_frequency=20.0, minimum_frequency=20.0))


def interferometers(noise="zero", acf=True, analysis_segment="imr", duration=DURATION,
                    segment_cut_time=-0.01):
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in ifos:
        ifo.minimum_frequency, ifo.maximum_frequency = 20, 448
    ifos.set_analysis_windows(
        duration=duration, start_time=START_TIME, analysis_segment=analysis_segment,
        segment_cut_time=segment_cut_time, cut_reference_parameters=INJECTION)
    random_state = np.random.default_rng(3)
    for ifo in ifos:
        window = (SAMPLING_FREQUENCY, ifo.analysis_window["duration"], ifo.analysis_window["start_time"])
        if noise == "zero":
            ifo.set_strain_data_from_zero_noise(*window)
        else:
            ifo.set_strain_data_from_power_spectral_density(*window, random_state=random_state)
    if acf:
        ifos.set_autocovariance_functions_from_power_spectral_densities()
    return ifos


SEGMENTS = dict(imr=dict(analysis_segment="imr"),
                inspiral=dict(analysis_segment="inspiral", duration=1.0),
                post_inspiral=dict(analysis_segment="post_inspiral", duration=0.5))


class TestGohbergSemencul(unittest.TestCase):
    def test_inverse_matches_dense_solve(self):
        rng = np.random.default_rng(1)
        lags = np.arange(400)
        acf = np.exp(-lags / 30.0) * np.cos(lags / 5.0)
        acf[0] += 0.1
        for number_of_samples in [5, 64, 333]:
            x, y = gohberg_semencul_vectors(acf[:number_of_samples])
            vector = rng.normal(size=number_of_samples)
            expected = np.linalg.solve(scipy.linalg.toeplitz(acf[:number_of_samples]), vector)
            np.testing.assert_allclose(gohberg_semencul_product(x, y, vector), expected,
                                       rtol=0, atol=1e-10 * np.max(abs(expected)))


class TestPlacement(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(256)
        self.signal = np.exp(-0.5 * ((self.times - 100) / 6.0) ** 2) * np.cos(0.4 * self.times)

    def test_fd_shift_whole_sample_equals_nearest(self):
        for placement in ["nearest", "fd_shift"]:
            placed = place_time_domain_signal(
                self.signal, merger_index=100, arrival_index=150,
                number_of_samples=300, placement=placement)
            if placement == "nearest":
                nearest = placed
        np.testing.assert_allclose(placed, nearest, atol=1e-12)

    def test_fd_shift_fraction_is_exact_for_band_limited_signal(self):
        delay = 50.3
        placed = place_time_domain_signal(
            self.signal, merger_index=100, arrival_index=100 + delay,
            number_of_samples=400, placement="fd_shift")
        times = np.arange(400) - delay
        expected = np.exp(-0.5 * ((times - 100) / 6.0) ** 2) * np.cos(0.4 * times)
        np.testing.assert_allclose(placed, expected, atol=1e-6)

    def test_nothing_wraps(self):
        placed = place_time_domain_signal(
            self.signal, merger_index=100, arrival_index=10,
            number_of_samples=256, placement="nearest")
        np.testing.assert_array_equal(placed[166:], 0)
        np.testing.assert_allclose(placed[:166], self.signal[90:])

    def test_fractional_shift_length(self):
        self.assertEqual(len(fractional_time_shift(self.signal, 0.25, pad=10)), 276)


class TestAutoCovarianceFunction(unittest.TestCase):
    def setUp(self):
        self.frequencies = np.arange(0, 512.125, 0.125)
        self.psd = 1e-46 * (1 + (50 / np.maximum(self.frequencies, 1)) ** 4)

    def test_sampling_frequency_from_grid_and_length(self):
        acf = AutoCovarianceFunction.from_power_spectral_density_array(
            self.frequencies, self.psd, 20, 400)
        self.assertEqual(acf.sampling_frequency, 1024)
        self.assertEqual(len(acf.acf_array), 8192)
        self.assertAlmostEqual(acf.meta_data["duration"], 8)

    def test_sampling_frequency_mismatch_warns(self):
        with self.assertLogs("bilby", level="WARNING"):
            acf = AutoCovarianceFunction.from_power_spectral_density_array(
                self.frequencies, self.psd, 20, 400, sampling_frequency=2048)
        self.assertEqual(acf.sampling_frequency, 1024)

    def test_grid_extended_to_zero_frequency(self):
        full = AutoCovarianceFunction.from_power_spectral_density_array(
            self.frequencies, self.psd, 20, 400)
        keep = self.frequencies >= 5
        cut = AutoCovarianceFunction.from_power_spectral_density_array(
            self.frequencies[keep], self.psd[keep], 20, 400)
        np.testing.assert_allclose(cut.acf_array, full.acf_array, rtol=1e-12)

    def test_patch_level(self):
        band = (self.frequencies >= 20) & (self.frequencies <= 400)
        for kwargs, level in [(dict(fill_multiplier=100), 100 * np.max(self.psd[band])),
                              (dict(fill_value=1e-41), 1e-41)]:
            acf = AutoCovarianceFunction.from_power_spectral_density_array(
                self.frequencies, self.psd, 20, 400, **kwargs)
            patched = 2 * np.fft.rfft(acf.acf_array).real / acf.sampling_frequency
            np.testing.assert_allclose(patched[band], self.psd[band], rtol=1e-8)
            np.testing.assert_allclose(patched[~band], level, rtol=1e-8)
            self.assertAlmostEqual(acf.fill_value / level, 1)

    def test_window_longer_than_acf_raises(self):
        acf = AutoCovarianceFunction.from_power_spectral_density_array(
            self.frequencies, self.psd, 20, 400)
        with self.assertRaises(ValueError):
            acf.get_acf_array(len(acf.acf_array) + 1)


class TestTimeDomainLikelihood(unittest.TestCase):
    def test_zero_noise_log_likelihood_ratio_is_half_snr_squared(self):
        for placement in ["nearest", "fd_shift"]:
            for segment, kwargs in SEGMENTS.items():
                with self.subTest(placement=placement, segment=segment):
                    ifos = interferometers(**kwargs)
                    generator = waveform_generator()
                    ifos.inject_signal_time_domain(
                        parameters=INJECTION, waveform_generator=generator, placement=placement)
                    likelihood = TimeDomainGravitationalWaveTransient(ifos, generator, placement=placement)
                    optimal_snr_squared = sum(ifo.meta_data["optimal_SNR"] ** 2 for ifo in ifos)
                    self.assertGreater(optimal_snr_squared, 10)
                    self.assertAlmostEqual(
                        likelihood.log_likelihood_ratio(INJECTION) / (optimal_snr_squared / 2), 1, 8)
                    self.assertAlmostEqual(
                        likelihood.noise_log_likelihood() / (-optimal_snr_squared / 2), 1, 8)

    def test_inspiral_and_post_inspiral_windows(self):
        inspiral = interferometers(acf=False, **SEGMENTS["inspiral"])
        post_inspiral = interferometers(acf=False, **SEGMENTS["post_inspiral"])
        for first, second in zip(inspiral, post_inspiral):
            cut_time = first.meta_data["analysis_window"]["cut_time"]
            self.assertAlmostEqual(cut_time, second.meta_data["analysis_window"]["cut_time"])
            self.assertAlmostEqual(first.strain_data.start_time + first.strain_data.duration, cut_time)
            self.assertAlmostEqual(second.strain_data.start_time, cut_time)
            self.assertEqual(len(first.time_domain_strain), SAMPLING_FREQUENCY)
            self.assertEqual(len(second.time_domain_strain), SAMPLING_FREQUENCY // 2)
        delay = (inspiral[1].time_delay_from_geocenter(INJECTION["ra"], INJECTION["dec"], INJECTION["geocent_time"])
                 - inspiral[0].time_delay_from_geocenter(INJECTION["ra"], INJECTION["dec"], INJECTION["geocent_time"]))
        self.assertAlmostEqual(inspiral[1].strain_data.start_time - inspiral[0].strain_data.start_time, delay, 6)

    def test_matches_dense_covariance(self):
        ifos = interferometers(noise="gaussian", **SEGMENTS["post_inspiral"])
        generator = waveform_generator()
        ifos.inject_signal_time_domain(parameters=INJECTION, waveform_generator=generator)
        likelihood = TimeDomainGravitationalWaveTransient(ifos, generator)
        parameters = dict(INJECTION, mass_1=37.0)
        expected = 0
        for ifo in ifos:
            data = ifo.analysis_data
            covariance = scipy.linalg.toeplitz(ifo.autocovariance_function.acf_array[:len(data)])
            signal = likelihood._compute_full_waveform(
                generator.time_domain_strain(parameters), ifo, parameters)
            expected += data @ np.linalg.solve(covariance, signal) \
                - 0.5 * signal @ np.linalg.solve(covariance, signal)
        self.assertAlmostEqual(likelihood.log_likelihood_ratio(parameters) / expected, 1, 6)

    def test_distance_marginalization(self):
        ifos = interferometers(noise="gaussian")
        generator = waveform_generator()
        ifos.inject_signal_time_domain(parameters=INJECTION, waveform_generator=generator)
        priors = bilby.gw.prior.BBHPriorDict()
        priors["luminosity_distance"] = bilby.core.prior.PowerLaw(
            alpha=2, minimum=200, maximum=3000, name="luminosity_distance")
        directory = tempfile.mkdtemp()
        try:
            marginalized = TimeDomainGravitationalWaveTransient(
                ifos, generator, priors=priors.copy(), distance_marginalization=True,
                distance_marginalization_lookup_table=os.path.join(directory, "table.npz"))
            plain = TimeDomainGravitationalWaveTransient(ifos, generator)
            distances = np.linspace(200, 3000, 2000)
            log_l = np.array([plain.log_likelihood_ratio(dict(INJECTION, luminosity_distance=d))
                              for d in distances])
            expected = np.log(trapezoid(
                np.exp(log_l - log_l.max()) * priors["luminosity_distance"].prob(distances),
                distances)) + log_l.max()
            self.assertAlmostEqual(marginalized.log_likelihood_ratio(dict(INJECTION)), expected, 1)
            sample = marginalized.generate_posterior_sample_from_marginalized_likelihood(dict(INJECTION))
            self.assertTrue(200 <= sample["luminosity_distance"] <= 3000)
        finally:
            shutil.rmtree(directory)

    def test_snrs_in_post_processing(self):
        ifos = interferometers(**SEGMENTS["inspiral"])
        generator = waveform_generator()
        ifos.inject_signal_time_domain(parameters=INJECTION, waveform_generator=generator)
        likelihood = TimeDomainGravitationalWaveTransient(ifos, generator)
        sample = dict(INJECTION)
        bilby.gw.conversion.compute_snrs(sample, likelihood)
        for ifo in ifos:
            self.assertAlmostEqual(sample[f"{ifo.name}_optimal_snr"], ifo.meta_data["optimal_SNR"], 8)

    def test_meta_data(self):
        ifos = interferometers(**SEGMENTS["inspiral"])
        likelihood = TimeDomainGravitationalWaveTransient(ifos, waveform_generator(), placement="fd_shift")
        meta_data = likelihood.meta_data
        self.assertEqual(meta_data["likelihood_domain"], "time")
        self.assertEqual(meta_data["placement"], "fd_shift")
        self.assertIn("autocovariance_function", meta_data["interferometers"]["H1"])
        self.assertEqual(meta_data["interferometers"]["H1"]["analysis_window"]["analysis_segment"], "inspiral")

    def test_errors(self):
        generator = waveform_generator()
        with self.assertRaises(ValueError):
            TimeDomainGravitationalWaveTransient(interferometers(acf=False), generator)
        frequency_domain_generator = bilby.gw.WaveformGenerator(
            duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole)
        with self.assertRaises(ValueError):
            TimeDomainGravitationalWaveTransient(interferometers(), frequency_domain_generator)
        for key in ["time_marginalization", "phase_marginalization", "calibration_marginalization"]:
            with self.assertRaises(NotImplementedError):
                TimeDomainGravitationalWaveTransient(interferometers(), generator, **{key: True})
        with self.assertRaises(ValueError):
            TimeDomainGravitationalWaveTransient(interferometers(), generator, placement="roll")
        ifos = interferometers()
        ifos[0].autocovariance_function = AutoCovarianceFunction(
            acf_array=ifos[0].autocovariance_function.acf_array, sampling_frequency=2048)
        with self.assertRaises(ValueError):
            TimeDomainGravitationalWaveTransient(ifos, generator)
        ifo = bilby.gw.detector.get_empty_interferometer("H1")
        with self.assertRaises(ValueError):
            ifo.autocovariance_function = AutoCovarianceFunction(np.ones(10), SAMPLING_FREQUENCY)
        ifos = interferometers()
        for ifo in ifos:
            ifo.set_strain_data_from_zero_noise(SAMPLING_FREQUENCY, DURATION, START_TIME + 1)
        with self.assertRaises(ValueError):
            TimeDomainGravitationalWaveTransient(ifos, generator)


def _has_model_time_waveforms():
    try:
        import lalsimulation
    except ImportError:
        return False
    return hasattr(lalsimulation, "SimIMRPhenomTPHMModelTimesTruncated")


def model_time_generator(approximant="IMRPhenomTPHM", **kwargs):
    arguments = dict(waveform_approximant=approximant, reference_frequency=20.0, minimum_frequency=20.0)
    arguments.update(kwargs)
    return bilby.gw.WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY, start_time=START_TIME,
        time_domain_source_model=bilby.gw.source.lal_binary_black_hole_model_time,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=arguments)


ALIGNED_INJECTION = dict(INJECTION, tilt_1=0.0, tilt_2=np.pi)


@unittest.skipUnless(_has_model_time_waveforms(), "lalsimulation without the model-time IMRPhenomT functions")
class TestModelTime(unittest.TestCase):
    def test_matches_lalsimulation_on_its_samples(self):
        # with geocent_time chosen so that LAL's own samples fall on the data samples,
        # "nearest" is the standard waveform without interpolation, and the model-time
        # waveform must agree (to the ~1e-4 sample left by the precision of GPS times;
        # the first sample can fall just outside the waveform's span)
        ifo = interferometers()[0]
        start_time = ifo.strain_data.time_array[ifo._analysis_slice().start]
        for approximant, injection in [("IMRPhenomT", ALIGNED_INJECTION), ("IMRPhenomTHM", ALIGNED_INJECTION),
                                       ("IMRPhenomTP", INJECTION), ("IMRPhenomTPHM", INJECTION)]:
            with self.subTest(approximant=approximant):
                generator = waveform_generator()
                generator.waveform_arguments.update(waveform_approximant=approximant, turn_on_window=0)
                epoch = generator.time_domain_strain(injection)["epoch"]
                parameters = dict(injection)
                for _ in range(3):
                    delay = ifo.time_delay_from_geocenter(
                        parameters["ra"], parameters["dec"], parameters["geocent_time"])
                    position = (parameters["geocent_time"] - start_time + delay + epoch) * SAMPLING_FREQUENCY
                    parameters["geocent_time"] -= (position - round(position)) / SAMPLING_FREQUENCY
                standard = ifo.get_time_domain_detector_response(
                    generator.time_domain_strain(parameters), parameters, placement="nearest")
                model_time = ifo.get_time_domain_detector_response(
                    model_time_generator(approximant).time_domain_strain(parameters), parameters,
                    placement="model_time")
                first = np.argmax(standard != 0) + 1
                self.assertTrue(np.all(model_time[:first - 1] == 0))
                self.assertLess(np.max(abs(model_time - standard)[first:]) / np.max(abs(standard)), 1e-3)

    def test_zero_noise_log_likelihood_ratio_is_half_snr_squared(self):
        for approximant, injection in [("IMRPhenomT", ALIGNED_INJECTION), ("IMRPhenomTHM", ALIGNED_INJECTION),
                                       ("IMRPhenomTP", INJECTION), ("IMRPhenomTPHM", INJECTION)]:
            with self.subTest(approximant=approximant):
                ifos = interferometers()
                generator = model_time_generator(approximant)
                ifos.inject_signal_time_domain(
                    parameters=injection, waveform_generator=generator, placement="model_time")
                likelihood = TimeDomainGravitationalWaveTransient(ifos, generator, placement="model_time")
                optimal_snr_squared = sum(ifo.meta_data["optimal_SNR"] ** 2 for ifo in ifos)
                self.assertGreater(optimal_snr_squared, 10)
                self.assertAlmostEqual(
                    likelihood.log_likelihood_ratio(injection) / (optimal_snr_squared / 2), 1, 8)

    def test_distance_marginalization(self):
        ifos = interferometers()
        generator = model_time_generator()
        ifos.inject_signal_time_domain(parameters=INJECTION, waveform_generator=generator, placement="model_time")
        priors = bilby.gw.prior.BBHPriorDict()
        priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
            minimum=200, maximum=3000, name="luminosity_distance")
        lookup = os.path.join(tempfile.mkdtemp(), "lookup.npz")
        likelihood = TimeDomainGravitationalWaveTransient(
            ifos, generator, placement="model_time", distance_marginalization=True, priors=priors,
            distance_marginalization_lookup_table=lookup)
        self.assertTrue(np.isfinite(likelihood.log_likelihood_ratio(INJECTION)))
        sample = likelihood.generate_posterior_sample_from_marginalized_likelihood(dict(INJECTION))
        self.assertTrue(200 <= sample["luminosity_distance"] <= 3000)
        shutil.rmtree(os.path.dirname(lookup))

    def test_errors(self):
        ifos = interferometers()
        with self.assertRaises(ValueError):  # model-time source, other placement
            TimeDomainGravitationalWaveTransient(ifos, model_time_generator()).log_likelihood_ratio(INJECTION)
        with self.assertRaises(ValueError):  # other source, model-time placement
            TimeDomainGravitationalWaveTransient(
                ifos, waveform_generator(), placement="model_time").log_likelihood_ratio(INJECTION)
        with self.assertRaises(ValueError):  # in-plane spins, non-precessing approximant
            model_time_generator("IMRPhenomT").time_domain_strain(INJECTION)
        with self.assertRaises(ValueError):
            model_time_generator("IMRPhenomXP").time_domain_strain(INJECTION)
        with self.assertLogs("bilby", level="WARNING"):
            TimeDomainGravitationalWaveTransient(ifos, model_time_generator(minimum_frequency=15.0),
                                                 placement="model_time")

    def test_modes(self):
        from bilby.gw.model_time_waveforms import modes_at_model_times
        import lal
        model_times = np.linspace(-3000, 100, 50)
        arguments = (36 * lal.MSUN_SI, 29 * lal.MSUN_SI, 0.3, 0.1, 0.2, -0.1, 0.2, 0.1,
                     1e9 * lal.PC_SI, 0.3, 1 / SAMPLING_FREQUENCY, 20.0, 20.0, lal.CreateDict())
        self.assertEqual(len(modes_at_model_times(model_times, "IMRPhenomTPHM", *arguments)), 32)
        self.assertEqual(set(modes_at_model_times(model_times, "IMRPhenomTP", *arguments)),
                         {(2, m) for m in range(-2, 3)})


class TestTimeDomainData(unittest.TestCase):
    def test_noise_realisation(self):
        first, second = interferometers(noise="gaussian"), interferometers(noise="gaussian")
        for one, two in zip(first, second):
            self.assertEqual(len(one.time_domain_strain), DURATION * SAMPLING_FREQUENCY)
            self.assertAlmostEqual(one.strain_data.start_time, START_TIME)
            np.testing.assert_array_equal(one.time_domain_strain, two.time_domain_strain)
            self.assertGreater(np.std(one.time_domain_strain), 0)

    def test_injection_adds_to_stored_time_series(self):
        ifos = interferometers(noise="gaussian")
        noise = {ifo.name: ifo.time_domain_strain.copy() for ifo in ifos}
        generator = waveform_generator()
        ifos.inject_signal_time_domain(parameters=INJECTION, waveform_generator=generator)
        polarizations = generator.time_domain_strain(INJECTION)
        for ifo in ifos:
            signal = ifo.get_time_domain_detector_response(polarizations, INJECTION)
            np.testing.assert_allclose(ifo.time_domain_strain, noise[ifo.name] + signal, rtol=0, atol=1e-30)
            self.assertEqual(ifo.meta_data["injection_domain"], "time")

    def test_analysis_data_is_the_window_slice(self):
        full = interferometers()
        full.inject_signal_time_domain(parameters=INJECTION, waveform_generator=waveform_generator())
        post_inspiral = interferometers(**SEGMENTS["post_inspiral"])
        post_inspiral.inject_signal_time_domain(parameters=INJECTION, waveform_generator=waveform_generator())
        for ifo, reference in zip(full, post_inspiral):
            ifo.analysis_window = (reference.analysis_window["start_time"], reference.analysis_window["duration"])
            self.assertEqual(len(ifo.time_domain_strain), DURATION * SAMPLING_FREQUENCY)
            np.testing.assert_allclose(
                ifo.time_array[ifo._analysis_slice()], reference.time_array[reference._analysis_slice()],
                atol=1e-6)
            np.testing.assert_allclose(ifo.analysis_data, reference.analysis_data, atol=1e-30)

    def test_injection_needs_autocovariance_function(self):
        ifos = interferometers(acf=False)
        with self.assertRaises(ValueError):
            ifos.inject_signal_time_domain(parameters=INJECTION, waveform_generator=waveform_generator())

    def _sine_data(self, start_time, duration, sampling_frequency=4096):
        ifo = bilby.gw.detector.get_empty_interferometer("H1")
        times = start_time + np.arange(duration * sampling_frequency) / sampling_frequency
        strain = np.sin(2 * np.pi * 60 * times) + np.sin(2 * np.pi * 1500 * times)
        ifo.strain_data.set_from_time_domain_strain(
            strain, sampling_frequency=sampling_frequency, duration=duration, start_time=start_time)
        return ifo

    def test_downsample(self):
        ifo = self._sine_data(1000.0, 16)
        ifo.analysis_window = (1006.0, 4)
        ifo.condition_strain_data(1024)
        self.assertEqual(ifo.sampling_frequency, 1024)
        self.assertAlmostEqual(ifo.time_array[ifo._analysis_slice()][0], 1006.0)
        self.assertEqual(len(ifo.analysis_data), 4096)
        # the 60 Hz line is kept and the 1500 Hz line (above the new Nyquist frequency) removed
        expected = np.sin(2 * np.pi * 60 * ifo.time_array)
        np.testing.assert_allclose(ifo.time_domain_strain, expected - np.mean(expected), atol=1e-6)

    def test_decimate(self):
        ifo = self._sine_data(1000.0001, 16)
        ifo.analysis_window = (1006.0, 4)
        ifo.condition_strain_data(1024, taper_and_trim=False, remove_mean=False)
        self.assertEqual(ifo.sampling_frequency, 1024)
        self.assertEqual(ifo.duration, 16)
        self.assertAlmostEqual(ifo.time_array[ifo._analysis_slice()][0], 1006.0, 3)
        self.assertEqual(len(ifo.analysis_data), 4096)
        # the 60 Hz line is kept (up to the filter ripple) and the 1500 Hz line removed
        expected = np.sin(2 * np.pi * 60 * ifo.time_array[ifo._analysis_slice()])
        np.testing.assert_allclose(ifo.analysis_data, expected, atol=1e-2)

    def test_downsample_data_slightly_off_grid(self):
        ifo = self._sine_data(1004.0001, 8)
        ifo.analysis_window = (1006.0, 4)
        ifo.condition_strain_data(1024)
        self.assertEqual(len(ifo.analysis_data), 4096)
        self.assertAlmostEqual(ifo.time_array[ifo._analysis_slice()][0], 1006.0, 3)

    def test_downsample_needs_buffer(self):
        ifo = self._sine_data(1005.0, 6)
        ifo.analysis_window = (1006.0, 4)
        with self.assertRaises(ValueError):
            ifo.condition_strain_data(1024)

    def test_data_not_conditioned_by_default(self):
        ifo = self._sine_data(1000.0, 16)
        ifo.analysis_window = (1006.0, 4)
        self.assertEqual(len(ifo.time_domain_strain), 16 * 4096)
        self.assertAlmostEqual(ifo.time_array[ifo._analysis_slice()][0], 1006.0)
        self.assertEqual(len(ifo.analysis_data), 4 * 4096)

    def test_network_downsample(self):
        ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
        ifos.set_strain_data_from_power_spectral_densities(4096, 16, 1000, random_state=2)
        ifos.set_analysis_windows(duration=4, start_time=1006)
        ifos.condition_strain_data(2048)
        for ifo in ifos:
            self.assertEqual(ifo.sampling_frequency, 2048)
            self.assertAlmostEqual(ifo.time_array[ifo._analysis_slice()][0], 1006.0)
            self.assertEqual(len(ifo.analysis_data), 4 * 2048)


class TestTimeDomainSourceModels(unittest.TestCase):
    def setUp(self):
        self.time_array = np.arange(DURATION * SAMPLING_FREQUENCY) / SAMPLING_FREQUENCY
        self.parameters = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(
            dict(INJECTION))[0]
        for key in ["ra", "dec", "psi", "geocent_time"]:
            self.parameters.pop(key)
        self.waveform_kwargs = dict(
            waveform_approximant="IMRPhenomXP", reference_frequency=20.0,
            minimum_frequency=20.0, turn_on_window=0)

    def test_output_on_own_grid(self):
        output = bilby.gw.source.lal_binary_black_hole_time_domain(
            self.time_array, **self.parameters, **self.waveform_kwargs)
        self.assertEqual(set(output), {"plus", "cross", "epoch"})
        self.assertEqual(len(output["plus"]), len(output["cross"]))
        self.assertLess(output["epoch"], 0)
        # the amplitude peak is close to the model's t=0
        peak = np.argmax(output["plus"] ** 2 + output["cross"] ** 2)
        self.assertLess(abs(output["epoch"] + peak / SAMPLING_FREQUENCY), 0.05)

    def test_matches_lalsimulation_with_iota(self):
        import lal
        import lalsimulation
        output = bilby.gw.source.lal_binary_black_hole_time_domain(
            self.time_array, **self.parameters, **self.waveform_kwargs)
        p = self.parameters
        iota, s1x, s1y, s1z, s2x, s2y, s2z = bilby.gw.conversion.bilby_to_lalsimulation_spins(
            theta_jn=p["theta_jn"], phi_jl=p["phi_jl"], tilt_1=p["tilt_1"], tilt_2=p["tilt_2"],
            phi_12=p["phi_12"], a_1=p["a_1"], a_2=p["a_2"],
            mass_1=p["mass_1"] * bilby.core.utils.solar_mass,
            mass_2=p["mass_2"] * bilby.core.utils.solar_mass,
            reference_frequency=20.0, phase=p["phase"])
        self.assertGreater(abs(iota - p["theta_jn"]), 1e-3)  # precessing: iota != theta_jn
        hplus, hcross = lalsimulation.SimInspiralChooseTDWaveform(
            p["mass_1"] * lal.MSUN_SI, p["mass_2"] * lal.MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z,
            p["luminosity_distance"] * 1e6 * lal.PC_SI, iota, p["phase"], 0.0, 0.0, 0.0,
            1.0 / SAMPLING_FREQUENCY, 20.0, 20.0, lal.CreateDict(),
            lalsimulation.GetApproximantFromString("IMRPhenomXP"))
        np.testing.assert_allclose(output["plus"], hplus.data.data, rtol=0,
                                   atol=1e-6 * np.max(abs(hplus.data.data)))
        np.testing.assert_allclose(output["cross"], hcross.data.data, rtol=0,
                                   atol=1e-6 * np.max(abs(hcross.data.data)))

    def test_binary_neutron_star(self):
        parameters = dict(
            mass_1=1.5, mass_2=1.3, luminosity_distance=50.0, a_1=0.02, tilt_1=0.0,
            phi_12=0.0, a_2=0.02, tilt_2=0.0, phi_jl=0.0, theta_jn=0.4, phase=1.3)
        kwargs = dict(waveform_approximant="IMRPhenomPv2_NRTidal", reference_frequency=50.0,
                      minimum_frequency=100.0)
        tidal = bilby.gw.source.lal_binary_neutron_star_time_domain(
            self.time_array, lambda_1=545, lambda_2=1346, **parameters, **kwargs)
        point = bilby.gw.source.lal_binary_neutron_star_time_domain(
            self.time_array, lambda_1=0, lambda_2=0, **parameters, **kwargs)
        self.assertIn("epoch", tidal)
        length = min(len(tidal["plus"]), len(point["plus"]))
        self.assertGreater(np.max(abs(tidal["plus"][-length:] - point["plus"][-length:])), 0)

    def test_unused_arguments_raise(self):
        with self.assertRaises(ValueError):
            bilby.gw.source.lal_binary_black_hole_time_domain(
                self.time_array, **self.parameters, **self.waveform_kwargs, not_an_argument=1)


class TestNoiseInputs(unittest.TestCase):
    def test_read_power_spectral_densities_from_hdf5(self):
        import h5py
        frequencies = np.arange(0, 1024.125, 0.125)
        psd = 1e-46 * (1 + (50 / np.maximum(frequencies, 1)) ** 4)
        directory = tempfile.mkdtemp()
        try:
            simple = os.path.join(directory, "simple.h5")
            with h5py.File(simple, "w") as ff:
                for ifo in ["h1", "l1"]:
                    ff.create_dataset(f"psds/{ifo}_psd/frequency", data=frequencies)
                    ff.create_dataset(f"psds/{ifo}_psd/spectrum", data=psd)
            pesummary = os.path.join(directory, "pesummary.h5")
            with h5py.File(pesummary, "w") as ff:
                for label in ["C01:IMRPhenomXPHM", "C01:SEOBNRv4PHM"]:
                    for ifo in ["H1", "L1"]:
                        ff.create_dataset(f"{label}/psds/{ifo}", data=np.array([frequencies, psd]).T)
            read = bilby.gw.detector.read_power_spectral_densities_from_hdf5
            for output in [read(simple, ["H1", "L1"]),
                           read(pesummary, ["H1", "L1"], label="C01:IMRPhenomXPHM")]:
                self.assertEqual(set(output), {"H1", "L1"})
                np.testing.assert_array_equal(output["L1"][0], frequencies)
                np.testing.assert_array_equal(output["L1"][1], psd)
            with self.assertRaises(ValueError):
                read(pesummary, ["H1"])  # two analyses: label needed
        finally:
            shutil.rmtree(directory)

    def test_welch_acf_agrees_with_psd_acf(self):
        ifo = bilby.gw.detector.get_empty_interferometer("H1")
        ifo.minimum_frequency, ifo.maximum_frequency = 20, 448
        ifo.set_strain_data_from_power_spectral_density(SAMPLING_FREQUENCY, 256, 0, random_state=11)
        noise = ifo.time_domain_strain.copy()
        ifo.analysis_window = (0, 1)
        ifo.set_autocovariance_function_from_power_spectral_density()
        from_data = AutoCovarianceFunction.from_time_domain_strain(
            noise, SAMPLING_FREQUENCY, analysis_duration=1, minimum_frequency=20, maximum_frequency=448)
        times = np.arange(SAMPLING_FREQUENCY) / SAMPLING_FREQUENCY
        signal = 1e-22 * np.exp(-0.5 * ((times - 0.5) / 0.05) ** 2) * np.sin(2 * np.pi * 120 * times)
        snr_squared = [gohberg_semencul_product(*acf.gohberg_semencul_vectors(len(times)), signal) @ signal
                       for acf in [ifo.autocovariance_function, from_data]]
        self.assertAlmostEqual(snr_squared[1] / snr_squared[0], 1, delta=0.1)


if __name__ == "__main__":
    unittest.main()
