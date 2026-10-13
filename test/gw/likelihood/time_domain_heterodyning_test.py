import json
import os
import shutil
import tempfile
import unittest

import numpy as np

import bilby
from bilby.gw.likelihood import (
    TimeDomainGravitationalWaveTransient,
    TimeDomainHeterodyningGravitationalWaveTransient,
)
from bilby.gw.time_domain_utils import (
    gohberg_semencul_product,
)

INJECTION = dict(
    mass_1=36.0, mass_2=29.0, a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=1000.0, theta_jn=1.2, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108,
)
ALIGNED_INJECTION = dict(INJECTION, tilt_1=0.0, tilt_2=np.pi)
APPROXIMANTS = [("IMRPhenomT", ALIGNED_INJECTION), ("IMRPhenomTHM", ALIGNED_INJECTION),
                ("IMRPhenomTP", INJECTION), ("IMRPhenomTPHM", INJECTION)]
SAMPLING_FREQUENCY = 1024
# the waveforms start at 20 Hz (~0.77 s before the merger; the heterodyned likelihood's
# waveforms at sampled parameters at 18 Hz, ~1 s before). The default window starts after
# both; WINDOW_WITH_START contains them.
DURATION = 1.0
START_TIME = INJECTION["geocent_time"] - 0.6
WINDOW_WITH_START = dict(duration=1.5, start_time=INJECTION["geocent_time"] - 1.2)


def _has_model_time_waveforms():
    import lalsimulation
    return hasattr(lalsimulation, "SimIMRPhenomTPHMCoprecModesHarmonicsModelTimes")


def generator(source_model, approximant="IMRPhenomTPHM", minimum_frequency=20.0):
    return bilby.gw.WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY, start_time=START_TIME,
        time_domain_source_model=source_model,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant=approximant, reference_frequency=20.0,
                                minimum_frequency=minimum_frequency))


def exact_generator(approximant="IMRPhenomTPHM"):
    return generator(bilby.gw.source.lal_binary_black_hole_model_time, approximant)


def heterodyning_generator(approximant="IMRPhenomTPHM", **kwargs):
    return generator(bilby.gw.source.lal_binary_black_hole_time_domain_heterodyning, approximant, **kwargs)


def interferometers(approximant="IMRPhenomTPHM", injection=INJECTION, duration=DURATION, start_time=START_TIME):
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in ifos:
        ifo.minimum_frequency, ifo.maximum_frequency = 20, 448
    ifos.set_analysis_windows(duration=duration, start_time=start_time, analysis_segment="imr",
                              cut_reference_parameters=injection)
    random_state = np.random.default_rng(3)
    for ifo in ifos:
        ifo.set_strain_data_from_power_spectral_density(
            SAMPLING_FREQUENCY, ifo.analysis_window["duration"], ifo.analysis_window["start_time"],
            random_state=random_state)
    ifos.set_autocovariance_functions_from_power_spectral_densities()
    ifos.inject_signal_time_domain(
        parameters=injection, waveform_generator=exact_generator(approximant), placement="model_time")
    return ifos


def perturbed(parameters, scale=1.0):
    parameters = dict(parameters)
    parameters["mass_1"] *= 1 + 0.002 * scale
    parameters["mass_2"] *= 1 - 0.002 * scale
    parameters["a_1"] += 0.02 * scale
    parameters["phase"] += 0.05 * scale
    parameters["psi"] -= 0.05 * scale
    parameters["geocent_time"] += 0.0002 * scale
    parameters["luminosity_distance"] *= 1 + 0.05 * scale
    return parameters


class TestBinEdges(unittest.TestCase):
    def test_edges(self):
        bin_edges = TimeDomainHeterodyningGravitationalWaveTransient.time_domain_bin_edges
        times = np.arange(-0.5, 0.3, 1 / 1024)
        edges = bin_edges(times)
        self.assertEqual(edges[0], 0)
        self.assertEqual(edges[-1], len(times) - 1)
        self.assertTrue(np.all(np.diff(edges) > 0))
        merger = np.searchsorted(times, 0)
        # two samples per bin from the merger until 0.02 s, ten until 0.1 s
        self.assertTrue(np.all(np.diff(edges[(times[edges] >= 0) & (times[edges] < 0.019)]) == 2))
        late = edges[(times[edges] > 0.03) & (times[edges] < 0.09)]
        self.assertTrue(np.all(np.diff(late) == 10))
        # smaller epsilon: more inspiral bins
        self.assertGreater(np.sum(bin_edges(times, epsilon=(0.1, 0.05)) < merger),
                           np.sum(edges < merger))


@unittest.skipUnless(_has_model_time_waveforms(), "lalsimulation without the model-time IMRPhenomT functions")
class TestHeterodyningTerms(unittest.TestCase):
    def test_terms_reproduce_polarizations(self):
        import lal
        from bilby.gw.model_time_waveforms import (
            heterodyning_terms_at_model_times,
            polarizations_at_model_times,
            total_mass_in_seconds,
        )
        mass_1, mass_2 = 36 * lal.MSUN_SI, 29 * lal.MSUN_SI
        model_times = np.linspace(-0.7, 0.05, 300) / total_mass_in_seconds(mass_1, mass_2)
        for approximant, spins in [("IMRPhenomT", (0, 0, 0.3, 0, 0, -0.2)),
                                   ("IMRPhenomTHM", (0, 0, 0.3, 0, 0, -0.2)),
                                   ("IMRPhenomTP", (0.4, 0.2, 0.3, -0.1, 0.3, -0.2)),
                                   ("IMRPhenomTPHM", (0.4, 0.2, 0.3, -0.1, 0.3, -0.2))]:
            with self.subTest(approximant=approximant):
                arguments = (model_times, approximant, mass_1, mass_2, *spins, 1e9 * lal.PC_SI, 0.7, 1.3,
                             1 / SAMPLING_FREQUENCY, 20.0, 20.0, lal.CreateDict())
                plus, cross = polarizations_at_model_times(*arguments)
                modes, harmonics = heterodyning_terms_at_model_times(*arguments)
                self.assertTrue(all(m >= 0 for _, m in modes))
                waveform = sum(harmonics[key][0] * mode + harmonics[key][1] * np.conj(mode)
                               for key, mode in modes.items())
                self.assertLess(np.max(abs(waveform - (plus - 1j * cross))) / np.max(abs(plus)), 1e-12)


@unittest.skipUnless(_has_model_time_waveforms(), "lalsimulation without the model-time IMRPhenomT functions")
class TestTimeDomainHeterodyning(unittest.TestCase):
    def test_matches_exact_likelihood(self):
        for approximant, injection in APPROXIMANTS:
            with self.subTest(approximant=approximant):
                ifos = interferometers(approximant, injection)
                exact = TimeDomainGravitationalWaveTransient(
                    ifos, exact_generator(approximant), placement="model_time")
                heterodyned = TimeDomainHeterodyningGravitationalWaveTransient(
                    ifos, heterodyning_generator(approximant), fiducial_parameters=injection)
                # aligned spins: exact at the fiducial point. Precessing spins: the harmonic factors
                # are linearized in each bin, and LAL computes the precession angles on its own time
                # grid from the minimum frequency, so waveforms from 18 Hz (heterodyned) and 20 Hz
                # (exact) differ slightly near the merger (more at low sampling frequency)
                aligned = approximant in ("IMRPhenomT", "IMRPhenomTHM")
                self.assertLess(abs(heterodyned.log_likelihood_ratio(injection)
                                    - exact.log_likelihood_ratio(injection)), 1e-6 if aligned else 0.05)
                parameters = perturbed(injection)
                self.assertGreater(exact.log_likelihood_ratio(injection) - exact.log_likelihood_ratio(parameters), 1)
                self.assertLess(abs(heterodyned.log_likelihood_ratio(parameters)
                                    - exact.log_likelihood_ratio(parameters)), 0.05 if aligned else 0.1)

    def test_use_float32(self):
        ifos = interferometers()
        values = list()
        for use_float32 in (False, True):
            likelihood = TimeDomainHeterodyningGravitationalWaveTransient(
                ifos, heterodyning_generator(), fiducial_parameters=INJECTION,
                use_float32=use_float32)
            values.append(likelihood.log_likelihood_ratio(perturbed(INJECTION)))
        self.assertLess(abs(values[0] - values[1]), 1e-3)

    def test_meta_data(self):
        ifos = interferometers()
        likelihood = TimeDomainHeterodyningGravitationalWaveTransient(
            ifos, heterodyning_generator(), fiducial_parameters=INJECTION)
        meta_data = likelihood.meta_data
        for ifo in ifos:
            np.testing.assert_array_equal(meta_data["bin_edges"][ifo.name], likelihood.bin_edges[ifo.name])
            self.assertEqual(len(meta_data["bin_edge_times"][ifo.name]), meta_data["number_of_bins"][ifo.name] + 1)
        json.dumps(meta_data, cls=bilby.core.utils.BilbyJsonEncoder)

    def test_full_waveform_is_piecewise_linear_model(self):
        ifos = interferometers()
        likelihood = TimeDomainHeterodyningGravitationalWaveTransient(
            ifos, heterodyning_generator(), fiducial_parameters=INJECTION)
        parameters = perturbed(INJECTION)
        signal = likelihood.waveform_generator.time_domain_strain(parameters)
        for ifo in ifos:
            template = likelihood._compute_full_waveform(signal, ifo, parameters)
            snrs = likelihood.calculate_snrs(signal, ifo, parameters=parameters)
            x, y = ifo.gohberg_semencul_vectors
            # <s, s> without the terms quadratic in the slopes: <s, s> - <s1, s1>, s1 = slope part of s
            r_tilde_1 = likelihood._r_tilde_coefficients(signal, [ifo], parameters)[0][:, 1]
            summary = likelihood.summary_data[ifo.name]
            h0 = np.array([likelihood.fiducial_modes[ifo.name][mode] for mode in likelihood.mode_keys])
            slopes = np.sum(
                (r_tilde_1[:, summary["bin_of_sample"]] * summary["time_from_bin_centre"] * h0).real, axis=0)
            expected = (template @ gohberg_semencul_product(x, y, template)
                        - slopes @ gohberg_semencul_product(x, y, slopes))
            self.assertAlmostEqual(snrs.optimal_snr_squared / expected, 1, 10)
            self.assertAlmostEqual(
                snrs.d_inner_h / (likelihood._data_times_inverse_covariance[ifo.name] @ template), 1, 10)

    def test_distance_marginalization(self):
        ifos = interferometers()
        priors = bilby.gw.prior.BBHPriorDict()
        priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
            minimum=200, maximum=3000, name="luminosity_distance")
        lookup = os.path.join(tempfile.mkdtemp(), "lookup.npz")
        exact = TimeDomainGravitationalWaveTransient(
            ifos, exact_generator(), placement="model_time", distance_marginalization=True,
            priors=priors.copy(), distance_marginalization_lookup_table=lookup)
        heterodyned = TimeDomainHeterodyningGravitationalWaveTransient(
            ifos, heterodyning_generator(), fiducial_parameters=INJECTION, distance_marginalization=True,
            priors=priors.copy(), distance_marginalization_lookup_table=lookup)
        parameters = dict(perturbed(INJECTION, 0.5), luminosity_distance=heterodyned._ref_dist)
        self.assertLess(abs(heterodyned.log_likelihood_ratio(parameters)
                            - exact.log_likelihood_ratio(parameters)), 0.05)
        sample = heterodyned.generate_posterior_sample_from_marginalized_likelihood(dict(INJECTION))
        self.assertTrue(200 <= sample["luminosity_distance"] <= 3000)
        shutil.rmtree(os.path.dirname(lookup))

    def test_time_reference(self):
        ifos = interferometers()
        parameters = dict(INJECTION)
        parameters["H1_time"] = parameters.pop("geocent_time") + ifos[0].time_delay_from_geocenter(
            INJECTION["ra"], INJECTION["dec"], INJECTION["geocent_time"])
        exact = TimeDomainGravitationalWaveTransient(
            ifos, exact_generator(), placement="model_time", time_reference="H1")
        heterodyned = TimeDomainHeterodyningGravitationalWaveTransient(
            ifos, heterodyning_generator(), fiducial_parameters=parameters, time_reference="H1")
        parameters["H1_time"] += 0.0002
        self.assertLess(abs(heterodyned.log_likelihood_ratio(parameters)
                            - exact.log_likelihood_ratio(parameters)), 0.05)

    def test_update_fiducial_parameters(self):
        ifos = interferometers("IMRPhenomT", ALIGNED_INJECTION)
        priors = bilby.core.prior.PriorDict(
            {key: value for key, value in ALIGNED_INJECTION.items() if key not in ("mass_1", "mass_2")})
        priors["chirp_mass"] = bilby.core.prior.Uniform(27, 29, "chirp_mass")
        priors["mass_ratio"] = bilby.core.prior.Uniform(0.5, 1, "mass_ratio")
        start = {key: value for key, value in ALIGNED_INJECTION.items() if key not in ("mass_1", "mass_2")}
        start.update(chirp_mass=28.4, mass_ratio=0.7)
        likelihood = TimeDomainHeterodyningGravitationalWaveTransient(
            ifos, heterodyning_generator("IMRPhenomT"), fiducial_parameters=start, priors=priors,
            update_fiducial_parameters=True, maximization_kwargs=dict(maxiter=5, popsize=5, seed=1))
        self.assertEqual(likelihood.parameters_to_be_updated, ["chirp_mass", "mass_ratio"])
        self.assertGreater(likelihood.log_likelihood_ratio(likelihood.fiducial_parameters),
                           likelihood.log_likelihood_ratio(start))

    def test_fiducial_waveform_and_bins_cover_the_window(self):
        ifos = interferometers(**WINDOW_WITH_START)
        likelihood = TimeDomainHeterodyningGravitationalWaveTransient(
            ifos, heterodyning_generator(), fiducial_parameters=INJECTION, sampled_minimum_frequency=10.0)
        self.assertLess(likelihood.fiducial_minimum_frequency, 18)
        for ifo in ifos:
            number_of_samples = len(ifo.analysis_data)
            self.assertTrue(np.all(likelihood.fiducial_modes[ifo.name][(2, 2)] != 0))
            self.assertEqual(likelihood.bin_edges[ifo.name][0], 0)
            self.assertEqual(likelihood.bin_edges[ifo.name][-1], number_of_samples - 1)
        self.assertEqual(likelihood.waveform_generator.waveform_arguments["fiducial"], 0)
        given = TimeDomainHeterodyningGravitationalWaveTransient(
            ifos, heterodyning_generator(), fiducial_parameters=INJECTION, sampled_minimum_frequency=10.0,
            fiducial_minimum_frequency=5.0)
        self.assertEqual(given.fiducial_minimum_frequency, 5.0)
        self.assertAlmostEqual(given.log_likelihood_ratio(perturbed(INJECTION)),
                               likelihood.log_likelihood_ratio(perturbed(INJECTION)), 2)

    def test_shared_waveform_arguments_are_not_changed(self):
        ifos = interferometers()
        arguments = dict(waveform_approximant="IMRPhenomTPHM", reference_frequency=20.0, minimum_frequency=20.0)
        generator = bilby.gw.WaveformGenerator(
            duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY, start_time=START_TIME,
            time_domain_source_model=bilby.gw.source.lal_binary_black_hole_time_domain_heterodyning,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=arguments)
        TimeDomainHeterodyningGravitationalWaveTransient(ifos, generator, fiducial_parameters=INJECTION)
        self.assertEqual(set(arguments), {"waveform_approximant", "reference_frequency", "minimum_frequency"})

    def test_errors(self):
        ifos = interferometers()
        with self.assertRaises(ValueError):  # fiducial_minimum_frequency above the derived one
            TimeDomainHeterodyningGravitationalWaveTransient(
                ifos, heterodyning_generator(), fiducial_parameters=INJECTION, fiducial_minimum_frequency=30.0)
        with self.assertLogs("bilby", level="WARNING"):  # above minimum_frequency - 2 Hz
            TimeDomainHeterodyningGravitationalWaveTransient(
                ifos, heterodyning_generator(), fiducial_parameters=INJECTION, sampled_minimum_frequency=19.0)
        with self.assertRaises(NotImplementedError):
            TimeDomainHeterodyningGravitationalWaveTransient(
                ifos, heterodyning_generator(), fiducial_parameters=INJECTION, time_marginalization=True)


if __name__ == "__main__":
    unittest.main()
