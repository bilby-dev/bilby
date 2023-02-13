import itertools
import os
import pytest
import unittest
from copy import deepcopy
from parameterized import parameterized

import numpy as np
import bilby


class TestMarginalizedLikelihood(unittest.TestCase):
    def setUp(self):
        np.random.seed(500)
        self.duration = 4
        self.sampling_frequency = 2048
        self.parameters = dict(
            mass_1=31.0,
            mass_2=29.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=1.7,
            phi_jl=0.3,
            luminosity_distance=4000.0,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3,
            geocent_time=1126259642.413,
            ra=1.375,
            dec=-1.2108,
        )

        self.interferometers = bilby.gw.detector.InterferometerList(["H1"])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            start_time=self.parameters["geocent_time"] - self.duration / 2,
        )

        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        )

        self.prior = bilby.gw.prior.BBHPriorDict()
        self.prior["geocent_time"] = bilby.prior.Uniform(
            minimum=self.parameters["geocent_time"] - self.duration / 2,
            maximum=self.parameters["geocent_time"] + self.duration / 2,
        )

    def test_cannot_instantiate_marginalised_likelihood_without_prior(self):
        self.assertRaises(
            ValueError,
            lambda: bilby.gw.likelihood.GravitationalWaveTransient(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                phase_marginalization=True,
            ),
        )

    def test_generating_default_time_prior(self):
        temp = self.prior.pop("geocent_time")
        new_prior = self.prior.copy()
        like = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            priors=new_prior,
            time_marginalization=True,
        )
        same = all(
            [
                temp.minimum == like.priors["geocent_time"].minimum,
                temp.maximum == like.priors["geocent_time"].maximum,
                new_prior["geocent_time"] == temp.minimum,
            ]
        )
        self.assertTrue(same)
        self.prior["geocent_time"] = temp

    def test_generating_default_phase_prior(self):
        temp = self.prior.pop("phase")
        new_prior = self.prior.copy()
        like = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            priors=new_prior,
            phase_marginalization=True,
        )
        same = all(
            [
                temp.minimum == like.priors["phase"].minimum,
                temp.maximum == like.priors["phase"].maximum,
                new_prior["phase"] == float(0),
            ]
        )
        self.assertTrue(same)
        self.prior["phase"] = temp

    def test_run_sampler_flags_if_marginalized_phase_is_sampled(self):
        like = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            priors=self.prior,
            phase_marginalization=True,
        )
        new_prior = self.prior.copy()
        new_prior["phase"] = bilby.prior.Uniform(minimum=0, maximum=2 * np.pi)
        for key, param in dict(
            mass_1=31.0,
            mass_2=29.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=1.7,
            phi_jl=0.3,
            theta_jn=0.4,
            psi=2.659,
            ra=1.375,
            dec=-1.2108,
        ).items():
            new_prior[key] = param
        with self.assertRaises(bilby.core.sampler.SamplingMarginalisedParameterError):
            bilby.run_sampler(like, new_prior)

    def test_run_sampler_flags_if_marginalized_time_is_sampled(self):
        like = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            priors=self.prior,
            time_marginalization=True,
        )
        new_prior = self.prior.copy()
        new_prior["geocent_time"] = bilby.prior.Uniform(minimum=0, maximum=1)
        for key, param in dict(
            mass_1=31.0,
            mass_2=29.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=1.7,
            phi_jl=0.3,
            theta_jn=0.4,
            psi=2.659,
            ra=1.375,
            dec=-1.2108,
        ).items():
            new_prior[key] = param
        with self.assertRaises(bilby.core.sampler.SamplingMarginalisedParameterError):
            bilby.run_sampler(like, new_prior)


class TestMarginalizations(unittest.TestCase):
    """
    Test all marginalised likelihoods matches brute force version.

    For time, this is strongly dependent on the specific time grid used.
    The `time_jitter` parameter makes this a weaker dependence during sampling.
    """
    _parameters = itertools.product(
        ["regular", "roq", "relbin"],
        ["luminosity_distance", "geocent_time", "phase"],
        [True, False],
        [True, False],
        [True, False],
    )

    lookup_phase = "distance_lookup_phase.npz"
    lookup_no_phase = "distance_lookup_no_phase.npz"
    path_to_roq_weights = "weights.npz"

    def setUp(self):
        np.random.seed(500)
        self.duration = 4
        self.sampling_frequency = 2048
        self.parameters = dict(
            mass_1=31.0,
            mass_2=29.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=1.7,
            phi_jl=0.3,
            luminosity_distance=4000.0,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3,
            geocent_time=1126259642.413,
            ra=1.375,
            dec=-1.2108,
            time_jitter=0,
            fiducial=0,
        )

        self.interferometers = bilby.gw.detector.InterferometerList(["H1"])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            start_time=1126259640,
        )

        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            start_time=1126259640,
            waveform_arguments=dict(
                reference_frequency=20.0,
                minimum_frequency=20.0,
                approximant="IMRPhenomPv2",
            )
        )
        self.interferometers.inject_signal(
            parameters=self.parameters, waveform_generator=self.waveform_generator
        )

        self.priors = bilby.gw.prior.BBHPriorDict()
        self.priors["geocent_time"] = bilby.prior.Uniform(
            minimum=self.parameters["geocent_time"] - 0.1,
            maximum=self.parameters["geocent_time"] + 0.1
        )

        trial_roq_paths = [
            "/roq_basis",
            os.path.join(os.path.expanduser("~"), "ROQ_data/IMRPhenomPv2/4s"),
            "/home/cbc/ROQ_data/IMRPhenomPv2/4s",
        ]
        roq_dir = None
        for path in trial_roq_paths:
            if os.path.isdir(path):
                roq_dir = path
                break
        if roq_dir is None:
            raise Exception("Unable to load ROQ basis: cannot proceed with tests")

        self.roq_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.binary_black_hole_roq,
            start_time=1126259640,
            waveform_arguments=dict(
                reference_frequency=20.0,
                minimum_frequency=20.0,
                approximant="IMRPhenomPv2",
                frequency_nodes_linear=np.load(f"{roq_dir}/fnodes_linear.npy"),
                frequency_nodes_quadratic=np.load(f"{roq_dir}/fnodes_quadratic.npy"),
            )
        )
        self.roq_linear_matrix_file = f"{roq_dir}/B_linear.npy"
        self.roq_quadratic_matrix_file = f"{roq_dir}/B_quadratic.npy"

        self.relbin_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole_relative_binning,
            start_time=1126259640,
            waveform_arguments=dict(
                reference_frequency=20.0,
                minimum_frequency=20.0,
                approximant="IMRPhenomPv2",
            )
        )

    def tearDown(self):
        del self.duration
        del self.sampling_frequency
        del self.parameters
        del self.interferometers
        del self.waveform_generator
        del self.roq_waveform_generator
        del self.priors

    @classmethod
    def tearDownClass(cls):
        """remove lookup tables so that they are not used accidentally in subsequent tests"""
        for filename in [cls.lookup_phase, cls.lookup_no_phase, cls.path_to_roq_weights]:
            if os.path.exists(filename):
                os.remove(filename)

    def likelihood_kwargs(self, kind, time_marginalization, phase_marginalization, distance_marginalization, priors):
        if priors is None:
            priors = deepcopy(self.priors)
        if distance_marginalization and phase_marginalization:
            lookup = TestMarginalizations.lookup_phase
        elif distance_marginalization:
            lookup = TestMarginalizations.lookup_no_phase
        else:
            lookup = None
        kwargs = dict(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            time_marginalization=time_marginalization,
            distance_marginalization_lookup_table=lookup,
            priors=priors,
        )
        if kind == "roq":
            kwargs.update(dict(
                linear_matrix=self.roq_linear_matrix_file,
                quadratic_matrix=self.roq_quadratic_matrix_file,
                waveform_generator=self.roq_waveform_generator,
            ))
            if os.path.exists(self.__class__.path_to_roq_weights):
                kwargs["weights"] = self.__class__.path_to_roq_weights
        elif kind == "relbin":
            kwargs["fiducial_parameters"] = deepcopy(self.parameters)
            kwargs["waveform_generator"] = self.relbin_waveform_generator
        return kwargs

    def get_likelihood(
        self,
        kind,
        time_marginalization=False,
        phase_marginalization=False,
        distance_marginalization=False,
        priors=None
    ):
        kwargs = self.likelihood_kwargs(
            kind, time_marginalization, phase_marginalization, distance_marginalization, priors
        )
        if kind == "regular":
            cls_ = bilby.gw.likelihood.GravitationalWaveTransient
        elif kind == "roq":
            cls_ = bilby.gw.likelihood.ROQGravitationalWaveTransient
        elif kind == "relbin":
            cls_ = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient
            kwargs["epsilon"] = 0.3
            self.parameters["fiducial"] = 0
        else:
            raise ValueError(f"kind {kind} not understood")
        like = cls_(**kwargs)
        if kind == "roq" and not os.path.exists(self.__class__.path_to_roq_weights):
            like.save_weights(self.__class__.path_to_roq_weights)
        like.parameters = self.parameters.copy()
        if time_marginalization:
            like.parameters["geocent_time"] = self.interferometers.start_time
        if distance_marginalization:
            like.parameters["luminosity_distance"] = like._ref_dist
        if phase_marginalization:
            like.parameters["phase"] = 0.0
        return like

    def _template(self, marginalized, non_marginalized, key, prior=None, values=None):
        if prior is None:
            prior = self.priors[key]
        if values is None:
            values = np.linspace(prior.minimum, prior.maximum, 1000)
        prior_values = prior.prob(values)
        ln_likes = np.empty(values.shape)
        for ii, value in enumerate(values):
            non_marginalized.parameters[key] = value
            ln_likes[ii] = non_marginalized.log_likelihood_ratio()
        like = np.exp(ln_likes - max(ln_likes))

        marg_like = np.log(np.trapz(like * prior_values, values)) + max(ln_likes)
        self.assertAlmostEqual(
            marg_like, marginalized.log_likelihood_ratio(), delta=0.5
        )

    @parameterized.expand(_parameters)
    def test_marginalisation(self, kind, key, distance, time, phase):
        if all([distance, time, phase]):
            pytest.skip()
        tested_args = dict(
            distance_marginalization=distance,
            time_marginalization=time,
            phase_marginalization=phase,
        )
        marg_key = f"{key.split('_')[-1]}_marginalization"
        if tested_args[marg_key]:
            pytest.skip()
        reference_args = tested_args.copy()
        reference_args[marg_key] = True
        self._template(
            self.get_likelihood(kind, **reference_args),
            self.get_likelihood(kind, **tested_args),
            key=key,
        )

    @parameterized.expand(["regular", "relbin"])
    def test_time_marginalisation_full_segment(self, kind):
        """
        Test time marginalised likelihood matches brute force version over
        just part of a segment.
        """
        priors = self.priors.copy()
        prior = bilby.prior.Uniform(
            minimum=self.interferometers.start_time,
            maximum=self.interferometers.start_time + self.interferometers.duration,
        )
        priors["geocent_time"] = prior
        self._template(
            self.get_likelihood(kind, time_marginalization=True, priors=priors.copy()),
            self.get_likelihood(kind, priors=priors.copy()),
            key="geocent_time",
            values=self.waveform_generator.time_array,
            prior=prior,
        )

    @parameterized.expand(
        itertools.product(["regular", "roq", "relbin"], *itertools.repeat([True, False], 3)),
        name_func=lambda func, num, param: (
            f"{func.__name__}_{num}__{param.args[0]}_" + "_".join([
                ["D", "P", "T"][ii] for ii, val
                in enumerate(param.args[1:]) if val
            ])
        )
    )
    def test_marginalization_reconstruction(self, kind, distance, phase, time):
        marginalizations = dict(
            geocent_time=time,
            luminosity_distance=distance,
            phase=phase,
        )
        like = self.get_likelihood(
            kind=kind,
            distance_marginalization=distance,
            time_marginalization=time,
            phase_marginalization=phase,
        )
        params = self.parameters.copy()
        reference_values = dict(
            luminosity_distance=self.priors["luminosity_distance"].rescale(0.5),
            geocent_time=self.interferometers.start_time,
            phase=0.0,
        )
        for key in marginalizations:
            if marginalizations[key]:
                params[key] = reference_values[key]
        like.parameters.update(params)
        output = like.generate_posterior_sample_from_marginalized_likelihood()
        for key in marginalizations:
            self.assertFalse(marginalizations[key] and reference_values[key] == output[key])
