import unittest
import numpy as np
import pandas as pd
from parameterized import parameterized

import bilby.hyper as hyp


def _toy_function(data, dataset, self, cls, a, b, c):
    return a


class _ToyClassNoVariableNames:
    def __call__(self, a, b, c):
        return a


class _ToyClassVariableNames:

    variable_names = ["a", "b", "c"]

    def __call__(self, **kwargs):
        return kwargs.get("a", 1)


class TestHyperLikelihood(unittest.TestCase):
    def setUp(self):
        self.keys = ["a", "b", "c"]
        self.lengths = [300, 400, 500]
        self.posteriors = list()
        for ii, length in enumerate(self.lengths):
            self.posteriors.append(
                pd.DataFrame({key: np.random.normal(0, 1, length) for key in self.keys})
            )
        self.log_evidences = [2, 2, 2]
        self.model = hyp.model.Model(list())
        self.sampling_model = hyp.model.Model(list())

    def tearDown(self):
        del self.keys
        del self.lengths
        del self.posteriors
        del self.log_evidences

    def test_evidence_factor_with_evidences(self):
        like = hyp.likelihood.HyperparameterLikelihood(
            self.posteriors,
            self.model,
            self.sampling_model,
            log_evidences=self.log_evidences,
        )
        self.assertEqual(like.evidence_factor, 6)

    def test_evidence_factor_without_evidences(self):
        like = hyp.likelihood.HyperparameterLikelihood(
            self.posteriors, self.model, self.sampling_model
        )
        self.assertTrue(np.isnan(like.evidence_factor))

    @parameterized.expand([
        ("func", _toy_function),
        ("class_no_names", _ToyClassNoVariableNames()),
        ("class_with_names", _ToyClassVariableNames()),
    ])
    def test_get_function_parameters(self, _, model):
        expected = dict(a=1, b=2, c=3)
        model = hyp.model.Model([model])
        model.parameters.update(expected)
        result = model._get_function_parameters(model.models[0])
        self.assertDictEqual(expected, result)

    def test_len_samples_with_max_samples(self):
        like = hyp.likelihood.HyperparameterLikelihood(
            self.posteriors,
            self.model,
            self.sampling_model,
            log_evidences=self.log_evidences,
            max_samples=10,
        )
        self.assertEqual(like.samples_per_posterior, 10)

    def test_len_samples_without_max_samples(self):
        like = hyp.likelihood.HyperparameterLikelihood(
            self.posteriors,
            self.model,
            self.sampling_model,
            log_evidences=self.log_evidences,
        )
        self.assertEqual(like.samples_per_posterior, min(self.lengths))

    def test_resample_with_max_samples(self):
        like = hyp.likelihood.HyperparameterLikelihood(
            self.posteriors,
            self.model,
            self.sampling_model,
            log_evidences=self.log_evidences,
        )
        resampled = like.resample_posteriors()
        self.assertEqual(resampled["a"].shape, (len(self.lengths), min(self.lengths)))

    def test_resample_without_max_samples(self):
        like = hyp.likelihood.HyperparameterLikelihood(
            self.posteriors,
            self.model,
            self.sampling_model,
            log_evidences=self.log_evidences,
        )
        resampled = like.resample_posteriors(10)
        self.assertEqual(resampled["a"].shape, (len(self.lengths), 10))


if __name__ == "__main__":
    unittest.main()
