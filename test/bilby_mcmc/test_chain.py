import os
import shutil
import unittest

import bilby
from bilby.bilby_mcmc.chain import Chain, Sample, calculate_tau
from bilby.bilby_mcmc.utils import LOGLKEY, LOGPKEY
from bilby.core.sampler.base_sampler import SamplerError
import numpy as np
import pandas as pd


class TestChain(unittest.TestCase):
    def setUp(self):
        self.initial_sample = self.create_random_sample()
        self.outdir = "chain_test"
        if os.path.isdir(self.outdir) is False:
            os.mkdir(self.outdir)

    def tearDown(self):
        if os.path.isdir(self.outdir):
            shutil.rmtree(self.outdir)

    def create_random_sample(self):
        return Sample({
            "a": np.random.normal(0, 1),
            "b": np.random.normal(0, 1),
            LOGLKEY: np.random.normal(0, 1),
            LOGPKEY: -1
        })

    def create_chain(self, n=1000):
        chain = Chain(initial_sample=self.initial_sample)
        for i in range(n):
            chain.append(self.create_random_sample())
        return chain

    def test_initialize(self):
        chain = Chain(initial_sample=self.initial_sample)
        self.assertEqual(chain.position, 0)

    def test_append(self):
        chain = Chain(initial_sample=self.initial_sample)
        chain.append(self.create_random_sample())
        self.assertEqual(chain.position, 1)
        self.assertEqual(len(chain.get_1d_array('a')), 2)

    def test_append_within_init_space(self):
        chain = Chain(initial_sample=self.initial_sample)
        N = chain.block_length - 1
        for i in range(N):
            chain.append(self.create_random_sample())

        self.assertEqual(chain.position, N)

        # N samples + 1 initial position
        self.assertEqual(len(chain.get_1d_array('a')), N + 1)

    def test_append_with_extending(self):
        block_length = 100
        chain = Chain(initial_sample=self.initial_sample, block_length=block_length)

        # Check the array is the block length
        self.assertEqual(len(chain._chain_array), block_length)
        for i in range(3 * block_length):
            chain.append(self.create_random_sample())

        # Check the array is now longer than the block length (successfully extended)
        self.assertEqual(len(chain._chain_array), 4 * block_length)

    def test_get_item(self):
        chain = self.create_chain()
        tenth_sample = chain[10]
        self.assertTrue(isinstance(tenth_sample, Sample))

        last_sample = chain[-1]
        self.assertEqual(last_sample, chain.current_sample)

        with self.assertRaises(SamplerError):
            chain[chain.position + 10]

    def test_set_item(self):
        chain = self.create_chain()
        s = self.create_random_sample()

        chain[10] = s
        self.assertEqual(s, chain[10])

        chain[-1] = s
        self.assertEqual(s, chain[-1])

    def test_random_sample(self):
        chain = self.create_chain()
        c1 = chain.random_sample
        c2 = chain.random_sample
        self.assertNotEqual(c1, c2)

    def test_fixed_discard(self):
        chain = self.create_chain()
        self.assertEqual(chain.fixed_discard, 0)
        chain.fixed_discard = 10
        self.assertEqual(chain.fixed_discard, 10)

    def test_minimum_index(self):
        chain = self.create_chain()
        # Test initialization
        self.assertEqual(chain.minimum_index, 1)

        chain._last_minimum_index = (chain.position, 10, "I")
        self.assertEqual(chain.minimum_index, 10)
        chain._last_minimum_index = (0, 0, "I")

        chain.fixed_discard = 200
        self.assertEqual(chain.minimum_index, 200)
        chain._last_minimum_index = (0, 0, "I")

        chain.fixed_discard = 100000
        self.assertEqual(chain.minimum_index, 100000)

    def test_tau(self):
        chain = self.create_chain(n=1000)
        self.assertGreaterEqual(chain.tau, chain.min_tau)
        self.assertLess(chain.tau, np.inf)
        self.assertEqual(chain.tau, chain.max_tau_dict[chain.position])
        self.assertEqual(chain.tau, chain.tau_last)

        # Check the cached tau calc works
        for i in range(5):
            chain.append(self.create_random_sample())
        chain.tau
        self.assertEqual(chain.cached_tau_count, 1)

    def test_nsamples(self):
        chain = self.create_chain(n=1000)
        self.assertGreaterEqual(chain.nsamples, 1)
        self.assertLessEqual(chain.nsamples, chain.position)

    def test_thin(self):
        chain = self.create_chain(n=1000)
        self.assertEqual(chain.thin, int(chain.thin_by_nact * chain.tau))

    def test_samples(self):
        chain = self.create_chain(n=1000)
        samples = chain.samples
        self.assertTrue(isinstance(samples, pd.DataFrame))
        self.assertTrue("a" in samples)
        self.assertTrue("b" in samples)
        self.assertTrue(LOGLKEY in samples)
        self.assertTrue(LOGPKEY in samples)

    def test_plot(self):
        chain = self.create_chain(n=1000)
        chain.plot(outdir=self.outdir, label="test")
        self.assertTrue(os.path.exists(f"{self.outdir}/test_checkpoint_trace.png"))
        priors = dict(
            a=bilby.core.prior.Uniform(-10, 10, latex_label='a'),
            b=bilby.core.prior.Uniform(-10, 10),
        )
        chain.thin_by_nact = 0.5
        chain.plot(outdir=self.outdir, label="test", priors=priors)
        self.assertTrue(os.path.exists(f"{self.outdir}/test_checkpoint_trace.png"))


class TestSample(unittest.TestCase):
    def setUp(self):
        self.sample_dict = dict(a=1, b=2)

    def tearDown(self):
        del self.sample_dict

    def test_init(self):
        s = Sample(self.sample_dict)
        self.assertEqual(s.keys, list(self.sample_dict.keys()))

    def test_dict_access(self):
        s = Sample(self.sample_dict)
        for key in s.keys:
            self.assertEqual(s[key], self.sample_dict[key])

    def test_list_access(self):
        s = Sample(self.sample_dict)
        slist = s.list
        self.assertEqual(slist, [self.sample_dict['a'], self.sample_dict['b']])

    def test_setitem(self):
        s = Sample(self.sample_dict)

        # Set existing parameter
        s['a'] = 100
        self.assertEqual(s['a'], 100)

        # Add parameter
        s['c'] = 100
        self.assertEqual(s['c'], 100)

    def test_parameter_only_dict(self):
        s = Sample(self.sample_dict)
        self.assertEqual(s.parameter_only_dict, dict(a=1, b=2))

    def test_update(self):
        sample_dict = dict(a=1, b=2)
        curr = Sample(sample_dict)
        prop = curr.copy()
        prop['a'] = 200
        self.assertEqual(prop['a'], 200)
        self.assertEqual(curr['a'], 1)


class TestACT(unittest.TestCase):
    def test_act_normal(self):
        x = np.random.normal(0, 1, 1000)
        tau = calculate_tau(x)
        self.assertLess(tau, 10)

    def test_act_identical(self):
        x = np.array([0] * 1000)
        tau = calculate_tau(x)
        self.assertEqual(tau, np.inf)

    def test_act_long(self):
        t = np.linspace(0, 1, 1000)
        x = np.sin(2 * np.pi * t)
        tau = calculate_tau(x)
        self.assertGreater(tau, 10)


if __name__ == "__main__":
    unittest.main()
