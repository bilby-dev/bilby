from __future__ import absolute_import, division

import bilby
import unittest
import numpy as np
import matplotlib.pyplot as plt


class TestFFT(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_nfft_frequencies(self):
        f = 2.1
        sampling_frequency = 10
        times = np.arange(0, 100, 1/sampling_frequency)
        tds = np.sin(2*np.pi*times * f + 0.4)
        fds, freqs = bilby.core.utils.nfft(tds, sampling_frequency)
        self.assertTrue(np.abs((f-freqs[np.argmax(np.abs(fds))])/f < 1e-15))

    def test_nfft_infft(self):
        sampling_frequency = 10
        tds = np.random.normal(0, 1, 10)
        fds, _ = bilby.core.utils.nfft(tds, sampling_frequency)
        tds2 = bilby.core.utils.infft(fds, sampling_frequency)
        self.assertTrue(np.all(np.abs((tds - tds2) / tds) < 1e-12))


if __name__ == '__main__':
    unittest.main()
