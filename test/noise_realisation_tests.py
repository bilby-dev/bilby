from __future__ import absolute_import
import numpy as np
import unittest
from test.context import tupak


class TestNoiseRealisation(unittest.TestCase):
    def test_averaged_noise(self):
        time_duration = 1.
        sampling_frequency = 4096.
        factor = np.sqrt(2./time_duration)
        n_avg = 1000
        psd_avg = 0
        interferometer = tupak.gw.detector.get_empty_interferometer('H1')
        for x in range(0, n_avg):
            interferometer.set_data(sampling_frequency, time_duration, from_power_spectral_density=True)
            psd_avg += abs(interferometer.data)**2

        psd_avg = psd_avg/n_avg
        asd_avg = np.sqrt(abs(psd_avg))

        a = interferometer.amplitude_spectral_density_array/factor
        b = asd_avg
        self.assertTrue(np.allclose(a, b, rtol=1e-1))

    def test_noise_normalisation(self):
        time_duration = 1.
        sampling_frequency = 4096.
        time_array = tupak.core.utils.create_time_series(sampling_frequency=sampling_frequency, duration=time_duration)

        # generate some toy-model signal for matched filtering SNR testing
        n_avg = 1000
        snr = np.zeros(n_avg)
        mu = np.exp(-(time_array-time_duration/2.)**2 / (2.*0.1**2)) * np.sin(2*np.pi*100*time_array)
        muf, frequency_array = tupak.core.utils.nfft(mu, sampling_frequency)
        for x in range(0, n_avg):
            interferometer = tupak.gw.detector.get_empty_interferometer('H1')
            interferometer.set_data(sampling_frequency, time_duration, from_power_spectral_density=True)
            hf_tmp = interferometer.data
            psd = interferometer.power_spectral_density
            snr[x] = tupak.core.utils.inner_product(hf_tmp, muf, frequency_array, psd) \
                     / np.sqrt(tupak.core.utils.inner_product(muf, muf, frequency_array, psd))

        self.assertTrue(np.isclose(np.std(snr), 1.00, atol=1e-1))


if __name__ == '__main__':
    unittest.main()
