from __future__ import absolute_import
import numpy as np
import unittest
import bilby


class TestNoiseRealisation(unittest.TestCase):
    def test_averaged_noise(self):
        time_duration = 1.
        sampling_frequency = 4096.
        factor = np.sqrt(2 / time_duration)
        n_avg = 1000
        psd_avg = 0
        interferometer = bilby.gw.detector.get_empty_interferometer('H1')
        for x in range(0, n_avg):
            interferometer.set_strain_data_from_power_spectral_density(
                sampling_frequency=sampling_frequency, duration=time_duration)
            psd_avg += abs(interferometer.strain_data.frequency_domain_strain)**2

        psd_avg = psd_avg / n_avg
        asd_avg = np.sqrt(abs(psd_avg)) * interferometer.frequency_mask

        a = np.nan_to_num(interferometer.amplitude_spectral_density_array / factor * interferometer.frequency_mask)
        b = asd_avg
        print(a, b)
        self.assertTrue(np.allclose(a, b, rtol=1e-1))

    def test_noise_normalisation(self):
        time_duration = 1.
        sampling_frequency = 4096.
        time_array = bilby.core.utils.create_time_series(sampling_frequency=sampling_frequency, duration=time_duration)

        # generate some toy-model signal for matched filtering SNR testing
        n_avg = 1000
        snr = np.zeros(n_avg)
        mu = np.exp(-(time_array-time_duration/2.)**2 / (2.*0.1**2)) * np.sin(2*np.pi*100*time_array)
        muf, frequency_array = bilby.core.utils.nfft(mu, sampling_frequency)
        for x in range(0, n_avg):
            interferometer = bilby.gw.detector.get_empty_interferometer('H1')
            interferometer.set_strain_data_from_power_spectral_density(
                sampling_frequency=sampling_frequency, duration=time_duration)
            hf_tmp = interferometer.strain_data.frequency_domain_strain
            psd = interferometer.power_spectral_density
            snr[x] = bilby.gw.utils.inner_product(hf_tmp, muf, frequency_array, psd) \
                     / np.sqrt(bilby.gw.utils.inner_product(muf, muf, frequency_array, psd))

        self.assertTrue(np.isclose(np.std(snr), 1.00, atol=1e-1))


if __name__ == '__main__':
    unittest.main()
