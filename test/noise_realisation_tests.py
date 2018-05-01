import numpy as np
import unittest
from context import peyote


class TestNoiseRealisation(unittest.TestCase):
    def test_averaged_noise(self):
        time_duration = 1.
        sampling_frequency = 4096.
        factor = np.sqrt(2./time_duration)
        navg = 1000
        for x in range(0, navg):
            H1 = peyote.detector.H1
            H1_hf_noise, frequencies = H1.power_spectral_density.get_noise_realisation(sampling_frequency, time_duration)
            H1.set_data(sampling_frequency, time_duration,frequency_domain_strain=H1_hf_noise)
            hf_tmp = H1.data
            if x==0:
                psd_avg = abs(hf_tmp)**2
            else:
                psd_avg = psd_avg + abs(hf_tmp)**2
        psd_avg = psd_avg/navg
        asd_avg = np.sqrt(abs(psd_avg))

        a = H1.amplitude_spectral_density_array/factor
        b = asd_avg
        self.assertTrue(np.isclose(a[2]/b[2], 1.00, atol=1e-2))

    def test_noise_normalisation(self):
        time_duration = 1.
        sampling_frequency = 4096.
        number_of_samples = int(np.round(time_duration*sampling_frequency))
        time_array = (1./sampling_frequency) * np.linspace(0, number_of_samples, number_of_samples)

        # generate some toy-model signal for matched filtering SNR testing
        navg = 10000
        snr = np.zeros(navg)
        mu = np.exp(-(time_array-time_duration/2.)**2 / (2.*0.1**2)) * np.sin(2*np.pi*100*time_array)
        muf, frequency_array = peyote.utils.nfft(mu, sampling_frequency)
        for x in range(0, navg):
            H1 = peyote.detector.H1
            H1_hf_noise, frequencies = H1.power_spectral_density.get_noise_realisation(sampling_frequency, time_duration)
            H1.set_data(sampling_frequency, time_duration,frequency_domain_strain=H1_hf_noise)
            hf_tmp = H1.data
            Sh = H1.power_spectral_density
            snr[x] = peyote.utils.inner_product(hf_tmp, muf, frequency_array, Sh) / np.sqrt(peyote.utils.inner_product(muf, muf, frequency_array, Sh))


        self.assertTrue(np.isclose(np.std(snr), 1.00, atol=1e-2))
