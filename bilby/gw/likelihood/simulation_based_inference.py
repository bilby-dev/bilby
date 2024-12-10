import inspect
import pickle
import os

import numpy as np
import sbi
import sbi.utils
import sbi.inference
import torch

from ...core.likelihood.base import Likelihood
from ...core.utils import logger, check_directory_exists_and_if_not_mkdir
from ...core.prior.base import Constraint
from ...core.likelihood import simulation_based_inference as sbibilby
from ...core.likelihood.simulation_based_inference import GenerateData


class GenerateWhitenedIFONoise(GenerateData):
    """
    TBD
    Parameters
    ==========
    num_data:
    """

    def __init__(self, ifo):
        super(GenerateWhitenedIFONoise, self).__init__(
            parameters=dict(sigma=None),
            call_parameter_key_list=["sigma"],
        )
        self.ifo = ifo
      
    def get_data(self, parameters: dict):
        self.parameters.update(parameters)
        sigma = self.parameters["sigma"]
        self.ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=self.ifo.sampling_frequency,
            duration=self.ifo.duration,
            start_time=self.ifo.start_time,
        )
        whitened_strain = self.ifo.whitened_time_domain_strain * np.array(sigma)
        
        return whitened_strain


class GenerateWhitenedSignal(GenerateData):
    """
    TBD

    Parameters
    ==========
    ifo:

    waveform_generator:

    bilby_prior:
    """

    def __init__(self, ifo, waveform_generator, signal_prior):
        call_parameter_key_list = signal_prior.non_fixed_keys
        parameters = signal_prior.sample()

        super(GenerateWhitenedSignal, self).__init__(
            parameters=parameters,
            call_parameter_key_list=call_parameter_key_list,
        )
        self.ifo = ifo
        self.waveform_generator = waveform_generator
        
    def get_data(self, parameters: dict):
        self.parameters.update(parameters)
        parameters = self.parameters

        # Simulate the GW signal
        waveform_polarizations = self.waveform_generator.time_domain_strain(parameters)
        signal = {}
        for mode in waveform_polarizations.keys():
            det_response = self.ifo.antenna_response(
                parameters['ra'],
                parameters['dec'],
                parameters['geocent_time'],
                parameters['psi'], mode)
            signal[mode] = waveform_polarizations[mode] * det_response
        ht = sum(signal.values())

        # Correct the time
        time_shift = self.ifo.time_delay_from_geocenter(
            parameters['ra'], parameters['dec'], parameters['geocent_time'])
        dt_geocent = parameters['geocent_time'] - self.ifo.strain_data.start_time
        dt = dt_geocent + time_shift
        nroll = int(dt * self.ifo.sampling_frequency)
        ht = np.roll(ht, nroll)

        # Whiten the time-domain signal
        frequency_window_factor = (
            np.sum(self.ifo.frequency_mask)
            / len(self.ifo.frequency_mask)
        )
        
        hf = np.fft.rfft(ht) / self.ifo.sampling_frequency
        ht_tilde = (
            np.fft.irfft(hf / (self.ifo.amplitude_spectral_density_array * np.sqrt(self.ifo.duration / 4)))
            * np.sqrt(np.sum(self.ifo.frequency_mask)) / frequency_window_factor
        )

        return ht_tilde