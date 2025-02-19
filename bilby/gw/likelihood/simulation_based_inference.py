import inspect
import pickle
import os

import numpy as np
import sbi
import sbi.utils
import sbi.inference
import torch

from scipy.signal.windows import tukey
from ...core.likelihood.base import Likelihood
from ...core.utils import logger, check_directory_exists_and_if_not_mkdir
from ...core.prior.base import Constraint
from ...core.likelihood import simulation_based_inference as sbibilby
from ...core.likelihood.simulation_based_inference import GenerateData
from  ..detector.networks import InterferometerList

class GenerateWhitenedIFONoise_fromGWF(GenerateData):
    """
    TBD
    Parameters
    ==========
    num_data:
    """

    def __init__(self, ifo, use_mask, times):
        super(GenerateWhitenedIFONoise_fromGWF, self).__init__(
            parameters=dict(sigma=None),
            call_parameter_key_list=["sigma"],
        )
        self.old_ifo=ifo
        self.use_mask=use_mask
        self.time_lower=times[0]
        self.time_upper=times[1]
    def get_data(self, parameters: dict):
        self.parameters.update(parameters)
        sigma = self.parameters["sigma"]
        self.ifon = InterferometerList(["H1"])
        self.ifon.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.old_ifo.sampling_frequency,
            duration=self.old_ifo.duration,
            start_time=self.old_ifo.start_time,
        )
        self.ifo=self.ifon[0]
        noise=self.ifo.strain_data.time_domain_strain
        
        roll_off=0.2
        alpha=2*roll_off/self.ifo.duration
        window = tukey(len(noise), alpha=alpha)
        window_factor = np.mean(window ** 2)
        frequency_window_factor = (np.sum(self.ifo.frequency_mask)/ len(self.ifo.frequency_mask))
        
        hf = np.fft.rfft(noise*window) / self.ifo.sampling_frequency*self.ifo.frequency_mask
        hf_whitened=hf/ (self.ifo.amplitude_spectral_density_array*np.sqrt(window_factor) * np.sqrt(self.ifo.duration / 4))
        ht_whitened=(np.fft.irfft(hf_whitened)*np.sqrt(np.sum(self.ifo.frequency_mask))/frequency_window_factor)
        whitened_strain = ht_whitened * np.array(sigma)
        #Taking only the piece necessary for the training
        if self.use_mask:
            window_start = self.ifo.start_time +(self.ifo.duration/2.)- self.time_lower
            window_end = self.ifo.start_time + (self.ifo.duration/2.) + self.time_upper
            mask = (self.ifo.time_array >= window_start) & (self.ifo.time_array <= window_end)
            whitened_strain=whitened_strain[mask]
        return whitened_strain


class GenerateWhitenedSignal_fromGWF(GenerateData):
    """
    TBD

    Parameters
    ==========
    ifo:

    waveform_generator:

    bilby_prior:
    """

    def __init__(self, ifo, waveform_generator, signal_prior, use_mask, times):
        call_parameter_key_list = signal_prior.non_fixed_keys
        parameters = signal_prior.sample()

        super(GenerateWhitenedSignal_fromGWF, self).__init__(
            parameters=parameters,
            call_parameter_key_list=call_parameter_key_list,
        )
        self.ifo = ifo
        self.waveform_generator = waveform_generator
        self.use_mask=use_mask
        self.time_lower=times[0]
        self.time_upper=times[1]
    def get_data(self, parameters: dict):
        self.parameters.update(parameters)
        parameters = self.parameters

        waveform_polarizations = self.waveform_generator.frequency_domain_strain(parameters)
        frequencies=None
        if frequencies is None:
            frequencies = self.ifo.frequency_array[self.ifo.frequency_mask]
            mask = self.ifo.frequency_mask
        
        time_data=self.ifo.strain_data.time_domain_strain
        roll_off=0.2
        alpha=2*roll_off/self.ifo.duration
        window = tukey(len(time_data), alpha=alpha)
        window_factor = np.mean(window ** 2)
        
        frequency_data = np.fft.rfft(time_data*window) / self.ifo.sampling_frequency*self.ifo.frequency_mask
        frequency_window_factor = (np.sum(self.ifo.frequency_mask)/ len(self.ifo.frequency_mask))
        frequency_data_whitened=frequency_data/ (self.ifo.amplitude_spectral_density_array*np.sqrt(window_factor) * np.sqrt(self.ifo.duration / 4))
        data_noise_whitened=(np.fft.irfft(frequency_data_whitened)*np.sqrt(np.sum(self.ifo.frequency_mask))/frequency_window_factor)
        signal = {}
        for mode in waveform_polarizations.keys():
            det_response = self.ifo.antenna_response(
                parameters['ra'],
                parameters['dec'],
                parameters['geocent_time'],
                parameters['psi'], mode)
        
            signal[mode] = waveform_polarizations[mode] * det_response
        signal_ifo = sum(signal.values()) * mask#*window_factor
        
        time_shift = self.ifo.time_delay_from_geocenter(
            parameters['ra'], parameters['dec'], parameters['geocent_time'])
        
        # Be careful to first subtract the two GPS times which are ~1e9 sec.
        # And then add the time_shift which varies at ~1e-5 sec
        dt_geocent = parameters['geocent_time'] - self.ifo.strain_data.start_time
        dt = dt_geocent + time_shift
        
        signal_ifo[mask] = signal_ifo[mask] * np.exp(-1j * 2 * np.pi * dt * frequencies)
        
        signal_ifo[mask] *= self.ifo.calibration_model.get_calibration_factor(
            frequencies, prefix='recalib_{}_'.format(self.ifo.name), **parameters
        )
        frequency_domain_signal= frequency_data+ signal_ifo
        
        ht_tilde = (
            np.fft.irfft(frequency_domain_signal / (self.ifo.amplitude_spectral_density_array*np.sqrt(window_factor) * np.sqrt(self.ifo.duration / 4)))
            * np.sqrt(np.sum(self.ifo.frequency_mask)) / frequency_window_factor
        )
        ht_whitened=ht_tilde-data_noise_whitened
        #Taking only the piece necessary for the training
        if self.use_mask:
            window_start = self.ifo.start_time +(self.ifo.duration/2.)- self.time_lower
            window_end = self.ifo.start_time + (self.ifo.duration/2.) + self.time_upper
            mask = (self.ifo.time_array >= window_start) & (self.ifo.time_array <= window_end)
            ht_whitened=ht_whitened[mask]

        return ht_whitened






class GenerateWhitenedSignal_fromPSD(GenerateData):
    """
    TBD

    Parameters
    ==========
    ifo:

    waveform_generator:

    bilby_prior:
    """

    def __init__(self, ifo, waveform_generator, signal_prior, use_mask, times):
        call_parameter_key_list = signal_prior.non_fixed_keys
        parameters = signal_prior.sample()

        super(GenerateWhitenedSignal_fromPSD, self).__init__(
            parameters=parameters,
            call_parameter_key_list=call_parameter_key_list,
        )
        self.ifo = ifo
        self.waveform_generator = waveform_generator
        self.use_mask=use_mask
        self.time_lower=times[0]
        self.time_upper=times[1]
    def get_data(self, parameters: dict):
        self.parameters.update(parameters)
        parameters = self.parameters

        frequency_signal=self.ifo.strain_data.frequency_domain_strain
        time_whitened=self.ifo.whitened_time_domain_strain
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(parameters)
        frequencies=None
        if frequencies is None:
            frequencies = self.ifo.frequency_array[self.ifo.frequency_mask]
            mask = self.ifo.frequency_mask
        else:
            mask = np.ones(len(frequencies), dtype=bool)
        
        signal = {}
        for mode in waveform_polarizations.keys():
            det_response =self.ifo.antenna_response(
                parameters['ra'],
                parameters['dec'],
                parameters['geocent_time'],
                parameters['psi'], mode)
        
            signal[mode] = waveform_polarizations[mode] * det_response
        signal_ifo = sum(signal.values()) * mask
        
        time_shift = self.ifo.time_delay_from_geocenter(
            parameters['ra'], parameters['dec'], parameters['geocent_time'])
        
        # Be careful to first subtract the two GPS times which are ~1e9 sec.
        # And then add the time_shift which varies at ~1e-5 sec
        dt_geocent = parameters['geocent_time'] - self.ifo.strain_data.start_time
        dt = dt_geocent + time_shift
        
        signal_ifo[mask] = signal_ifo[mask] * np.exp(-1j * 2 * np.pi * dt * frequencies)
        
        signal_ifo[mask] *= self.ifo.calibration_model.get_calibration_factor(
            frequencies, prefix='recalib_{}_'.format(self.ifo.name), **parameters
        )
        frequency_domain_signal= frequency_signal+ signal_ifo
        frequency_window_factor = (
                np.sum(self.ifo.frequency_mask)
                / len(self.ifo.frequency_mask)
            )   
        
        ht_tilde = (
            np.fft.irfft(frequency_domain_signal / (self.ifo.amplitude_spectral_density_array * np.sqrt(self.ifo.duration / 4)))
            * np.sqrt(np.sum(self.ifo.frequency_mask)) / frequency_window_factor
        )
        ht_whitened=ht_tilde-time_whitened
        #Taking only the piece necessary for the training
        if self.use_mask:
            window_start = self.ifo.start_time +(self.ifo.duration/2.)- self.time_lower
            window_end = self.ifo.start_time + (self.ifo.duration/2.) + self.time_upper
            mask = (self.ifo.time_array >= window_start) & (self.ifo.time_array <= window_end)
            ht_whitened=ht_whitened[mask]

        return ht_whitened


class GenerateWhitenedIFONoise_fromPSD(GenerateData):
    """
    TBD
    Parameters
    ==========
    num_data:
    """

    def __init__(self, ifo, use_mask, times):
        super(GenerateWhitenedIFONoise_fromPSD, self).__init__(
            parameters=dict(sigma=None),
            call_parameter_key_list=["sigma"],
        )
        self.ifo = ifo
        self.use_mask=use_mask
        self.time_lower=times[0]
        self.time_upper=times[1]
    def get_data(self, parameters: dict):
        self.parameters.update(parameters)
        sigma = self.parameters["sigma"]
        self.ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=self.ifo.sampling_frequency,
            duration=self.ifo.duration,
            start_time=self.ifo.start_time,
        )
        whitened_strain = self.ifo.whitened_time_domain_strain * np.array(sigma)
        #Taking only the piece necessary for the training
        if self.use_mask:
            window_start = self.ifo.start_time +(self.ifo.duration/2.)- self.time_lower
            window_end = self.ifo.start_time + (self.ifo.duration/2.) + self.time_upper
            mask = (self.ifo.time_array >= window_start) & (self.ifo.time_array <= window_end)
            whitened_strain=whitened_strain[mask]
        return whitened_strain