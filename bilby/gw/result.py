from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import os

from ..core.result import Result as CoreResult
from ..core.utils import logger
from .utils import (plot_spline_pos,
                    spline_angle_xform)


class CompactBinaryCoalesenceResult(CoreResult):
    def __init__(self, **kwargs):
        super(CompactBinaryCoalesenceResult, self).__init__(**kwargs)

    def __get_from_nested_meta_data(self, *keys):
        dictionary = self.meta_data
        try:
            for k in keys:
                item = dictionary[k]
                dictionary = item
            return item
        except KeyError:
            raise ValueError(
                "No information stored for {}".format('/'.join(keys)))

    @property
    def time_marginalization(self):
        """ Boolean for if the likelihood used time marginalization """
        return self.__get_from_nested_meta_data(
            'likelihood', 'time_marginalization')

    @property
    def phase_marginalization(self):
        """ Boolean for if the likelihood used phase marginalization """
        return self.__get_from_nested_meta_data(
            'likelihood', 'phase_marginalization')

    @property
    def distance_marginalization(self):
        """ Boolean for if the likelihood used distance marginalization """
        return self.__get_from_nested_meta_data(
            'likelihood', 'distance_marginalization')

    @property
    def waveform_approximant(self):
        """ String of the waveform approximant """
        return self.__get_from_nested_meta_data(
            'likelihood', 'waveform_arguments', 'waveform_approximant')

    @property
    def reference_frequency(self):
        """ Float of the reference frequency """
        return self.__get_from_nested_meta_data(
            'likelihood', 'waveform_arguments', 'reference_frequency')

    @property
    def frequency_domain_source_model(self):
        """ The frequency domain source model (function)"""
        return self.__get_from_nested_meta_data(
            'likelihood', 'frequency_domain_source_model')

    def detector_injection_properties(self, detector):
        """ Returns a dictionary of the injection properties for each detector

        The injection properties include the parameters injected, and
        information about the signal to noise ratio (SNR) given the noise
        properties.

        Parameters
        ----------
        detector: str [H1, L1, V1]
            Detector name

        Returns
        -------
        injection_properties: dict
            A dictionary of the injection properties

        """
        try:
            return self.__get_from_nested_meta_data(
                'likelihood', 'interferometers', detector)
        except ValueError:
            logger.info("No injection for detector {}".format(detector))
            return None

    def plot_calibration_posterior(self, level=.9):
        """ Plots the calibration amplitude and phase uncertainty.
        Adapted from the LALInference version in bayespputils

        Parameters
        ----------
        level: float,  percentage for confidence levels

        Returns
        -------
        saves a plot to outdir+label+calibration.png

        """
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(15, 15), dpi=500)
        posterior = self.posterior

        font_size = 32
        outdir = self.outdir

        parameters = posterior.keys()
        ifos = np.unique([param.split('_')[1] for param in parameters if 'recalib_' in param])
        if ifos.size == 0:
            logger.info("No calibration parameters found. Aborting calibration plot.")
            return

        for ifo in ifos:
            if ifo == 'H1':
                color = 'r'
            elif ifo == 'L1':
                color = 'g'
            elif ifo == 'V1':
                color = 'm'
            else:
                color = 'c'

            # Assume spline control frequencies are constant
            freq_params = np.sort([param for param in parameters if
                                   'recalib_{0}_frequency_'.format(ifo) in param])

            logfreqs = np.log([posterior[param].iloc[0] for param in freq_params])

            # Amplitude calibration model
            plt.sca(ax1)
            amp_params = np.sort([param for param in parameters if
                                  'recalib_{0}_amplitude_'.format(ifo) in param])
            if len(amp_params) > 0:
                amplitude = 100 * np.column_stack([posterior[param] for param in amp_params])
                plot_spline_pos(logfreqs, amplitude, color=color, level=level,
                                label="{0} (mean, {1}%)".format(ifo.upper(), int(level * 100)))

            # Phase calibration model
            plt.sca(ax2)
            phase_params = np.sort([param for param in parameters if
                                    'recalib_{0}_phase_'.format(ifo) in param])
            if len(phase_params) > 0:
                phase = np.column_stack([posterior[param] for param in phase_params])
                plot_spline_pos(logfreqs, phase, color=color, level=level,
                                label="{0} (mean, {1}%)".format(ifo.upper(), int(level * 100)),
                                xform=spline_angle_xform)

        ax1.tick_params(labelsize=.75 * font_size)
        ax2.tick_params(labelsize=.75 * font_size)
        plt.legend(loc='upper right', prop={'size': .75 * font_size}, framealpha=0.1)
        ax1.set_xscale('log')
        ax2.set_xscale('log')

        ax2.set_xlabel('Frequency (Hz)', fontsize=font_size)
        ax1.set_ylabel('Amplitude (%)', fontsize=font_size)
        ax2.set_ylabel('Phase (deg)', fontsize=font_size)

        filename = os.path.join(outdir, self.label + '_calibration.png')
        fig.tight_layout()
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)


CBCResult = CompactBinaryCoalesenceResult
