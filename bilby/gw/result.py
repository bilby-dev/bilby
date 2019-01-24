from __future__ import division

from ..core.result import Result as CoreResult
from ..core.utils import logger


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


CBCResult = CompactBinaryCoalesenceResult
