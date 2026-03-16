from copy import deepcopy

import numpy as np
from scipy.optimize import differential_evolution

from .base import GravitationalWaveTransient
from ...core.utils import logger
from ...core.prior.base import Constraint
from ...core.prior import DeltaFunction
from ...core.likelihood import _fallback_to_parameters
from ..utils import noise_weighted_inner_product

class EarthRotationGravitationalWaveTransient(GravitationalWaveTransient):
    """
    A gravitational-wave transient likelihood object which incorporates the effect of earth rotation.

    More documentation
    """
    def __init__(
        self, interferometers, waveform_generator, earth_rotation=False, time_marginalization=False,
        distance_marginalization=False, phase_marginalization=False, calibration_marginalization=False,
        priors=None, distance_marginalization_lookup_table=None, calibration_lookup_table=None,
        number_of_response_curves=1000, starting_index=0, jitter_time=True, reference_frame="sky",
        time_reference="geocenter"
    ):

        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            time_marginalization=time_marginalization,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            calibration_marginalization=calibration_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            calibration_lookup_table=calibration_lookup_table,
            number_of_response_curves=number_of_response_curves,
            starting_index=starting_index,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference,
        )
        self.earth_rotation = earth_rotation

    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters=None):
        """
        Project the waveform polarizations against the interferometer
        response, optionally applying an Earth rotation correction.

        Parameters
        ==========
        signal_polarizations: dict
            Dictionary containing the waveform evaluated at
            :code:`interferometer.frequency_array`.
        interferometer: bilby.gw.detector.Interferometer
            Interferometer to compute the response with respect to.
        """
        parameters = _fallback_to_parameters(self, parameters)
        if not self.earth_rotation:
            signal = interferometer.get_detector_response(signal_polarizations, parameters)

        if self.earth_rotation:
            # apply Earth rotation correction to signal here
            signal = interferometer.get_detector_response(signal_polarizations, parameters, earth_rotation=self.earth_rotation)
        return signal

        



                 