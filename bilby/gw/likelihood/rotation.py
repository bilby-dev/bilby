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


    def __init__(self, interferometers,
                 waveform_generator,
                 fiducial_parameters=None,
                 parameter_bounds=None,
                 maximization_kwargs=None,
                 update_fiducial_parameters=False,
                 distance_marginalization=False,
                 time_marginalization=False,
                 phase_marginalization=False,
                 priors=None,
                 distance_marginalization_lookup_table=None,
                 jitter_time=True,
                 reference_frame="sky",
                 time_reference="geocenter"):


        super(EarthRotationGravitationalWaveTransient, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            time_marginalization=time_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference)
            

                 