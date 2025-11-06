from ..source import lal_binary_black_hole
from ..waveform_generator import WaveformGenerator
from .base import GravitationalWaveTransient
from .basic import BasicGravitationalWaveTransient
from .multiband import MBGravitationalWaveTransient
from .relative import RelativeBinningGravitationalWaveTransient
from .roq import BilbyROQParamsRangeError, ROQGravitationalWaveTransient

__all__ = [
    BasicGravitationalWaveTransient,
    GravitationalWaveTransient,
    MBGravitationalWaveTransient,
    RelativeBinningGravitationalWaveTransient,
    BilbyROQParamsRangeError,
    ROQGravitationalWaveTransient,
    "get_binary_black_hole_likelihood",
]


def get_binary_black_hole_likelihood(interferometers):
    """A wrapper to quickly set up a likelihood for BBH parameter estimation

    Parameters
    ==========
    interferometers: {bilby.gw.detector.InterferometerList, list}
        A list of `bilby.detector.Interferometer` instances, typically the
        output of either `bilby.detector.get_interferometer_with_open_data`
        or `bilby.detector.get_interferometer_with_fake_noise_and_injection`

    Returns
    =======
    bilby.GravitationalWaveTransient: The likelihood to pass to `run_sampler`

    """
    waveform_generator = WaveformGenerator(
        duration=interferometers.duration,
        sampling_frequency=interferometers.sampling_frequency,
        frequency_domain_source_model=lal_binary_black_hole,
        waveform_arguments={"waveform_approximant": "IMRPhenomPv2", "reference_frequency": 50},
    )
    return GravitationalWaveTransient(interferometers, waveform_generator)
