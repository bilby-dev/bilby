""" IMRPhenomT-family waveforms at arbitrary times.

These call the lalsimulation functions that evaluate IMRPhenomT, IMRPhenomTHM,
IMRPhenomTP and IMRPhenomTPHM at a user-supplied array of model times, instead
of on LAL's own time grid. Model time is t / M, with M = m1 + m2 (detector
frame) in seconds, and t = 0 at the peak of the (2,2) amplitude of the
non-precessing waveform.

- truncate=False: the waveform at every requested time; every time must lie in
  the span of the standard waveform (from f_min to the end of the ringdown).
- truncate=True: zero at times outside that span.

The functions need a lalsuite build that has these functions; an ImportError is
raised only when one of them is used.
"""
import numpy as np

APPROXIMANTS = ("IMRPhenomT", "IMRPhenomTHM", "IMRPhenomTP", "IMRPhenomTPHM")
PRECESSING_APPROXIMANTS = ("IMRPhenomTP", "IMRPhenomTPHM")

# lalsimulation function names, e.g. SimIMRPhenomTPHMModelTimes(Truncated)
_POLARIZATION_FUNCTION = "Sim{approximant}ModelTimes"
_MODE_FUNCTION = "Sim{approximant}ModesModelTimes"


def _lalsimulation_function(template, approximant, truncate):
    import lalsimulation as lalsim
    name = template.format(approximant=approximant) + ("Truncated" if truncate else "")
    try:
        return getattr(lalsim, name)
    except AttributeError:
        raise ImportError(
            f"lalsimulation has no {name}: model-time waveforms need a lalsuite build "
            "with the time-array IMRPhenomT-family functions") from None


def total_mass_in_seconds(mass_1, mass_2):
    """ Total mass in seconds, as lalsimulation computes it.

    Parameters
    ==========
    mass_1, mass_2: float
        Component masses (kg).
    """
    import lal
    return (mass_1 / lal.MSUN_SI + mass_2 / lal.MSUN_SI) * lal.MTSUN_SI


def _time_sequence(model_times):
    import lal
    model_times = np.asarray(model_times, dtype=float)
    sequence = lal.CreateREAL8Sequence(len(model_times))
    sequence.data = model_times
    return sequence


def check_arguments(approximant, spin_1x, spin_1y, spin_2x, spin_2y):
    """ Raise a ValueError for an approximant without model-time functions, or
    for in-plane spins with a non-precessing approximant (as lalsimulation's
    ChooseTDWaveform and ChooseFDWaveform do). """
    if approximant not in APPROXIMANTS:
        raise ValueError(
            f"model-time waveforms are available for {', '.join(APPROXIMANTS)}, not {approximant!r}")
    if approximant not in PRECESSING_APPROXIMANTS and (spin_1x != 0 or spin_1y != 0 or spin_2x != 0 or spin_2y != 0):
        raise ValueError(
            f"Non-zero transverse spins were given, but {approximant} is a non-precessing approximant.")


def _spin_arguments(approximant, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z):
    check_arguments(approximant, spin_1x, spin_1y, spin_2x, spin_2y)
    if approximant in PRECESSING_APPROXIMANTS:
        return (spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)
    return (spin_1z, spin_2z)


def polarizations_at_model_times(
        model_times, approximant, mass_1, mass_2, spin_1x, spin_1y, spin_1z,
        spin_2x, spin_2y, spin_2z, luminosity_distance, inclination, phase,
        delta_time, minimum_frequency, reference_frequency, waveform_dictionary,
        truncate=True):
    """ Plus and cross polarizations at the given model times.

    Parameters
    ==========
    model_times: array_like
        Times in units of the total mass, t = 0 at the peak (any order, any spacing).
    approximant: str
        IMRPhenomT, IMRPhenomTHM, IMRPhenomTP or IMRPhenomTPHM.
    mass_1, mass_2: float
        Component masses (kg).
    spin_1x, ..., spin_2z: float
        Dimensionless spin components at the reference frequency (lalsimulation frame).
        The in-plane components must be zero for IMRPhenomT and IMRPhenomTHM.
    luminosity_distance: float
        Luminosity distance (m).
    inclination, phase: float
        Inclination and reference phase (rad), as for ChooseTDWaveform.
    delta_time: float
        Sampling interval (s) of the standard waveform; the precessing models take
        their angles on the time grid of the standard waveform.
    minimum_frequency, reference_frequency: float
        f_min and f_ref (Hz).
    waveform_dictionary: lal.Dict
    truncate: bool
        If True (default), zero outside the span of the standard waveform.

    Returns
    =======
    plus, cross: array_like
    """
    spins = _spin_arguments(approximant, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)
    function = _lalsimulation_function(_POLARIZATION_FUNCTION, approximant, truncate)
    hplus, hcross = function(
        mass_1, mass_2, *spins, luminosity_distance, inclination, delta_time,
        minimum_frequency, reference_frequency, phase, _time_sequence(model_times),
        waveform_dictionary)
    return hplus.data.data, hcross.data.data


def modes_at_model_times(
        model_times, approximant, mass_1, mass_2, spin_1x, spin_1y, spin_1z,
        spin_2x, spin_2y, spin_2z, luminosity_distance, phase, delta_time,
        minimum_frequency, reference_frequency, waveform_dictionary, truncate=True):
    """ Spherical-harmonic modes h_lm at the given model times.

    For IMRPhenomTP/TPHM these are the modes in the inertial frame of the
    orbital angular momentum at the reference frequency (L0 frame), i.e.,
    already rotated with the precession angles; they contain every m of each l.
    For IMRPhenomT only (2, 2) and (2, -2) are returned.

    Parameters are those of :code:`polarizations_at_model_times`, without the
    inclination.

    Returns
    =======
    dict: {(l, m): complex array}
    """
    spins = _spin_arguments(approximant, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)
    function = _lalsimulation_function(_MODE_FUNCTION, approximant, truncate)
    hlms = function(
        mass_1, mass_2, *spins, luminosity_distance, delta_time, minimum_frequency,
        reference_frequency, phase, _time_sequence(model_times), waveform_dictionary)
    modes = dict()
    while hlms is not None:
        modes[(hlms.l, hlms.m)] = hlms.mode.data.data.copy()
        hlms = hlms.next
    return modes
