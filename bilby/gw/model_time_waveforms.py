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
_COPRECESSING_HARMONICS_FUNCTION = "Sim{approximant}CoprecModesHarmonicsModelTimes"


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
    return _modes_dict(hlms)


def _modes_dict(hlms, nonnegative=False):
    modes = dict()
    while hlms is not None:
        if not nonnegative or hlms.m >= 0:
            modes[(hlms.l, hlms.m)] = hlms.mode.data.data.copy()
        hlms = hlms.next
    return modes


def heterodyning_terms_at_model_times(
        model_times, approximant, mass_1, mass_2, spin_1x, spin_1y, spin_1z,
        spin_2x, spin_2y, spin_2z, luminosity_distance, inclination, phase,
        delta_time, minimum_frequency, reference_frequency, waveform_dictionary,
        truncate=True):
    """ Modes and harmonic factors for the time-domain heterodyned likelihood.

    The waveform is written as

    .. math::

        h_+ - i h_\\times = \\sum_{\\ell, m \\geq 0} \\left[ Z_{\\ell m} h_{\\ell m}
        + \\bar{Z}_{\\ell m} h_{\\ell m}^* \\right].

    - IMRPhenomT/THM: :math:`h_{\\ell m}` are the modes with m > 0,
      :math:`Z_{\\ell m} = Y_{\\ell m}` and
      :math:`\\bar{Z}_{\\ell m} = (-1)^\\ell Y_{\\ell, -m}`, with
      :math:`Y_{\\ell m} = {}_{-2}Y_{\\ell m}(\\iota, \\pi/2 - \\phi)`.
    - IMRPhenomTP/TPHM: :math:`h_{\\ell m}` are the co-precessing modes with
      m >= 0, :math:`Z_{\\ell m}(t) = \\sum_{m'} Y_{\\ell m'} R^\\ell_{m m'}(t)`,
      with R the rotation from the co-precessing frame to the L0 frame, and
      :math:`\\bar{Z}_{\\ell m} = (-1)^\\ell Z_{\\ell, -m}` (zero for m = 0). The modes and
      Z are computed by lalsimulation (Sim...CoprecModesHarmonicsModelTimes).

    The modes of the precessing models vary slowly in time, so their ratios to
    the modes of a nearby (fiducial) waveform are smooth; the factors Z carry
    the precession and are evaluated exactly.

    Parameters are those of :code:`polarizations_at_model_times`.

    Returns
    =======
    modes: dict
        {(l, m): complex array}; for IMRPhenomTP/TPHM, modes that are zero
        at every requested time are left out.
    harmonics: dict
        {(l, m): (Z, Zbar)}, complex scalars or arrays.
    """
    spins = _spin_arguments(approximant, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)
    if approximant not in PRECESSING_APPROXIMANTS:
        modes = modes_at_model_times(
            model_times, approximant, mass_1, mass_2, spin_1x, spin_1y, spin_1z,
            spin_2x, spin_2y, spin_2z, luminosity_distance, phase, delta_time,
            minimum_frequency, reference_frequency, waveform_dictionary, truncate=truncate)
        modes = {(ell, m): mode for (ell, m), mode in modes.items() if m > 0}
        harmonics = {
            (ell, m): (_spherical_harmonic(ell, m, inclination, phase),
                       (-1) ** ell * _spherical_harmonic(ell, -m, inclination, phase))
            for ell, m in modes}
        return modes, harmonics

    function = _lalsimulation_function(_COPRECESSING_HARMONICS_FUNCTION, approximant, truncate)
    hlms, factors = function(
        mass_1, mass_2, *spins, luminosity_distance, inclination, delta_time, minimum_frequency,
        reference_frequency, phase, _time_sequence(model_times), waveform_dictionary)
    modes = {key: mode for key, mode in _modes_dict(hlms, nonnegative=True).items() if np.any(mode)}
    factors = _modes_dict(factors)
    harmonics = {
        (ell, m): (factors[(ell, m)], (-1) ** ell * factors[(ell, -m)] if m > 0 else 0)
        for ell, m in modes}
    return modes, harmonics


def _spherical_harmonic(ell, m, inclination, phase):
    import lal
    return lal.SpinWeightedSphericalHarmonic(inclination, np.pi / 2 - phase, -2, ell, m)
