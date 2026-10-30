"""
IMRPhenomT-family waveforms evaluated at arbitrary times.

Wrappers of the lalsimulation functions that evaluate IMRPhenomT
(Estellés et al. arXiv:2004.08302), IMRPhenomTHM (arXiv:2012.11923),
IMRPhenomTP and IMRPhenomTPHM (arXiv:2105.05872) at given model times
t / M, where M is the total mass in seconds and t = 0 at the peak of the (2, 2)
amplitude. These functions require a lalsuite build that provides them.
"""
import numpy as np

APPROXIMANTS = ("IMRPhenomT", "IMRPhenomTHM", "IMRPhenomTP", "IMRPhenomTPHM")
PRECESSING_APPROXIMANTS = ("IMRPhenomTP", "IMRPhenomTPHM")

# lalsimulation function names, e.g. SimIMRPhenomTPHMModelTimes(Truncated)
_POLARIZATION_FUNCTION = "Sim{approximant}ModelTimes"
_MODE_FUNCTION = "Sim{approximant}ModesModelTimes"
_COPRECESSING_HARMONICS_FUNCTION = "Sim{approximant}CoprecModesHarmonicsModelTimes"


def _get_approximant(template, approximant, truncate):
    import lalsimulation as lalsim
    name = template.format(approximant=approximant) + ("Truncated" if truncate else "")
    try:
        return getattr(lalsim, name)
    except AttributeError:
        raise ImportError(
            f"lalsimulation has no {name}: model-time waveforms need a lalsuite build "
            "with the time-array IMRPhenomT-family functions") from None


def _time_sequence(model_times):
    import lal
    model_times = np.asarray(model_times, dtype=float)
    sequence = lal.CreateREAL8Sequence(len(model_times))
    sequence.data = model_times
    return sequence


def spin_arguments(approximant, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z):
    """ The spin arguments of the lalsimulation model-time functions: all six
    components for IMRPhenomTP and IMRPhenomTPHM, the z components for
    IMRPhenomT and IMRPhenomTHM

    Raises a ValueError for other approximants, or for in-plane spins with a
    non-precessing approximant.
    """
    if approximant not in APPROXIMANTS:
        raise ValueError(
            f"model-time waveforms are available for {', '.join(APPROXIMANTS)}, not {approximant!r}")
    if approximant in PRECESSING_APPROXIMANTS:
        return (spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)
    if spin_1x != 0 or spin_1y != 0 or spin_2x != 0 or spin_2y != 0:
        raise ValueError(
            f"Non-zero transverse spins were given, but {approximant} is a non-precessing approximant.")
    return (spin_1z, spin_2z)


def polarizations_at_model_times(
        model_times, approximant, mass_1, mass_2, spin_1x, spin_1y, spin_1z,
        spin_2x, spin_2y, spin_2z, luminosity_distance, inclination, phase,
        delta_time, minimum_frequency, reference_frequency, waveform_dictionary,
        truncate=True):
    """ The plus and cross polarizations at given model times

    Parameters
    ==========
    model_times: array_like
        The times t / M, with t = 0 at the peak of the (2, 2) amplitude
    approximant: str
        IMRPhenomT, IMRPhenomTHM, IMRPhenomTP or IMRPhenomTPHM
    mass_1, mass_2: float
        The component masses (in kg)
    spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z: float
        The dimensionless spin components in the lalsimulation frame
    luminosity_distance: float
        The luminosity distance (in m)
    inclination: float
        The inclination (in rad)
    phase: float
        The reference phase (in rad)
    delta_time: float
        The sampling interval (in s) of the waveform on which the precession angles are computed
    minimum_frequency: float
        The minimum frequency (in Hz)
    reference_frequency: float
        The reference frequency (in Hz)
    waveform_dictionary: lal.Dict
        The lal waveform dictionary
    truncate: bool
        If True, the waveform is zero at times outside its span, from the
        minimum frequency to the end of the ringdown

    Returns
    =======
    dict: The plus and cross polarizations
    """
    spins = spin_arguments(approximant, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)
    function = _get_approximant(_POLARIZATION_FUNCTION, approximant, truncate)
    hplus, hcross = function(
        mass_1, mass_2, *spins, luminosity_distance, inclination, delta_time,
        minimum_frequency, reference_frequency, phase, _time_sequence(model_times),
        waveform_dictionary)
    return dict(plus=hplus.data.data, cross=hcross.data.data)


def modes_at_model_times(
        model_times, approximant, mass_1, mass_2, spin_1x, spin_1y, spin_1z,
        spin_2x, spin_2y, spin_2z, luminosity_distance, phase, delta_time,
        minimum_frequency, reference_frequency, waveform_dictionary, truncate=True):
    """ The spherical-harmonic modes at given model times; for IMRPhenomTP and
    IMRPhenomTPHM, in the frame of the orbital angular momentum at the reference
    frequency

    Parameters are as in :code:`polarizations_at_model_times`, without the inclination.

    Returns
    =======
    dict: The modes, with keys (l, m)
    """
    spins = spin_arguments(approximant, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)
    function = _get_approximant(_MODE_FUNCTION, approximant, truncate)
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
    """ The modes and harmonic factors used by
    :code:`bilby.gw.likelihood.TimeDomainHeterodyningGravitationalWaveTransient`
    at given model times: the modes with m > 0 for IMRPhenomT and IMRPhenomTHM,
    the co-precessing modes with m >= 0 for IMRPhenomTP and IMRPhenomTPHM. See
    Sharma et al. arXiv:2601.11239.

    Parameters are as in :code:`polarizations_at_model_times`.

    Returns
    =======
    modes: dict
        The modes, with keys (l, m)
    harmonics: dict
        The harmonic factors of each mode and of its complex conjugate, with keys (l, m)
    """
    spins = spin_arguments(approximant, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)
    if approximant not in PRECESSING_APPROXIMANTS:
        import lal
        modes = modes_at_model_times(
            model_times, approximant, mass_1, mass_2, spin_1x, spin_1y, spin_1z,
            spin_2x, spin_2y, spin_2z, luminosity_distance, phase, delta_time,
            minimum_frequency, reference_frequency, waveform_dictionary, truncate=truncate)
        modes = {(ell, m): mode for (ell, m), mode in modes.items() if m > 0}
        # Y_lm = -2Y_lm(inclination, pi / 2 - phase)
        harmonics = {
            (ell, m): (lal.SpinWeightedSphericalHarmonic(inclination, np.pi / 2 - phase, -2, ell, m),
                       (-1) ** ell * lal.SpinWeightedSphericalHarmonic(inclination, np.pi / 2 - phase, -2, ell, -m))
            for ell, m in modes}
        return modes, harmonics

    function = _get_approximant(_COPRECESSING_HARMONICS_FUNCTION, approximant, truncate)
    hlms, factors = function(
        mass_1, mass_2, *spins, luminosity_distance, inclination, delta_time, minimum_frequency,
        reference_frequency, phase, _time_sequence(model_times), waveform_dictionary)
    modes = {key: mode for key, mode in _modes_dict(hlms, nonnegative=True).items() if np.any(mode)}
    factors = _modes_dict(factors)
    harmonics = {
        (ell, m): (factors[(ell, m)], (-1) ** ell * factors[(ell, -m)] if m > 0 else 0)
        for ell, m in modes}
    return modes, harmonics
