import os

import numpy as np
import math

from ...core import utils
from ...core.utils import logger, safe_file_dump
from ..geometry import zenith_azimuth_to_theta_phi
from .interferometer import Interferometer
from .psd import PowerSpectralDensity


class InterferometerList(list):
    """A list of Interferometer objects"""

    def __init__(self, interferometers):
        """Instantiate a InterferometerList

        The InterferometerList is a list of Interferometer objects, each
        object has the data used in evaluating the likelihood

        Parameters
        ==========
        interferometers: iterable
            The list of interferometers
        """

        super(InterferometerList, self).__init__()
        if isinstance(interferometers, str):
            raise TypeError("Input must not be a string")
        for ifo in interferometers:
            if isinstance(ifo, str):
                ifo = get_empty_interferometer(ifo)
            if not isinstance(ifo, (Interferometer, TriangularInterferometer)):
                raise TypeError(
                    "Input list of interferometers are not all Interferometer objects"
                )
            else:
                self.append(ifo)
        self._check_interferometers()

    @property
    def reference_time(self):
        return self._reference_time

    @reference_time.setter
    def reference_time(self, time):
        self._reference_time = time
        for ifo in self:
            ifo.reference_time = time

    def _check_interferometers(self):
        """Verify IFOs 'duration', 'start_time', 'sampling_frequency' are the same.

        If every interferometer has a time-domain analysis window, 'start_time'
        is not checked.

        If the above attributes are not the same, then the attributes are checked to
        see if they are the same up to 5 decimal places.

        If both checks fail, then a ValueError is raised.
        """
        consistent_attributes = ["duration", "start_time", "sampling_frequency"]
        if len(self) > 0 and all(getattr(ifo, "analysis_window", None) is not None for ifo in self):
            # time-domain analysis windows can start at different times in each detector
            consistent_attributes = ["duration", "sampling_frequency"]
        for attribute in consistent_attributes:
            x = [
                getattr(interferometer.strain_data, attribute)
                for interferometer in self
            ]
            try:
                if not all(y == x[0] for y in x):
                    ifo_strs = [
                        "{ifo}[{attribute}]={value}".format(
                            ifo=ifo.name,
                            attribute=attribute,
                            value=getattr(ifo.strain_data, attribute),
                        )
                        for ifo in self
                    ]
                    raise ValueError(
                        "The {} of all interferometers are not the same: {}".format(
                            attribute, ", ".join(ifo_strs)
                        )
                    )
            except ValueError as e:
                if not all(math.isclose(y, x[0], abs_tol=1e-5) for y in x):
                    raise ValueError(e)
                else:
                    logger.warning(e)

    def set_strain_data_from_power_spectral_densities(
        self, sampling_frequency, duration, start_time=0, *, random_state=None
    ):
        """Set the `Interferometer.strain_data` from the power spectral densities of the detectors

        This uses the `interferometer.power_spectral_density` object to set
        the `strain_data` to a noise realization. See
        `bilby.gw.detector.InterferometerStrainData` for further information.

        Parameters
        ==========
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data

        """
        for interferometer in self:
            interferometer.set_strain_data_from_power_spectral_density(
                sampling_frequency=sampling_frequency,
                duration=duration,
                start_time=start_time,
                random_state=random_state,
            )

    def set_strain_data_from_zero_noise(
        self, sampling_frequency, duration, start_time=0
    ):
        """Set the `Interferometer.strain_data` to zero in each detector

        See :py:meth:`bilby.gw.detector.InterferometerStrainData.set_from_zero_noise`
        for further  information.

        Parameters
        ==========
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data

        """
        for interferometer in self:
            interferometer.set_strain_data_from_zero_noise(
                sampling_frequency=sampling_frequency,
                duration=duration,
                start_time=start_time,
            )

    def inject_signal(
        self,
        parameters=None,
        injection_polarizations=None,
        waveform_generator=None,
        raise_error=True,
    ):
        """ Inject a signal into noise in each of the three detectors.

        Parameters
        ==========
        parameters: dict
            Parameters of the injection.
        injection_polarizations: dict
           Polarizations of waveform to inject, output of
           `waveform_generator.frequency_domain_strain()`. If
           `waveform_generator` is also given, the injection_polarizations will
           be calculated directly and this argument can be ignored.
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject. If
            `injection_polarizations` is given, this will be ignored.
        raise_error: bool
            Whether to raise an error if the injected signal does not fit in
            the segment.

        Notes
        =====
        if your signal takes a substantial amount of time to generate, or
        you experience buggy behaviour. It is preferable to provide the
        injection_polarizations directly.

        Returns
        =======
        injection_polarizations: dict

        """
        if injection_polarizations is None:
            if waveform_generator is not None:
                injection_polarizations = waveform_generator.frequency_domain_strain(
                    parameters
                )
            else:
                raise ValueError(
                    "inject_signal needs one of waveform_generator or "
                    "injection_polarizations."
                )

        all_injection_polarizations = list()
        for interferometer in self:
            all_injection_polarizations.append(
                interferometer.inject_signal(
                    parameters=parameters,
                    injection_polarizations=injection_polarizations,
                    raise_error=raise_error,
                )
            )

        return all_injection_polarizations

    def set_analysis_windows(self, duration, start_time=None, analysis_segment="imr",
                             segment_cut_time=None, cut_reference_parameters=None):
        """ Set the analysis window of each detector.

        For "imr" every detector is given [`start_time`, `start_time` +
        `duration`). For "inspiral" and "post_inspiral" the window of each
        detector ends or starts at its cut time

        .. math::

            t_{\\\\rm cut} = t_c + \\\\Delta t_{\\\\rm ifo}(\\\\alpha, \\\\delta) + \\\\tau_{\\\\rm cut},

        where :math:`t_c`, :math:`\\\\alpha` and :math:`\\\\delta` are the
        geocent_time, ra and dec in `cut_reference_parameters`,
        :math:`\\\\Delta t_{\\\\rm ifo}` is the time delay from the geocenter and
        :math:`\\\\tau_{\\\\rm cut}` is `segment_cut_time`. All detectors have the
        same `duration`.

        Parameters
        ==========
        duration: float
            Duration of the analysis window (s).
        start_time: float, optional
            GPS start time of the window (s). Required for "imr".
        analysis_segment: str
            "imr" (default), "inspiral" (the window ends at the cut) or
            "post_inspiral" (the window starts at the cut).
        segment_cut_time: float, optional
            Cut time relative to the arrival of the merger at each detector
            (s). Required for "inspiral" and "post_inspiral".
        cut_reference_parameters: dict, optional
            Dictionary with 'geocent_time', 'ra' and 'dec' fixing the cut
            time. Required for "inspiral" and "post_inspiral".
        """
        for interferometer in self:
            if analysis_segment == "imr":
                window_start = start_time
                cut_time = None
            elif analysis_segment in ("inspiral", "post_inspiral"):
                reference = cut_reference_parameters
                delay = interferometer.time_delay_from_geocenter(
                    reference["ra"], reference["dec"], reference["geocent_time"])
                cut_time = reference["geocent_time"] + delay + segment_cut_time
                window_start = cut_time - duration if analysis_segment == "inspiral" else cut_time
            else:
                raise ValueError(
                    "analysis_segment must be 'imr', 'inspiral' or 'post_inspiral', "
                    f"not {analysis_segment!r}")
            interferometer.analysis_window = (window_start, duration)
            interferometer.meta_data["analysis_window"].update(
                analysis_segment=analysis_segment, segment_cut_time=segment_cut_time,
                cut_time=cut_time)

    def set_autocovariance_functions_from_power_spectral_densities(self, **kwargs):
        """Set the autocovariance function of each detector from its power
        spectral density.

        See :py:meth:`bilby.gw.detector.Interferometer.set_autocovariance_function_from_power_spectral_density`
        for the keyword arguments.
        """
        for interferometer in self:
            interferometer.set_autocovariance_function_from_power_spectral_density(**kwargs)

    def condition_strain_data(self, sampling_frequency, trim=0.25, taper_and_trim=True,
                              remove_mean=True):
        """Downsample the time-domain strain data of each detector around its
        analysis window.

        See :py:meth:`bilby.gw.detector.Interferometer.condition_strain_data`.

        Parameters
        ==========
        sampling_frequency: float
            The new sampling frequency (Hz).
        trim: float
            Fraction of the data at each end that is removed (default 0.25).
        taper_and_trim: bool
            Use the Tukey window, buffer and trimming (default True); if
            False, use :code:`scipy.signal.decimate`.
        remove_mean: bool
            Subtract the mean of the result (default True).
        """
        for interferometer in self:
            interferometer.condition_strain_data(
                sampling_frequency, trim=trim, taper_and_trim=taper_and_trim,
                remove_mean=remove_mean)

    def inject_signal_time_domain(
        self,
        parameters=None,
        injection_polarizations=None,
        waveform_generator=None,
        placement="nearest",
    ):
        """ Inject a time-domain signal into the time-domain strain data of each
        detector (no Fourier transform, frequency mask or window).

        Parameters
        ==========
        parameters: dict
            Parameters of the injection.
        injection_polarizations: dict
           Output of `waveform_generator.time_domain_strain()` for a time-domain
           source model. If `waveform_generator` is also given, this is used.
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator with a time-domain source model. The waveform is
            generated once and projected onto every detector (for
            placement="model_time", it is evaluated at each detector's samples).
        placement: str
            "nearest" (default), "fd_shift" or "model_time"; see
            :code:`bilby.gw.detector.Interferometer.get_time_domain_detector_response`.

        Returns
        =======
        injection_polarizations: list
            One dict per detector (the same polarizations for each).
        """
        if injection_polarizations is None:
            if waveform_generator is not None:
                if waveform_generator.time_domain_source_model is None:
                    raise ValueError(
                        "inject_signal_time_domain needs a waveform generator with "
                        "a time_domain_source_model")
                injection_polarizations = waveform_generator.time_domain_strain(
                    parameters
                )
            else:
                raise ValueError(
                    "inject_signal_time_domain needs one of waveform_generator or "
                    "injection_polarizations."
                )

        all_injection_polarizations = list()
        for interferometer in self:
            all_injection_polarizations.append(
                interferometer.inject_signal_time_domain(
                    parameters=parameters,
                    injection_polarizations=injection_polarizations,
                    placement=placement,
                )
            )

        return all_injection_polarizations

    def save_data(self, outdir, label=None):
        """Creates a save file for the data in plain text format

        Parameters
        ==========
        outdir: str
            The output directory in which the data is supposed to be saved
        label: str
            The string labelling the data
        """
        for interferometer in self:
            interferometer.save_data(outdir=outdir, label=label)

    def plot_data(self, signal=None, outdir=".", label=None):
        if utils.command_line_args.bilby_test_mode:
            return

        for interferometer in self:
            interferometer.plot_data(signal=signal, outdir=outdir, label=label)

    def plot_time_domain_data(
        self, outdir=".", label=None, bandpass_frequencies=(50, 250),
        notches=None, start_end=None, t0=None
    ):
        """Plots the strain data in the time domain for each of the
        interfeormeters

        Parameters
        ==========
        outdir: str
            The output directory in which the plots should be saved.
        label: str
            The string labelling the data.
        bandpass_frequencies: tuple, optional
            A tuple of the (low, high) frequencies to use when bandpassing
            data, if None no bandpass is applied.
        notches: list, optional
            A list of frequencies specifying any lines to notch.
        start_end: tuple, optional
            A tuple of the (start, end) range of GPS times to plot.
        t0: float, optional
            If given, the reference time to subtract from the time series
            plotting.
        """
        if utils.command_line_args.bilby_test_mode:
            return

        for interferometer in self:
            interferometer.plot_time_domain_data(
                outdir=outdir,
                label=label,
                bandpass_frequencies=bandpass_frequencies,
                notches=notches,
                start_end=start_end,
                t0=t0
            )

    @property
    def number_of_interferometers(self):
        return len(self)

    @property
    def duration(self):
        return self[0].strain_data.duration

    @property
    def start_time(self):
        return self[0].strain_data.start_time

    @property
    def sampling_frequency(self):
        return self[0].strain_data.sampling_frequency

    @property
    def frequency_array(self):
        return self[0].strain_data.frequency_array

    def append(self, interferometer):
        if isinstance(interferometer, InterferometerList):
            super(InterferometerList, self).extend(interferometer)
        else:
            super(InterferometerList, self).append(interferometer)
        self._check_interferometers()

    def extend(self, interferometers):
        super(InterferometerList, self).extend(interferometers)
        self._check_interferometers()

    def insert(self, index, interferometer):
        super(InterferometerList, self).insert(index, interferometer)
        self._check_interferometers()

    @property
    def meta_data(self):
        """Dictionary of the per-interferometer meta_data"""
        return {
            interferometer.name: interferometer.meta_data for interferometer in self
        }

    @staticmethod
    def _filename_from_outdir_label_extension(outdir, label, extension="h5"):
        return os.path.join(outdir, label + f".{extension}")

    _save_docstring = """ Saves the object to a {format} file

    {extra}

    Parameters
    ==========
    outdir: str, optional
        Output directory name of the file
    label: str, optional
        Output file name, is 'ifo_list' if not given otherwise. A list of
        the included interferometers will be appended.
    """

    _load_docstring = """ Loads in an InterferometerList object from a {format} file

    Parameters
    ==========
    filename: str
        If given, try to load from this filename

    """

    def to_pickle(self, outdir="outdir", label="ifo_list"):
        utils.check_directory_exists_and_if_not_mkdir(outdir)
        label = label + "_" + "".join(ifo.name for ifo in self)
        filename = self._filename_from_outdir_label_extension(
            outdir, label, extension="pkl"
        )
        safe_file_dump(self, filename, "dill")

    @classmethod
    def from_pickle(cls, filename=None):
        import dill

        with open(filename, "rb") as ff:
            res = dill.load(ff)
        if res.__class__ != cls:
            raise TypeError("The loaded object is not an InterferometerList")
        return res

    to_pickle.__doc__ = _save_docstring.format(
        format="pickle", extra=".. versionadded:: 1.1.0"
    )
    from_pickle.__doc__ = _load_docstring.format(format="pickle")

    def set_array_backend(self, xp):
        for ifo in self:
            ifo.set_array_backend(xp)

    @property
    def array_backend(self):
        return self[0].array_backend


class TriangularInterferometer(InterferometerList):
    def __init__(
        self,
        name,
        power_spectral_density,
        minimum_frequency,
        maximum_frequency,
        length,
        latitude,
        longitude,
        elevation,
        xarm_azimuth,
        yarm_azimuth,
        xarm_tilt=0.0,
        yarm_tilt=0.0,
    ):
        super(TriangularInterferometer, self).__init__([])
        self.name = name
        # for attr in ['power_spectral_density', 'minimum_frequency', 'maximum_frequency']:
        if isinstance(power_spectral_density, PowerSpectralDensity):
            power_spectral_density = [power_spectral_density] * 3
        if isinstance(minimum_frequency, float) or isinstance(minimum_frequency, int):
            minimum_frequency = [minimum_frequency] * 3
        if isinstance(maximum_frequency, float) or isinstance(maximum_frequency, int):
            maximum_frequency = [maximum_frequency] * 3

        brng = 90 - xarm_azimuth

        for ii in range(3):
            self.append(
                Interferometer(
                    "{}{}".format(name, ii + 1),
                    power_spectral_density[ii],
                    minimum_frequency[ii],
                    maximum_frequency[ii],
                    length,
                    latitude,
                    longitude,
                    elevation,
                    xarm_azimuth,
                    yarm_azimuth,
                    xarm_tilt,
                    yarm_tilt,
                )
            )

            phi1 = np.radians(latitude)
            phi2 = np.arcsin(
                np.sin(phi1) * np.cos(length * 1e3 / utils.radius_of_earth) +
                np.cos(phi1) * np.sin(length * 1e3 / utils.radius_of_earth) * np.cos(np.radians(brng))
            )
            latitude = np.degrees(phi2)

            lam1 = np.radians(longitude)
            lam2 = lam1 + np.arctan2(
                np.sin(np.radians(brng)) * np.sin(length * 1e3 / utils.radius_of_earth) * np.cos(phi1),
                np.cos(length * 1e3 / utils.radius_of_earth) - np.sin(phi1) * np.sin(phi2)
            )
            longitude = np.degrees(lam2)

            brng += 240
            xarm_azimuth += 240
            yarm_azimuth += 240


_LEGACY_DETECTOR_NAMES = {
    # GEO600 was renamed to its LAL/channel-name prefix, G1, so that
    # InterferometerList(["G1"]) matches the "G1:..." channel names found
    # in GEO frame files. This alias keeps the old identifier working.
    "GEO600": "G1",
}


def get_empty_interferometer(name):
    """
    Get an interferometer with standard parameters for known detectors.

    These objects do not have any noise instantiated.

    The available instruments are:
        H1, L1, V1, G1, CE

    ``GEO600`` is accepted as a deprecated alias for ``G1``.

    Detector positions taken from:
        L1/H1: LIGO-T980044-10
        V1/G1: arXiv:gr-qc/0008066 [45]
        CE: located at the site of H1

    Detector sensitivities:
        H1/L1/V1: https://dcc.ligo.org/LIGO-P1200087-v42/public
        G1: http://www.geo600.org/1032083/GEO600_Sensitivity_Curves
        CE: https://dcc.ligo.org/LIGO-P1600143/public


    Parameters
    ==========
    name: str
        Interferometer identifier.

    Returns
    =======
    interferometer: Interferometer
        Interferometer instance
    """
    if name in _LEGACY_DETECTOR_NAMES:
        new_name = _LEGACY_DETECTOR_NAMES[name]
        logger.warning(
            "Interferometer name '{}' is deprecated, use '{}' instead.".format(
                name, new_name
            )
        )
        name = new_name
    filename = os.path.join(
        os.path.dirname(__file__), "detectors", "{}.interferometer".format(name)
    )
    try:
        return load_interferometer(filename)
    except OSError:
        raise ValueError("Interferometer {} not implemented".format(name))


def load_interferometer(filename):
    """Load an interferometer from a file."""
    parameters = dict()
    with open(filename, "r") as parameter_file:
        lines = parameter_file.readlines()
        for line in lines:
            if line[0] == "#" or line[0] == "\n":
                continue
            split_line = line.split("=")
            key = split_line[0].strip()
            value = eval("=".join(split_line[1:]))
            parameters[key] = value
    if "shape" not in parameters.keys():
        ifo = Interferometer(**parameters)
        logger.debug("Assuming L shape for {}".format("name"))
    elif parameters["shape"].lower() in ["l", "ligo"]:
        parameters.pop("shape")
        ifo = Interferometer(**parameters)
    elif parameters["shape"].lower() in ["triangular", "triangle"]:
        parameters.pop("shape")
        ifo = TriangularInterferometer(**parameters)
    else:
        raise IOError(
            "{} could not be loaded. Invalid parameter 'shape'.".format(filename)
        )
    return ifo


@zenith_azimuth_to_theta_phi.dispatch
def zenith_azimuth_to_theta_phi(zenith, azimuth, ifos: InterferometerList | list):
    delta_x = ifos[0].geometry.vertex - ifos[1].geometry.vertex
    return zenith_azimuth_to_theta_phi(zenith, azimuth, delta_x)
