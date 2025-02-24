import os

import numpy as np
import math

from ...core import utils
from ...core.utils import logger, safe_file_dump
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

    def _check_interferometers(self):
        """Verify IFOs 'duration', 'start_time', 'sampling_frequency' are the same.

        If the above attributes are not the same, then the attributes are checked to
        see if they are the same up to 5 decimal places.

        If both checks fail, then a ValueError is raised.
        """
        consistent_attributes = ["duration", "start_time", "sampling_frequency"]
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
        self, sampling_frequency, duration, start_time=0
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

            xarm_azimuth += 240
            yarm_azimuth += 240

            latitude += (
                np.arctan(
                    length
                    * np.sin(xarm_azimuth * np.pi / 180)
                    * 1e3
                    / utils.radius_of_earth
                )
                * 180
                / np.pi
            )
            longitude += (
                np.arctan(
                    length
                    * np.cos(xarm_azimuth * np.pi / 180)
                    * 1e3
                    / utils.radius_of_earth
                )
                * 180
                / np.pi
            )


def get_empty_interferometer(name):
    """
    Get an interferometer with standard parameters for known detectors.

    These objects do not have any noise instantiated.

    The available instruments are:
        H1, L1, V1, GEO600, CE

    Detector positions taken from:
        L1/H1: LIGO-T980044-10
        V1/GEO600: arXiv:gr-qc/0008066 [45]
        CE: located at the site of H1

    Detector sensitivities:
        H1/L1/V1: https://dcc.ligo.org/LIGO-P1200087-v42/public
        GEO600: http://www.geo600.org/1032083/GEO600_Sensitivity_Curves
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
