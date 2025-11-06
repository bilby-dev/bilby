import json
import os
import pickle

import numpy as np

from ..core.result import Result as CoreResult
from ..core.utils import (
    check_directory_exists_and_if_not_mkdir,
    infft,
    latex_plot_format,
    logger,
    safe_file_dump,
    safe_save_figure,
)
from .detector import Interferometer, get_empty_interferometer
from .utils import asd_from_freq_series, plot_spline_pos, spline_angle_xform


class CompactBinaryCoalescenceResult(CoreResult):
    """
    Result class with additional methods and attributes specific to analyses
    of compact binaries.
    """

    def __init__(self, **kwargs):
        if "meta_data" not in kwargs:
            kwargs["meta_data"] = dict()
        if "global_meta_data" not in kwargs:
            from ..core.utils.meta_data import global_meta_data

            kwargs["meta_data"]["global_meta_data"] = global_meta_data
        # Ensure cosmology is always stored in the meta_data
        if "cosmology" not in kwargs["meta_data"]["global_meta_data"]:
            from .cosmology import get_cosmology

            kwargs["meta_data"]["global_meta_data"]["cosmology"] = get_cosmology()

        super().__init__(**kwargs)

    def __get_from_nested_meta_data(self, *keys):
        dictionary = self.meta_data
        try:
            item = None
            for k in keys:
                item = dictionary[k]
                dictionary = item
            return item
        except KeyError:
            raise AttributeError("No information stored for {}".format("/".join(keys)))

    @property
    def sampling_frequency(self):
        """Sampling frequency in Hertz"""
        return self.__get_from_nested_meta_data("likelihood", "sampling_frequency")

    @property
    def duration(self):
        """Duration in seconds"""
        return self.__get_from_nested_meta_data("likelihood", "duration")

    @property
    def start_time(self):
        """Start time in seconds"""
        return self.__get_from_nested_meta_data("likelihood", "start_time")

    @property
    def time_marginalization(self):
        """Boolean for if the likelihood used time marginalization"""
        return self.__get_from_nested_meta_data("likelihood", "time_marginalization")

    @property
    def phase_marginalization(self):
        """Boolean for if the likelihood used phase marginalization"""
        return self.__get_from_nested_meta_data("likelihood", "phase_marginalization")

    @property
    def distance_marginalization(self):
        """Boolean for if the likelihood used distance marginalization"""
        return self.__get_from_nested_meta_data("likelihood", "distance_marginalization")

    @property
    def interferometers(self):
        """List of interferometer names"""
        return [name for name in self.__get_from_nested_meta_data("likelihood", "interferometers")]

    @property
    def waveform_approximant(self):
        """String of the waveform approximant"""
        return self.__get_from_nested_meta_data("likelihood", "waveform_arguments", "waveform_approximant")

    @property
    def waveform_generator_class(self):
        """Dict of waveform arguments"""
        return self.__get_from_nested_meta_data("likelihood", "waveform_generator_class")

    @property
    def waveform_arguments(self):
        """Dict of waveform arguments"""
        return self.__get_from_nested_meta_data("likelihood", "waveform_arguments")

    @property
    def reference_frequency(self):
        """Float of the reference frequency"""
        return self.__get_from_nested_meta_data("likelihood", "waveform_arguments", "reference_frequency")

    @property
    def frequency_domain_source_model(self):
        """The frequency domain source model (function)"""
        return self.__get_from_nested_meta_data("likelihood", "frequency_domain_source_model")

    @property
    def time_domain_source_model(self):
        """The time domain source model (function)"""
        return self.__get_from_nested_meta_data("likelihood", "time_domain_source_model")

    @property
    def parameter_conversion(self):
        """The frequency domain source model (function)"""
        return self.__get_from_nested_meta_data("likelihood", "parameter_conversion")

    @property
    def cosmology(self):
        """The global cosmology used in the analysis.

        Will return None if the result does not include global meta data.
        Inclusion of the the global meta is controlled by the
        :code:`BILBY_INCLUDE_GLOBAL_META_DATA` environment variable.

        .. versionadded:: 2.5.0
        """
        try:
            return self.__get_from_nested_meta_data("global_meta_data", "cosmology")
        except AttributeError as e:
            logger.warning(
                "No cosmology found in result. "
                "This is likely due to the result not containing "
                f"global meta data. Error: {e}."
            )
            return None

    def detector_injection_properties(self, detector):
        """Returns a dictionary of the injection properties for each detector

        The injection properties include the parameters injected, and
        information about the signal to noise ratio (SNR) given the noise
        properties.

        Parameters
        ==========
        detector: str [H1, L1, V1]
            Detector name

        Returns
        =======
        injection_properties: dict
            A dictionary of the injection properties

        """
        try:
            return self.__get_from_nested_meta_data("likelihood", "interferometers", detector)
        except AttributeError:
            logger.info(f"No injection for detector {detector}")
            return None

    @latex_plot_format
    def plot_calibration_posterior(self, level=0.9, format="png"):
        """Plots the calibration amplitude and phase uncertainty.
        Adapted from the LALInference version in bayespputils

        Plot is saved to {self.outdir}/{self.label}_calibration.{format}

        Parameters
        ==========
        level: float
            Quantile for confidence levels, default=0.9, i.e., 90% interval
        format: str
            Format to save the plot, default=png, options are png/pdf
        """
        import matplotlib.pyplot as plt

        if format not in ["png", "pdf"]:
            raise ValueError("Format should be one of png or pdf")

        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(15, 15), dpi=500)
        posterior = self.posterior

        font_size = 32
        outdir = self.outdir

        parameters = posterior.keys()
        ifos = np.unique([param.split("_")[1] for param in parameters if "recalib_" in param])
        if ifos.size == 0:
            logger.info("No calibration parameters found. Aborting calibration plot.")
            return

        for ifo in ifos:
            if ifo == "H1":
                color = "r"
            elif ifo == "L1":
                color = "g"
            elif ifo == "V1":
                color = "m"
            else:
                color = "c"

            # Assume spline control frequencies are constant
            freq_params = np.sort([param for param in parameters if f"recalib_{ifo}_frequency_" in param])

            logfreqs = np.log([posterior[param].iloc[0] for param in freq_params])

            # Amplitude calibration model
            plt.sca(ax1)
            amp_params = np.sort([param for param in parameters if f"recalib_{ifo}_amplitude_" in param])
            if len(amp_params) > 0:
                amplitude = 100 * np.column_stack([posterior[param] for param in amp_params])
                plot_spline_pos(
                    logfreqs,
                    amplitude,
                    color=color,
                    level=level,
                    label=rf"{ifo.upper()} (mean, {int(level * 100)}$\%$)",
                )

            # Phase calibration model
            plt.sca(ax2)
            phase_params = np.sort([param for param in parameters if f"recalib_{ifo}_phase_" in param])
            if len(phase_params) > 0:
                phase = np.column_stack([posterior[param] for param in phase_params])
                plot_spline_pos(
                    logfreqs,
                    phase,
                    color=color,
                    level=level,
                    label=rf"{ifo.upper()} (mean, {int(level * 100)}$\%$)",
                    xform=spline_angle_xform,
                )

        ax1.tick_params(labelsize=0.75 * font_size)
        ax2.tick_params(labelsize=0.75 * font_size)
        plt.legend(loc="upper right", prop={"size": 0.75 * font_size}, framealpha=0.1)
        ax1.set_xscale("log")
        ax2.set_xscale("log")

        ax2.set_xlabel("Frequency [Hz]", fontsize=font_size)
        ax1.set_ylabel(r"Amplitude [$\%$]", fontsize=font_size)
        ax2.set_ylabel("Phase [deg]", fontsize=font_size)

        filename = os.path.join(outdir, self.label + "_calibration." + format)
        fig.tight_layout()
        safe_save_figure(fig=fig, filename=filename, format=format, dpi=600, bbox_inches="tight")
        logger.debug(f"Calibration figure saved to {filename}")
        plt.close()

    def plot_waveform_posterior(
        self, interferometers=None, level=0.9, n_samples=None, format="png", start_time=None, end_time=None
    ):
        """
        Plot the posterior for the waveform in the frequency domain and
        whitened time domain for all detectors.

        If the strain data is passed that will be plotted.

        If injection parameters can be found, the injection will be plotted.

        Parameters
        ==========
        interferometers: (list, bilby.gw.detector.InterferometerList, optional)
        level: float, optional
            symmetric confidence interval to show, default is 90%
        n_samples: int, optional
            number of samples to use to calculate the median/interval
            default is all
        format: str, optional
            format to save the figure in, default is png
        start_time: float, optional
            the amount of time before merger to begin the time domain plot.
            the merger time is defined as the mean of the geocenter time
            posterior. Default is - 0.4
        end_time: float, optional
            the amount of time before merger to end the time domain plot.
            the merger time is defined as the mean of the geocenter time
            posterior. Default is 0.2
        """
        if interferometers is None:
            interferometers = self.interferometers
        elif not isinstance(interferometers, list):
            raise TypeError("interferometers must be a list or InterferometerList")
        for ifo in interferometers:
            self.plot_interferometer_waveform_posterior(
                interferometer=ifo,
                level=level,
                n_samples=n_samples,
                save=True,
                format=format,
                start_time=start_time,
                end_time=end_time,
            )

    @latex_plot_format
    def plot_interferometer_waveform_posterior(
        self, interferometer, level=0.9, n_samples=None, save=True, format="png", start_time=None, end_time=None
    ):
        """
        Plot the posterior for the waveform in the frequency domain and
        whitened time domain.

        If the strain data is passed that will be plotted.

        If injection parameters can be found, the injection will be plotted.

        Parameters
        ==========
        interferometer: (str, bilby.gw.detector.interferometer.Interferometer)
            detector to use, if an Interferometer object is passed the data
            will be overlaid on the posterior
        level: float, optional
            symmetric confidence interval to show, default is 90%
        n_samples: int, optional
            number of samples to use to calculate the median/interval
            default is all
        save: bool, optional
            whether to save the image, default=True
            if False, figure handle is returned
        format: str, optional
            format to save the figure in, default is png
        start_time: float, optional
            the amount of time before merger to begin the time domain plot.
            the merger time is defined as the mean of the geocenter time
            posterior. Default is - 0.4
        end_time: float, optional
            the amount of time before merger to end the time domain plot.
            the merger time is defined as the mean of the geocenter time
            posterior. Default is 0.2

        Returns
        =======
        fig: figure-handle, only is save=False

        Notes
        -----
        To reduce the memory footprint we decimate the frequency domain
        waveforms to have ~4000 entries. This should be sufficient for decent
        resolution.
        """

        DATA_COLOR = "#ff7f0e"
        WAVEFORM_COLOR = "#1f77b4"
        INJECTION_COLOR = "#000000"

        if format == "html":
            try:
                import plotly.graph_objects as go
                from plotly.offline import plot
                from plotly.subplots import make_subplots
            except ImportError:
                logger.warning(
                    "HTML plotting requested, but plotly cannot be imported, "
                    "falling back to png format for waveform plot."
                )
                format = "png"

        if isinstance(interferometer, str):
            interferometer = get_empty_interferometer(interferometer)
            interferometer.set_strain_data_from_zero_noise(
                sampling_frequency=self.sampling_frequency, duration=self.duration, start_time=self.start_time
            )
            PLOT_DATA = False
        elif not isinstance(interferometer, Interferometer):
            raise TypeError("interferometer must be either str or Interferometer")
        else:
            PLOT_DATA = True
        logger.info(f"Generating waveform figure for {interferometer.name}")

        if n_samples is None:
            samples = self.posterior
        elif n_samples > len(self.posterior):
            logger.debug(
                f"Requested more waveform samples ({n_samples}) than we have posterior samples ({len(self.posterior)})!"
            )
            samples = self.posterior
        else:
            samples = self.posterior.sample(n_samples, replace=False)

        if start_time is None:
            start_time = -0.4
        start_time = np.mean(samples.geocent_time) + start_time
        if end_time is None:
            end_time = 0.2
        end_time = np.mean(samples.geocent_time) + end_time
        if format == "html":
            start_time = -np.inf
            end_time = np.inf
        time_idxs = (interferometer.time_array >= start_time) & (interferometer.time_array <= end_time)
        frequency_idxs = np.where(interferometer.frequency_mask)[0]
        logger.debug(f"Frequency mask contains {len(frequency_idxs)} values")
        frequency_idxs = frequency_idxs[:: max(1, len(frequency_idxs) // 4000)]
        logger.debug(f"Downsampling frequency mask to {len(frequency_idxs)} values")
        plot_times = interferometer.time_array[time_idxs]
        plot_times -= interferometer.strain_data.start_time
        start_time -= interferometer.strain_data.start_time
        end_time -= interferometer.strain_data.start_time
        plot_frequencies = interferometer.frequency_array[frequency_idxs]

        waveform_generator = self.waveform_generator_class(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            start_time=self.start_time,
            frequency_domain_source_model=self.frequency_domain_source_model,
            time_domain_source_model=self.time_domain_source_model,
            parameter_conversion=self.parameter_conversion,
            waveform_arguments=self.waveform_arguments,
        )

        if format == "html":
            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.5, 0.5],
            )
            fig.update_layout(
                template="plotly_white",
                font=dict(
                    family="Computer Modern",
                ),
            )
        else:
            import matplotlib.pyplot as plt
            from matplotlib import rcParams

            old_font_size = rcParams["font.size"]
            rcParams["font.size"] = 20
            fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[1.5, 1]), figsize=(16, 12.5))

        if PLOT_DATA:
            if format == "html":
                fig.add_trace(
                    go.Scatter(
                        x=plot_frequencies,
                        y=asd_from_freq_series(
                            interferometer.frequency_domain_strain[frequency_idxs],
                            1 / interferometer.strain_data.duration,
                        ),
                        fill=None,
                        mode="lines",
                        line_color=DATA_COLOR,
                        opacity=0.5,
                        name="Data",
                        legendgroup="data",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=plot_frequencies,
                        y=interferometer.amplitude_spectral_density_array[frequency_idxs],
                        fill=None,
                        mode="lines",
                        line_color=DATA_COLOR,
                        opacity=0.8,
                        name="ASD",
                        legendgroup="asd",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=plot_times,
                        y=interferometer.whitened_time_domain_strain[time_idxs],
                        fill=None,
                        mode="lines",
                        line_color=DATA_COLOR,
                        opacity=0.5,
                        name="Data",
                        legendgroup="data",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )
            else:
                axs[0].loglog(
                    plot_frequencies,
                    asd_from_freq_series(
                        interferometer.frequency_domain_strain[frequency_idxs], 1 / interferometer.strain_data.duration
                    ),
                    color=DATA_COLOR,
                    label="Data",
                    alpha=0.3,
                )
                axs[0].loglog(
                    plot_frequencies,
                    interferometer.amplitude_spectral_density_array[frequency_idxs],
                    color=DATA_COLOR,
                    label="ASD",
                )
                axs[1].plot(
                    plot_times, interferometer.whitened_time_domain_strain[time_idxs], color=DATA_COLOR, alpha=0.3
                )
            logger.debug("Plotted interferometer data.")

        fd_waveforms = list()
        td_waveforms = list()
        for params in samples.to_dict(orient="records"):
            wf_pols = waveform_generator.frequency_domain_strain(params)
            fd_waveform = interferometer.get_detector_response(wf_pols, params)
            fd_waveforms.append(fd_waveform[frequency_idxs])
            whitened_fd_waveform = interferometer.whiten_frequency_series(fd_waveform)
            td_waveform = interferometer.get_whitened_time_series_from_whitened_frequency_series(whitened_fd_waveform)[
                time_idxs
            ]
            td_waveforms.append(td_waveform)
        fd_waveforms = asd_from_freq_series(fd_waveforms, 1 / interferometer.strain_data.duration)
        td_waveforms = np.array(td_waveforms)

        delta = (1 + level) / 2
        upper_percentile = delta * 100
        lower_percentile = (1 - delta) * 100
        logger.debug(f"Plotting posterior between the {lower_percentile} and {upper_percentile} percentiles")

        if format == "html":
            fig.add_trace(
                go.Scatter(
                    x=plot_frequencies,
                    y=np.median(fd_waveforms, axis=0),
                    fill=None,
                    mode="lines",
                    line_color=WAVEFORM_COLOR,
                    opacity=1,
                    name="Median reconstructed",
                    legendgroup="median",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_frequencies,
                    y=np.percentile(fd_waveforms, lower_percentile, axis=0),
                    fill=None,
                    mode="lines",
                    line_color=WAVEFORM_COLOR,
                    opacity=0.1,
                    name=f"{upper_percentile - lower_percentile:.2f}% credible interval",
                    legendgroup="uncertainty",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_frequencies,
                    y=np.percentile(fd_waveforms, upper_percentile, axis=0),
                    fill="tonexty",
                    mode="lines",
                    line_color=WAVEFORM_COLOR,
                    opacity=0.1,
                    name=f"{upper_percentile - lower_percentile:.2f}% credible interval",
                    legendgroup="uncertainty",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_times,
                    y=np.median(td_waveforms, axis=0),
                    fill=None,
                    mode="lines",
                    line_color=WAVEFORM_COLOR,
                    opacity=1,
                    name="Median reconstructed",
                    legendgroup="median",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_times,
                    y=np.percentile(td_waveforms, lower_percentile, axis=0),
                    fill=None,
                    mode="lines",
                    line_color=WAVEFORM_COLOR,
                    opacity=0.1,
                    name=f"{upper_percentile - lower_percentile:.2f}% credible interval",
                    legendgroup="uncertainty",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_times,
                    y=np.percentile(td_waveforms, upper_percentile, axis=0),
                    fill="tonexty",
                    mode="lines",
                    line_color=WAVEFORM_COLOR,
                    opacity=0.1,
                    name=f"{upper_percentile - lower_percentile:.2f}% credible interval",
                    legendgroup="uncertainty",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        else:
            lower_limit = np.mean(fd_waveforms, axis=0)[0] / 1e3
            axs[0].loglog(
                plot_frequencies, np.mean(fd_waveforms, axis=0), color=WAVEFORM_COLOR, label="Mean reconstructed"
            )
            axs[0].fill_between(
                plot_frequencies,
                np.percentile(fd_waveforms, lower_percentile, axis=0),
                np.percentile(fd_waveforms, upper_percentile, axis=0),
                color=WAVEFORM_COLOR,
                label=rf"{int(upper_percentile - lower_percentile)}% credible interval",
                alpha=0.3,
            )
            axs[1].plot(plot_times, np.mean(td_waveforms, axis=0), color=WAVEFORM_COLOR)
            axs[1].fill_between(
                plot_times,
                np.percentile(td_waveforms, lower_percentile, axis=0),
                np.percentile(td_waveforms, upper_percentile, axis=0),
                color=WAVEFORM_COLOR,
                alpha=0.3,
            )

        if self.injection_parameters is not None:
            try:
                hf_inj = waveform_generator.frequency_domain_strain(self.injection_parameters)
                hf_inj_det = interferometer.get_detector_response(hf_inj, self.injection_parameters)
                ht_inj_det = infft(
                    hf_inj_det
                    * np.sqrt(2.0 / interferometer.sampling_frequency)
                    / interferometer.amplitude_spectral_density_array,
                    self.sampling_frequency,
                )[time_idxs]
                if format == "html":
                    fig.add_trace(
                        go.Scatter(
                            x=plot_frequencies,
                            y=asd_from_freq_series(hf_inj_det[frequency_idxs], 1 / interferometer.strain_data.duration),
                            fill=None,
                            mode="lines",
                            line=dict(color=INJECTION_COLOR, dash="dot"),
                            name="Injection",
                            legendgroup="injection",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=plot_times,
                            y=ht_inj_det,
                            fill=None,
                            mode="lines",
                            line=dict(color=INJECTION_COLOR, dash="dot"),
                            name="Injection",
                            legendgroup="injection",
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )
                else:
                    axs[0].loglog(
                        plot_frequencies,
                        asd_from_freq_series(hf_inj_det[frequency_idxs], 1 / interferometer.strain_data.duration),
                        color=INJECTION_COLOR,
                        label="Injection",
                        linestyle=":",
                    )
                    axs[1].plot(plot_times, ht_inj_det, color=INJECTION_COLOR, linestyle=":")
                logger.debug("Plotted injection.")
            except IndexError as e:
                logger.info(f"Failed to plot injection with message {e}.")

        f_domain_x_label = "$f [\\mathrm{Hz}]$"
        f_domain_y_label = "$\\mathrm{ASD} \\left[\\mathrm{Hz}^{-1/2}\\right]$"
        t_domain_x_label = f"$t - {interferometer.strain_data.start_time} [s]$"
        t_domain_y_label = "Whitened Strain"
        if format == "html":
            fig.update_xaxes(title_text=f_domain_x_label, type="log", row=1)
            fig.update_yaxes(title_text=f_domain_y_label, type="log", row=1)
            fig.update_xaxes(title_text=t_domain_x_label, type="linear", row=2)
            fig.update_yaxes(title_text=t_domain_y_label, type="linear", row=2)
        else:
            axs[0].set_xlim(interferometer.minimum_frequency, interferometer.maximum_frequency)
            axs[1].set_xlim(start_time, end_time)
            axs[0].set_ylim(lower_limit)
            axs[0].set_xlabel(f_domain_x_label)
            axs[0].set_ylabel(f_domain_y_label)
            axs[1].set_xlabel(t_domain_x_label)
            axs[1].set_ylabel(t_domain_y_label)
            axs[0].legend(loc="lower left", ncol=2)

        if save:
            filename = os.path.join(self.outdir, self.label + f"_{interferometer.name}_waveform.{format}")
            if format == "html":
                plot(fig, filename=filename, include_mathjax="cdn", auto_open=False)
            else:
                plt.tight_layout()
                safe_save_figure(fig=fig, filename=filename, format=format, dpi=600)
                plt.close()
            logger.debug(f"Waveform figure saved to {filename}")
            rcParams["font.size"] = old_font_size
        else:
            rcParams["font.size"] = old_font_size
            return fig

    def plot_skymap(
        self,
        maxpts=None,
        trials=5,
        jobs=1,
        enable_multiresolution=True,
        objid=None,
        instruments=None,
        geo=False,
        dpi=600,
        transparent=False,
        colorbar=False,
        contour=[50, 90],
        annotate=True,
        cmap="cylon",
        load_pickle=False,
    ):
        """Generate a fits file and sky map from a result

        Code adapted from ligo.skymap.tool.ligo_skymap_from_samples and
        ligo.skymap.tool.plot_skymap. Note, the use of this additionally
        required the installation of ligo.skymap.

        Parameters
        ==========
        maxpts: int
            Maximum number of samples to use, if None all samples are used
        trials: int
            Number of trials at each clustering number
        jobs: int
            Number of multiple threads
        enable_multiresolution: bool
            Generate a multiresolution HEALPix map (default: True)
        objid: str
            Event ID to store in FITS header
        instruments: str
            Name of detectors
        geo: bool
            Plot in geographic coordinates (lat, lon) instead of RA, Dec
        dpi: int
            Resolution of figure in fots per inch
        transparent: bool
            Save image with transparent background
        colorbar: bool
            Show colorbar
        contour: list
            List of contour levels to use
        annotate: bool
            Annotate image with details
        cmap: str
            Name of the colormap to use
        load_pickle: bool, str
            If true, load the cached pickle file (default name), or the
            pickle-file give as a path.
        """
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        try:
            import healpy as hp
            from astropy.time import Time
            from ligo.skymap import bayestar, io, kde, plot, postprocess, version
        except ImportError as e:
            logger.info(f"Unable to generate skymap: error {e}")
            return

        check_directory_exists_and_if_not_mkdir(self.outdir)

        logger.info("Reading samples for skymap")
        data = self.posterior

        if maxpts is not None and maxpts < len(data):
            logger.info("Taking random subsample of chain")
            data = data.sample(maxpts)

        default_obj_filename = os.path.join(self.outdir, f"{self.label}_skypost.obj")

        if load_pickle is False:
            try:
                pts = data[["ra", "dec", "luminosity_distance"]].values
                confidence_levels = kde.Clustered2Plus1DSkyKDE
                distance = True
            except KeyError:
                logger.warning("The results file does not contain luminosity_distance")
                pts = data[["ra", "dec"]].values
                confidence_levels = kde.Clustered2DSkyKDE
                distance = False

            logger.info("Initialising skymap class")
            skypost = confidence_levels(pts, trials=trials, jobs=jobs)
            logger.info(f"Pickling skymap to {default_obj_filename}")
            safe_file_dump(skypost, default_obj_filename, "pickle")

        else:
            if isinstance(load_pickle, str):
                obj_filename = load_pickle
            else:
                obj_filename = default_obj_filename
            logger.info(f"Reading from pickle {obj_filename}")
            with open(obj_filename, "rb") as file:
                skypost = pickle.load(file)
            skypost.jobs = jobs
            distance = isinstance(skypost, kde.Clustered2Plus1DSkyKDE)

        logger.info("Making skymap")
        hpmap = skypost.as_healpix()
        if not enable_multiresolution:
            hpmap = bayestar.rasterize(hpmap)

        hpmap.meta.update(io.fits.metadata_for_version_module(version))
        hpmap.meta["creator"] = "bilby"
        hpmap.meta["origin"] = "LIGO/Virgo"
        hpmap.meta["gps_creation_time"] = Time.now().gps
        hpmap.meta["history"] = ""
        if objid is not None:
            hpmap.meta["objid"] = objid
        if instruments:
            hpmap.meta["instruments"] = instruments
        if distance:
            hpmap.meta["distmean"] = np.mean(data["luminosity_distance"])
            hpmap.meta["diststd"] = np.std(data["luminosity_distance"])

        try:
            time = data["geocent_time"]
            hpmap.meta["gps_time"] = time.mean()
        except KeyError:
            logger.warning("Cannot determine the event time from geocent_time")

        fits_filename = os.path.join(self.outdir, f"{self.label}_skymap.fits")
        logger.info(f"Saving skymap fits-file to {fits_filename}")
        io.write_sky_map(fits_filename, hpmap, nest=True)

        skymap, metadata = io.fits.read_sky_map(fits_filename, nest=None)
        nside = hp.npix2nside(len(skymap))

        # Convert sky map from probability to probability per square degree.
        deg2perpix = hp.nside2pixarea(nside, degrees=True)
        probperdeg2 = skymap / deg2perpix

        if geo:
            obstime = Time(metadata["gps_time"], format="gps").utc.isot
            ax = plt.axes(projection="geo degrees mollweide", obstime=obstime)
        else:
            ax = plt.axes(projection="astro hours mollweide")
        ax.grid()

        # Plot sky map.
        vmax = probperdeg2.max()
        img = ax.imshow_hpx((probperdeg2, "ICRS"), nested=metadata["nest"], vmin=0.0, vmax=vmax, cmap=cmap)

        # Add colorbar.
        if colorbar:
            cb = plot.colorbar(img)
            cb.set_label(r"prob. per deg$^2$")

        if contour is not None:
            confidence_levels = 100 * postprocess.find_greedy_credible_levels(skymap)
            contours = ax.contour_hpx(
                (confidence_levels, "ICRS"), nested=metadata["nest"], colors="k", linewidths=0.5, levels=contour
            )
            fmt = r"%g\%%" if rcParams["text.usetex"] else "%g%%"
            plt.clabel(contours, fmt=fmt, fontsize=6, inline=True)

        # Add continents.
        if geo:
            geojson_filename = os.path.join(os.path.dirname(plot.__file__), "ne_simplified_coastline.json")
            with open(geojson_filename) as geojson_file:
                geoms = json.load(geojson_file)["geometries"]
            verts = [coord for geom in geoms for coord in zip(*geom["coordinates"])]
            plt.plot(*verts, color="0.5", linewidth=0.5, transform=ax.get_transform("world"))

        # Add a white outline to all text to make it stand out from the background.
        plot.outline_text(ax)

        if annotate:
            text = []
            try:
                objid = metadata["objid"]
            except KeyError:
                pass
            else:
                text.append(f"event ID: {objid}")
            if contour:
                pp = np.round(contour).astype(int)
                ii = np.round(np.searchsorted(np.sort(confidence_levels), contour) * deg2perpix).astype(int)
                for i, p in zip(ii, pp):
                    text.append(f"{p:d}% area: {i:d} deg$^2$")
            ax.text(1, 1, "\n".join(text), transform=ax.transAxes, ha="right")

        filename = os.path.join(self.outdir, f"{self.label}_skymap.png")
        logger.info(f"Generating 2D projected skymap to {filename}")
        safe_save_figure(fig=plt.gcf(), filename=filename, dpi=dpi)


CBCResult = CompactBinaryCoalescenceResult
