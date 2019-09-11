from __future__ import division

import json
import pickle
import os

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

from ..core.result import Result as CoreResult
from ..core.utils import infft, logger, check_directory_exists_and_if_not_mkdir
from .utils import plot_spline_pos, spline_angle_xform, asd_from_freq_series
from .waveform_generator import WaveformGenerator
from .detector import get_empty_interferometer, Interferometer
from .source import lal_binary_black_hole
from .conversion import convert_to_lal_binary_black_hole_parameters


class CompactBinaryCoalescenceResult(CoreResult):
    def __init__(self, **kwargs):
        super(CompactBinaryCoalescenceResult, self).__init__(**kwargs)

    def __get_from_nested_meta_data(self, *keys):
        dictionary = self.meta_data
        try:
            item = None
            for k in keys:
                item = dictionary[k]
                dictionary = item
            return item
        except KeyError:
            raise AttributeError(
                "No information stored for {}".format('/'.join(keys)))

    @property
    def sampling_frequency(self):
        """ Sampling frequency in Hertz"""
        return self.__get_from_nested_meta_data(
            'likelihood', 'sampling_frequency')

    @property
    def duration(self):
        """ Duration in seconds """
        return self.__get_from_nested_meta_data(
            'likelihood', 'duration')

    @property
    def start_time(self):
        """ Start time in seconds """
        return self.__get_from_nested_meta_data(
            'likelihood', 'start_time')

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
    def interferometers(self):
        """ List of interferometer names """
        return [name for name in self.__get_from_nested_meta_data(
            'likelihood', 'interferometers')]

    @property
    def waveform_approximant(self):
        """ String of the waveform approximant """
        return self.__get_from_nested_meta_data(
            'likelihood', 'waveform_arguments', 'waveform_approximant')

    @property
    def waveform_arguments(self):
        """ Dict of waveform arguments """
        return self.__get_from_nested_meta_data(
            'likelihood', 'waveform_arguments')

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
        except AttributeError:
            logger.info("No injection for detector {}".format(detector))
            return None

    def plot_calibration_posterior(self, level=.9):
        """ Plots the calibration amplitude and phase uncertainty.
        Adapted from the LALInference version in bayespputils

        Parameters
        ----------
        level: float,  percentage for confidence levels

        Returns
        -------
        saves a plot to outdir+label+calibration.png

        """
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(15, 15), dpi=500)
        posterior = self.posterior

        font_size = 32
        outdir = self.outdir

        parameters = posterior.keys()
        ifos = np.unique([param.split('_')[1] for param in parameters if 'recalib_' in param])
        if ifos.size == 0:
            logger.info("No calibration parameters found. Aborting calibration plot.")
            return

        for ifo in ifos:
            if ifo == 'H1':
                color = 'r'
            elif ifo == 'L1':
                color = 'g'
            elif ifo == 'V1':
                color = 'm'
            else:
                color = 'c'

            # Assume spline control frequencies are constant
            freq_params = np.sort([param for param in parameters if
                                   'recalib_{0}_frequency_'.format(ifo) in param])

            logfreqs = np.log([posterior[param].iloc[0] for param in freq_params])

            # Amplitude calibration model
            plt.sca(ax1)
            amp_params = np.sort([param for param in parameters if
                                  'recalib_{0}_amplitude_'.format(ifo) in param])
            if len(amp_params) > 0:
                amplitude = 100 * np.column_stack([posterior[param] for param in amp_params])
                plot_spline_pos(logfreqs, amplitude, color=color, level=level,
                                label="{0} (mean, {1}%)".format(ifo.upper(), int(level * 100)))

            # Phase calibration model
            plt.sca(ax2)
            phase_params = np.sort([param for param in parameters if
                                    'recalib_{0}_phase_'.format(ifo) in param])
            if len(phase_params) > 0:
                phase = np.column_stack([posterior[param] for param in phase_params])
                plot_spline_pos(logfreqs, phase, color=color, level=level,
                                label="{0} (mean, {1}%)".format(ifo.upper(), int(level * 100)),
                                xform=spline_angle_xform)

        ax1.tick_params(labelsize=.75 * font_size)
        ax2.tick_params(labelsize=.75 * font_size)
        plt.legend(loc='upper right', prop={'size': .75 * font_size}, framealpha=0.1)
        ax1.set_xscale('log')
        ax2.set_xscale('log')

        ax2.set_xlabel('Frequency (Hz)', fontsize=font_size)
        ax1.set_ylabel('Amplitude (%)', fontsize=font_size)
        ax2.set_ylabel('Phase (deg)', fontsize=font_size)

        filename = os.path.join(outdir, self.label + '_calibration.png')
        fig.tight_layout()
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    def plot_waveform_posterior(
            self, interferometers=None, level=0.9, n_samples=None,
            format='png', start_time=None, end_time=None):
        """
        Plot the posterior for the waveform in the frequency domain and
        whitened time domain for all detectors.

        If the strain data is passed that will be plotted.

        If injection parameters can be found, the injection will be plotted.

        Parameters
        ----------
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
            raise TypeError(
                'interferometers must be a list or InterferometerList')
        for ifo in interferometers:
            self.plot_interferometer_waveform_posterior(
                interferometer=ifo, level=level, n_samples=n_samples,
                save=True, format=format, start_time=start_time,
                end_time=end_time)

    def plot_interferometer_waveform_posterior(
            self, interferometer, level=0.9, n_samples=None, save=True,
            format='png', start_time=None, end_time=None):
        """
        Plot the posterior for the waveform in the frequency domain and
        whitened time domain.

        If the strain data is passed that will be plotted.

        If injection parameters can be found, the injection will be plotted.

        Parameters
        ----------
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
        -------
        fig: figure-handle, only is save=False

        Notes
        -----
        To reduce the memory footprint we decimate the frequency domain
        waveforms to have ~4000 entries. This should be sufficient for decent
        resolution.
        """
        if isinstance(interferometer, str):
            interferometer = get_empty_interferometer(interferometer)
            interferometer.set_strain_data_from_zero_noise(
                sampling_frequency=self.sampling_frequency,
                duration=self.duration, start_time=self.start_time)
        elif not isinstance(interferometer, Interferometer):
            raise TypeError(
                'interferometer must be either str or Interferometer')
        logger.info("Generating waveform figure for {}".format(
            interferometer.name))

        if n_samples is None:
            n_samples = len(self.posterior)
        elif n_samples > len(self.posterior):
            logger.debug(
                "Requested more waveform samples ({}) than we have "
                "posterior samples ({})!".format(
                    n_samples, len(self.posterior)
                )
            )
            n_samples = len(self.posterior)

        if start_time is None:
            start_time = - 0.4
        start_time = np.mean(self.posterior.geocent_time) + start_time
        if end_time is None:
            end_time = 0.2
        end_time = np.mean(self.posterior.geocent_time) + end_time
        time_idxs = (
            (interferometer.time_array >= start_time) &
            (interferometer.time_array <= end_time)
        )
        frequency_idxs = np.where(interferometer.frequency_mask)[0]
        logger.debug("Frequency mask contains {} values".format(
            len(frequency_idxs))
        )
        frequency_idxs = frequency_idxs[::max(1, len(frequency_idxs) // 4000)]
        logger.debug("Downsampling frequency mask to {} values".format(
            len(frequency_idxs))
        )
        plot_times = interferometer.time_array[time_idxs]
        plot_frequencies = interferometer.frequency_array[frequency_idxs]

        waveform_generator = WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            start_time=self.start_time,
            frequency_domain_source_model=lal_binary_black_hole,
            parameter_conversion=convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=self.waveform_arguments)
        fig, axs = plt.subplots(2, 1)
        if self.injection_parameters is not None:
            try:
                hf_inj = waveform_generator.frequency_domain_strain(
                    self.injection_parameters)
                hf_inj_det = interferometer.get_detector_response(
                    hf_inj, self.injection_parameters)
                axs[0].loglog(
                    plot_frequencies,
                    asd_from_freq_series(
                        hf_inj_det[frequency_idxs],
                        1 / interferometer.strain_data.duration),
                    color='k', label='injected', linestyle='--')
                axs[1].plot(
                    plot_times,
                    infft(hf_inj_det /
                          interferometer.amplitude_spectral_density_array,
                          self.sampling_frequency)[time_idxs],
                    color='k', linestyle='--')
                logger.debug('Plotted injection.')
            except IndexError:
                logger.info('Failed to plot injection.')

        fd_waveforms = list()
        td_waveforms = list()
        for ii in range(n_samples):
            params = dict(self.posterior.iloc[ii])
            wf_pols = waveform_generator.frequency_domain_strain(params)
            fd_waveform = interferometer.get_detector_response(wf_pols, params)
            fd_waveforms.append(fd_waveform[frequency_idxs])
            td_waveform = infft(
                fd_waveform / interferometer.amplitude_spectral_density_array,
                self.sampling_frequency)[time_idxs]
            td_waveforms.append(td_waveform)
        fd_waveforms = asd_from_freq_series(
            fd_waveforms,
            1 / interferometer.strain_data.duration)
        td_waveforms = np.array(td_waveforms)

        delta = (1 + level) / 2
        upper_percentile = delta * 100
        lower_percentile = (1 - delta) * 100
        logger.debug(
            'Plotting posterior between the {} and {} percentiles'.format(
                lower_percentile, upper_percentile
            )
        )

        axs[0].loglog(
            plot_frequencies,
            np.median(fd_waveforms, axis=0), color='r', label='Median')
        axs[0].fill_between(
            plot_frequencies,
            np.percentile(fd_waveforms, lower_percentile, axis=0),
            np.percentile(fd_waveforms, upper_percentile, axis=0),
            color='r', label='{} % Interval'.format(
                int(upper_percentile - lower_percentile)),
            alpha=0.3)
        axs[1].plot(
            plot_times, np.median(td_waveforms, axis=0),
            color='r')
        axs[1].fill_between(
            plot_times, np.percentile(
                td_waveforms, lower_percentile, axis=0),
            np.percentile(td_waveforms, upper_percentile, axis=0), color='r',
            alpha=0.3)

        try:
            axs[0].loglog(
                plot_frequencies,
                asd_from_freq_series(
                    interferometer.frequency_domain_strain[frequency_idxs],
                    1 / interferometer.strain_data.duration),
                color='b', label='Data', alpha=0.3)
            axs[0].loglog(
                plot_frequencies,
                interferometer.amplitude_spectral_density_array[frequency_idxs],
                color='b', label='PSD')
            axs[1].plot(
                plot_times, infft(
                    interferometer.whitened_frequency_domain_strain,
                    sampling_frequency=interferometer.strain_data.sampling_frequency)[time_idxs],
                color='b', alpha=0.3)
            logger.debug('Plotted interferometer data.')
        except AttributeError:
            pass

        axs[0].set_xlim(interferometer.minimum_frequency,
                        interferometer.maximum_frequency)
        axs[1].set_xlim(start_time, end_time)

        axs[0].set_xlabel('$f$ [$Hz$]')
        axs[1].set_xlabel('$t$ [$s$]')
        axs[0].set_ylabel('$\\tilde{h}(f)$ [Hz$^{-\\frac{1}{2}}$]')
        axs[1].set_ylabel('Whitened strain')
        axs[0].legend(loc='lower left')

        plt.tight_layout()
        if save:
            filename = os.path.join(
                self.outdir,
                self.label + '_{}_waveform.{}'.format(
                    interferometer.name, format))
            plt.savefig(filename, format=format, dpi=600)
            logger.debug("Figure saved to {}".format(filename))
            plt.close()
        else:
            return fig

    def plot_skymap(
            self, maxpts=None, trials=5, jobs=1, enable_multiresolution=True,
            objid=None, instruments=None, geo=False, dpi=600,
            transparent=False, colorbar=False, contour=[50, 90],
            annotate=True, cmap='cylon', load_pickle=False):
        """ Generate a fits file and sky map from a result

        Code adapted from ligo.skymap.tool.ligo_skymap_from_samples and
        ligo.skymap.tool.plot_skymap. Note, the use of this additionally
        required the installation of ligo.skymap.

        Parameters
        ----------
        maxpts: int
            Number of samples to use, if None all samples are used
        trials: int
            Number of trials at each clustering number
        jobs: int
            Number of multiple threads
        enable_multiresolution: bool
            Generate a multiresolution HEALPix map (default: True)
        objid: st
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

        try:
            from astropy.time import Time
            from ligo.skymap import io, version, plot, postprocess, bayestar, kde
            import healpy as hp
        except ImportError as e:
            logger.info("Unable to generate skymap: error {}".format(e))
            return

        check_directory_exists_and_if_not_mkdir(self.outdir)

        logger.info('Reading samples for skymap')
        data = self.posterior

        if maxpts is not None and maxpts < len(data):
            logger.info('Taking random subsample of chain')
            data = data.sample(maxpts)

        default_obj_filename = os.path.join(self.outdir, '{}_skypost.obj'.format(self.label))

        if load_pickle is False:
            try:
                pts = data[['ra', 'dec', 'luminosity_distance']].values
                cls = kde.Clustered2Plus1DSkyKDE
                distance = True
            except KeyError:
                logger.warning("The results file does not contain luminosity_distance")
                pts = data[['ra', 'dec']].values
                cls = kde.Clustered2DSkyKDE
                distance = False

            logger.info('Initialising skymap class')
            skypost = cls(pts, trials=trials, multiprocess=jobs)
            logger.info('Pickling skymap to {}'.format(default_obj_filename))
            with open(default_obj_filename, 'wb') as out:
                pickle.dump(skypost, out)

        else:
            if isinstance(load_pickle, str):
                obj_filename = load_pickle
            else:
                obj_filename = default_obj_filename
            logger.info('Reading from pickle {}'.format(obj_filename))
            with open(obj_filename, 'rb') as file:
                skypost = pickle.load(file)
            skypost.multiprocess = jobs

        logger.info('Making skymap')
        hpmap = skypost.as_healpix()
        if not enable_multiresolution:
            hpmap = bayestar.rasterize(hpmap)

        hpmap.meta.update(io.fits.metadata_for_version_module(version))
        hpmap.meta['creator'] = "bilby"
        hpmap.meta['origin'] = 'LIGO/Virgo'
        hpmap.meta['gps_creation_time'] = Time.now().gps
        hpmap.meta['history'] = ""
        if objid is not None:
            hpmap.meta['objid'] = objid
        if instruments:
            hpmap.meta['instruments'] = instruments
        if distance:
            hpmap.meta['distmean'] = np.mean(data['luminosity_distance'])
            hpmap.meta['diststd'] = np.std(data['luminosity_distance'])

        try:
            time = data['geocent_time']
            hpmap.meta['gps_time'] = time.mean()
        except KeyError:
            logger.warning('Cannot determine the event time from geocent_time')

        fits_filename = os.path.join(self.outdir, "{}_skymap.fits".format(self.label))
        logger.info('Saving skymap fits-file to {}'.format(fits_filename))
        io.write_sky_map(fits_filename, hpmap, nest=True)

        skymap, metadata = io.fits.read_sky_map(fits_filename, nest=None)
        nside = hp.npix2nside(len(skymap))

        # Convert sky map from probability to probability per square degree.
        deg2perpix = hp.nside2pixarea(nside, degrees=True)
        probperdeg2 = skymap / deg2perpix

        if geo:
            obstime = Time(metadata['gps_time'], format='gps').utc.isot
            ax = plt.axes(projection='geo degrees mollweide', obstime=obstime)
        else:
            ax = plt.axes(projection='astro hours mollweide')
        ax.grid()

        # Plot sky map.
        vmax = probperdeg2.max()
        img = ax.imshow_hpx(
            (probperdeg2, 'ICRS'), nested=metadata['nest'], vmin=0., vmax=vmax,
            cmap=cmap)

        # Add colorbar.
        if colorbar:
            cb = plot.colorbar(img)
            cb.set_label(r'prob. per deg$^2$')

        if contour is not None:
            cls = 100 * postprocess.find_greedy_credible_levels(skymap)
            cs = ax.contour_hpx(
                (cls, 'ICRS'), nested=metadata['nest'],
                colors='k', linewidths=0.5, levels=contour)
            fmt = r'%g\%%' if rcParams['text.usetex'] else '%g%%'
            plt.clabel(cs, fmt=fmt, fontsize=6, inline=True)

        # Add continents.
        if geo:
            geojson_filename = os.path.join(
                os.path.dirname(plot.__file__), 'ne_simplified_coastline.json')
            with open(geojson_filename, 'r') as geojson_file:
                geoms = json.load(geojson_file)['geometries']
            verts = [coord for geom in geoms
                     for coord in zip(*geom['coordinates'])]
            plt.plot(*verts, color='0.5', linewidth=0.5,
                     transform=ax.get_transform('world'))

        # Add a white outline to all text to make it stand out from the background.
        plot.outline_text(ax)

        if annotate:
            text = []
            try:
                objid = metadata['objid']
            except KeyError:
                pass
            else:
                text.append('event ID: {}'.format(objid))
            if contour:
                pp = np.round(contour).astype(int)
                ii = np.round(np.searchsorted(np.sort(cls), contour) *
                              deg2perpix).astype(int)
                for i, p in zip(ii, pp):
                    text.append(
                        u'{:d}% area: {:d} deg$^2$'.format(p, i, grouping=True))
            ax.text(1, 1, '\n'.join(text), transform=ax.transAxes, ha='right')

        filename = os.path.join(self.outdir, "{}_skymap.png".format(self.label))
        logger.info("Generating 2D projected skymap to {}".format(filename))
        plt.savefig(filename, dpi=500)


class CompactBinaryCoalesenceResult(CompactBinaryCoalescenceResult):

    def __init__(self, **kwargs):
        logger.warning('CompactBinaryCoalesenceResult is deprecated use '
                       'CompactBinaryCoalescenceResult')
        super(CompactBinaryCoalesenceResult, self).__init__(**kwargs)


CBCResult = CompactBinaryCoalescenceResult
