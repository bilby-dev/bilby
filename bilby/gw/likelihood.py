from __future__ import division

import gc
import os
import json
import copy

import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp
from scipy.special import i0e

from ..core import likelihood
from ..core.utils import (
    logger, UnsortedInterp2d, BilbyJsonEncoder, decode_bilby_json,
    create_frequency_series, create_time_series, speed_of_light,
    radius_of_earth)
from ..core.prior import Interped, Prior, Uniform
from .detector import InterferometerList
from .prior import BBHPriorDict
from .source import lal_binary_black_hole
from .utils import noise_weighted_inner_product, build_roq_weights, blockwise_dot_product
from .waveform_generator import WaveformGenerator
from collections import namedtuple


class GravitationalWaveTransient(likelihood.Likelihood):
    """ A gravitational-wave transient likelihood object

    This is the usual likelihood object to use for transient gravitational
    wave parameter estimation. It computes the log-likelihood in the frequency
    domain assuming a colored Gaussian noise model described by a power
    spectral density. See Thrane & Talbot (2019), arxiv.org/abs/1809.02293.


    Parameters
    ----------
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal,
        given some set of parameters
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood.
        This uses a look up table calculated at run time.
        The distance prior is set to be a delta function at the minimum
        distance allowed in the prior being marginalised over.
    time_marginalization: bool, optional
        If true, marginalize over time in the likelihood.
        This uses a FFT to calculate the likelihood over a regularly spaced
        grid.
        In order to cover the whole space the prior is set to be uniform over
        the spacing of the array of times.
        If using time marginalisation and jitter_time is True a "jitter"
        parameter is added to the prior which modifies the position of the
        grid of times.
    phase_marginalization: bool, optional
        If true, marginalize over phase in the likelihood.
        This is done analytically using a Bessel function.
        The phase prior is set to be a delta function at phase=0.
    priors: dict, optional
        If given, used in the distance and phase marginalization.
    distance_marginalization_lookup_table: (dict, str), optional
        If a dict, dictionary containing the lookup_table, distance_array,
        (distance) prior_array, and reference_distance used to construct
        the table.
        If a string the name of a file containing these quantities.
        The lookup table is stored after construction in either the
        provided string or a default location:
        '.distance_marginalization_lookup_dmin{}_dmax{}_n{}.npz'
    jitter_time: bool, optional
        Whether to introduce a `time_jitter` parameter. This avoids either
        missing the likelihood peak, or introducing biases in the
        reconstructed time posterior due to an insufficient sampling frequency.
        Default is False, however using this parameter is strongly encouraged.

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters

    """

    _CalculatedSNRs = namedtuple('CalculatedSNRs',
                                 ['d_inner_h',
                                  'optimal_snr_squared',
                                  'complex_matched_filter_snr',
                                  'd_inner_h_squared_tc_array'])

    def __init__(self, interferometers, waveform_generator,
                 time_marginalization=False, distance_marginalization=False,
                 phase_marginalization=False, priors=None,
                 distance_marginalization_lookup_table=None,
                 jitter_time=True):

        self.waveform_generator = waveform_generator
        likelihood.Likelihood.__init__(self, dict())
        self.interferometers = InterferometerList(interferometers)
        self.time_marginalization = time_marginalization
        self.distance_marginalization = distance_marginalization
        self.phase_marginalization = phase_marginalization
        self.priors = priors
        self._check_set_duration_and_sampling_frequency_of_waveform_generator()
        self.jitter_time = jitter_time

        if self.time_marginalization:
            self._check_prior_is_set(key='geocent_time')
            self._setup_time_marginalization()
            priors['geocent_time'] = float(self.interferometers.start_time)
            if self.jitter_time:
                priors['time_jitter'] = Uniform(
                    minimum=- self._delta_tc / 2, maximum=self._delta_tc / 2,
                    boundary='periodic')
        elif self.jitter_time:
            logger.info(
                "Time jittering requested with non-time-marginalised "
                "likelihood, ignoring.")
            self.jitter_time = False

        if self.phase_marginalization:
            self._check_prior_is_set(key='phase')
            self._bessel_function_interped = None
            self._setup_phase_marginalization()
            priors['phase'] = float(0)

        if self.distance_marginalization:
            self._lookup_table_filename = None
            self._check_prior_is_set(key='luminosity_distance')
            self._distance_array = np.linspace(
                self.priors['luminosity_distance'].minimum,
                self.priors['luminosity_distance'].maximum, int(1e4))
            self.distance_prior_array = np.array(
                [self.priors['luminosity_distance'].prob(distance)
                 for distance in self._distance_array])
            self._setup_distance_marginalization(
                distance_marginalization_lookup_table)
            priors['luminosity_distance'] = float(self._ref_dist)

    def __repr__(self):
        return self.__class__.__name__ + '(interferometers={},\n\twaveform_generator={},\n\ttime_marginalization={}, ' \
                                         'distance_marginalization={}, phase_marginalization={}, priors={})'\
            .format(self.interferometers, self.waveform_generator, self.time_marginalization,
                    self.distance_marginalization, self.phase_marginalization, self.priors)

    def _check_set_duration_and_sampling_frequency_of_waveform_generator(self):
        """ Check the waveform_generator has the same duration and
        sampling_frequency as the interferometers. If they are unset, then
        set them, if they differ, raise an error
        """

        attributes = ['duration', 'sampling_frequency', 'start_time']
        for attr in attributes:
            wfg_attr = getattr(self.waveform_generator, attr)
            ifo_attr = getattr(self.interferometers, attr)
            if wfg_attr is None:
                logger.debug(
                    "The waveform_generator {} is None. Setting from the "
                    "provided interferometers.".format(attr))
            elif wfg_attr != ifo_attr:
                logger.warning(
                    "The waveform_generator {} is not equal to that of the "
                    "provided interferometers. Overwriting the "
                    "waveform_generator.".format(attr))
            setattr(self.waveform_generator, attr, ifo_attr)

    def calculate_snrs(self, waveform_polarizations, interferometer):
        """
        Compute the snrs

        Parameters
        ----------
        waveform_polarizations: dict
            A dictionary of waveform polarizations and the corresponding array
        interferometer: bilby.gw.detector.Interferometer
            The bilby interferometer object

        """
        signal = interferometer.get_detector_response(
            waveform_polarizations, self.parameters)
        d_inner_h = interferometer.inner_product(signal=signal)
        optimal_snr_squared = interferometer.optimal_snr_squared(signal=signal)
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

        if self.time_marginalization:
            d_inner_h_squared_tc_array =\
                4 / self.waveform_generator.duration * np.fft.fft(
                    signal[0:-1] *
                    interferometer.frequency_domain_strain.conjugate()[0:-1] /
                    interferometer.power_spectral_density_array[0:-1])
        else:
            d_inner_h_squared_tc_array = None

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h, optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_squared_tc_array=d_inner_h_squared_tc_array)

    def _check_prior_is_set(self, key):
        if key not in self.priors or not isinstance(
                self.priors[key], Prior):
            logger.warning(
                'Prior not provided for {}, using the BBH default.'.format(key))
            if key == 'geocent_time':
                self.priors[key] = Uniform(
                    self.interferometers.start_time,
                    self.interferometers.start_time + self.interferometers.duration)
            else:
                self.priors[key] = BBHPriorDict()[key]

    @property
    def priors(self):
        return self._prior

    @priors.setter
    def priors(self, priors):
        if priors is not None:
            self._prior = priors.copy()
        elif any([self.time_marginalization, self.phase_marginalization,
                  self.distance_marginalization]):
            raise ValueError("You can't use a marginalized likelihood without specifying a priors")
        else:
            self._prior = None

    def noise_log_likelihood(self):
        log_l = 0
        for interferometer in self.interferometers:
            mask = interferometer.frequency_mask
            log_l -= noise_weighted_inner_product(
                interferometer.frequency_domain_strain[mask],
                interferometer.frequency_domain_strain[mask],
                interferometer.power_spectral_density_array[mask],
                self.waveform_generator.duration) / 2
        return float(np.real(log_l))

    def log_likelihood_ratio(self):
        waveform_polarizations =\
            self.waveform_generator.frequency_domain_strain(self.parameters)

        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        d_inner_h = 0.
        optimal_snr_squared = 0.
        complex_matched_filter_snr = 0.
        if self.time_marginalization:
            if self.jitter_time:
                self.parameters['geocent_time'] += self.parameters['time_jitter']
            d_inner_h_tc_array = np.zeros(
                self.interferometers.frequency_array[0:-1].shape,
                dtype=np.complex128)

        for interferometer in self.interferometers:
            per_detector_snr = self.calculate_snrs(
                waveform_polarizations=waveform_polarizations,
                interferometer=interferometer)

            d_inner_h += per_detector_snr.d_inner_h
            optimal_snr_squared += np.real(per_detector_snr.optimal_snr_squared)
            complex_matched_filter_snr += per_detector_snr.complex_matched_filter_snr

            if self.time_marginalization:
                d_inner_h_tc_array += per_detector_snr.d_inner_h_squared_tc_array

        if self.time_marginalization:
            log_l = self.time_marginalized_likelihood(
                d_inner_h_tc_array=d_inner_h_tc_array,
                h_inner_h=optimal_snr_squared)
            if self.jitter_time:
                self.parameters['geocent_time'] -= self.parameters['time_jitter']

        elif self.distance_marginalization:
            log_l = self.distance_marginalized_likelihood(
                d_inner_h=d_inner_h, h_inner_h=optimal_snr_squared)

        elif self.phase_marginalization:
            log_l = self.phase_marginalized_likelihood(
                d_inner_h=d_inner_h, h_inner_h=optimal_snr_squared)

        else:
            log_l = np.real(d_inner_h) - optimal_snr_squared / 2

        return float(log_l.real)

    def generate_posterior_sample_from_marginalized_likelihood(self):
        """
        Reconstruct the distance posterior from a run which used a likelihood
        which explicitly marginalised over time/distance/phase.

        See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

        Return
        ------
        sample: dict
            Returns the parameters with new samples.

        Notes
        -----
        This involves a deepcopy of the signal to avoid issues with waveform
        caching, as the signal is overwritten in place.
        """
        if any([self.phase_marginalization, self.distance_marginalization,
                self.time_marginalization]):
            signal_polarizations = copy.deepcopy(
                self.waveform_generator.frequency_domain_strain(
                    self.parameters))
        else:
            return self.parameters
        if self.time_marginalization:
            new_time = self.generate_time_sample_from_marginalized_likelihood(
                signal_polarizations=signal_polarizations)
            self.parameters['geocent_time'] = new_time
        if self.distance_marginalization:
            new_distance = self.generate_distance_sample_from_marginalized_likelihood(
                signal_polarizations=signal_polarizations)
            self.parameters['luminosity_distance'] = new_distance
        if self.phase_marginalization:
            new_phase = self.generate_phase_sample_from_marginalized_likelihood(
                signal_polarizations=signal_polarizations)
            self.parameters['phase'] = new_phase
        return self.parameters.copy()

    def generate_time_sample_from_marginalized_likelihood(
            self, signal_polarizations=None):
        """
        Generate a single sample from the posterior distribution for coalescence
        time when using a likelihood which explicitly marginalises over time.

        In order to resolve the posterior we artifically upsample to 16kHz.

        See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

        Parameters
        ----------
        signal_polarizations: dict, optional
            Polarizations modes of the template.

        Returns
        -------
        new_time: float
            Sample from the time posterior.
        """
        if self.jitter_time:
            self.parameters['geocent_time'] += self.parameters['time_jitter']
        if signal_polarizations is None:
            signal_polarizations = \
                self.waveform_generator.frequency_domain_strain(self.parameters)

        n_time_steps = int(self.waveform_generator.duration * 16384)
        d_inner_h = np.zeros(n_time_steps, dtype=np.complex)
        psd = np.ones(n_time_steps)
        signal_long = np.zeros(n_time_steps, dtype=np.complex)
        data = np.zeros(n_time_steps, dtype=np.complex)
        h_inner_h = np.zeros(1)
        for ifo in self.interferometers:
            ifo_length = len(ifo.frequency_domain_strain)
            signal = ifo.get_detector_response(
                signal_polarizations, self.parameters)
            signal_long[:ifo_length] = signal
            data[:ifo_length] = np.conj(ifo.frequency_domain_strain)
            psd[:ifo_length] = ifo.power_spectral_density_array
            d_inner_h += np.fft.fft(signal_long * data / psd)
            h_inner_h += ifo.optimal_snr_squared(signal=signal).real

        if self.distance_marginalization:
            time_log_like = self.distance_marginalized_likelihood(
                d_inner_h, h_inner_h)
        elif self.phase_marginalization:
            time_log_like = (self._bessel_function_interped(abs(d_inner_h)) -
                             h_inner_h.real / 2)
        else:
            time_log_like = (d_inner_h.real - h_inner_h.real / 2)

        times = create_time_series(
            sampling_frequency=16384,
            starting_time=self.parameters['geocent_time'] - self.waveform_generator.start_time,
            duration=self.waveform_generator.duration)
        times = times % self.waveform_generator.duration
        times += self.waveform_generator.start_time

        time_prior_array = self.priors['geocent_time'].prob(times)
        time_post = (
            np.exp(time_log_like - max(time_log_like)) * time_prior_array)

        keep = (time_post > max(time_post) / 1000)
        time_post = time_post[keep]
        times = times[keep]

        new_time = Interped(times, time_post).sample()
        return new_time

    def generate_distance_sample_from_marginalized_likelihood(
            self, signal_polarizations=None):
        """
        Generate a single sample from the posterior distribution for luminosity
        distance when using a likelihood which explicitly marginalises over
        distance.

        See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

        Parameters
        ----------
        signal_polarizations: dict, optional
            Polarizations modes of the template.
            Note: These are rescaled in place after the distance sample is
                  generated to allow further parameter reconstruction to occur.

        Returns
        -------
        new_distance: float
            Sample from the distance posterior.
        """
        if signal_polarizations is None:
            signal_polarizations = \
                self.waveform_generator.frequency_domain_strain(self.parameters)
        d_inner_h = 0
        h_inner_h = 0
        for interferometer in self.interferometers:
            per_detector_snr = self.calculate_snrs(
                signal_polarizations, interferometer)

            d_inner_h += per_detector_snr.d_inner_h
            h_inner_h += per_detector_snr.optimal_snr_squared

        d_inner_h_dist = (
            d_inner_h * self.parameters['luminosity_distance'] /
            self._distance_array)

        h_inner_h_dist = (
            h_inner_h * self.parameters['luminosity_distance']**2 /
            self._distance_array**2)

        if self.phase_marginalization:
            distance_log_like = (
                self._bessel_function_interped(abs(d_inner_h_dist)) -
                h_inner_h_dist.real / 2)
        else:
            distance_log_like = (d_inner_h_dist.real - h_inner_h_dist.real / 2)

        distance_post = (np.exp(distance_log_like - max(distance_log_like)) *
                         self.distance_prior_array)

        new_distance = Interped(
            self._distance_array, distance_post).sample()

        self._rescale_signal(signal_polarizations, new_distance)
        return new_distance

    def generate_phase_sample_from_marginalized_likelihood(
            self, signal_polarizations=None):
        """
        Generate a single sample from the posterior distribution for phase when
        using a likelihood which explicitly marginalises over phase.

        See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

        Parameters
        ----------
        signal_polarizations: dict, optional
            Polarizations modes of the template.

        Returns
        -------
        new_phase: float
            Sample from the phase posterior.

        Notes
        -----
        This is only valid when assumes that mu(phi) \propto exp(-2i phi).
        """
        if signal_polarizations is None:
            signal_polarizations = \
                self.waveform_generator.frequency_domain_strain(self.parameters)
        d_inner_h = 0
        h_inner_h = 0
        for interferometer in self.interferometers:
            per_detector_snr = self.calculate_snrs(
                signal_polarizations, interferometer)

            d_inner_h += per_detector_snr.d_inner_h
            h_inner_h += per_detector_snr.optimal_snr_squared

        phases = np.linspace(0, 2 * np.pi, 101)
        phasor = np.exp(-2j * phases)
        phase_log_post = d_inner_h * phasor - h_inner_h / 2
        phase_post = np.exp(phase_log_post.real - max(phase_log_post.real))
        new_phase = Interped(phases, phase_post).sample()
        return new_phase

    def distance_marginalized_likelihood(self, d_inner_h, h_inner_h):
        d_inner_h_ref, h_inner_h_ref = self._setup_rho(
            d_inner_h, h_inner_h)
        if self.phase_marginalization:
            d_inner_h_ref = np.abs(d_inner_h_ref)
        else:
            d_inner_h_ref = np.real(d_inner_h_ref)
        return self._interp_dist_margd_loglikelihood(
            d_inner_h_ref, h_inner_h_ref)

    def phase_marginalized_likelihood(self, d_inner_h, h_inner_h):
        d_inner_h = self._bessel_function_interped(abs(d_inner_h))
        return d_inner_h - h_inner_h / 2

    def time_marginalized_likelihood(self, d_inner_h_tc_array, h_inner_h):
        if self.distance_marginalization:
            log_l_tc_array = self.distance_marginalized_likelihood(
                d_inner_h=d_inner_h_tc_array, h_inner_h=h_inner_h)
        elif self.phase_marginalization:
            log_l_tc_array = self.phase_marginalized_likelihood(
                d_inner_h=d_inner_h_tc_array,
                h_inner_h=h_inner_h)
        else:
            log_l_tc_array = np.real(d_inner_h_tc_array) - h_inner_h / 2
        times = self._times
        if self.jitter_time:
            times = self._times + self.parameters['time_jitter']
        time_prior_array = self.priors['geocent_time'].prob(times) * self._delta_tc
        return logsumexp(log_l_tc_array, b=time_prior_array)

    def _setup_rho(self, d_inner_h, optimal_snr_squared):
        optimal_snr_squared_ref = (optimal_snr_squared.real *
                                   self.parameters['luminosity_distance'] ** 2 /
                                   self._ref_dist ** 2.)
        d_inner_h_ref = (d_inner_h * self.parameters['luminosity_distance'] /
                         self._ref_dist)
        return d_inner_h_ref, optimal_snr_squared_ref

    def log_likelihood(self):
        return self.log_likelihood_ratio() + self.noise_log_likelihood()

    @property
    def _delta_distance(self):
        return self._distance_array[1] - self._distance_array[0]

    @property
    def _ref_dist(self):
        """ Smallest distance contained in priors """
        return self._distance_array[0]

    @property
    def _optimal_snr_squared_ref_array(self):
        """ Optimal filter snr at fiducial distance of ref_dist Mpc """
        return np.logspace(-5, 10, self._dist_margd_loglikelihood_array.shape[0])

    @property
    def _d_inner_h_ref_array(self):
        """ Matched filter snr at fiducial distance of ref_dist Mpc """
        if self.phase_marginalization:
            return np.logspace(-5, 10, self._dist_margd_loglikelihood_array.shape[1])
        else:
            return np.hstack((-np.logspace(3, -3, self._dist_margd_loglikelihood_array.shape[1] / 2),
                              np.logspace(-3, 10, self._dist_margd_loglikelihood_array.shape[1] / 2)))

    def _setup_distance_marginalization(self, lookup_table=None):
        if isinstance(lookup_table, str) or lookup_table is None:
            self.cached_lookup_table_filename = lookup_table
            lookup_table = self.load_lookup_table(
                self.cached_lookup_table_filename)
        if isinstance(lookup_table, dict):
            if self._test_cached_lookup_table(lookup_table):
                self._dist_margd_loglikelihood_array = lookup_table[
                    'lookup_table']
            else:
                self._create_lookup_table()
        else:
            self._create_lookup_table()
        self._interp_dist_margd_loglikelihood = UnsortedInterp2d(
            self._d_inner_h_ref_array, self._optimal_snr_squared_ref_array,
            self._dist_margd_loglikelihood_array, kind='cubic')

    @property
    def cached_lookup_table_filename(self):
        if self._lookup_table_filename is None:
            self._lookup_table_filename = (
                '.distance_marginalization_lookup.npz')
        return self._lookup_table_filename

    @cached_lookup_table_filename.setter
    def cached_lookup_table_filename(self, filename):
        if isinstance(filename, str):
            if filename[-4:] != '.npz':
                filename += '.npz'
        self._lookup_table_filename = filename

    def load_lookup_table(self, filename):
        if os.path.exists(filename):
            loaded_file = dict(np.load(filename))
            match, failure = self._test_cached_lookup_table(loaded_file)
            if match:
                logger.info('Loaded distance marginalisation lookup table from '
                            '{}.'.format(filename))
                return loaded_file
            else:
                logger.info('Loaded distance marginalisation lookup table does '
                            'not match for {}.'.format(failure))
                return None
        elif isinstance(filename, str):
            logger.info('Distance marginalisation file {} does not '
                        'exist'.format(filename))
            return None
        else:
            return None

    def cache_lookup_table(self):
        np.savez(self.cached_lookup_table_filename,
                 distance_array=self._distance_array,
                 prior_array=self.distance_prior_array,
                 lookup_table=self._dist_margd_loglikelihood_array,
                 reference_distance=self._ref_dist,
                 phase_marginalization=self.phase_marginalization)

    def _test_cached_lookup_table(self, loaded_file):
        pairs = dict(
            distance_array=self._distance_array,
            prior_array=self.distance_prior_array,
            reference_distance=self._ref_dist,
            phase_marginalization=self.phase_marginalization)
        for key in pairs:
            if key not in loaded_file:
                return False, key
            elif not np.array_equal(np.atleast_1d(loaded_file[key]),
                                    np.atleast_1d(pairs[key])):
                return False, key
        return True, None

    def _create_lookup_table(self):
        """ Make the lookup table """
        logger.info('Building lookup table for distance marginalisation.')

        self._dist_margd_loglikelihood_array = np.zeros((400, 800))
        for ii, optimal_snr_squared_ref in enumerate(self._optimal_snr_squared_ref_array):
            optimal_snr_squared_array = (
                optimal_snr_squared_ref * self._ref_dist ** 2. /
                self._distance_array ** 2)
            for jj, d_inner_h_ref in enumerate(self._d_inner_h_ref_array):
                d_inner_h_array = (
                    d_inner_h_ref * self._ref_dist / self._distance_array)
                if self.phase_marginalization:
                    d_inner_h_array =\
                        self._bessel_function_interped(abs(d_inner_h_array))
                self._dist_margd_loglikelihood_array[ii][jj] = \
                    logsumexp(d_inner_h_array - optimal_snr_squared_array / 2,
                              b=self.distance_prior_array * self._delta_distance)
        log_norm = logsumexp(0. / self._distance_array,
                             b=self.distance_prior_array * self._delta_distance)
        self._dist_margd_loglikelihood_array -= log_norm
        self.cache_lookup_table()

    def _setup_phase_marginalization(self):
        self._bessel_function_interped = interp1d(
            np.logspace(-5, 10, int(1e6)), np.logspace(-5, 10, int(1e6)) +
            np.log([i0e(snr) for snr in np.logspace(-5, 10, int(1e6))]),
            bounds_error=False, fill_value=(0, np.nan))

    def _setup_time_marginalization(self):
        self._delta_tc = 2 / self.waveform_generator.sampling_frequency
        self._times =\
            self.interferometers.start_time + np.linspace(
                0, self.interferometers.duration,
                int(self.interferometers.duration / 2 *
                    self.waveform_generator.sampling_frequency + 1))[1:]
        self.time_prior_array = \
            self.priors['geocent_time'].prob(self._times) * self._delta_tc

    @property
    def interferometers(self):
        return self._interferometers

    @interferometers.setter
    def interferometers(self, interferometers):
        self._interferometers = InterferometerList(interferometers)

    def _rescale_signal(self, signal, new_distance):
        for mode in signal:
            signal[mode] *= self._ref_dist / new_distance

    @property
    def meta_data(self):
        return dict(
            interferometers=self.interferometers.meta_data,
            time_marginalization=self.time_marginalization,
            phase_marginalization=self.phase_marginalization,
            distance_marginalization=self.distance_marginalization,
            waveform_arguments=self.waveform_generator.waveform_arguments,
            frequency_domain_source_model=str(
                self.waveform_generator.frequency_domain_source_model),
            sampling_frequency=self.waveform_generator.sampling_frequency,
            duration=self.waveform_generator.duration,
            start_time=self.waveform_generator.start_time)


class BasicGravitationalWaveTransient(likelihood.Likelihood):

    def __init__(self, interferometers, waveform_generator):
        """

        A likelihood object, able to compute the likelihood of the data given
        some model parameters

        The simplest frequency-domain gravitational wave transient likelihood. Does
        not include distance/phase marginalization.


        Parameters
        ----------
        interferometers: list
            A list of `bilby.gw.detector.Interferometer` instances - contains the
            detector data and power spectral densities
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            An object which computes the frequency-domain strain of the signal,
            given some set of parameters

        """
        likelihood.Likelihood.__init__(self, dict())
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator

    def __repr__(self):
        return self.__class__.__name__ + '(interferometers={},\n\twaveform_generator={})'\
            .format(self.interferometers, self.waveform_generator)

    def noise_log_likelihood(self):
        """ Calculates the real part of noise log-likelihood

        Returns
        -------
        float: The real part of the noise log likelihood

        """
        log_l = 0
        for interferometer in self.interferometers:
            log_l -= 2. / self.waveform_generator.duration * np.sum(
                abs(interferometer.frequency_domain_strain) ** 2 /
                interferometer.power_spectral_density_array)
        return log_l.real

    def log_likelihood(self):
        """ Calculates the real part of log-likelihood value

        Returns
        -------
        float: The real part of the log likelihood

        """
        log_l = 0
        waveform_polarizations =\
            self.waveform_generator.frequency_domain_strain(
                self.parameters.copy())
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)
        for interferometer in self.interferometers:
            log_l += self.log_likelihood_interferometer(
                waveform_polarizations, interferometer)
        return log_l.real

    def log_likelihood_interferometer(self, waveform_polarizations,
                                      interferometer):
        """

        Parameters
        ----------
        waveform_polarizations: dict
            Dictionary containing the desired waveform polarization modes and the related strain
        interferometer: bilby.gw.detector.Interferometer
            The Interferometer object we want to have the log-likelihood for

        Returns
        -------
        float: The real part of the log-likelihood for this interferometer

        """
        signal_ifo = interferometer.get_detector_response(
            waveform_polarizations, self.parameters)

        log_l = - 2. / self.waveform_generator.duration * np.vdot(
            interferometer.frequency_domain_strain - signal_ifo,
            (interferometer.frequency_domain_strain - signal_ifo) /
            interferometer.power_spectral_density_array)
        return log_l.real


class ROQGravitationalWaveTransient(GravitationalWaveTransient):
    """A reduced order quadrature likelihood object

    This uses the method described in Smith et al., (2016) Phys. Rev. D 94,
    044031. A public repository of the ROQ data is available from
    https://git.ligo.org/lscsoft/ROQ_data.

    Parameters
    ----------
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal,
        given some set of parameters
    linear_matrix: str, array_like
        Either a string point to the file from which to load the linear_matrix
        array, or the array itself.
    quadratic_matrix: str, array_like
        Either a string point to the file from which to load the
        quadratic_matrix array, or the array itself.
    roq_params: str, array_like
        Parameters describing the domain of validity of the ROQ basis.
    priors: dict, bilby.prior.PriorDict
        A dictionary of priors containing at least the geocent_time prior
    distance_marginalization_lookup_table: (dict, str), optional
        If a dict, dictionary containing the lookup_table, distance_array,
        (distance) prior_array, and reference_distance used to construct
        the table.
        If a string the name of a file containing these quantities.
        The lookup table is stored after construction in either the
        provided string or a default location:
        '.distance_marginalization_lookup_dmin{}_dmax{}_n{}.npz'

    """
    def __init__(self, interferometers, waveform_generator, priors,
                 weights=None, linear_matrix=None, quadratic_matrix=None,
                 roq_params=None,
                 distance_marginalization=False, phase_marginalization=False,
                 distance_marginalization_lookup_table=None):
        GravitationalWaveTransient.__init__(
            self, interferometers=interferometers,
            waveform_generator=waveform_generator, priors=priors,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            time_marginalization=False,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=False)

        if isinstance(roq_params, np.ndarray) or roq_params is None:
            self.roq_params = roq_params
        elif isinstance(roq_params, str):
            self.roq_params = np.genfromtxt(roq_params, names=True)
        else:
            raise TypeError("roq_params should be array or str")
        if isinstance(weights, dict):
            self.weights = weights
        elif isinstance(weights, str):
            self.weights = self.load_weights(weights)
        else:
            self.weights = dict()
            if isinstance(linear_matrix, str):
                logger.info(
                    "Loading linear matrix from {}".format(linear_matrix))
                linear_matrix = np.load(linear_matrix).T
            if isinstance(quadratic_matrix, str):
                logger.info(
                    "Loading quadratic_matrix from {}".format(quadratic_matrix))
                quadratic_matrix = np.load(quadratic_matrix).T
            self._set_weights(linear_matrix=linear_matrix,
                              quadratic_matrix=quadratic_matrix)
        self.frequency_nodes_linear =\
            waveform_generator.waveform_arguments['frequency_nodes_linear']
        self.frequency_nodes_quadratic = \
            waveform_generator.waveform_arguments['frequency_nodes_quadratic']

    def calculate_snrs(self, waveform_polarizations, interferometer):
        """
        Compute the snrs for ROQ

        Parameters
        ----------
        waveform_polarizations: waveform
        interferometer: bilby.gw.detector.Interferometer

        """

        f_plus = interferometer.antenna_response(
            self.parameters['ra'], self.parameters['dec'],
            self.parameters['geocent_time'], self.parameters['psi'], 'plus')
        f_cross = interferometer.antenna_response(
            self.parameters['ra'], self.parameters['dec'],
            self.parameters['geocent_time'], self.parameters['psi'], 'cross')

        dt = interferometer.time_delay_from_geocenter(
            self.parameters['ra'], self.parameters['dec'],
            self.parameters['geocent_time'])
        ifo_time = self.parameters['geocent_time'] + dt - \
            interferometer.strain_data.start_time

        calib_linear = interferometer.calibration_model.get_calibration_factor(
            self.frequency_nodes_linear,
            prefix='recalib_{}_'.format(interferometer.name), **self.parameters)
        calib_quadratic = interferometer.calibration_model.get_calibration_factor(
            self.frequency_nodes_quadratic,
            prefix='recalib_{}_'.format(interferometer.name), **self.parameters)

        h_plus_linear = f_plus * waveform_polarizations['linear']['plus'] * calib_linear
        h_cross_linear = f_cross * waveform_polarizations['linear']['cross'] * calib_linear
        h_plus_quadratic = (
            f_plus * waveform_polarizations['quadratic']['plus'] * calib_quadratic)
        h_cross_quadratic = (
            f_cross * waveform_polarizations['quadratic']['cross'] * calib_quadratic)

        indices, in_bounds = self._closest_time_indices(
            ifo_time, self.weights['time_samples'])
        if not in_bounds:
            logger.debug("SNR calculation error: requested time at edge of ROQ time samples")
            return self._CalculatedSNRs(
                d_inner_h=np.nan_to_num(-np.inf), optimal_snr_squared=0,
                complex_matched_filter_snr=np.nan_to_num(-np.inf),
                d_inner_h_squared_tc_array=None)

        d_inner_h_tc_array = np.einsum(
            'i,ji->j', np.conjugate(h_plus_linear + h_cross_linear),
            self.weights[interferometer.name + '_linear'][indices])

        d_inner_h = interp1d(
            self.weights['time_samples'][indices],
            d_inner_h_tc_array, kind='cubic', assume_sorted=True)(ifo_time)

        optimal_snr_squared = \
            np.vdot(np.abs(h_plus_quadratic + h_cross_quadratic)**2,
                    self.weights[interferometer.name + '_quadratic'])

        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)
        d_inner_h_squared_tc_array = None

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h, optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_squared_tc_array=d_inner_h_squared_tc_array)

    @staticmethod
    def _closest_time_indices(time, samples):
        """
        Get the closest five times

        Parameters
        ----------
        time: float
            Time to check
        samples: array-like
            Available times

        Returns
        -------
        indices: list
            Indices nearest to time
        in_bounds: bool
            Whether the indices are for valid times
        """
        closest = np.argmin(abs(samples - time))
        indices = [closest + ii for ii in [-2, -1, 0, 1, 2]]
        in_bounds = (indices[0] >= 0) & (indices[-1] < samples.size)
        return indices, in_bounds

    def _set_weights(self, linear_matrix, quadratic_matrix):
        """ Setup the time-dependent ROQ weights.

        Parameters
        ----------
        linear_matrix, quadratic_matrix: array_like
            Arrays of the linear and quadratic basis

        """

        time_space = self._get_time_resolution()
        # Maximum delay time to geocentre + 5 steps
        earth_light_crossing_time = radius_of_earth / speed_of_light + 5 * time_space
        delta_times = np.arange(
            self.priors['geocent_time'].minimum - earth_light_crossing_time,
            self.priors['geocent_time'].maximum + earth_light_crossing_time,
            time_space)
        time_samples = delta_times - self.interferometers.start_time
        self.weights['time_samples'] = time_samples
        logger.info("Using {} ROQ time samples".format(len(time_samples)))

        for ifo in self.interferometers:
            if self.roq_params is not None:
                if ifo.maximum_frequency > self.roq_params['fhigh']:
                    raise ValueError("Requested maximum frequency larger than ROQ basis fhigh")
                # Generate frequencies for the ROQ
                roq_frequencies = create_frequency_series(
                    sampling_frequency=self.roq_params['fhigh'] * 2,
                    duration=self.roq_params['seglen'])
                roq_mask = roq_frequencies >= self.roq_params['flow']
                roq_frequencies = roq_frequencies[roq_mask]
                overlap_frequencies, ifo_idxs, roq_idxs = np.intersect1d(
                    ifo.frequency_array[ifo.frequency_mask], roq_frequencies,
                    return_indices=True)
            else:
                overlap_frequencies = ifo.frequency_array[ifo.frequency_mask]
                roq_idxs = np.arange(linear_matrix.shape[0], dtype=int)
                ifo_idxs = [True] * sum(ifo.frequency_mask)
                if sum(ifo_idxs) != len(roq_idxs):
                    raise ValueError(
                        "Mismatch between ROQ basis and frequency array for "
                        "{}".format(ifo.name))
            logger.info(
                "Building ROQ weights for {} with {} frequencies between {} "
                "and {}.".format(
                    ifo.name, len(overlap_frequencies),
                    min(overlap_frequencies), max(overlap_frequencies)))

            logger.debug("Preallocate array")
            tc_shifted_data = np.zeros((
                len(self.weights['time_samples']), len(overlap_frequencies)),
                dtype=complex)

            logger.debug("Calculate shifted data")
            data = ifo.frequency_domain_strain[ifo.frequency_mask][ifo_idxs]
            prefactor = (
                data /
                ifo.power_spectral_density_array[ifo.frequency_mask][ifo_idxs]
            )
            for j in range(len(self.weights['time_samples'])):
                tc_shifted_data[j] = prefactor * np.exp(
                    2j * np.pi * overlap_frequencies * time_samples[j])

            # to not kill all computers this minimises the memory usage of the
            # required inner products
            max_block_gigabytes = 4
            max_elements = int((max_block_gigabytes * 2 ** 30) / 8)

            logger.debug("Apply dot product")
            self.weights[ifo.name + '_linear'] = blockwise_dot_product(
                tc_shifted_data,
                linear_matrix[roq_idxs],
                max_elements) * 4 / ifo.strain_data.duration

            del tc_shifted_data, overlap_frequencies
            gc.collect()

            self.weights[ifo.name + '_quadratic'] = build_roq_weights(
                1 /
                ifo.power_spectral_density_array[ifo.frequency_mask][ifo_idxs],
                quadratic_matrix[roq_idxs].real,
                1 / ifo.strain_data.duration)

            logger.info("Finished building weights for {}".format(ifo.name))

    def save_weights(self, filename, format='npz'):
        if format not in filename:
            filename += "." + format
        logger.info("Saving ROQ weights to {}".format(filename))
        if format == 'json':
            with open(filename, 'w') as file:
                json.dump(self.weights, file, indent=2, cls=BilbyJsonEncoder)
        elif format == 'npz':
            np.savez(filename, **self.weights)

    @staticmethod
    def load_weights(filename, format=None):
        if format is None:
            format = filename.split(".")[-1]
        if format not in ["json", "npz"]:
            raise IOError("Format {} not recongized.".format(format))
        logger.info("Loading ROQ weights from {}".format(filename))
        if format == "json":
            with open(filename, 'r') as file:
                weights = json.load(file, object_hook=decode_bilby_json)
        elif format == "npz":
            # Wrap in dict to load data into memory
            weights = dict(np.load(filename))
        return weights

    def _get_time_resolution(self):
        """
        This method estimates the time resolution given the optimal SNR of the
        signal in the detector. This is then used when constructing the weights
        for the ROQ.

        A minimum resolution is set by assuming the SNR in each detector is at
        least 10. When the SNR is not available the SNR is assumed to be 30 in
        each detector.

        Returns
        -------
        delta_t: float
            Time resolution
        """

        def calc_fhigh(freq, psd, scaling=20.):
            """

            Parameters
            ----------
            freq: array-like
                Frequency array
            psd: array-like
                Power spectral density
            scaling: float
                SNR dependent scaling factor

            Returns
            -------
            f_high: float
                The maximum frequency which must be considered
            """
            integrand1 = np.power(freq, -7. / 3) / psd
            integral1 = integrate.simps(integrand1, freq)
            integrand3 = np.power(freq, 2. / 3.) / (psd * integral1)
            f_3_bar = integrate.simps(integrand3, freq)

            f_high = scaling * f_3_bar**(1 / 3)

            return f_high

        def c_f_scaling(snr):
            return (np.pi**2 * snr**2 / 6)**(1 / 3)

        inj_snr_sq = 0
        for ifo in self.interferometers:
            inj_snr_sq += min(10, getattr(ifo.meta_data, 'optimal_SNR', 30))**2

        psd = ifo.power_spectral_density_array[ifo.frequency_mask]
        freq = ifo.frequency_array[ifo.frequency_mask]
        fhigh = calc_fhigh(freq, psd, scaling=c_f_scaling(inj_snr_sq**0.5))

        delta_t = fhigh**-1

        # Apply a safety factor to ensure the time step is short enough
        delta_t = delta_t / 5

        logger.info("ROQ time-step = {}".format(delta_t))
        return delta_t

    def _rescale_signal(self, signal, new_distance):
        for kind in ['linear', 'quadratic']:
            for mode in signal[kind]:
                signal[kind][mode] *= self._ref_dist / new_distance


def get_binary_black_hole_likelihood(interferometers):
    """ A rapper to quickly set up a likelihood for BBH parameter estimation

    Parameters
    ----------
    interferometers: {bilby.gw.detector.InterferometerList, list}
        A list of `bilby.detector.Interferometer` instances, typically the
        output of either `bilby.detector.get_interferometer_with_open_data`
        or `bilby.detector.get_interferometer_with_fake_noise_and_injection`

    Returns
    -------
    bilby.GravitationalWaveTransient: The likelihood to pass to `run_sampler`

    """
    waveform_generator = WaveformGenerator(
        duration=interferometers.duration,
        sampling_frequency=interferometers.sampling_frequency,
        frequency_domain_source_model=lal_binary_black_hole,
        waveform_arguments={'waveform_approximant': 'IMRPhenomPv2',
                            'reference_frequency': 50})
    return GravitationalWaveTransient(interferometers, waveform_generator)
