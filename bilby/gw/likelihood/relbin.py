import numpy as np
from .base import GravitationalWaveTransient
from ...core.utils import logger, create_time_series
from ...core.prior.base import Constraint
from ...core.prior import Interped, DeltaFunction
from ..utils import noise_weighted_inner_product, ln_i0
from scipy.optimize import differential_evolution
import copy


class RelativeBinningGravitationalWaveTransient(GravitationalWaveTransient):
    """A gravitational-wave transient likelihood object which uses the relative
    binning procedure to calculate a fast likelihood. See IAS paper:


    Parameters
    ----------
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal,
        given some set of parameters
    fiducial_parameters: dict, optional
        A starting guess for initial parameters of the event for finding the
        maximum likelihood (fiducial) waveform.
    parameter_bounds: dict, optional
        Dictionary of bounds (lists) for the initial parameters when finding
        the initial maximum likelihood (fiducial) waveform.
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
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location.
        - "sky": sample in RA/dec, this is the default
        - e.g., "H1L1", ["H1", "L1"], InterferometerList(["H1", "L1"]):
          sample in azimuth and zenith, `azimuth` and `zenith` defined in the
          frame where the z-axis is aligned the the vector connecting H1
          and L1.
    time_reference: str, optional
        Name of the reference for the sampled time parameter.
        - "geocent"/"geocenter": sample in the time at the Earth's center,
          this is the default
        - e.g., "H1": sample in the time of arrival at H1
    chi: float, optional
        Tunable parameter which limits the perturbation of alpha when setting
        up the bin range. See https://arxiv.org/abs/1806.08792.
    epsilon: float, optional
        Tunable parameter which limits the differential phase change in each
        bin when setting up the bin range. See https://arxiv.org/abs/1806.08792.

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters.
    """

    def __init__(self, interferometers,
                 waveform_generator,
                 fiducial_parameters={}, parameter_bounds=None,
                 maximization_kwargs=dict(),
                 update_fiducial_parameters=False,
                 distance_marginalization=False,
                 time_marginalization=False,
                 phase_marginalization=False,
                 priors=None,
                 distance_marginalization_lookup_table=None,
                 jitter_time=True,
                 reference_frame="sky",
                 time_reference="geocenter",
                 chi=1,
                 epsilon=0.5):

        super(RelativeBinningGravitationalWaveTransient, self).__init__(
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

        self.fiducial_parameters = fiducial_parameters
        self.chi = chi
        self.epsilon = epsilon
        self.gamma = np.array([-5 / 3, -2 / 3, 1, 5 / 3, 7 / 3])
        self.fiducial_waveform_obtained = False
        self.check_if_bins_are_setup = False
        self.fiducial_polarizations = None
        self.per_detector_fiducial_waveforms = {}
        self.bin_freqs = dict()
        self.bin_inds = dict()
        self.set_fiducial_waveforms(self.fiducial_parameters)
        logger.info("Initial fiducial waveforms set up")
        self.setup_bins()
        self.compute_summary_data()
        logger.info("Summary Data Obtained")

        if update_fiducial_parameters:
            # write a check to make sure prior is not None
            logger.info("Using scipy optimization to find maximum likelihood parameters.")
            self.parameters_to_be_updated = [key for key in self.priors if not isinstance(
                self.priors[key], (DeltaFunction, Constraint))]
            logger.info("Parameters over which likelihood is maximized: {}".format(self.parameters_to_be_updated))
            if parameter_bounds is None:
                logger.info("No parameter bounds were given. Using priors instead.")
                self.parameter_bounds = self.get_bounds_from_priors(self.priors)
            else:
                self.parameter_bounds = self.get_parameter_list_from_dictionary(parameter_bounds)
            self.fiducial_parameters = self.find_maximum_likelihood_parameters(
                self.parameter_bounds, maximization_kwargs=maximization_kwargs)

    def __repr__(self):
        return self.__class__.__name__ + '(interferometers={},\n\twaveform_generator={},\n\fiducial_parameters={},' \
            .format(self.interferometers, self.waveform_generator, self.fiducial_parameters)

    def setup_bins(self):
        frequency_array = self.waveform_generator.frequency_array
        gamma = self.gamma
        for interferometer in self.interferometers:
            frequency_array_useful = frequency_array[np.intersect1d(
                np.where(frequency_array >= interferometer.minimum_frequency),
                np.where(frequency_array <= interferometer.maximum_frequency))]

            d_alpha = self.chi * 2 * np.pi / np.abs(
                (interferometer.minimum_frequency ** gamma) * np.heaviside(
                    -gamma, 1) - (interferometer.maximum_frequency ** gamma) * np.heaviside(
                    gamma, 1))
            d_phi = np.sum(np.array([np.sign(gamma[i]) * d_alpha[i] * (
                frequency_array_useful ** gamma[i]) for i in range(len(gamma))]), axis=0)
            d_phi_from_start = d_phi - d_phi[0]
            self.number_of_bins = int(d_phi_from_start[-1] // self.epsilon)
            self.bin_freqs[interferometer.name] = np.zeros(self.number_of_bins + 1)
            self.bin_inds[interferometer.name] = np.zeros(self.number_of_bins + 1, dtype=np.int)

            for i in range(self.number_of_bins + 1):
                bin_index = np.where(d_phi_from_start >= ((i / self.number_of_bins) * d_phi_from_start[-1]))[0][0]
                bin_freq = frequency_array_useful[bin_index]
                self.bin_freqs[interferometer.name][i] = bin_freq
                self.bin_inds[interferometer.name][i] = np.where(frequency_array >= bin_freq)[0][0]

            logger.info("Set up {} bins for {} between {} Hz and {} Hz".format(
                self.number_of_bins, interferometer.name, interferometer.minimum_frequency,
                interferometer.maximum_frequency))
            self.waveform_generator.waveform_arguments["frequency_bin_edges"] = self.bin_freqs[interferometer.name]
        return

    def set_fiducial_waveforms(self, parameters):
        parameters["fiducial"] = 1
        self.fiducial_polarizations = self.waveform_generator.frequency_domain_strain(
            parameters)

        maximum_nonzero_index = np.where(self.fiducial_polarizations["plus"] != 0j)[0][-1]
        logger.info("Maximum Nonzero Index is {}".format(maximum_nonzero_index))
        maximum_nonzero_frequency = self.waveform_generator.frequency_array[maximum_nonzero_index]
        logger.info("Maximum Nonzero Frequency is {}".format(maximum_nonzero_frequency))

        if self.fiducial_polarizations is None:
            return np.nan_to_num(-np.inf)

        for interferometer in self.interferometers:
            logger.info("Maximum Frequency is {}".format(interferometer.maximum_frequency))
            if interferometer.maximum_frequency > maximum_nonzero_frequency:
                interferometer.maximum_frequency = maximum_nonzero_frequency

            self.per_detector_fiducial_waveforms[interferometer.name] = (
                interferometer.get_detector_response(
                    self.fiducial_polarizations, parameters))
        parameters["fiducial"] = 0
        return

    def log_likelihood(self):
        return self.log_likelihood_ratio() + self.noise_log_likelihood()

    def log_likelihood_ratio(self):
        self.parameters.update(self.get_sky_frame_parameters())

        d_inner_h = 0.
        optimal_snr_squared = 0.
        complex_matched_filter_snr = 0.
        waveform_ratio = self.compute_waveform_ratio(self.parameters)

        if self.time_marginalization:
            if self.jitter_time:
                self.parameters['geocent_time'] += self.parameters['time_jitter']
            d_inner_h_tc_array = np.zeros(
                self.interferometers.frequency_array[0:-1].shape,
                dtype=np.complex128)

        for interferometer in self.interferometers:
            per_detector_snr = self.calculate_snrs_relative_binning(
                waveform_ratio[interferometer.name], interferometer)
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

    def find_maximum_likelihood_parameters(self, parameter_bounds,
                                           iterations=1, maximization_kwargs=dict()):

        for i in range(iterations):
            logger.info("Optimizing fiducial parameters. Iteration : {}".format(i + 1))
            output = differential_evolution(self.lnlike_scipy_maximize,
                                            bounds=parameter_bounds, **maximization_kwargs)
            updated_parameters_list = output['x']
            updated_parameters = self.get_parameter_dictionary_from_list(updated_parameters_list)
            self.set_fiducial_waveforms(updated_parameters)
            self.compute_summary_data()

        logger.info("Fiducial waveforms updated")
        logger.info("Summary Data updated")
        return updated_parameters

    def lnlike_scipy_maximize(self, parameter_list):
        self.parameters = self.get_parameter_dictionary_from_list(parameter_list)
        return -self.log_likelihood_ratio()

    def get_parameter_dictionary_from_list(self, parameter_list):
        parameter_dictionary = dict(zip(self.parameters_to_be_updated, parameter_list))
        excluded_parameter_keys = set(self.fiducial_parameters) - set(self.parameters_to_be_updated)
        for key in excluded_parameter_keys:
            parameter_dictionary[key] = self.fiducial_parameters[key]
        return parameter_dictionary

    def get_parameter_list_from_dictionary(self, parameter_dict):
        return [parameter_dict[k] for k in self.parameters_to_be_updated]

    def get_bounds_from_priors(self, priors):
        bounds = []
        for key in self.parameters_to_be_updated:
            bounds.append([priors[key].minimum, priors[key].maximum])
        return bounds

    def compute_summary_data(self):
        summary_data = dict()

        for interferometer in self.interferometers:
            mask = interferometer.frequency_mask
            masked_frequency_array = interferometer.frequency_array[mask]
            masked_bin_inds = []
            for edge in self.bin_freqs[interferometer.name]:
                index = np.where(masked_frequency_array == edge)[0][0]
                masked_bin_inds.append(index)
            masked_strain = interferometer.frequency_domain_strain[mask]
            masked_h0 = self.per_detector_fiducial_waveforms[interferometer.name][mask]
            masked_psd = interferometer.power_spectral_density_array[mask]
            a0, b0, a1, b1 = np.zeros((4, self.number_of_bins), dtype=np.complex)

            for i in range(self.number_of_bins):

                central_frequency_i = 0.5 * \
                    (masked_frequency_array[masked_bin_inds[i]] + masked_frequency_array[masked_bin_inds[i + 1]])
                masked_strain_i = masked_strain[masked_bin_inds[i]:masked_bin_inds[i + 1]]
                masked_h0_i = masked_h0[masked_bin_inds[i]:masked_bin_inds[i + 1]]
                masked_psd_i = masked_psd[masked_bin_inds[i]:masked_bin_inds[i + 1]]
                masked_frequency_i = masked_frequency_array[masked_bin_inds[i]:masked_bin_inds[i + 1]]

                a0[i] = noise_weighted_inner_product(
                    masked_h0_i,
                    masked_strain_i,
                    masked_psd_i,
                    self.waveform_generator.duration)

                b0[i] = noise_weighted_inner_product(
                    masked_h0_i,
                    masked_h0_i,
                    masked_psd_i,
                    self.waveform_generator.duration)

                a1[i] = noise_weighted_inner_product(
                    masked_h0_i,
                    masked_strain_i * (masked_frequency_i - central_frequency_i),
                    masked_psd_i,
                    self.waveform_generator.duration)

                b1[i] = noise_weighted_inner_product(
                    masked_h0_i,
                    masked_h0_i * (masked_frequency_i - central_frequency_i),
                    masked_psd_i,
                    self.waveform_generator.duration)

            summary_data[interferometer.name] = dict(a0=a0, a1=a1, b0=b0, b1=b1)

        self.summary_data = summary_data

    def compute_waveform_ratio(self, parameters):
        waveform_ratio = dict()
        self.waveform_generator.parameters = parameters
        new_polarizations = self.waveform_generator.frequency_domain_strain(parameters)
        for interferometer in self.interferometers:
            h = interferometer.get_detector_response_relative_binning(
                new_polarizations, parameters, self.bin_freqs[interferometer.name])
            h0 = self.per_detector_fiducial_waveforms[interferometer.name][self.bin_inds[interferometer.name]]
            waveform_ratio_per_detector = h / h0

            r0 = (waveform_ratio_per_detector[1:] + waveform_ratio_per_detector[:-1]) / 2
            r1 = (waveform_ratio_per_detector[1:] - waveform_ratio_per_detector[:-1]) / (
                self.bin_freqs[interferometer.name][1:] - self.bin_freqs[interferometer.name][:-1])

            waveform_ratio[interferometer.name] = [r0, r1]

        return waveform_ratio

    def calculate_snrs_relative_binning(self, waveform_ratio_per_detector, interferometer):
        summary_data_per_interferometer = self.summary_data[interferometer.name]
        a0 = summary_data_per_interferometer["a0"]
        a1 = summary_data_per_interferometer["a1"]
        b0 = summary_data_per_interferometer["b0"]
        b1 = summary_data_per_interferometer["b1"]

        r0, r1 = waveform_ratio_per_detector

        d_inner_h = np.sum(a0 * np.conjugate(r0) + a1 * np.conjugate(r1))
        h_inner_h = np.sum(b0 * np.abs(r0) ** 2 + 2 * b1 * np.real(
            r0 * np.conjugate(r1)))
        optimal_snr_squared = h_inner_h
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared ** 0.5)

        if self.time_marginalization:
            f = interferometer.frequency_array
            duplicated_r0, duplicated_r1, duplicated_fm = np.zeros((3, f.shape[0]), dtype=np.complex)

            for i in range(self.number_of_bins):
                ind = self.bin_inds[interferometer.name]
                fm = 0.5 * (self.bin_freqs[interferometer.name][1:] + self.bin_freqs[interferometer.name][:-1])
                duplicated_fm[ind[i]:ind[i + 1]] = fm[i]
                duplicated_r0[ind[i]:ind[i + 1]] = r0[i]
                duplicated_r1[ind[i]:ind[i + 1]] = r1[i]

            duplicated_r = duplicated_r0 + duplicated_r1 * (f - duplicated_fm)

            d_inner_h_squared_tc_array = 4 / self.waveform_generator.duration * np.fft.fft(
                self.per_detector_fiducial_waveforms[interferometer.name][0:-1] *
                interferometer.frequency_domain_strain.conjugate()[0:-1] *
                duplicated_r[0:-1] / interferometer.power_spectral_density_array[0:-1])

        else:
            d_inner_h_squared_tc_array = None

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h, optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=None, optimal_snr_squared_array=None,
            d_inner_h_squared_tc_array=d_inner_h_squared_tc_array)

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
            waveform_ratio = copy.deepcopy(
                self.compute_waveform_ratio(self.parameters))
        else:
            return self.parameters
        if self.time_marginalization:
            new_time = self.generate_time_sample_from_marginalized_likelihood(
                waveform_ratio=waveform_ratio)
            self.parameters['geocent_time'] = new_time
        if self.distance_marginalization:
            new_distance = self.generate_distance_sample_from_marginalized_likelihood(
                waveform_ratio=waveform_ratio)
            self.parameters['luminosity_distance'] = new_distance
        if self.phase_marginalization:
            new_phase = self.generate_phase_sample_from_marginalized_likelihood(
                waveform_ratio=waveform_ratio)
            self.parameters['phase'] = new_phase
        return self.parameters.copy()

    def generate_time_sample_from_marginalized_likelihood(
            self, waveform_ratio=None):
        """
        Generate a single sample from the posterior distribution for coalescence
        time when using a likelihood which explicitly marginalises over time.

        In order to resolve the posterior we artifically upsample to 16kHz.

        See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

        Parameters
        ----------
        waveform_ratio: dict, optional
            contains waveform ratios for all interferometers.

        Returns
        -------
        new_time: float
            Sample from the time posterior.
        """
        self.parameters.update(self.get_sky_frame_parameters())
        if self.jitter_time:
            self.parameters['geocent_time'] += self.parameters['time_jitter']
        if waveform_ratio is None:
            waveform_ratio = \
                self.compute_waveform_ratio(self.parameters)

        times = create_time_series(
            sampling_frequency=16384,
            starting_time=self.parameters['geocent_time'] - self.waveform_generator.start_time,
            duration=self.waveform_generator.duration)
        times = times % self.waveform_generator.duration
        times += self.waveform_generator.start_time

        prior = self.priors["geocent_time"]
        in_prior = (times >= prior.minimum) & (times < prior.maximum)
        times = times[in_prior]

        n_time_steps = int(self.waveform_generator.duration * 16384)
        d_inner_h = np.zeros(len(times), dtype=np.complex)
        psd = np.ones(n_time_steps)
        signal_long = np.zeros(n_time_steps, dtype=np.complex)
        data = np.zeros(n_time_steps, dtype=np.complex)
        h_inner_h = np.zeros(1)
        for ifo in self.interferometers:
            f = ifo.frequency_array
            ifo_length = len(f)
            mask = ifo.frequency_mask
            r0, r1 = waveform_ratio[ifo.name]
            duplicated_r0, duplicated_r1, duplicated_fm = np.zeros((3, ifo_length), dtype=np.complex)
            for i in range(self.number_of_bins):
                ind = self.bin_inds[ifo.name]
                fm = 0.5 * (self.bin_freqs[ifo.name][1:] + self.bin_freqs[ifo.name][:-1])
                duplicated_fm[ind[i]:ind[i + 1]] = fm[i]
                duplicated_r0[ind[i]:ind[i + 1]] = r0[i]
                duplicated_r1[ind[i]:ind[i + 1]] = r1[i]
            duplicated_r = duplicated_r0 + duplicated_r1 * (f - duplicated_fm)
            signal = duplicated_r * self.per_detector_fiducial_waveforms[ifo.name]
            signal_long[:ifo_length] = signal
            data[:ifo_length] = np.conj(ifo.frequency_domain_strain)
            psd[:ifo_length][mask] = ifo.power_spectral_density_array[mask]
            d_inner_h += np.fft.fft(signal_long * data / psd)[in_prior]
            h_inner_h += ifo.optimal_snr_squared(signal=signal).real

        if self.distance_marginalization:
            time_log_like = self.distance_marginalized_likelihood(
                d_inner_h, h_inner_h)
        elif self.phase_marginalization:
            time_log_like = (ln_i0(abs(d_inner_h)) -
                             h_inner_h.real / 2)
        else:
            time_log_like = (d_inner_h.real - h_inner_h.real / 2)

        time_prior_array = self.priors['geocent_time'].prob(times)
        time_post = (
            np.exp(time_log_like - max(time_log_like)) * time_prior_array)

        keep = (time_post > max(time_post) / 1000)
        if sum(keep) < 3:
            keep[1:-1] = keep[1:-1] | keep[2:] | keep[:-2]
        time_post = time_post[keep]
        times = times[keep]

        new_time = Interped(times, time_post).sample()
        return new_time

    def generate_distance_sample_from_marginalized_likelihood(
            self, waveform_ratio=None):
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
        self.parameters.update(self.get_sky_frame_parameters())
        if waveform_ratio is None:
            waveform_ratio = \
                self.compute_waveform_ratio(self.parameters)
        d_inner_h, h_inner_h = self._calculate_inner_products(waveform_ratio)

        d_inner_h_dist = (
            d_inner_h * self.parameters['luminosity_distance'] /
            self._distance_array)

        h_inner_h_dist = (
            h_inner_h * self.parameters['luminosity_distance']**2 /
            self._distance_array**2)

        if self.phase_marginalization:
            distance_log_like = (
                ln_i0(abs(d_inner_h_dist)) -
                h_inner_h_dist.real / 2)
        else:
            distance_log_like = (d_inner_h_dist.real - h_inner_h_dist.real / 2)

        distance_post = (np.exp(distance_log_like - max(distance_log_like)) *
                         self.distance_prior_array)

        new_distance = Interped(
            self._distance_array, distance_post).sample()

        self._rescale_signal(waveform_ratio, new_distance)
        return new_distance

    def _rescale_signal(self, waveform_ratio, new_distance):
        for ifo in self.interferometers:
            for i in range(2):
                waveform_ratio[ifo.name][i] *= self._ref_dist / new_distance

    def _calculate_inner_products(self, waveform_ratio):
        d_inner_h = 0
        h_inner_h = 0
        for interferometer in self.interferometers:
            waveform_ratio_per_detector = waveform_ratio[interferometer.name]
            per_detector_snr = self.calculate_snrs_relative_binning(
                waveform_ratio_per_detector, interferometer)

            d_inner_h += per_detector_snr.d_inner_h
            h_inner_h += per_detector_snr.optimal_snr_squared
        return d_inner_h, h_inner_h

    def generate_phase_sample_from_marginalized_likelihood(
            self, waveform_ratio=None):
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
        self.parameters.update(self.get_sky_frame_parameters())
        if waveform_ratio is None:
            waveform_ratio = \
                self.compute_waveform_ratio(self.parameters)
        d_inner_h, h_inner_h = self._calculate_inner_products(waveform_ratio)

        phases = np.linspace(0, 2 * np.pi, 101)
        phasor = np.exp(-2j * phases)
        phase_log_post = d_inner_h * phasor - h_inner_h / 2
        phase_post = np.exp(phase_log_post.real - max(phase_log_post.real))
        new_phase = Interped(phases, phase_post).sample()
        return new_phase
