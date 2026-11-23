from copy import deepcopy
from itertools import combinations_with_replacement

import numpy as np
import scipy.interpolate
import scipy.sparse
from scipy.optimize import differential_evolution

from .base import GravitationalWaveTransient
from ...compat.utils import array_module
from ...core.utils import logger, solar_mass
from ...core.prior.base import Constraint
from ...core.prior import DeltaFunction
from ..conversion import (
    bilby_to_lalsimulation_spins,
    chirp_mass_and_mass_ratio_to_component_masses,
    component_masses_to_chirp_mass,
    component_masses_to_symmetric_mass_ratio,
    symmetric_mass_ratio_to_mass_ratio,
)
from ..utils import noise_weighted_inner_product


class RelativeBinningGravitationalWaveTransient(GravitationalWaveTransient):
    """A gravitational-wave transient likelihood object which uses the relative
    binning procedure to calculate a fast likelihood. See Zackay et al.
    arXiv1806.08792

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
        maximum likelihood (fiducial) waveform. These should be specified in
        the same parameter basis as the one that sampling is carried out in.
        For example, if sampling in `mass_1` and `mass_2`, the fiducial
        parameters should also be provided in `mass_1` and `mass_2.`
        If no fiducial parameters are passed, a random point will be drawn from
        the prior.
    update_fiducial_parameters: bool
        Whether to try to update the fiducial parameters by finding the maximum
        likelihood point using scipy differential evolution. Default is
        :code:`False`.
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
          frame where the z-axis is aligned with the vector connecting H1
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

    Notes
    -----
    The relative binning likelihood does not currently support calibration marginalization.
    """

    def __init__(
        self,
        interferometers,
        waveform_generator,
        fiducial_parameters=None,
        parameter_bounds=None,
        maximization_kwargs=None,
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
        epsilon=0.5,
    ):
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
            time_reference=time_reference,
        )

        if fiducial_parameters is None:
            logger.info("Drawing fiducial parameters from prior.")
            fiducial_parameters = priors.sample()
        self.fiducial_parameters = fiducial_parameters.copy()
        self._unset_fiducial()
        if self.time_marginalization:
            self.fiducial_parameters["geocent_time"] = interferometers.start_time
        if self.distance_marginalization:
            self.fiducial_parameters["luminosity_distance"] = self._ref_dist
        if self.phase_marginalization:
            self.fiducial_parameters["phase"] = 0.0
        self.chi = chi
        self.epsilon = epsilon
        self.gamma = np.array([-5 / 3, -2 / 3, 1, 5 / 3, 7 / 3])
        self.maximum_frequency = waveform_generator.frequency_array[-1]
        self.fiducial_waveform_obtained = False
        self.check_if_bins_are_setup = False
        self.fiducial_polarizations = None
        self.per_detector_fiducial_waveforms = dict()
        self.per_detector_fiducial_waveform_points = dict()
        self.set_fiducial_waveforms(self.fiducial_parameters)
        logger.info("Initial fiducial waveforms set up")
        self.setup_bins()
        self.compute_summary_data()
        logger.info("Summary Data Obtained")

        if update_fiducial_parameters:
            # write a check to make sure prior is not None
            logger.info(
                "Using scipy optimization to find maximum likelihood parameters."
            )
            self.parameters_to_be_updated = [
                key
                for key in priors
                if not isinstance(priors[key], (DeltaFunction, Constraint, float, int))
            ]
            logger.info(
                f"Parameters over which likelihood is maximized: {self.parameters_to_be_updated}"
            )
            if parameter_bounds is None:
                logger.info("No parameter bounds were given. Using priors instead.")
                self.parameter_bounds = self.get_bounds_from_priors(priors)
            else:
                self.parameter_bounds = self.get_parameter_list_from_dictionary(
                    parameter_bounds
                )
            self.fiducial_parameters = self.find_maximum_likelihood_parameters(
                self.parameter_bounds, maximization_kwargs=maximization_kwargs
            )
        logger.info(
            f"Fiducial likelihood: {self.log_likelihood_ratio(self.fiducial_parameters):.2f}"
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(interferometers={},\n\twaveform_generator={},\n\fiducial_parameters={},".format(
                self.interferometers, self.waveform_generator, self.fiducial_parameters
            )
        )

    def setup_bins(self):
        """
        Setup the frequency bins following the method in
        https://arxiv.org/abs/1806.08792.

        If :code:`epsilon` is too small, the naive bins can be smaller than
        the frequency spacing of the data. We require that bins are at least
        as wide as this spacing.
        """
        frequency_array = self.waveform_generator.frequency_array
        gamma = self.gamma[:, np.newaxis]
        maximum_frequency = frequency_array[0]
        minimum_frequency = frequency_array[-1]
        for interferometer in self.interferometers:
            maximum_frequency = max(maximum_frequency, interferometer.maximum_frequency)
            minimum_frequency = min(minimum_frequency, interferometer.minimum_frequency)
        maximum_frequency = min(maximum_frequency, self.maximum_frequency)
        frequency_array_useful = frequency_array[
            (frequency_array >= minimum_frequency)
            & (frequency_array <= maximum_frequency)
        ]

        d_alpha = (
            self.chi
            * 2
            * np.pi
            / np.abs(
                (minimum_frequency**gamma) * np.heaviside(-gamma, 1)
                - (maximum_frequency**gamma) * np.heaviside(gamma, 1)
            )
        )
        d_phi = np.sum(np.sign(gamma) * d_alpha * frequency_array_useful**gamma, axis=0)
        d_phi_from_start = d_phi - d_phi[0]
        number_of_bins = int(d_phi_from_start[-1] // self.epsilon)
        bin_inds = list()
        bin_freqs = list()

        last_index = -1
        for i in range(number_of_bins + 1):
            bin_index = np.where(
                d_phi_from_start >= ((i / number_of_bins) * d_phi_from_start[-1])
            )[0][0]
            if bin_index == last_index:
                continue
            bin_freq = frequency_array_useful[bin_index]
            last_index = bin_index
            bin_index = np.where(frequency_array >= bin_freq)[0][0]
            bin_inds.append(bin_index)
            bin_freqs.append(bin_freq)
        self.bin_inds = np.array(bin_inds, dtype=int)
        self.bin_sizes = np.diff(bin_inds)
        self.bin_sizes[-1] += 1
        self.bin_freqs = np.array(bin_freqs)
        self.number_of_bins = len(self.bin_inds) - 1
        logger.debug(
            f"Set up {self.number_of_bins} bins "
            f"between {minimum_frequency} Hz and {maximum_frequency} Hz"
        )
        self.waveform_generator.waveform_arguments["frequency_bin_edges"] = (
            self.bin_freqs
        )
        self.bin_widths = self.bin_freqs[1:] - self.bin_freqs[:-1]
        self.bin_centers = (self.bin_freqs[1:] + self.bin_freqs[:-1]) / 2
        for interferometer in self.interferometers:
            name = interferometer.name
            self.per_detector_fiducial_waveform_points[name] = (
                self.per_detector_fiducial_waveforms[name][self.bin_inds]
            )

    def set_fiducial_waveforms(self, parameters):
        parameters = parameters.copy()
        self._set_fiducial()
        parameters.update(self.get_sky_frame_parameters(parameters=parameters))
        self.fiducial_polarizations = self.waveform_generator.frequency_domain_strain(
            parameters
        )
        self._unset_fiducial()

        maximum_nonzero_index = np.where(self.fiducial_polarizations["plus"] != 0j)[0][
            -1
        ]
        logger.debug(f"Maximum Nonzero Index is {maximum_nonzero_index}")
        maximum_nonzero_frequency = self.waveform_generator.frequency_array[
            maximum_nonzero_index
        ]
        logger.debug(f"Maximum Nonzero Frequency is {maximum_nonzero_frequency}")
        self.maximum_frequency = maximum_nonzero_frequency

        if self.fiducial_polarizations is None:
            raise ValueError(f"Cannot compute fiducial waveforms for {parameters}")

        for interferometer in self.interferometers:
            logger.debug(f"Maximum Frequency is {interferometer.maximum_frequency}")
            wf = interferometer.get_detector_response(
                self.fiducial_polarizations, parameters
            )
            wf *= interferometer.frequency_array <= self.maximum_frequency
            self.per_detector_fiducial_waveforms[interferometer.name] = wf

    def find_maximum_likelihood_parameters(
        self, parameter_bounds, iterations=5, maximization_kwargs=None
    ):
        if maximization_kwargs is None:
            maximization_kwargs = dict()
        parameters = deepcopy(self.fiducial_parameters)
        self._unset_fiducial()
        updated_parameters_list = self.get_parameter_list_from_dictionary(
            self.fiducial_parameters
        )
        old_fiducial_ln_likelihood = self.log_likelihood_ratio(self.fiducial_parameters)
        logger.info(f"Fiducial ln likelihood ratio: {old_fiducial_ln_likelihood:.2f}")
        for it in range(iterations):
            logger.info(f"Optimizing fiducial parameters. Iteration : {it + 1}")
            output = differential_evolution(
                self.lnlike_scipy_maximize,
                bounds=parameter_bounds,
                args=(parameters,),
                x0=updated_parameters_list,
                **maximization_kwargs,
            )
            updated_parameters_list = output["x"]
            updated_parameters = deepcopy(self.fiducial_parameters)
            updated_parameters.update(
                self.get_parameter_dictionary_from_list(updated_parameters_list)
            )
            self.set_fiducial_waveforms(updated_parameters)
            self.setup_bins()
            self.compute_summary_data()
            new_fiducial_ln_likelihood = self.log_likelihood_ratio(updated_parameters)
            logger.info(
                f"Fiducial ln likelihood ratio: {new_fiducial_ln_likelihood:.2f}"
            )
            if new_fiducial_ln_likelihood - old_fiducial_ln_likelihood < 0.1:
                break
            old_fiducial_ln_likelihood = new_fiducial_ln_likelihood

        logger.info("Fiducial waveforms updated")
        logger.info("Summary Data updated")
        return updated_parameters

    def lnlike_scipy_maximize(self, parameter_list, parameters):
        parameters.update(self.get_parameter_dictionary_from_list(parameter_list))
        return -self.log_likelihood_ratio(parameters)

    def get_parameter_dictionary_from_list(self, parameter_list):
        parameter_dictionary = dict(zip(self.parameters_to_be_updated, parameter_list))
        excluded_parameter_keys = set(self.fiducial_parameters) - set(
            self.parameters_to_be_updated
        )
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
            for edge in self.bin_freqs:
                index = np.where(masked_frequency_array == edge)[0][0]
                masked_bin_inds.append(index)
            # For the last bin, make sure to include
            # the last point in the frequency array,
            # if it's not already included
            if masked_bin_inds[-1] < len(masked_frequency_array) - 1:
                masked_bin_inds[-1] += 1

            masked_strain = interferometer.frequency_domain_strain[mask]
            masked_h0 = np.asarray(
                self.per_detector_fiducial_waveforms[interferometer.name][mask]
            )
            masked_psd = interferometer.power_spectral_density_array[mask]
            duration = interferometer.duration
            a0, b0, a1, b1 = np.zeros((4, self.number_of_bins), dtype=complex)

            for i in range(self.number_of_bins):
                start_idx = masked_bin_inds[i]
                end_idx = masked_bin_inds[i + 1]
                start = masked_frequency_array[start_idx]
                stop = masked_frequency_array[end_idx]
                idxs = slice(start_idx, end_idx)

                strain = masked_strain[idxs]
                h0 = masked_h0[idxs]
                psd = masked_psd[idxs]

                frequencies = masked_frequency_array[idxs]
                central_frequency = (start + stop) / 2
                delta_frequency = frequencies - central_frequency

                a0[i] = noise_weighted_inner_product(h0, strain, psd, duration)
                b0[i] = noise_weighted_inner_product(h0, h0, psd, duration)
                a1[i] = noise_weighted_inner_product(
                    h0, strain * delta_frequency, psd, duration
                )
                b1[i] = noise_weighted_inner_product(
                    h0, h0 * delta_frequency, psd, duration
                )

            summary_data[interferometer.name] = (a0, a1, b0, b1)

        self.summary_data = summary_data

    def compute_waveform_ratio_per_interferometer(
        self, waveform_polarizations, interferometer, parameters
    ):
        name = interferometer.name
        strain = interferometer.get_detector_response(
            waveform_polarizations=waveform_polarizations,
            parameters=parameters,
            frequencies=self.bin_freqs,
        )
        reference_strain = self.per_detector_fiducial_waveform_points[name]
        waveform_ratio = strain / reference_strain

        r0 = (waveform_ratio[1:] + waveform_ratio[:-1]) / 2
        r1 = (waveform_ratio[1:] - waveform_ratio[:-1]) / self.bin_widths

        return [r0, r1]

    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters):
        fiducial_waveform = self.per_detector_fiducial_waveforms[interferometer.name]
        r0, r1 = self.compute_waveform_ratio_per_interferometer(
            waveform_polarizations=signal_polarizations,
            interferometer=interferometer,
            parameters=parameters,
        )

        idxs = slice(self.bin_inds[0], self.bin_inds[-1] + 1)
        duplicated_r0 = np.repeat(r0, self.bin_sizes)
        duplicated_r1 = np.repeat(r1, self.bin_sizes)
        duplicated_fm = np.repeat(self.bin_centers, self.bin_sizes)

        f = interferometer.frequency_array
        full_waveform_ratio = np.zeros(f.shape[0], dtype=complex)
        full_waveform_ratio[idxs] = duplicated_r0 + duplicated_r1 * (
            f[idxs] - duplicated_fm
        )
        return fiducial_waveform * full_waveform_ratio

    def calculate_snrs(
        self, waveform_polarizations, interferometer, *, return_array=True, parameters
    ):
        r0, r1 = self.compute_waveform_ratio_per_interferometer(
            waveform_polarizations=waveform_polarizations,
            interferometer=interferometer,
            parameters=parameters,
        )
        a0, a1, b0, b1 = self.summary_data[interferometer.name]
        d_inner_h = (a0 * r0.conj() + a1 * r1.conj()).sum()
        h_inner_h = (b0 * abs(r0) ** 2 + 2 * b1 * (r0 * r1.conj()).real).sum()
        optimal_snr_squared = h_inner_h
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

        if return_array and self.time_marginalization:
            xp = array_module(r0)
            full_waveform = self._compute_full_waveform(
                signal_polarizations=waveform_polarizations,
                interferometer=interferometer,
                parameters=parameters,
            )
            d_inner_h_array = (
                4
                / self.waveform_generator.duration
                * xp.fft.fft(
                    full_waveform[0:-1]
                    * interferometer.frequency_domain_strain.conj()[0:-1]
                    / interferometer.power_spectral_density_array[0:-1]
                )
            )

        else:
            d_inner_h_array = None

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared.real,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=d_inner_h_array,
        )

    def _set_fiducial(self):
        """
        Set the internal waveform generator to generate fiducial waveforms,
        e.g., for setting up the reference point.

        This is handled automatically within the likelihood for most applications
        and users shouldn't need to use this method.
        """
        self.waveform_generator.waveform_arguments["fiducial"] = 1
        self.waveform_generator._cache["parameters"] = None

    def _unset_fiducial(self):
        """
        Set the internal waveform generator to generate waveforms at the binning
        frequencies.

        This is handled automatically within the likelihood for most applications
        and users shouldn't need to use this method.
        """
        self.waveform_generator.waveform_arguments["fiducial"] = 0
        self.waveform_generator._cache["parameters"] = None


_SAFE_KEYS = [
    "mass_1",
    "mass_2",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "phase",
    "luminosity_distance",
    "ra",
    "dec",
    "psi",
    "geocent_time",
]


def get_safe_fiducial_parameters(parameters, waveform_generator, eta_max=0.24):
    """
    Return fiducial parameters that avoid zeros in single harmonics of the
    fiducial waveform, following :code:`ReferenceWaveformFinder._get_safe_par_dic`
    of cogwheel (Roulet et al., (2022) Phys. Rev. D 106, 123015).

    The in-plane spin components at the reference frequency are set to zero,
    the inclination of the orbital angular momentum is set to 1 or pi - 1,
    whichever is on the same side of pi / 2, and the symmetric mass ratio is
    capped at :code:`eta_max` at constant chirp mass.

    Parameters
    ----------
    parameters: dict
        Fiducial parameters, in any basis that the parameter conversion of
        :code:`waveform_generator` accepts.
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
        Provides the parameter conversion and the reference frequency.
    eta_max: float, optional
        Largest symmetric mass ratio of the returned parameters.

    Returns
    -------
    safe_parameters: dict
        The parameters in the basis :code:`mass_1`, :code:`mass_2`, :code:`a_1`,
        :code:`a_2`, :code:`tilt_1`, :code:`tilt_2`, :code:`phi_12`,
        :code:`phi_jl`, :code:`theta_jn`, :code:`phase`,
        :code:`luminosity_distance`, :code:`ra`, :code:`dec`, :code:`psi`,
        :code:`geocent_time`, together with any calibration parameters and
        :code:`time_jitter` of the input.
    """
    converted_parameters, _ = waveform_generator.parameter_conversion(parameters.copy())
    reference_frequency = waveform_generator.waveform_arguments["reference_frequency"]

    def lalsimulation_spins(sample):
        return bilby_to_lalsimulation_spins(
            theta_jn=sample["theta_jn"],
            phi_jl=sample["phi_jl"],
            tilt_1=sample["tilt_1"],
            tilt_2=sample["tilt_2"],
            phi_12=sample["phi_12"],
            a_1=sample["a_1"],
            a_2=sample["a_2"],
            mass_1=sample["mass_1"] * solar_mass,
            mass_2=sample["mass_2"] * solar_mass,
            reference_frequency=reference_frequency,
            phase=sample["phase"],
        )

    safe_parameters = {key: converted_parameters[key] for key in _SAFE_KEYS}
    iota, _, _, spin_1z, _, _, spin_2z = lalsimulation_spins(safe_parameters)
    target_iota = 1.0 if iota < np.pi / 2 else np.pi - 1.0

    if (
        component_masses_to_symmetric_mass_ratio(
            safe_parameters["mass_1"], safe_parameters["mass_2"]
        )
        > eta_max
    ):
        chirp_mass = component_masses_to_chirp_mass(
            safe_parameters["mass_1"], safe_parameters["mass_2"]
        )
        safe_parameters["mass_1"], safe_parameters["mass_2"] = (
            chirp_mass_and_mass_ratio_to_component_masses(
                chirp_mass, symmetric_mass_ratio_to_mass_ratio(eta_max)
            )
        )

    safe_parameters.update(
        a_1=abs(spin_1z),
        a_2=abs(spin_2z),
        tilt_1=0.0 if spin_1z >= 0 else np.pi,
        tilt_2=0.0 if spin_2z >= 0 else np.pi,
        phi_12=0.0,
        phi_jl=0.0,
        theta_jn=target_iota,
    )
    # J and L are antiparallel if the aligned spins outweigh L; theta_jn is then pi - iota
    if not np.isclose(lalsimulation_spins(safe_parameters)[0], target_iota):
        safe_parameters["theta_jn"] = np.pi - target_iota

    for key in parameters:
        if key.startswith("recalib_") or key == "time_jitter":
            safe_parameters[key] = parameters[key]
    return safe_parameters


class ModeByModeRelativeBinningGravitationalWaveTransient(
    RelativeBinningGravitationalWaveTransient
):
    """A gravitational-wave transient likelihood object which uses relative
    binning separately for each azimuthal index m of the co-precessing modes
    and for each polarization, following Leslie, Dai and Pratten
    arXiv:2109.09872 and the implementation in cogwheel (Roulet et al., (2022)
    Phys. Rev. D 106, 123015).

    For each detector, harmonic m and polarization p, the ratio of the
    waveform :math:`h_{m,p}` to the detector response :math:`h_{0,m}` of
    harmonic m of the fiducial waveform is interpolated between the bin edges
    with a spline of degree :code:`spline_degree`. The detector antenna
    pattern multiplies :math:`h_{m,p}` after the interpolation.

    The fiducial parameters are made safe with
    :func:`get_safe_fiducial_parameters`, which avoids zeros in single
    harmonics of the fiducial waveform. The bins end at the last frequency at
    which every harmonic of the fiducial waveform is nonzero, as the bins of
    :code:`RelativeBinningGravitationalWaveTransient` end where its fiducial
    waveform does.

    The waveform generator must use
    :func:`bilby.gw.source.lal_binary_black_hole_relative_binning_by_mode`, or
    another source model that returns arrays of shape (number of harmonics,
    number of frequencies) for each polarization.

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
        If no fiducial parameters are passed, a random point will be drawn from
        the prior.
    parameter_bounds: dict, optional
        Dictionary of bounds (lists) for the initial parameters when finding
        the initial maximum likelihood (fiducial) waveform.
    maximization_kwargs: dict, optional
        Keyword arguments passed to :code:`scipy.optimize.differential_evolution`.
    update_fiducial_parameters: bool
        Whether to try to update the fiducial parameters by finding the maximum
        likelihood point using scipy differential evolution. Default is
        :code:`False`.
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood.
    time_marginalization: bool, optional
        Not supported, must be :code:`False`.
    phase_marginalization: bool, optional
        If true, marginalize over phase in the likelihood.
    priors: dict, optional
        If given, used in the distance and phase marginalization.
    distance_marginalization_lookup_table: (dict, str), optional
        See :code:`RelativeBinningGravitationalWaveTransient`.
    jitter_time: bool, optional
        See :code:`RelativeBinningGravitationalWaveTransient`.
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location.
    time_reference: str, optional
        Name of the reference for the sampled time parameter.
    chi: float, optional
        Tunable parameter which limits the perturbation of alpha when setting
        up the bin range. See https://arxiv.org/abs/1806.08792.
    epsilon: float, optional
        Tunable parameter which limits the differential phase change in each
        bin when setting up the bin range. See https://arxiv.org/abs/1806.08792.
    spline_degree: int, optional
        Degree of the spline that interpolates the waveform ratios between
        bin edges, 1 to 5. Default is 3.
    safe_fiducial_parameters: bool, optional
        Whether to replace the fiducial parameters with
        :func:`get_safe_fiducial_parameters` of them. Default is :code:`True`.

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters.

    Notes
    -----
    Time marginalization and calibration marginalization are not supported.
    """

    def __init__(
        self,
        interferometers,
        waveform_generator,
        fiducial_parameters=None,
        parameter_bounds=None,
        maximization_kwargs=None,
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
        epsilon=0.5,
        spline_degree=3,
        safe_fiducial_parameters=True,
    ):
        if time_marginalization:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support time marginalization."
            )
        self.spline_degree = spline_degree
        self.safe_fiducial_parameters = safe_fiducial_parameters
        self.fiducial_waveform_parameters = None
        self.per_detector_fiducial_mode_waveforms = dict()
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            fiducial_parameters=fiducial_parameters,
            parameter_bounds=parameter_bounds,
            maximization_kwargs=maximization_kwargs,
            update_fiducial_parameters=update_fiducial_parameters,
            distance_marginalization=distance_marginalization,
            time_marginalization=time_marginalization,
            phase_marginalization=phase_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference,
            chi=chi,
            epsilon=epsilon,
        )

    def set_fiducial_waveforms(self, parameters):
        """
        Compute the detector response of each harmonic of the fiducial
        waveform on the full frequency grid.

        If :code:`safe_fiducial_parameters` is set, the waveform is computed
        at :func:`get_safe_fiducial_parameters` of :code:`parameters`. The
        maximum frequency of the bins is set to the last frequency at which
        every harmonic of the fiducial waveform is nonzero.

        Parameters
        ----------
        parameters: dict
            The fiducial parameters.
        """
        parameters = parameters.copy()
        parameters.update(self.get_sky_frame_parameters(parameters=parameters))
        if self.safe_fiducial_parameters:
            parameters = get_safe_fiducial_parameters(
                parameters, self.waveform_generator
            )
        self._set_fiducial()
        self.fiducial_polarizations = self.waveform_generator.frequency_domain_strain(
            parameters
        )
        self._unset_fiducial()
        if self.fiducial_polarizations is None:
            raise ValueError(f"Cannot compute fiducial waveforms for {parameters}")
        self.fiducial_waveform_parameters = parameters

        every_harmonic_nonzero = np.all(
            self.fiducial_polarizations["plus"] != 0, axis=0
        )
        self.maximum_frequency = self.waveform_generator.frequency_array[
            np.where(every_harmonic_nonzero)[0][-1]
        ]

        for interferometer in self.interferometers:
            waveforms = interferometer.get_detector_response(
                self.fiducial_polarizations, parameters
            )
            self.per_detector_fiducial_mode_waveforms[interferometer.name] = waveforms
            self.per_detector_fiducial_waveforms[interferometer.name] = waveforms.sum(
                axis=0
            )

    def compute_summary_data(self):
        r"""
        Compute the summary data of each interferometer: for each harmonic m,
        the weights :math:`D_{m,j}` of :math:`\langle h|d \rangle`, and for
        each pair of harmonics m <= m', the weights :math:`H_{mm',j}` of
        :math:`\langle h|h \rangle`, at bin edge j.

        With :math:`S_j(f)` the spline that is one at bin edge j and zero at
        the others,

        .. math::

            D_{m,j} = \frac{4}{T} \sum_f S_j(f) \frac{h_{0,m}^*(f) d(f)}{P(f)}
            \Big/ h_{0,m}^*(f_j)

            H_{mm',j} = \frac{4}{T} \sum_f S_j(f)
            \frac{h_{0,m}(f) h_{0,m'}^*(f)}{P(f)}
            \Big/ \left(h_{0,m}(f_j) h_{0,m'}^*(f_j)\right)

        with :math:`H_{mm',j}` doubled for :math:`m \neq m'`.
        """
        names = [interferometer.name for interferometer in self.interferometers]
        number_of_harmonics = len(self.per_detector_fiducial_mode_waveforms[names[0]])
        self.harmonic_pairs = tuple(
            np.array(indices)
            for indices in zip(
                *combinations_with_replacement(range(number_of_harmonics), 2)
            )
        )
        m_inds, mprime_inds = self.harmonic_pairs
        inverse_psds = dict()
        for interferometer in self.interferometers:
            inverse_psds[interferometer.name] = np.zeros(
                len(interferometer.frequency_array)
            )
            inverse_psds[interferometer.name][interferometer.frequency_mask] = (
                1
                / interferometer.power_spectral_density_array[
                    interferometer.frequency_mask
                ]
            )

        frequency_array = self.interferometers[0].frequency_array
        self._set_splines(frequency_array)
        summary_data = dict()
        for interferometer in self.interferometers:
            name = interferometer.name
            h0 = self.per_detector_fiducial_mode_waveforms[name]
            h0_bins = h0[:, self.bin_inds]
            if np.any(h0_bins == 0):
                raise ValueError(
                    "A harmonic of the fiducial waveform is zero at a bin edge; "
                    "use safe_fiducial_parameters=True or other fiducial parameters."
                )

            inverse_psd = inverse_psds[name]
            strain = interferometer.frequency_domain_strain
            d_inner_h_weights = (
                self._get_summary_weights(
                    h0.conj() * strain * inverse_psd, interferometer.duration
                )
                / h0_bins.conj()
            )
            h_inner_h_weights = self._get_summary_weights(
                h0[m_inds] * h0[mprime_inds].conj() * inverse_psd,
                interferometer.duration,
            ) / (h0_bins[m_inds] * h0_bins[mprime_inds].conj())
            h_inner_h_weights[m_inds != mprime_inds] *= 2
            summary_data[name] = (d_inner_h_weights, h_inner_h_weights)

        self.summary_data = summary_data

    def _set_splines(self, frequency_array):
        """
        Set :code:`_spline_coefficients`, whose row j holds the B-spline
        coefficients of the spline of degree :code:`spline_degree` that is one
        at bin edge j and zero at the other edges, and :code:`_basis_splines`,
        a sparse array whose rows are the B-spline basis elements on
        :code:`frequency_array`.

        Parameters
        ----------
        frequency_array: array_like
            The frequencies at which the basis elements are evaluated.
        """
        number_of_edges = len(self.bin_freqs)
        coefficients = np.empty((number_of_edges, number_of_edges))
        for index, values in enumerate(np.eye(number_of_edges)):
            knots, spline_coefficients, _ = scipy.interpolate.splrep(
                self.bin_freqs, values, s=0, k=self.spline_degree
            )
            coefficients[index] = spline_coefficients[:number_of_edges]
        self._spline_coefficients = coefficients

        rows, columns, values = list(), list(), list()
        for index in range(number_of_edges):
            element_knots = knots[index : index + self.spline_degree + 2]
            element = scipy.interpolate.BSpline.basis_element(element_knots)
            start = np.searchsorted(frequency_array, element_knots[0], side="left")
            stop = np.searchsorted(frequency_array, element_knots[-1], side="right")
            rows.append(np.full(stop - start, index))
            columns.append(np.arange(start, stop))
            # A basis element vanishes at its last knot; evaluate just below it so the last element is one there
            values.append(
                element(
                    np.minimum(
                        frequency_array[start:stop], np.nextafter(element_knots[-1], 0)
                    )
                )
            )
        self._basis_splines = scipy.sparse.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(number_of_edges, len(frequency_array)),
        )

    def _get_summary_weights(self, integrand, duration):
        r"""
        Return the weights :math:`W_j = (4 / T) \sum_f g(f) S_j(f)`, so that
        :math:`(4 / T) \sum_f g(f) r(f) \approx \sum_j W_j r(f_j)` for a
        smooth function r, exactly if r is a spline of degree
        :code:`spline_degree` with knots at the bin edges.

        Parameters
        ----------
        integrand: array_like
            :math:`g(f)`, of shape (..., number of frequencies).
        duration: float
            The duration T of the data.

        Returns
        -------
        weights: array_like
            Of shape (..., number of bin edges).
        """
        projected = (self._basis_splines @ np.atleast_2d(integrand).T).T
        weights = 4 / duration * projected @ self._spline_coefficients.T
        return weights.reshape(integrand.shape[:-1] + (len(self.bin_freqs),))

    def calculate_snrs(
        self, waveform_polarizations, interferometer, *, return_array=True, parameters
    ):
        r"""
        Compute the inner products of the waveform with the data and with
        itself from the summary data,

        .. math::

            h_m(f_j) = \sum_p F_p h_{m,p}(f_j)

            \langle h|d \rangle = \sum_m \sum_j D_{m,j} h_m^*(f_j)

            \langle h|h \rangle = \mathrm{Re} \sum_{m \leq m'}
            \sum_j H_{mm',j} h_m(f_j) h_{m'}^*(f_j)

        with :math:`F_p` the antenna pattern of polarization p,
        :math:`h_{m,p}` harmonic m of polarization p, and :math:`h_m`
        harmonic m of the strain in the interferometer, time shifted and
        calibrated, from
        :meth:`bilby.gw.detector.Interferometer.get_detector_response`.

        Parameters
        ----------
        waveform_polarizations: dict
            Polarizations at the bin edges, each of shape
            (number of harmonics, number of bin edges).
        interferometer: bilby.gw.detector.Interferometer
            The interferometer.
        return_array: bool
            Unused, as time marginalization is not supported.
        parameters: dict
            Parameters of the signal.

        Returns
        -------
        calculated_snrs: _CalculatedSNRs
            The inner products, with :code:`d_inner_h_array` set to None.
        """
        # The strain of each harmonic at the bin edges, of shape (harmonics, bin edges)
        strain = interferometer.get_detector_response(
            waveform_polarizations=waveform_polarizations,
            parameters=parameters,
            frequencies=self.bin_freqs,
        )
        d_inner_h_weights, h_inner_h_weights = self.summary_data[interferometer.name]
        m_inds, mprime_inds = self.harmonic_pairs
        d_inner_h = (d_inner_h_weights * strain.conj()).sum()
        h_inner_h = (
            (h_inner_h_weights * strain[m_inds] * strain[mprime_inds].conj()).sum().real
        )
        optimal_snr_squared = h_inner_h
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=None,
        )

    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not compute the full waveform."
        )
