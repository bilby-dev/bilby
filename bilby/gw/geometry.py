from plum import dispatch

from .time import greenwich_mean_sidereal_time
from ..compat.utils import array_module, promote_to_array
from ..core.utils.constants import msun_time_si

__all__ = [
    "antenna_response",
    "calculate_arm",
    "detector_tensor",
    "get_polarization_tensor",
    "get_polarization_tensor_multiple_modes",
    "rotation_matrix_from_delta",
    "three_by_three_matrix_contraction",
    "time_delay_geocentric",
    "time_delay_from_geocenter",
    "zenith_azimuth_to_theta_phi",
]


@dispatch
def antenna_response(detector_tensor, ra, dec, time, psi, mode):
    """"""
    xp = array_module(detector_tensor)
    polarization_tensor = get_polarization_tensor(*promote_to_array((ra, dec, time, psi), xp), mode)
    return three_by_three_matrix_contraction(detector_tensor, polarization_tensor)


@dispatch
def calculate_arm(arm_tilt, arm_azimuth, longitude, latitude):
    """"""
    xp = array_module(arm_tilt)
    e_long = xp.asarray([-xp.sin(longitude), xp.cos(longitude), longitude * 0])
    e_lat = xp.asarray(
        [
            -xp.sin(latitude) * xp.cos(longitude),
            -xp.sin(latitude) * xp.sin(longitude),
            xp.cos(latitude),
        ]
    )
    e_h = xp.asarray(
        [
            xp.cos(latitude) * xp.cos(longitude),
            xp.cos(latitude) * xp.sin(longitude),
            xp.sin(latitude),
        ]
    )

    return (
        xp.cos(arm_tilt) * xp.cos(arm_azimuth) * e_long
        + xp.cos(arm_tilt) * xp.sin(arm_azimuth) * e_lat
        + xp.sin(arm_tilt) * e_h
    )


@dispatch
def detector_tensor(x, y):
    """"""
    xp = array_module(x)
    return (xp.outer(x, x) - xp.outer(y, y)) / 2


@dispatch
def get_polarization_tensor(ra, dec, time, psi, mode):
    """"""
    from functools import partial

    xp = array_module(ra)

    gmst = greenwich_mean_sidereal_time(time) % (2 * xp.pi)
    phi = ra - gmst
    theta = xp.atleast_1d(xp.pi / 2 - dec).squeeze()
    u = xp.asarray(
        [
            xp.cos(phi) * xp.cos(theta),
            xp.cos(theta) * xp.sin(phi),
            -xp.sin(theta) * xp.ones_like(phi),
        ]
    )
    v = xp.asarray([
        -xp.sin(phi), xp.cos(phi), xp.zeros_like(phi)
    ]) * xp.ones_like(theta)
    omega = xp.asarray([
        xp.sin(xp.pi - theta) * xp.cos(xp.pi + phi),
        xp.sin(xp.pi - theta) * xp.sin(xp.pi + phi),
        xp.cos(xp.pi - theta) * xp.ones_like(phi),
    ])
    m = -u * xp.sin(psi) - v * xp.cos(psi)
    n = -u * xp.cos(psi) + v * xp.sin(psi)
    if xp.__name__ == "mlx.core":
        einsum_shape = "i,j->ij"
    else:
        einsum_shape = "i...,j...->ij..."
    product = partial(xp.einsum, einsum_shape)

    match mode.lower():
        case "plus":
            return product(m, m) - product(n, n)
        case "cross":
            return product(m, n) + product(n, m)
        case "breathing":
            return product(m, m) + product(n, n)
        case "longitudinal":
            return product(omega, omega)
        case "x":
            return product(m, omega) + product(omega, m)
        case "y":
            return product(n, omega) + product(omega, n)
        case _:
            raise ValueError(f"{mode} not a polarization mode!")


@dispatch
def get_polarization_tensor_multiple_modes(ra, dec, time, psi, modes):
    """"""
    return [get_polarization_tensor(ra, dec, time, psi, mode) for mode in modes]


@dispatch
def rotation_matrix_from_delta(delta_x):
    """"""
    xp = array_module(delta_x)
    delta_x = delta_x / (delta_x**2).sum() ** 0.5
    alpha = xp.arctan2(-delta_x[1] * delta_x[2], delta_x[0])
    beta = xp.arccos(delta_x[2])
    gamma = xp.arctan2(delta_x[1], delta_x[0])
    rotation_1 = xp.asarray(
        [
            [xp.cos(alpha), -xp.sin(alpha), xp.zeros(alpha.shape)],
            [xp.sin(alpha), xp.cos(alpha), xp.zeros(alpha.shape)],
            [xp.zeros(alpha.shape), xp.zeros(alpha.shape), xp.ones(alpha.shape)],
        ]
    )
    rotation_2 = xp.asarray(
        [
            [xp.cos(beta), xp.zeros(beta.shape), xp.sin(beta)],
            [xp.zeros(beta.shape), xp.ones(beta.shape), xp.zeros(beta.shape)],
            [-xp.sin(beta), xp.zeros(beta.shape), xp.cos(beta)],
        ]
    )
    rotation_3 = xp.asarray(
        [
            [xp.cos(gamma), -xp.sin(gamma), xp.zeros(gamma.shape)],
            [xp.sin(gamma), xp.cos(gamma), xp.zeros(gamma.shape)],
            [xp.zeros(gamma.shape), xp.zeros(gamma.shape), xp.ones(gamma.shape)],
        ]
    )
    return rotation_3 @ rotation_2 @ rotation_1


@dispatch
def three_by_three_matrix_contraction(a, b):
    """"""
    xp = array_module(a)
    return xp.einsum("ij,ij->", a, b)


@dispatch
def time_delay_geocentric(detector1, detector2, ra, dec, time):
    """"""
    xp = array_module(detector1)
    gmst = greenwich_mean_sidereal_time(time) % (2 * xp.pi)
    speed_of_light = 299792458.0
    phi = ra - gmst
    theta = xp.pi / 2 - dec
    omega = xp.asarray(
        [xp.sin(theta) * xp.cos(phi), xp.sin(theta) * xp.sin(phi), xp.cos(theta)]
    )
    delta_d = detector2 - detector1
    return omega @ delta_d / speed_of_light


@dispatch
def time_delay_from_geocenter(detector1, ra, dec, time):
    """"""
    xp = array_module(detector1)
    return time_delay_geocentric(detector1, xp.zeros(3), ra, dec, time)


@dispatch
def zenith_azimuth_to_theta_phi(zenith, azimuth, delta_x):
    """"""
    xp = array_module(delta_x)
    omega_prime = xp.stack(
        [
            xp.sin(zenith) * xp.cos(azimuth),
            xp.sin(zenith) * xp.sin(azimuth),
            xp.cos(zenith),
        ]
    )
    rotation_matrix = rotation_matrix_from_delta(delta_x)
    omega = rotation_matrix @ omega_prime
    theta = xp.arccos(omega[2])
    phi = xp.arctan2(omega[1], omega[0]) % (2 * xp.pi)
    return theta, phi


def transform_precessing_spins(
    theta_jn,
    phi_jl,
    tilt_1,
    tilt_2,
    phi_12,
    chi_1,
    chi_2,
    mass_1,
    mass_2,
    f_ref,
    phase,
):
    """
    A direct reimplementation of
    :code:`lalsimulation.SimInspiralTransformPrecessingNewInitialConditions`.

    Parameters
    ----------
    theta_jn: float | xp.ndarray
        Zenith angle between J and N (rad).
    phi_jl: float | xp.ndarray
        Azimuthal angle of L_N on its cone about J (rad).
    tilt_1: float | xp.ndarray
        Zenith angle between S1 and LNhat (rad).
    tilt_2: float | xp.ndarray
        Zenith angle between S2 and LNhat (rad).
    phi_12: float | xp.ndarray
        Difference in azimuthal angle between S1, S2 (rad).
    chi_1: float | xp.ndarray
        Dimensionless spin of body 1.
    chi_2: float | xp.ndarray
        Dimensionless spin of body 2.
    mass_1: float | xp.ndarray
        Mass of body 1 (solar masses).
    mass_2: float | xp.ndarray
        Mass of body 2 (solar masses).
    f_ref: float | xp.ndarray
        Reference GW frequency (Hz).
    phase: float | xp.ndarray
        Reference orbital phase.

    Returns
    -------
    tuple
        (iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)
        - iota: Inclination angle of L_N
        - spin_1x, spin_1y, spin_1z: Components of spin 1
        - spin_2x, spin_2y, spin_2z: Components of spin 2
    """

    xp = array_module(theta_jn)
    pi = xp.pi

    # Helper rotation functions
    def rotate_z(angle, vec):
        """Rotate vector about z-axis"""
        cos_a = xp.cos(angle)
        sin_a = xp.sin(angle)
        x_new = cos_a * vec[0] - sin_a * vec[1]
        y_new = sin_a * vec[0] + cos_a * vec[1]
        return xp.stack([x_new, y_new, vec[2]], axis=0)

    def rotate_y(angle, vec):
        """Rotate vector about y-axis"""
        cos_a = xp.cos(angle)
        sin_a = xp.sin(angle)
        x_new = cos_a * vec[0] + sin_a * vec[2]
        z_new = -sin_a * vec[0] + cos_a * vec[2]
        return xp.stack([x_new, vec[1], z_new], axis=0)

    # Starting frame: LNhat is along the z-axis
    ln_hat = xp.stack([
        xp.zeros_like(theta_jn),
        xp.zeros_like(theta_jn),
        xp.ones_like(theta_jn)
    ], axis=0)

    # Initial spin unit vectors
    s1_hat = xp.stack([
        xp.sin(tilt_1) * xp.cos(phase),
        xp.sin(tilt_1) * xp.sin(phase),
        xp.cos(tilt_1)
    ], axis=0)

    s2_hat = xp.stack([
        xp.sin(tilt_2) * xp.cos(phi_12 + phase),
        xp.sin(tilt_2) * xp.sin(phi_12 + phase),
        xp.cos(tilt_2)
    ], axis=0)

    # Compute physical parameters
    m_total = mass_1 + mass_2
    eta = mass_1 * mass_2 / (m_total * m_total)

    # v parameter at reference point (c=G=1 units)
    v0 = (m_total * msun_time_si * pi * f_ref) ** (1/3)

    # Compute angular momentum magnitude using PN expressions
    # L/M = eta * v^(-1) * (1 + v^2 * L_2PN)
    # L_2PN = 3/2 + 1/6 * eta
    l_2pn = 1.5 + eta / 6.0
    l_mag = eta * m_total * m_total / v0 * (1.0 + v0 * v0 * l_2pn)

    # Spin vectors with proper magnitudes
    s1 = mass_1 * mass_1 * chi_1 * s1_hat
    s2 = mass_2 * mass_2 * chi_2 * s2_hat

    # Total angular momentum J = L + S1 + S2
    l_vec = xp.stack([xp.zeros_like(theta_jn), xp.zeros_like(theta_jn), l_mag], axis=0)
    j = l_vec + s1 + s2

    # Normalize J to get Jhat and find its angles
    j_norm = xp.sqrt(xp.sum(j * j, axis=0))
    j_hat = j / j_norm

    theta_0 = xp.arccos(j_hat[2])
    phi_0 = xp.arctan2(j_hat[1], j_hat[0])

    # Rotation 1: Rotate about z-axis by -phi_0 to put Jhat in x-z plane
    angle = -phi_0
    s1_hat = rotate_z(angle, s1_hat)
    s2_hat = rotate_z(angle, s2_hat)

    # Rotation 2: Rotate about y-axis by -theta_0 to put Jhat along z-axis
    angle = -theta_0
    ln_hat = rotate_y(angle, ln_hat)
    s1_hat = rotate_y(angle, s1_hat)
    s2_hat = rotate_y(angle, s2_hat)

    # Rotation 3: Rotate about z-axis by (phi_jl - pi) to put L at desired azimuth
    angle = phi_jl - pi
    ln_hat = rotate_z(angle, ln_hat)
    s1_hat = rotate_z(angle, s1_hat)
    s2_hat = rotate_z(angle, s2_hat)

    # Compute inclination: angle between L and N
    n = xp.stack([
        xp.zeros_like(theta_jn),
        xp.sin(theta_jn),
        xp.cos(theta_jn)
    ], axis=0)
    iota = xp.arccos(xp.sum(n * ln_hat, axis=0))

    # Rotation 4-5: Bring L into the z-axis
    theta_lj = xp.arccos(ln_hat[2])
    phi_l = xp.arctan2(ln_hat[1], ln_hat[0])

    angle = -phi_l
    s1_hat = rotate_z(angle, s1_hat)
    s2_hat = rotate_z(angle, s2_hat)
    n = rotate_z(angle, n)

    angle = -theta_lj
    s1_hat = rotate_y(angle, s1_hat)
    s2_hat = rotate_y(angle, s2_hat)
    n = rotate_y(angle, n)

    # Rotation 6: Bring N into y-z plane with positive y component
    phi_n = xp.arctan2(n[1], n[0])

    angle = pi / 2.0 - phi_n - phase
    s1_hat = rotate_z(angle, s1_hat)
    s2_hat = rotate_z(angle, s2_hat)

    # Return final spin components
    spin_1 = s1_hat * chi_1
    spin_2 = s2_hat * chi_2

    return iota, *spin_1, *spin_2
