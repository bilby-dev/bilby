from plum import dispatch

from .time import greenwich_mean_sidereal_time
from ..compat.utils import array_module, promote_to_array


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
    """
    Calculate the antenna response for a detector.

    Parameters
    ==========
    detector_tensor: array-like
        The detector tensor (3x3 matrix).
    ra: float or array-like
        Right ascension of the source in radians.
    dec: float or array-like
        Declination of the source in radians.
    time: float or array-like
        GPS time of the observation.
    psi: float or array-like
        Polarization angle in radians.
    mode: str
        Polarization mode ('plus', 'cross', 'breathing', 'longitudinal', 'x', 'y').

    Returns
    =======
    array-like
        The antenna response (scalar or array depending on input).
    """
    xp = array_module(detector_tensor)
    polarization_tensor = get_polarization_tensor(*promote_to_array((ra, dec, time, psi), xp), mode)
    return three_by_three_matrix_contraction(detector_tensor, polarization_tensor)


@dispatch
def calculate_arm(arm_tilt, arm_azimuth, longitude, latitude):
    """
    Calculate arm unit vector from tilt, azimuth, and location.

    Parameters
    ==========
    arm_tilt: float or array-like
        Tilt angle of the arm from horizontal in radians.
    arm_azimuth: float or array-like
        Azimuth angle of the arm in radians.
    longitude: float or array-like
        Longitude of the detector in radians.
    latitude: float or array-like
        Latitude of the detector in radians.

    Returns
    =======
    array-like
        3D unit vector (shape (3,) or (3, ...)) representing the arm direction.
    """
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
    """
    Calculate the detector tensor from x and y arm components.

    Parameters
    ==========
    x: array-like
        3D unit vector for the x arm.
    y: array-like
        3D unit vector for the y arm.

    Returns
    =======
    array-like
        3x3 detector tensor with components
        :math:`d_{ij} = (x_i x_j - y_i y_j) / 2`.
    """
    xp = array_module(x)
    return (xp.outer(x, x) - xp.outer(y, y)) / 2


@dispatch
def get_polarization_tensor(ra, dec, time, psi, mode):
    """
    Calculate the polarization tensor for a given sky location and mode.

    Parameters
    ==========
    ra: float or array-like
        Right ascension of the source in radians.
    dec: float or array-like
        Declination of the source in radians.
    time: float or array-like
        GPS time of the observation.
    psi: float or array-like
        Polarization angle in radians.
    mode: str
        Polarization mode: 'plus', 'cross', 'breathing', 'longitudinal',
        'x', or 'y'.

    Returns
    =======
    array-like
        3x3 polarization tensor for the specified mode.
    """
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
    """
    Calculate polarization tensors for multiple modes.

    Parameters
    ==========
    ra: float or array-like
        Right ascension of the source in radians.
    dec: float or array-like
        Declination of the source in radians.
    time: float or array-like
        GPS time of the observation.
    psi: float or array-like
        Polarization angle in radians.
    modes: list of str
        List of polarization modes to calculate.

    Returns
    =======
    list
        List of 3x3 polarization tensors, one for each mode.
    """
    return [get_polarization_tensor(ra, dec, time, psi, mode) for mode in modes]


@dispatch
def rotation_matrix_from_delta(delta_x):
    r"""
    Calculate rotation matrix from a delta vector.

    Parameters
    ==========
    delta_x: array-like
        3D vector :math:`\vec{\Delta}x` representing the separation
        or orientation.

    Returns
    =======
    array-like
        3x3 rotation matrix that rotates the z-axis to align with
        :math:`\vec{\Delta}x` direction.
    """
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
    """
    Perform contraction of two 3x3 matrices.

    Parameters
    ==========
    a: array-like
        First 3x3 matrix.
    b: array-like
        Second 3x3 matrix.

    Returns
    =======
    float or array-like
        Scalar result of the einsum contraction :math:`a_{ij} b_{ij}`.
    """
    xp = array_module(a)
    return xp.einsum("ij,ij->", a, b)


@dispatch
def time_delay_geocentric(detector1, detector2, ra, dec, time):
    r"""
    Calculate time delay between two detectors for a source direction.

    Parameters
    ==========
    detector1: array-like
        3D position vector of the first detector in meters.
    detector2: array-like
        3D position vector of the second detector in meters.
    ra: float or array-like
        Right ascension of the source in radians.
    dec: float or array-like
        Declination of the source in radians.
    time: float or array-like
        GPS time of the observation.

    Returns
    =======
    float or array-like
        Time delay :math:`\Delta t = \hat{\omega} \cdot (\vec{d}_2 - \vec{d}_1) / c`
        in seconds, where :math:`\hat{\omega}` is the unit vector to the
        source and :math:`c` is the speed of light.
    """
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
    r"""
    Calculate time delay from geocenter to a detector.

    Parameters
    ==========
    detector1: array-like
        3D position vector of the detector in meters.
    ra: float or array-like
        Right ascension of the source in radians.
    dec: float or array-like
        Declination of the source in radians.
    time: float or array-like
        GPS time of the observation.

    Returns
    =======
    float or array-like
        Time delay :math:`\Delta t = \hat{\omega} \cdot \vec{d} / c` in
        seconds, where :math:`\vec{d}` is the detector position and
        :math:`c` is the speed of light.
    """
    xp = array_module(detector1)
    return time_delay_geocentric(detector1, xp.zeros(3), ra, dec, time)


@dispatch
def zenith_azimuth_to_theta_phi(zenith, azimuth, delta_x):
    """
    Convert zenith/azimuth angles to theta/phi in a rotated frame.

    Parameters
    ==========
    zenith: float or array-like
        Zenith angle in radians.
    azimuth: float or array-like
        Azimuth angle in radians.
    delta_x: array-like
        3D vector defining the rotation frame.

    Returns
    =======
    tuple of array-like
        (theta, phi) angles in the rotated frame, both in radians.
    """
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
