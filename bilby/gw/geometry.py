import numpy as np
from plum import dispatch
from bilby_rust import geometry as _geometry

from .time import greenwich_mean_sidereal_time
from ..compat.types import Real, ArrayLike
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
    """"""
    xp = array_module(detector_tensor)
    polarization_tensor = get_polarization_tensor(*promote_to_array((ra, dec, time, psi), xp), mode)
    return three_by_three_matrix_contraction(detector_tensor, polarization_tensor)


@dispatch
def calculate_arm(arm_tilt, arm_azimuth, longitude, latitude):
    """"""
    xp = array_module(arm_tilt)
    e_long = xp.array([-xp.sin(longitude), xp.cos(longitude), longitude * 0])
    e_lat = xp.array(
        [
            -xp.sin(latitude) * xp.cos(longitude),
            -xp.sin(latitude) * xp.sin(longitude),
            xp.cos(latitude),
        ]
    )
    e_h = xp.array(
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
    u = xp.array(
        [
            xp.cos(phi) * xp.cos(theta),
            xp.cos(theta) * xp.sin(phi),
            -xp.sin(theta) * xp.ones_like(phi),
        ]
    )
    v = xp.array([
        -xp.sin(phi), xp.cos(phi), xp.zeros_like(phi)
    ]) * xp.ones_like(theta)
    omega = xp.array([
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
    rotation_1 = xp.array(
        [
            [xp.cos(alpha), -xp.sin(alpha), xp.zeros(alpha.shape)],
            [xp.sin(alpha), xp.cos(alpha), xp.zeros(alpha.shape)],
            [xp.zeros(alpha.shape), xp.zeros(alpha.shape), xp.ones(alpha.shape)],
        ]
    )
    rotation_2 = xp.array(
        [
            [xp.cos(beta), xp.zeros(beta.shape), xp.sin(beta)],
            [xp.zeros(beta.shape), xp.ones(beta.shape), xp.zeros(beta.shape)],
            [-xp.sin(beta), xp.zeros(beta.shape), xp.cos(beta)],
        ]
    )
    rotation_3 = xp.array(
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
    omega = xp.array(
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
    omega_prime = xp.array(
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



# @dispatch(precedence=1)
# def antenna_response(detector_tensor: np.ndarray, ra: FloatOrInt, dec: FloatOrInt, time: FloatOrInt, psi: FloatOrInt, mode: str):
#     return _geometry.antenna_response(detector_tensor, ra, dec, time, psi, mode)


@dispatch(precedence=1)
def calculate_arm(arm_tilt: Real, arm_azimuth: Real, longitude: Real, latitude: Real):
    return _geometry.calculate_arm(arm_tilt, arm_azimuth, longitude, latitude)


@dispatch(precedence=1)
def detector_tensor(x: ArrayLike, y: ArrayLike):
    return _geometry.detector_tensor(x, y)


@dispatch(precedence=1)
def get_polarization_tensor(ra: Real, dec: Real, time: Real, psi: Real, mode: str):
    return _geometry.get_polarization_tensor(ra, dec, time, psi, mode)


# @dispatch(precedence=1)
# def get_polarization_tensor_multiple_modes(ra: FloatOrInt, dec: FloatOrInt, time: FloatOrInt, psi: FloatOrInt, modes: list[str]):
#     return [geometry.get_polarization_tensor(ra, dec, time, psi, mode) for mode in modes]


@dispatch(precedence=1)
def rotation_matrix_from_delta(delta: ArrayLike):
    return _geometry.rotation_matrix_from_delta_x(delta)


# @dispatch(precedence=1)
# def three_by_three_matrix_contraction(a: ArrayLike, b: ArrayLike):
#     return _geometry.three_by_three_matrix_contraction(a, b)


@dispatch(precedence=1)
def time_delay_geocentric(detector1: ArrayLike, detector2: ArrayLike, ra, dec, time):
    return _geometry.time_delay_geocentric(detector1, detector2, ra, dec, time)


@dispatch(precedence=1)
def time_delay_from_geocenter(detector1: ArrayLike, ra: Real, dec: Real, time: Real):
    return _geometry.time_delay_from_geocenter(detector1, ra, dec, time)


@dispatch(precedence=1)
def time_delay_from_geocenter(detector1: ArrayLike, ra: Real, dec: Real, time: ArrayLike):
    return _geometry.time_delay_from_geocenter_vectorized(detector1, ra, dec, time)


@dispatch(precedence=1)
def zenith_azimuth_to_theta_phi(zenith: Real, azimuth: Real, delta_x: np.ndarray):
    theta, phi = _geometry.zenith_azimuth_to_theta_phi(zenith, azimuth, delta_x)
    return theta, phi % (2 * np.pi)

