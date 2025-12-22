import numpy as np
from bilby_cython import time as _time, geometry as _geometry
from plum import dispatch

from ...compat.types import Real, ArrayLike


@dispatch(precedence=1)
def gps_time_to_utc(gps_time: Real):
    return _time.gps_time_to_utc(gps_time)


@dispatch(precedence=1)
def greenwich_mean_sidereal_time(gps_time: Real | ArrayLike):
    return _time.greenwich_mean_sidereal_time(gps_time)


@dispatch(precedence=1)
def greenwich_sidereal_time(gps_time: Real, equation_of_equinoxes: Real):
    return _time.greenwich_sidereal_time(gps_time, equation_of_equinoxes)


@dispatch(precedence=1)
def n_leap_seconds(gps_time: Real):
    return _time.n_leap_seconds(gps_time)


@dispatch(precedence=1)
def utc_to_julian_day(utc_time: Real):
    return _time.utc_to_julian_day(utc_time)


@dispatch(precedence=1)
def calculate_arm(arm_tilt: Real, arm_azimuth: Real, longitude: Real, latitude: Real):
    return _geometry.calculate_arm(arm_tilt, arm_azimuth, longitude, latitude)


@dispatch(precedence=1)
def detector_tensor(x: ArrayLike, y: ArrayLike):
    return _geometry.detector_tensor(x, y)


@dispatch(precedence=1)
def get_polarization_tensor(ra: Real, dec: Real, time: Real, psi: Real, mode: str):
    return _geometry.get_polarization_tensor(ra, dec, time, psi, mode)


@dispatch(precedence=1)
def rotation_matrix_from_delta(delta: ArrayLike):
    return _geometry.rotation_matrix_from_delta_x(delta)


@dispatch(precedence=1)
def time_delay_geocentric(detector1: ArrayLike, detector2: ArrayLike, ra, dec, time):
    return _geometry.time_delay_geocentric(detector1, detector2, ra, dec, time)


@dispatch(precedence=1)
def time_delay_from_geocenter(detector1: ArrayLike, ra: Real, dec: Real, time: Real | ArrayLike):
    return _geometry.time_delay_from_geocenter(detector1, ra, dec, time)


@dispatch(precedence=1)
def zenith_azimuth_to_theta_phi(zenith: Real, azimuth: Real, delta_x: np.ndarray):
    theta, phi = _geometry.zenith_azimuth_to_theta_phi(zenith, azimuth, delta_x)
    return theta, phi % (2 * np.pi)
