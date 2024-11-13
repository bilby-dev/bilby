from typing import Union

import numpy as np
from plum import dispatch
from bilby_rust import time as _time

from ..compat.types import Real, ArrayLike
from ..compat.utils import array_module


__all__ = [
    "datetime",
    "gps_time_to_utc",
    "greenwich_mean_sidereal_time",
    "greenwich_sidereal_time",
    "n_leap_seconds",
    "utc_to_julian_day",
    "LEAP_SECONDS",
]


class datetime:
    """
    A barebones datetime class for use in the GPS to GMST conversion.
    """

    def __init__(
        self,
        year: int = 0,
        month: int = 0,
        day: int = 0,
        hour: int = 0,
        minute: int = 0,
        second: float = 0,
    ):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second

    def __repr__(self):
        return f"{self.year}-{self.month}-{self.day} {self.hour}:{self.minute}:{self.second}"

    def __add__(self, other):
        """
        Add two datetimes together.
        Note that this does not handle overflow and can lead to unphysical
        values for the various attributes.
        """
        return datetime(
            self.year + other.year,
            self.month + other.month,
            self.day + other.day,
            self.hour + other.hour,
            self.minute + other.minute,
            self.second + other.second,
        )

    @property
    def julian_day(self):
        return (
            367 * self.year
            - 7 * (self.year + (self.month + 9) // 12) // 4
            + 275 * self.month // 9
            + self.day
            + self.second / SECONDS_PER_DAY
            + JULIAN_GPS_EPOCH
        )


GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
JULIAN_GPS_EPOCH = 1721013.5
EPOCH_J2000_0_JD = 2451545.0
DAYS_PER_CENTURY = 36525.0
SECONDS_PER_DAY = 86400.0
LEAP_SECONDS = [
    46828800,
    78364801,
    109900802,
    173059203,
    252028804,
    315187205,
    346723206,
    393984007,
    425520008,
    457056009,
    504489610,
    551750411,
    599184012,
    820108813,
    914803214,
    1025136015,
    1119744016,
    1167264017,
]


@dispatch
def gps_time_to_utc(gps_time):
    """
    Convert GPS time to UTC.

    Parameters
    ----------
    gps_time : float
        GPS time in seconds.

    Returns
    -------
    datetime
        UTC time.
    """
    return GPS_EPOCH + datetime(second=gps_time - n_leap_seconds(gps_time))


@dispatch
def greenwich_mean_sidereal_time(gps_time):
    """
    Calculate the Greenwich Mean Sidereal Time.

    This is a thin wrapper around :py:func:`greenwich_sidereal_time` with the
    equation of the equinoxes set to zero.

    Parameters
    ----------
    gps_time : float
        GPS time in seconds.
    
    Returns
    -------
    float
        Greenwich Mean Sidereal Time in radians.
    """
    return greenwich_sidereal_time(gps_time, gps_time * 0)


@dispatch
def greenwich_sidereal_time(gps_time, equation_of_equinoxes):
    """
    Calculate the Greenwich Sidereal Time.

    Parameters
    ----------
    gps_time : float
        GPS time in seconds.
    equation_of_equinoxes : float
        Equation of the equinoxes in seconds.
    
    Returns
    -------
    float
    """
    julian_day = utc_to_julian_day(gps_time_to_utc(gps_time // 1))
    t_hi = (julian_day - EPOCH_J2000_0_JD) / DAYS_PER_CENTURY
    t_lo = (gps_time % 1) / (DAYS_PER_CENTURY * SECONDS_PER_DAY)

    t = t_hi + t_lo

    sidereal_time = (
        equation_of_equinoxes + (-6.2e-6 * t + 0.093104) * t**2 + 67310.54841
    )
    sidereal_time += 8640184.812866 * t_lo
    sidereal_time += 3155760000.0 * t_lo
    sidereal_time += 8640184.812866 * t_hi
    sidereal_time += 3155760000.0 * t_hi

    return sidereal_time * 2 * np.pi / SECONDS_PER_DAY


@dispatch
def n_leap_seconds(gps_time, leap_seconds):
    """
    Calculate the number of leap seconds that have occurred up to a given GPS time.

    Parameters
    ----------
    gps_time : float
        GPS time in seconds.
    leap_seconds : array_like
        GPS time of leap seconds.
    
    Returns
    -------
    float
        Number of leap seconds    
    """
    xp = array_module(gps_time)
    return xp.sum(gps_time > leap_seconds[:, None], axis=0).squeeze()


@dispatch
def n_leap_seconds(gps_time: Union[np.ndarray, float, int]):
    """
    Calculate the number of leap seconds that have occurred up to a given GPS time.

    Parameters
    ----------
    gps_time : float
        GPS time in seconds.
    
    Returns
    -------
    float
        Number of leap seconds
    """
    xp = array_module(gps_time)
    return n_leap_seconds(gps_time, xp.array(LEAP_SECONDS))


@dispatch
def utc_to_julian_day(utc_time):
    """
    Convert UTC time to Julian day.

    Parameters
    ----------
    utc_time : datetime
        UTC time.
    
    Returns
    -------
    float
        Julian day.

    """
    return utc_time.julian_day


@dispatch(precedence=1)
def gps_time_to_utc(gps_time: Real):
    return _time.gps_time_to_utc(gps_time)


@dispatch(precedence=1)
def greenwich_mean_sidereal_time(gps_time: Real):
    return _time.greenwich_mean_sidereal_time(gps_time)


@dispatch(precedence=1)
def greenwich_mean_sidereal_time(gps_time: ArrayLike):
    return _time.greenwich_mean_sidereal_time_vectorized(gps_time)


@dispatch(precedence=1)
def greenwich_sidereal_time(gps_time: Real, equation_of_equinoxes: Real):
    return _time.greenwich_sidereal_time(gps_time, equation_of_equinoxes)


@dispatch(precedence=1)
def n_leap_seconds(gps_time: Real):
    return _time.n_leap_seconds(gps_time)


@dispatch(precedence=1)
def utc_to_julian_day(utc_time: Real):
    return _time.utc_to_julian_day(utc_time)

