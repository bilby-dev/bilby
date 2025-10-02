import jax.numpy as jnp
from jax import Array
from plum import dispatch

from ..time import LEAP_SECONDS as _LEAP_SECONDS, n_leap_seconds

__all__ = ["n_leap_seconds"]

LEAP_SECONDS = jnp.array(_LEAP_SECONDS)


@dispatch
def n_leap_seconds(date: Array):
    """
    Find the number of leap seconds required for the specified date.
    """
    return n_leap_seconds(date, LEAP_SECONDS)

