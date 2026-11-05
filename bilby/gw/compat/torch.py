import torch
from plum import dispatch

from ..time import (
    LEAP_SECONDS as _LEAP_SECONDS,
    n_leap_seconds as _n_leap_seconds,
)

__all__ = ["n_leap_seconds"]

LEAP_SECONDS = torch.tensor(_LEAP_SECONDS)


@dispatch
def n_leap_seconds(date: torch.Tensor) -> torch.Tensor:
    """
    Find the number of leap seconds required for the specified date.
    """
    return _n_leap_seconds(date, LEAP_SECONDS)
