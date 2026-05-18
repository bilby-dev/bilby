from ...compat.utils import BILBY_ARRAY_API

try:
    from .cython import gps_time_to_utc
except ModuleNotFoundError:
    pass

if BILBY_ARRAY_API:
    try:
        from .jax import n_leap_seconds
    except ModuleNotFoundError:
        pass

    try:
        from .torch import n_leap_seconds
    except ModuleNotFoundError:
        pass