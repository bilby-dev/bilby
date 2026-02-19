try:
    from .jax import n_leap_seconds
except ModuleNotFoundError:
    pass


try:
    from .cython import gps_time_to_utc
except ModuleNotFoundError:
    pass

try:
    from .torch import n_leap_seconds
except ModuleNotFoundError:
    pass