from collections.abc import Iterable

import numpy as np
from array_api_compat import array_namespace

from ..core.utils.log import logger

__all__ = ["array_module", "promote_to_array"]


def array_module(arr):
    try:
        return array_namespace(arr)
    except TypeError:
        if isinstance(arr, dict):
            try:
                return array_namespace(*[val for val in arr.values() if not isinstance(val, str)])
            except TypeError:
                return np
        elif arr.__class__.__module__ == "builtins" and isinstance(arr, Iterable):
            return array_namespace(arr)
        elif arr.__class__.__module__ == "builtins":
            return np
        elif arr.__module__.startswith("pandas"):
            return np
        else:
            logger.warning(
                f"Unknown array module for type: {type(arr)} Defaulting to numpy."
            )
            return np


def promote_to_array(args, backend, skip=None):
    if skip is None:
        skip = len(args)
    else:
        skip = len(args) - skip
    if backend.__name__ != "numpy":
        args = tuple(backend.array(arg) for arg in args[:skip]) + args[skip:]
    return args


def xp_wrap(func):

    def wrapped(self, *args, **kwargs):
        if "xp" not in kwargs:
            try:
                kwargs["xp"] = array_module(*args)
            except TypeError:
                pass
        return func(self, *args, **kwargs)

    return wrapped


class BackendNotImplementedError(NotImplementedError):
    pass
