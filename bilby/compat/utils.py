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
            try:
                return array_namespace(*arr)
            except TypeError:
                return np
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


def xp_wrap(func, no_xp=False):
    """
    A decorator that will figure out the array module from the input
    arguments and pass it to the function as the 'xp' keyword argument.

    Parameters
    ==========
    func: function
        The function to be decorated.
    no_xp: bool
        If True, the decorator will not attempt to add the 'xp' keyword
        argument and so the wrapper is a no-op.
    
    Returns
    =======
    function
        The decorated function.
    """

    def wrapped(self, *args, xp=None, **kwargs):
        if not no_xp and xp is None:
            try:
                if len(args) > 0:
                    array_module = array_namespace(*args)
                elif len(kwargs) > 0:
                    array_module = array_namespace(*kwargs.values())
                else:
                    array_module = np
                kwargs["xp"] = array_module
            except TypeError:
                kwargs["xp"] = np
        elif not no_xp:
            kwargs["xp"] = xp
        return func(self, *args, **kwargs)

    return wrapped


class BackendNotImplementedError(NotImplementedError):
    pass
