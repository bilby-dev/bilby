import numpy as np
from array_api_compat import array_namespace

__all__ = ["array_module", "promote_to_array"]


def array_module(arr):
    if arr.__class__.__module__ == "builtins":
        return np
    else:
        return array_namespace(arr)


def promote_to_array(args, backend, skip=None):
    if skip is None:
        skip = len(args)
    else:
        skip = len(args) - skip
    if backend.__name__ != "numpy":
        args = tuple(backend.array(arg) for arg in args[:skip]) + args[skip:]
    return args

