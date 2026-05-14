import inspect
from collections.abc import Iterable

import numpy as np
from array_api_compat import array_namespace, is_numpy_namespace

from ..core.utils.log import logger

__all__ = ["array_module", "promote_to_array"]


def array_module(arr):
    """
    Infer the array module (namespace) from the input argument.
    This is a generalization of the :code:`array_api_compat.array_namespace`
    function that can handle a wider variety of inputs, including some nested
    structures.

    This function determines which array library backend is being used
    by inspecting the input argument. It handles various input types and
    fallback mechanisms to ensure a valid array module is always returned.

    The inference logic proceeds as follows:
    1. If a single-element tuple is provided, extract the element first.
    2. Attempt to use the array_api_compat.array_namespace() function
       directly, which works for most array-like objects.
    3. If that fails, handle special cases:
       - Dictionaries: extract values and infer from non-string values
       - Builtin iterables (list, tuple, etc.): infer from elements
       - Builtin scalars: default to numpy
       - Pandas objects: default to numpy (treated as numpy-compatible)
       - Unknown types: log a warning and default to numpy

    This is a best-effort function, but will not cover all possible edge cases.

    Parameters
    ==========
    arr: array-like, tuple, dict, or other type
        The input argument to infer the array module from. Can be:
        - An array object (numpy, cupy, jax.numpy, etc.)
        - A tuple of arrays (single-element unwrapped)
        - A dictionary with array values
        - An iterable containing arrays
        - A builtin scalar or type

    Returns
    =======
    module
        The array namespace module (e.g., numpy, cupy, jax.numpy, etc.).
        Defaults to numpy if the module cannot be determined.

    Examples
    ========
    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> array_module(np.array([1, 2, 3]))
    <module 'numpy' ...>

    >>> array_module(jnp.array([1, 2, 3]))
    <module 'jax.numpy' ...>

    >>> array_module({'data': np.array([1, 2, 3])})
    <module 'numpy' ...>

    >>> array_module([np.array([1]), np.array([2])])
    <module 'numpy' ...>

    >>> array_module([1, jnp.array([2])])
    <module 'jax.numpy' ...>

    >>> array_module(5)
    <module 'numpy' ...>
    """
    if isinstance(arr, tuple) and len(arr) == 1:
        arr = arr[0]
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


def promote_to_array(args, xp, skip=None):
    """
    Promote arguments to arrays using the specified array module.

    Parameters
    ==========
    args: tuple
        Tuple of arguments to promote.
    xp: module
        The array module (namespace) to use for promotion.
    skip: int, optional
        Number of trailing arguments to skip promotion for.
        Defaults to None (promote all arguments).

    Returns
    =======
    tuple
        Arguments with the first (len(args) - skip) elements promoted to
        arrays using the specified module.

    Notes
    =====
    This function cannot handle manual specification of devices. Arrays
    are promoted to the default device of the specified array module. This
    may be added in future if there is a need.
    """
    if skip is None:
        skip = len(args)
    else:
        skip = len(args) - skip
    if not is_numpy_namespace(xp):
        args = tuple(xp.array(arg) for arg in args[:skip]) + args[skip:]
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
    def parse_args_kwargs_for_xp(*args, xp=None, **kwargs):
        if not no_xp and xp is None:
            try:
                # if the user specified the target arrays in kwargs
                # we need to be able to support this, if there is
                # only one kwargs, pass it through alone, this is
                # sometimes a dictionary of arrays so this is needed
                # to remove a level of nesting
                if len(args) > 0:
                    xp = array_module(args)
                elif len(kwargs) == 1:
                    xp = array_module(next(iter(kwargs.values())))
                elif len(kwargs) > 1:
                    xp = array_module(kwargs)
                else:
                    xp = np
                kwargs["xp"] = xp
            except TypeError as e:
                print("type failed", e)
                kwargs["xp"] = np
        elif not no_xp:
            kwargs["xp"] = xp
        return args, kwargs

    sig = inspect.signature(func)
    if any(name in sig.parameters for name in ("self", "cls")):
        def wrapped(self, *args, xp=None, **kwargs):
            args, kwargs = parse_args_kwargs_for_xp(*args, xp=xp, **kwargs)
            return func(self, *args, **kwargs)
    else:
        def wrapped(*args, xp=None, **kwargs):
            args, kwargs = parse_args_kwargs_for_xp(*args, xp=xp, **kwargs)
            return func(*args, **kwargs)

    return wrapped


class BackendNotImplementedError(NotImplementedError):
    pass
