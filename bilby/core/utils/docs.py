def docstring(docstr, sep="\n"):
    """
    Decorator: Append to a function's docstring.

    This is required for e.g., :code:`classmethods` as the :code:`__doc__`
    can't be changed after.

    Parameters
    ==========
    docstr: str
        The docstring
    sep: str
        Separation character for appending the existing docstring.
    """
    def _decorator(func):
        if func.__doc__ is None:
            func.__doc__ = docstr
        else:
            func.__doc__ = sep.join([func.__doc__, docstr])
        return func
    return _decorator
