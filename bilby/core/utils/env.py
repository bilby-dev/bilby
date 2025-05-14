

def string_to_boolean(value):
    """Convert a string to a boolean.

    Supports True/False (case-insensitive), and 1/0.
    """
    value = value.strip().lower()
    if value in ['true', '1']:
        return True
    elif value in ['false', '0']:
        return False
    else:
        raise ValueError(
            f"Invalid value for boolean: {value}"
        )
