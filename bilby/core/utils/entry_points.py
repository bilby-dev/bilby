from importlib.metadata import entry_points


def get_entry_points(group):
    """Return a dictionary of entry points for a given group

    Parameters
    ----------
    group: str
        Entry points you wish to query
    """
    return {custom.name: custom for custom in entry_points(group=group)}
