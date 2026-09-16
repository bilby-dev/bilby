from . import random
from .log import logger


class GlobalMetaData(dict):
    """A class to store global meta data.

    This class is a singleton, meaning that only one instance can exist at a time.
    """

    _instance = None

    def __init__(self, mapping=None, /, **kwargs):
        if mapping is None:
            mapping = {}
        else:
            mapping = dict(mapping)
        mapping.update(kwargs)
        for key, item in mapping.items():
            self.__setitem__(key, item)

    def __setitem__(self, key, item):
        if key in self:
            logger.debug(
                f"Overwriting meta data key {key} with value {item}. "
                f"Old value was {self[key]}"
            )
        else:
            logger.debug(f"Setting meta data key {key} with value {item}")
        return super().__setitem__(key, item)

    def update(self, *args, **kwargs):
        for key, item in dict(*args, **kwargs).items():
            self.__setitem__(key, item)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        else:
            logger.warning(
                "GlobalMetaData has already been instantiated. "
                "Returning the existing instance."
            )
        return cls._instance


global_meta_data = GlobalMetaData({
    "rng": random.rng
})
