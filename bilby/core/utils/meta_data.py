from collections import UserDict

from . import random
from .log import logger


class GlobalMetaData(UserDict):
    """A class to store global meta data.

    This class is a singleton, meaning that only one instance can exist at a time.
    """

    _instance = None

    def __setitem__(self, key, item):
        if key in self:
            logger.debug(
                f"Overwriting meta data key {key} with value {item}. "
                f"Old value was {self[key]}"
            )
        else:
            logger.debug(f"Setting meta data key {key} with value {item}")
        return super().__setitem__(key, item)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


global_meta_data = GlobalMetaData({
    "rng": random.rng
})
