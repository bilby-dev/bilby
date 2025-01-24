from . import random


class GlobalMetaData(dict):
    """A class to store global meta data.

    This class is a singleton, meaning that only one instance can exist at a time.
    """

    _instance = None

    def add_to_meta_data(self, key, value):
        self[key] = value

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


global_meta_data = GlobalMetaData({
    "rng": random.rng
})
