from numpy.random import default_rng, SeedSequence


def __getattr__(name):
    if name == "rng":
        return Generator.rng


class Generator:
    rng = default_rng()


def seed(seed):
    from .meta_data import global_meta_data

    Generator.rng = default_rng(seed)
    global_meta_data["rng"] = Generator.rng
    global_meta_data["seed"] = seed


def generate_seeds(nseeds):
    return SeedSequence(Generator.rng.integers(0, 2**63 - 1, size=4)).spawn(nseeds)
