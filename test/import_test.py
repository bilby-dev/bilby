"""
Tests the number of packages imported with a top-level :code:`import bilby`
statement.
"""

import sys

import bilby  # noqa

unique_packages = set(sys.modules)

unwanted = {
    "lal", "lalsimulation", "matplotlib",
    "h5py", "dill", "tqdm", "tables", "deepdish", "corner",
}

for filename in ["sampler_requirements.txt", "optional_requirements.txt"]:
    with open(filename, "r") as ff:
        packages = ff.readlines()
        for package in packages:
            package = package.split(">")[0].split("<")[0].split("=")[0].strip()
            unwanted.add(package)

if not unique_packages.isdisjoint(unwanted):
    raise ImportError(
        f"{' '.join(unique_packages.intersection(unwanted))} imported with Bilby"
    )
