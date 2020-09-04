import matplotlib

matplotlib.use("Agg")

import unittest
import os
import shutil
import logging

# Required to run the tests
from past.builtins import execfile
import bilby.core.utils

# Imported to ensure the examples run
import numpy as np  # noqa: F401
import inspect  # noqa: F401

bilby.core.utils.command_line_args.bilby_test_mode = True


class ExampleTest(unittest.TestCase):
    outdir = "outdir"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(dir_path, os.path.pardir))

    @classmethod
    def setUpClass(cls):
        if os.path.isdir(cls.outdir):
            try:
                shutil.rmtree(cls.outdir)
            except OSError:
                logging.warning("{} not removed prior to tests".format(cls.outdir))

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.outdir):
            try:
                shutil.rmtree(cls.outdir)
            except OSError:
                logging.warning("{} not removed prior to tests".format(cls.outdir))

    def test_examples(self):
        """ Loop over examples to check they run """
        examples = [
            "examples/core_examples/linear_regression.py",
            "examples/core_examples/linear_regression_unknown_noise.py",
        ]
        for filename in examples:
            print("Testing {}".format(filename))
            execfile(filename)

    def test_gw_examples(self):
        """ Loop over examples to check they run """
        examples = [
            "examples/gw_examples/injection_examples/fast_tutorial.py",
            "examples/gw_examples/data_examples/GW150914.py",
        ]
        for filename in examples:
            print("Testing {}".format(filename))
            execfile(filename)


if __name__ == "__main__":
    unittest.main()
