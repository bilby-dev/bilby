import unittest
import os
import shutil
import logging

# Required to run the tests
from past.builtins import execfile
from context import tupak

# Imported to ensure the examples run
import numpy as np
import inspect

tupak.utils.command_line_args.test = True


class Test(unittest.TestCase):
    outdir = 'outdir'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(dir_path, os.path.pardir))

    @classmethod
    def setUpClass(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning(
                    "{} not removed prior to tests".format(self.outdir))

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning(
                    "{} not removed prior to tests".format(self.outdir))

    def test_examples(self):
        """ Loop over examples to check they run """
        examples = ['examples/injection_examples/basic_tutorial.py',
                    'examples/injection_examples/change_sampled_parameters.py',
                    'examples/injection_examples/marginalized_likelihood.py',
                    'examples/injection_examples/create_your_own_time_domain_source_model.py',
                    'examples/other_examples/linear_regression.py',
                    ]
        for filename in examples:
            print("Testing {}".format(filename))
            execfile(filename)


if __name__ == '__main__':
    unittest.main()


