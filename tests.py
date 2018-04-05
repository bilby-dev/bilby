import unittest
import numpy as np
import os
import shutil
import peyote
import logging


class Test(unittest.TestCase):
    outdir = 'TestDir'
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Run a script to produce standard data
    msd = {}  # A dictionary of variables saved in make_standard_data.py
    execfile(dir_path + '/tutorials/make_standard_data.py', msd)

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

    def test_make_standard_data(self):
        " Load in the saved standard data and compare with new data "

        # Load in the saved standard data
        frequencies_saved, hf_real_saved, hf_imag_saved = np.loadtxt(
            self.dir_path + '/tutorials/standard_data.txt').T
        hf_signal_and_noise_saved = hf_real_saved + 1j * hf_imag_saved

        self.assertTrue(
            all(self.msd['frequencies'] == frequencies_saved))
        self.assertTrue(
            all(self.msd['hf_signal_and_noise'] == hf_signal_and_noise_saved))

    def test_calculate_likelihood_for_standard_data(self):
        pS = self.msd['simulation_parameters']
        pN = self.msd['simulation_parameters'].copy()
        pN['luminosity_distance'] = 1e6

        likelihood = peyote.likelihood.likelihood(
            [self.msd['IFO']], self.msd['source'])
        loglS = likelihood.loglikelihood(pS)
        loglN = likelihood.loglikelihood(pN)
        self.assertAlmostEqual(
            loglS - loglN,
            36573752.328521997, 5)


if __name__ == '__main__':
    unittest.main()
