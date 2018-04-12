import unittest
import numpy as np
import os
import shutil
import peyote
import logging
from past.builtins import execfile


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

        self.assertTrue(np.array_equal(self.msd['frequencies'], frequencies_saved))
        self.assertAlmostEqual(all(self.msd['hf_signal_and_noise'] - hf_signal_and_noise_saved), 0.00000000, 5)

    def test_recover_luminosity_distance(self):
        likelihood = peyote.likelihood.Likelihood(
            [self.msd['IFO']], self.msd['waveform_generator'])

        prior = self.msd['simulation_parameters'].copy()
        dL = self.msd['simulation_parameters']['luminosity_distance']
        prior['luminosity_distance'] = peyote.parameter.Parameter(
            'luminosity_distance',
            prior=peyote.prior.Uniform(lower=dL - 10, upper=dL + 10))

        result = peyote.run_sampler(likelihood, prior, sampler='nestle',
                                    verbose=False)
        self.assertAlmostEqual(np.mean(result.samples), dL,
                               delta=np.std(result.samples))


if __name__ == '__main__':
    unittest.main()


