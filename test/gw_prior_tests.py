from __future__ import division, absolute_import
import unittest
import tupak
import os
import sys


class TestBBHPriorSet(unittest.TestCase):

    def setUp(self):
        self.prior_dict = dict()
        self.base_directory =\
            '/'.join(os.path.dirname(
                os.path.abspath(sys.argv[0])).split('/')[:-1])
        self.filename =\
            self.base_directory + '/tupak/gw/prior_files/GW150914.prior'
        self.default_prior = tupak.gw.prior.BBHPriorSet(
            filename=self.base_directory\
                     + '/tupak/gw/prior_files/binary_black_holes.prior')

    def tearDown(self):
        del self.prior_dict
        del self.filename

    def test_create_default_prior(self):
        default = tupak.gw.prior.BBHPriorSet()
        minima = all([self.default_prior[key].minimum == default[key].minimum
                      for key in default.keys()])
        maxima = all([self.default_prior[key].maximum == default[key].maximum
                      for key in default.keys()])
        names = all([self.default_prior[key].name == default[key].name
                     for key in default.keys()])

        self.assertTrue(all([minima, maxima, names]))

    def test_create_from_dict(self):
        tupak.gw.prior.BBHPriorSet(dictionary=self.prior_dict)

    def test_create_from_filename(self):
        tupak.gw.prior.BBHPriorSet(filename=self.filename)


if __name__ == '__main__':
    unittest.main()