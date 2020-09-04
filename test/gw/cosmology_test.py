import unittest

from astropy.cosmology import WMAP9, Planck15
from bilby.gw import cosmology


class TestSetCosmology(unittest.TestCase):
    def setUp(self):
        pass

    def test_setting_cosmology_with_string(self):
        cosmology.set_cosmology("WMAP9")
        self.assertEqual(cosmology.COSMOLOGY[1], "WMAP9")
        cosmology.set_cosmology("Planck15")

    def test_setting_cosmology_with_astropy_object(self):
        cosmology.set_cosmology(WMAP9)
        self.assertEqual(cosmology.COSMOLOGY[1], "WMAP9")
        cosmology.set_cosmology(Planck15)

    def test_setting_cosmology_with_default(self):
        cosmology.set_cosmology()
        self.assertEqual(cosmology.COSMOLOGY[1], cosmology.DEFAULT_COSMOLOGY.name)

    def test_setting_cosmology_with_flat_lambda_cdm_dict(self):
        cosmo_dict = dict(H0=67.7, Om0=0.3)
        cosmology.set_cosmology(cosmo_dict)
        self.assertEqual(cosmology.COSMOLOGY[1][:13], "FlatLambdaCDM")

    def test_setting_cosmology_with_lambda_cdm_dict(self):
        cosmo_dict = dict(H0=67.7, Om0=0.3, Ode0=0.7)
        cosmology.set_cosmology(cosmo_dict)
        self.assertEqual(cosmology.COSMOLOGY[1][:9], "LambdaCDM")

    def test_setting_cosmology_with_w_cdm_dict(self):
        cosmo_dict = dict(H0=67.7, Om0=0.3, Ode0=0.7, w0=-1.0)
        cosmology.set_cosmology(cosmo_dict)
        self.assertEqual(cosmology.COSMOLOGY[1][:4], "wCDM")


class TestGetCosmology(unittest.TestCase):
    def setUp(self):
        pass

    def test_getting_cosmology_with_string(self):
        self.assertEqual(cosmology.get_cosmology("WMAP9").name, "WMAP9")

    def test_getting_cosmology_with_default(self):
        self.assertEqual(cosmology.get_cosmology(), cosmology.COSMOLOGY[0])


if __name__ == "__main__":
    unittest.main()
