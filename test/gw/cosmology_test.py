import unittest

from astropy.cosmology import WMAP9, Planck15
from bilby.gw import cosmology
import lal


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
        self.assertEqual(cosmology.get_cosmology().name, "Planck15")


class TestPlanck15LALCosmology(unittest.TestCase):

    def setUp(self):
        pass

    def test_redshift_to_luminosity_distance(self):
        z = 1.0
        cosmo = cosmology.get_cosmology("Planck15_LAL")
        dist_bilby = cosmo.luminosity_distance(z)
        # Change to use CreateDefaultCosmologicalParameters once it is
        # available in a release
        omega = lal.CreateCosmologicalParameters(
            h=0.679,
            om=0.3065,
            ol=1 - 0.3065,
            w0=-1.0,
            w1=0.0,
            w2=0.0
        )
        dist_lal = lal.LuminosityDistance(omega=omega, z=z)
        # Results are the same to within 9 decimal places
        self.assertAlmostEqual(dist_bilby.value, dist_lal)


if __name__ == "__main__":
    unittest.main()
