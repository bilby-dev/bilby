import unittest
import numpy
import lalsimulation as lalsim
from bilby.gw.eos import SpectralDecompositionEOS, EOSFamily, TabularEOS
from bilby.core import utils


KNOWN_TOV_RESULT = 1
KNOWN_EOS_PRIOR_RESULTS = 1
ENERGY_FROM_PRESSURE_RESULTS = 0.04047517810698063
EOS_FROM_TABLE = EOSFamily(TabularEOS('AP4', True))
EOS_FROM_SPRECTRAL_DECOMPOSITION = EOSFamily(SpectralDecompositionEOS(gammas=[0.8651, 0.1548, -0.0151, -0.0002],
                                                                      p0=1.5e33,
                                                                      e0=2.02e14,
                                                                      xmax=7.04))
MASSES_TO_TEST = numpy.linspace(0.9, 2.0, 12)
PRESSURE = 1e-11
ENTHALPY = 1.2
ENERGY_DENSITY = 9.541419e-11

# Setup for MPA1 Comparison
PRESSURE_0 = 5.3716e32
ENERGY_DENSITY_0 = 1.1555e35 / (utils.speed_of_light * 100) ** 2.
XMAX = 12.3081

GAMMAS = [1.0215, 0.1653, -0.0235, -0.0004]
LALSIM_MPA1 = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(GAMMAS[0], GAMMAS[1], GAMMAS[2], GAMMAS[3])
BILBY_MPA1 = SpectralDecompositionEOS(GAMMAS, PRESSURE_0, ENERGY_DENSITY_0, XMAX)

MPA1_PRESSURES = BILBY_MPA1.e_pdat.T[0]
LALSIM_MPA1_ENERGY_DENSITY = [lalsim.SimNeutronStarEOSEnergyDensityOfPressureGeometerized(PRESSURE, LALSIM_MPA1)
                              for PRESSURE in MPA1_PRESSURES[:-1]]
BILBY_MPA1_ENERGY_DENSITY = BILBY_MPA1.e_pdat.T[1][:-1]


class TestEOSFamily(unittest.TestCase):
    def test_spectral_decomposition_energy_from_pressure(self):
        self.assertAlmostEqual(EOS_FROM_TABLE.eos.energy_from_pressure(PRESSURE),
                               3.2736497985232014e-10)

        self.assertAlmostEqual(EOS_FROM_SPRECTRAL_DECOMPOSITION.eos.energy_from_pressure(PRESSURE),
                               3.270622527256167e-10)

    def test_spectral_decomposition_pressure_from_pseudo_enthalpy(self):
        self.assertAlmostEqual(EOS_FROM_TABLE.eos.pressure_from_pseudo_enthalpy(ENTHALPY),
                               2.7338376042831513e-09)

        self.assertAlmostEqual(EOS_FROM_SPRECTRAL_DECOMPOSITION.eos.pressure_from_pseudo_enthalpy(ENTHALPY),
                               2.754018499535077e-09)

    def test_spectral_decomposition_energy_density_from_pseudo_enthalpy(self):
        self.assertAlmostEqual(EOS_FROM_TABLE.eos.energy_density_from_pseudo_enthalpy(ENTHALPY),
                               2.9486942467903607e-09)

        self.assertAlmostEqual(EOS_FROM_SPRECTRAL_DECOMPOSITION.eos.energy_density_from_pseudo_enthalpy(ENTHALPY),
                               3.0402598495601078e-09)

    def test_spectral_decomposition_pseudo_enthalpy_from_energy_density(self):
        self.assertAlmostEqual(EOS_FROM_TABLE.eos.pseudo_enthalpy_from_energy_density(ENERGY_DENSITY),
                               0.024415755781136812)
        print(EOS_FROM_SPRECTRAL_DECOMPOSITION.eos.pseudo_enthalpy_from_energy_density(ENERGY_DENSITY))
        self.assertAlmostEqual(EOS_FROM_SPRECTRAL_DECOMPOSITION.eos.pseudo_enthalpy_from_energy_density(ENERGY_DENSITY),
                               0.02420629785967365)


class TestBilbyLALSimComparison(unittest.TestCase):
    def test_spectral_decomposition_MPA1(self):
        numpy.testing.assert_allclose(LALSIM_MPA1_ENERGY_DENSITY, BILBY_MPA1_ENERGY_DENSITY, rtol=1e6)
