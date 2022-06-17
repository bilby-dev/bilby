#!/usr/bin/env python
"""
A script to demonstrate how to create and plot EoS's with Bilby
"""
from bilby.gw import eos

# In this script we're going to use Bilby to plot the MPA1 EoS from tabulated data
# and plot its representation as a spectral decomposition EoS.

# First, we specify the spectral parameter values for the MPA1 EoS.
MPA1_gammas = [1.0215, 0.1653, -0.0235, -0.0004]
MPA1_p0 = 1.51e33  # Pressure in CGS
MPA1_e0_c2 = 2.04e14  # \epsilon_0 / c^2 in CGS
MPA1_xmax = 6.63  # Dimensionless ending pressure

# Create the spectral decomposition EoS class
MPA1_spectral = eos.SpectralDecompositionEOS(
    MPA1_gammas, p0=MPA1_p0, e0=MPA1_e0_c2, xmax=MPA1_xmax, npts=100
)

# And create another from tabulated data
MPA1_tabulated = eos.TabularEOS("MPA1")

# Now let's plot them
# To do so, we specify a representation and plot ranges.
MPA1_spectral_plot = MPA1_spectral.plot(
    "pressure-energy_density", xlim=[1e22, 1e36], ylim=[1e9, 1e36]
)
MPA1_tabular_plot = MPA1_tabulated.plot(
    "pressure-energy_density", xlim=[1e22, 1e36], ylim=[1e9, 1e36]
)
MPA1_spectral_plot.savefig("spectral_mpa1.pdf")
MPA1_tabular_plot.savefig("tabular_mpa1.pdf")
