from __future__ import division, print_function
import numpy as np
import peyote


class Parameter:

    def __init__(self, name, prior=None, value=None, latex_label=None, fixed=False):
        self.name = name

        if prior is None:
            self.set_default_prior()
        else:
            self.prior = prior

        if value is None:
            self.set_default_values()
        else:
            self.value = value

        if latex_label is None:
            self.latex_label = name
        else:
            self.set_default_latex_label()

        if fixed:
            self.is_fixed = True
        else:
            self.is_fixed = False

    def fix(self, value=None):
        """
        Specify parameter as fixed, this will not be sampled.
        """
        self.is_fixed = True
        if value is not None:
            self.value = value
        elif self.value == np.nan:
            raise ValueError("You can't fix the value to be np.nan. You need to assign it a legal value")
        self.prior = None

    def set_value(self, value):
        """Set a value for the parameter"""
        self.value = value
        return None

    def set_default_prior(self):

        if self.name == 'mass_1':
            self.prior = peyote.prior.PowerLaw(alpha=0, bounds=(5, 100))
        if self.name == 'mass_2':
            self.prior = peyote.prior.PowerLaw(alpha=0, bounds=(5, 100))
        if self.name == 'mchirp':
            self.prior = peyote.prior.PowerLaw(alpha=0, bounds=(5, 100))
        if self.name == 'q':
            self.prior = peyote.prior.PowerLaw(alpha=0, bounds=(0, 1))

        # spins
        if self.name == 'a1':
            self.prior = peyote.prior.PowerLaw(alpha=0, bounds=(0, 1))
        if self.name == 'a2':
            self.prior = peyote.prior.PowerLaw(alpha=0, bounds=(0, 1))
        if self.name == 'tilt1':
            self.prior = peyote.prior.Sine()
        if self.name == 'tilt2':
            self.prior = peyote.prior.Sine()
        if self.name == 'phi1':
            self.prior = peyote.prior.PowerLaw(alpha=0, bounds=(0, 2 * np.pi))
        if self.name == 'phi2':
            self.prior = peyote.prior.PowerLaw(alpha=0, bounds=(0, 2 * np.pi))

        # extrinsic
        if self.name == 'luminosity_distance':
            self.prior = peyote.prior.PowerLaw(alpha=2, bounds=(1e2, 5e3))
        if self.name == 'dec':
            self.prior = peyote.prior.Cosine()
        if self.name == 'ra':
            self.prior = peyote.prior.PowerLaw(alpha=0, bounds=(0, 2 * np.pi))
        if self.name == 'iota':
            self.prior = peyote.prior.Sine()
        if self.name == 'psi':
            self.prior = peyote.prior.PowerLaw(alpha=0, bounds=(0, 2 * np.pi))
        if self.name == 'phase':
            self.prior = peyote.prior.PowerLaw(alpha=0, bounds=(0, 2 * np.pi))

    def set_default_values(self):
        # spins
        if self.name == 'a1':
            self.value = 0
        elif self.name == 'a2':
            self.value = 0
        elif self.name == 'tilt1':
            self.value = 0
        elif self.name == 'tilt2':
            self.value = 0
        elif self.name == 'phi1':
            self.value = 0
        elif self.name == 'phi2':
            self.value = 0
        else:
            self.value = np.nan

    def set_default_latex_label(self):
        if self.name == 'mass_1':
            self.latex_label = '$m_1$'
        elif self.name == 'mass_2':
            self.latex_label = '$m_2$'
        elif self.name == 'mchirp':
            self.latex_label = '$\mathcal{M}$'
        elif self.name == 'q':
            self.latex_label = 'q'
        elif self.name == 'a1':
            self.latex_label = 'a_1'
        elif self.name == 'a2':
            self.latex_label = 'a_2'
        elif self.name == 'tilt1':
            self.latex_label = 'tilt_1'
        elif self.name == 'tilt2':
            self.latex_label = 'tilt_2'
        elif self.name == 'phi1':
            self.latex_label = '\phi_1'
        elif self.name == 'phi2':
            self.latex_label = '\phi_2'
        elif self.name == 'luminosity_distance':
            self.latex_label = 'd'
        elif self.name == 'dec':
            self.latex_label = '\mathrm{DEC}'
        elif self.name == 'ra':
            self.latex_label = '\mathrm{RA}'
        elif self.name == 'iota':
            self.latex_label = '\iota'
        elif self.name == 'psi':
            self.latex_label = '\psi'
        elif self.name == 'phase':
            self.latex_label = '\phi'
        elif self.name == 'tc':
            self.latex_label = 't_c'
        else:
            self.latex_label = self.name
