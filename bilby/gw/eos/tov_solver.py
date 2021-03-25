# Monica Rizzo, 2019

import numpy as np


class IntegrateTOV:
    """Class that given an initial pressure a mass radius value and a k2-love number
    """

    def __init__(self, eos, eps_0):
        self.eos = eos
        # determine central values
        pseudo_enthalpy0 = self.eos.pseudo_enthalpy_from_energy_density(eps_0)

        # evolve first step analytically
        self.pseudo_enthalpy = pseudo_enthalpy0 - 1e-3 * pseudo_enthalpy0

        mass0, radius0 = self.__mass_radius_cent(pseudo_enthalpy0, self.pseudo_enthalpy)

        # k2 integration starting vals
        H0 = radius0 ** 2
        B0 = 2. * radius0

        self.y = np.array([mass0, radius0, H0, B0])

    def __mass_radius_cent(self, pseudo_enthalpy0, pseudo_enthalpy):
        """
        Calculate approximate mass/radius central values (for starting integration)

        Eqns 7 + 8 from http://www.ccom.ucsd.edu/~lindblom/Publications/50_1992ApJ...398..569L.pdf
        """

        eps_c = self.eos.energy_density_from_pseudo_enthalpy(pseudo_enthalpy0)
        p_c = self.eos.pressure_from_pseudo_enthalpy(pseudo_enthalpy0)
        depsdh_c = self.eos.dedh(pseudo_enthalpy0)

        radius = ((3. * (pseudo_enthalpy0 - pseudo_enthalpy)) / (2. * np.pi * (eps_c + 3. * p_c))) ** 0.5 \
            * (1. - 0.25 * (eps_c - 3. * p_c - (3. / 5.) * depsdh_c) *
               ((pseudo_enthalpy0 - pseudo_enthalpy) / (eps_c + 3. * p_c)))

        mass = (4. * np.pi) / 3. * eps_c * radius ** 3 * (1. - (3. / 5.) *
                                                          depsdh_c * (pseudo_enthalpy0 - pseudo_enthalpy) / eps_c)

        return mass, radius

    def __tov_eqns(self, h, y):
        """
        Taken from http://www.ccom.ucsd.edu/~lindblom/Publications/50_1992ApJ...398..569L.pdf

        TOV equations in terms of pseudo-enthalpy; evolves in mass and radius
        H and B equations taken from:

        https://journals.aps.org/prd/pdf/10.1103/PhysRevD.81.123016

        Transformed (hopefully to evolve in H(h) and B(h)
        """

        m = y[0]
        r = y[1]
        H = y[2]
        B = y[3]
        eps = self.eos.energy_density_from_pseudo_enthalpy(h, interp_type='CubicSpline')
        p = self.eos.pressure_from_pseudo_enthalpy(h, interp_type='CubicSpline')
        depsdp = self.eos.dedp(p, interp_type='CubicSpline')

        dmdh = (- (4. * np.pi * eps * r ** 3 * (r - 2. * m)) /
                (m + 4. * np.pi * r ** 3 * p))

        drdh = -(r * (r - 2. * m)) / (m + 4. * np.pi * r ** 3 * p)

        dHdh = B * drdh

        # taken from Damour & Nagar
        e_lam = (1. - 2. * m / r) ** (-1)

        C1 = 2. / r + e_lam * (2. * m / r ** 2. + 4. * np.pi * r * (p - eps))
        C0 = (e_lam * (- 6. / r ** 2. + 4. * np.pi * (eps + p) *
              depsdp + 4. * np.pi * (5. * eps + 9. * p)) -
              (2. * (m + 4. * np.pi * r ** 3. * p) /
              (r ** 2. - 2. * m * r)) ** 2.)

        dBdh = -(C1 * B + C0 * H) * drdh

        y_dot = np.array([dmdh, drdh, dHdh, dBdh])

        return y_dot

    def __calc_k2(self, R, Beta, H, C):
        """
        Using evolved quantities, calculate the k2 love

        https://journals.aps.org/prd/pdf/10.1103/PhysRevD.81.123016

           https://journals.aps.org/prd/pdf/10.1103/PhysRevD.80.084035
        """

        y = (R * Beta) / H

        num = ((8. / 5.) * (1. - 2. * C) ** 2 *
               C ** 5 * (2. * C * (y - 1.) - y + 2.))
        denom = (2. * C * (4. * (y + 1.) * C ** 4 + (6. * y - 4.) * C ** 3 +
                 (26. - 22. * y) * C ** 2 + 3. * (5. * y - 8.) *
                 C - 3. * y + 6.) - 3. * (1. - 2 * C) ** 2 *
                 (2. * C * (y - 1.) - y + 2.) *
                 np.log(1. / (1. - 2. * C)))

        return num / denom

    def integrate_TOV(self):
        """
        Evolves TOV+k2 equations and returns final quantities
        """
        from scipy.integrate import solve_ivp

        # integration settings the same as in lalsimulation
        rel_err = 1e-4
        abs_err = 0.0

        result = solve_ivp(self.__tov_eqns, (self.pseudo_enthalpy, 1e-16), self.y, rtol=rel_err,
                           atol=abs_err)
        m_fin = result.y[0, -1]
        r_fin = result.y[1, -1]
        H_fin = result.y[2, -1]
        B_fin = result.y[3, -1]

        k_2 = self.__calc_k2(r_fin, B_fin, H_fin, m_fin / r_fin)

        return m_fin, r_fin, k_2
