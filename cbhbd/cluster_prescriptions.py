import numpy
from numpy import log, sqrt, pi, exp, tanh, arctan, arctanh
from scipy.special import erf, hyp2f1, spence, gammainc, gammaincc, gamma, beta, betainc
from scipy.integrate import quad

# List of some prescriptions used in cluster.py, moved here to improve readability.

def _get_cluster_prescriptions(self):
    # Define models for stellar evolution for ssp=False. Coefficients can capture different dependences on metallicity, or an explicit dependence can be inserted.
    self.sev_dict = {
        'constant': lambda Z, t: self.nu,  # M(t) is a power-law.
        'power_law': lambda Z, t: self.nu * (t / self.tsev) ** (- self.c1),
        # M(t) is exponential. nu=0.087, tsev=1.49, c1=0.046 agree with a Kroupa IMF.
        'exponential': lambda Z, t: self.nu * exp(-self.c1 * t / self.tsev),
        # M(t) depends on exponential integral Ei.
        'logarithmic': lambda Z, t: self.nu * (1 + self.c1 * log(t / self.tsev) - self.c2 * log(t / self.tsev) ** 2)
        # M(t) is power-law products. nu=0.079, tsev=1.4, c1=0.004, c2=0.003 seem to work for a Kroupa IMF
    }

    # Define the tidal model dictionary.
    self.tidal_models = {
        # Both the half-mass and the tidal radius should be in [pc]. All the rest parameters are fixed. In a different scenario, they should be treated as variables.
        'Power_Law': lambda rh, rt: 3 * self.zeta / 5 * (rh / rt / self.Rht) ** self.n,
        # Simple insertion. The case of n=1.5 suggests \dot Mst is independend of half-mass radius.
        'Exponential': lambda rh, rt: self.zeta * self.xi0 * exp(rh / rt / self.Rht),
        # General formula. The evaporation rate is constant for dense clusters.
        'Constant': lambda rh, rt: self.xi0  # Constant evaporation rate.
    }

    # Galactic model dictionary.
    self.galactic_model_dict = {
        # Dictionary for spherically symmetric galactic potentials. The index is used for the tidal radius only and it is dimensionless. The potential has units are [pc^2/Myr^2].
        # Derivative of the potential is [pc/Myr^2] and is used to specify the velocity profile. X2 is the ratio of the velocity profile squared over twice the velocity dispersion squared for isotropic models, currently used for tidal spiraling only. Density is in [Msun/pc^3] and is used in tidal spiraling.
        # Distance r is inserted in kpc everywhere. X2 is valid only for a Maxwellian distribution. Current treatment does not allow for combinations of galactic potentials at different ranges of the galactocentric distance.
        # Add rt, E, L in the future for eccentric orbits. They need information for rapo, rperi.

        'SIS': {
            'rt_index': lambda r: 2,
            'Phi': lambda r: (1.023 * self.Vc) ** 2 * log(r / self.Rmax),
            'dPhi_dr': lambda r: 1e-3 * (1.023 * self.Vc) ** 2 / r,
            'd2Phi_dr2': lambda r: - 1e-6 * (1.023 * self.Vc) ** 2 / r ** 2,
            'X2': lambda r: 1,
            'rho': lambda r: 1e-6 * (1.023 * self.Vc) ** 2 / (4 * pi * self.G * r ** 2),
            'mu_r': lambda r: 0
        },  # Singular isothermal sphere. It is the default.

        'Point_mass': {
            'rt_index': lambda r: 3,
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / r,
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / r ** 2,
            'd2Phi_dr2': lambda r: - 2e-9 * self.G * self.Mg / r ** 3,
            'X2': lambda r: self.X0,
            'rho': lambda r: self.rhoG,
            'mu_r': lambda r: 0
        },  # Point mass galaxy.

        'Hernquist': {
            'rt_index': lambda r: (3 * r + self.rp) / (r + self.rp),
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / (r + self.rp),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / (r + self.rp) ** 2,
            'd2Phi_dr2': lambda r: - 2e-9 * self.G * self.Mg / (r + self.rp) ** 3,
            'X2': lambda r: 6 * r / self.rp / ((1 + r / self.rp) ** 2 * (
                    12 * r / self.rp * (1 + r / self.rp) ** 3 * log(1 + self.rp / r) - r / (r + self.rp) * (
                    25 + 52 * r / self.rp + 42 * (r / self.rp) ** 2 + 12 * (r / self.rp) ** 3))),
            'rho': lambda r: 1e-9 * self.Mg / (2 * pi * self.rp ** 2) * 1 / (r * (1 + r / self.rp) ** 3),
            'mu_r': lambda r: (- 2 * r + self.rp) / (r + self.rp)
        },  # Hernquist model.

        'Plummer': {
            'rt_index': lambda r: 3 * r ** 2 / (r ** 2 + self.rp ** 2),
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / sqrt(r ** 2 + self.rp ** 2),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg * r / (r ** 2 + self.rp ** 2) ** (3 / 2),
            'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg * (2 * r ** 2 - self.rp ** 2) / (
                    r ** 2 + self.rp ** 2) ** (5 / 2),
            'X2': lambda r: 3 * (r / self.rp) ** 2 / (1 + (r / self.rp) ** 2),
            'rho': lambda r: 1e-9 * 3 * self.Mg * self.rp ** 2 / (4 * pi * (self.rp ** 2 + r ** 2) ** (5 / 2)),
            'mu_r': lambda r: (- 3 * r ** 2 + 2 * self.rp ** 2) / (r ** 2 + self.rp ** 2)
        },  # Plummer model.

        'Jaffe': {
            'rt_index': lambda r: (3 * r + 2 * self.rp) / (r + self.rp),
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / self.rp * log(1 + self.rp / r),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / (r * (r + self.rp)),
            'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg * (self.rp + 2 * r) / (r ** 2 * (self.rp + r) ** 2),
            'X2': lambda r: ((1 + r / self.rp) * (
                    12 * (r / self.rp) ** 2 * (1 + r / self.rp) ** 2 * log(1 + self.rp / r) - 12 * (
                    r / self.rp) ** 3 - 18 * (r / self.rp) ** 2 - 4 * r / self.rp + 1)) ** (-1),
            'rho': lambda r: 1e-9 * self.Mg * self.rp / (4 * pi * r ** 2 * (r + self.rp) ** 2),
            'mu_r': lambda r: -2 * r / (r + self.rp)
        },  # Jaffe model.

        'NFW': {
            'rt_index': lambda r: 1 + (
                    2 * log(1 + r / self.rp) - 2 * r / (r + self.rp) - (r / (r + self.rp)) ** 2) / (
                                          log(1 + r / self.rp) - r / (r + self.rp)),
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / r * log(1 + r / self.rp) / (
                    log(1 + self.Rmax / self.rp) - self.Rmax / (self.rp + self.Rmax)),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg * (log(1 + r / self.rp) - r / (r + self.rp)) / r ** 2 / (
                    log(1 + self.Rmax / self.rp) - self.Rmax / (self.rp + self.Rmax)),
            'd2Phi_dr2': lambda r: - 1e-9 * 2 * self.G * self.Mg / r ** 3 * (
                    log(1 + r / self.rp) - r / (r + self.rp) - 0.5 * (r / (r + self.rp)) ** 2) / (
                                           log(1 + self.Rmax / self.rp) - self.Rmax / (self.rp + self.Rmax)),
            'X2': lambda r: (log(1 + r / self.rp) - r / (r + self.rp)) / (
                    (r / self.rp) ** 2 * (1 + r / self.rp) ** 2 * (
                    pi ** 2 + log(1 + self.rp / r) + 3 * log(1 + r / self.rp) ** 2 + 6 * spence(
                1 + r / self.rp)) + (1 + r / self.rp) ** 2 * log(1 + r / self.rp) - r / self.rp * (
                            1 + 9 * r / self.rp + 7 * (r / self.rp) ** 2 + 2 * log(1 + r / self.rp) * (
                            3 * (r / self.rp) ** 2 + 5 * r / self.rp + 2))),
            'rho': lambda r: 1e-9 / (r * (r + self.rp) ** 2) * self.Mg / (
                    4 * pi * (log(1 + self.Rmax / self.rp) - self.Rmax / (self.rp + self.Rmax))),
            'mu_r': lambda r: (1 - r / self.rp) / (1 + r / self.rp)
        },  # Navarro-Frenk-White model. At infinity it diverges so a truncation is inserted at Rmax.

        'Isochrone': {
            'rt_index': lambda r: 1 + 1 / (self.rp + sqrt(r ** 2 + self.rp ** 2)) / sqrt(r ** 2 + self.rp ** 2) * (
                    2 * r ** 2 - self.rp ** 2 * (self.rp + sqrt(r ** 2 + self.rp ** 2)) / sqrt(
                self.rp ** 2 + r ** 2)),
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / (self.rp + sqrt(r ** 2 + self.rp ** 2)),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg * r / (
                    (self.rp + sqrt(self.rp ** 2 + r ** 2)) ** 2 * sqrt(self.rp ** 2 + r ** 2)),
            'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg * (
                    (2 * r ** 2 - self.rp ** 2) * sqrt(r ** 2 + self.rp ** 2) - self.rp ** 3) / (
                                           (self.rp + sqrt(self.rp ** 2 + r ** 2)) ** 3 * (
                                           self.rp ** 2 + r ** 2) ** (3 / 2)),
            'X2': lambda r: (r / self.rp) ** 2 * (1 + 2 / 3 * (r / self.rp) ** 2 + sqrt(1 + (r / self.rp) ** 2)) / (
                    2 * (1 + (r / self.rp) ** 2) ** 2 * (1 + sqrt(1 + (r / self.rp) ** 2)) ** 5 * (
                    2 / 3 * log(1 + sqrt(1 + (r / self.rp) ** 2)) - 1 / 3 * log(
                1 + (r / self.rp) ** 2) + 1 / 6 * 1 / (
                            1 + sqrt(1 + (r / self.rp) ** 2)) ** 2 + 1 / 9 * 1 / (
                            1 + sqrt(1 + (r / self.rp) ** 2)) ** 3 - 2 / 3 / sqrt(
                1 + (r / self.rp) ** 2) + 1 / 6 / (1 + (r / self.rp) ** 2))),
            'rho': lambda r: 1e-9 * 3 * self.Mg * (
                    1 + 2 / 3 * (r / self.rp) ** 2 + sqrt(1 + (r / self.rp) ** 2)) / (
                                     4 * pi * self.rp ** 3 * (1 + (r / self.rp) ** 2) ** (3 / 2) * (
                                     1 + sqrt(1 + (r / self.rp) ** 2)) ** 3),
            'mu_r': lambda r: (- 3 / (r / self.rp) - (r / self.rp) * (1 - (r / self.rp) ** 2) / (
                    1 + (r / self.rp) ** 2) ** 2 - 2 * (r / self.rp) / (numpy.sqrt(1 + (r / self.rp) ** 2) * (
                    1 + numpy.sqrt(1 + (r / self.rp) ** 2))) + 2 * (r / self.rp) ** 2 * (
                                       (r / self.rp) * (1 + numpy.sqrt(1 + (r / self.rp) ** 2)) + 1 + (
                                       r / self.rp) ** 2) / ((1 + (r / self.rp) ** 2) ** (3 / 2) * (
                    1 + numpy.sqrt(1 + (r / self.rp) ** 2)) ** 2))
        },  # Isochrone potential.

        'Dehnen': {
            'rt_index': lambda r: (3 * r + self.gammap * self.rp) / (r + self.rp),
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / (self.rp * (2 - self.gammap)) * (
                    1 - (r / (r + self.rp)) ** (2 - self.gammap)),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg * r ** (1 - self.gammap) / (r + self.rp) ** (
                    3 - self.gammap),
            'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg / (
                    r ** self.gammap * (r + self.rp) ** (4 - self.gammap)) * (
                                           2 * r + (self.gammap - 1) * self.rp),
            'X2': lambda r: 5 / (2 * hyp2f1(1, 7 - 2 * self.gammap, 6, 1 / (1 + r / self.rp))),
            'rho': lambda r: (3 - self.gammap) * 1e-9 * self.Mg * self.rp / (
                    4 * pi * r ** self.gammap * (r + self.rp) ** (4 - self.gammap)),
            'mu_r': lambda r: (-2 * r + self.rp * (2 - self.gammap)) / (r + self.rp)
        },
        # Dehnen model. Generalization of the Jaffe and Hernquist. Because of the form, it is finite as infinity. Avoid values for gammap equal to (7 + k) / 2 where k is integer.

        'Veltmann': {
            'rt_index': lambda r: (3 * r ** self.gammap + (2 - self.gammap) * self.rp ** self.gammap) / (
                    r ** self.gammap + self.rp ** self.gammap),
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / (r ** self.gammap + self.rp ** self.gammap) ** (
                    1 / self.gammap),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg * r ** (self.gammap - 1) / (
                    r ** self.gammap + self.rp ** self.gammap) ** (1 + 1 / self.gammap),
            'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg * r ** (self.gammap - 2) * (
                    2 * r ** self.gammap + (1 - self.gammap) * self.rp ** self.gammap) / (
                                           r ** self.gammap + self.rp ** self.gammap) ** (2 + 1 / self.gammap),
            'X2': lambda r: (4 + self.gammap) / (2 * hyp2f1(1, 3 + 2 / self.gammap, 2 * (1 + 2 / self.gammap),
                                                            1 / (1 + (r / self.rp) ** self.gammap))),
            'rho': lambda r: 1e-9 * (1 + self.gammap) * self.Mg * self.rp ** self.gammap / (
                    4 * pi * r ** (2 - self.gammap) * (self.rp ** self.gammap + r ** self.gammap) ** (
                    2 + 1 / self.gammap)),
            'mu_r': lambda r: (- (self.gammap + 1) * r ** self.gammap + self.gammap * self.rp ** self.gammap) / (
                    r ** self.gammap + self.rp ** self.gammap)
        },  # Veltmann model. Generalization of the Hernquist and Plummer models. It is finite as infinity.

        'Soft': {
            'rt_index': lambda r: (3 * r + 8 * self.rp) / (r + 2 * self.rp),
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / r * (1 + self.rp / r),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / r ** 2 * (1 + 2 * self.rp / r),
            'd2Phi_dr2': lambda r: - 1e-9 * 2 * self.G * self.Mg / r ** 3 * (1 + 3 * self.rp / r),
            'X2': lambda r: (15 * (2 + r / self.rp)) / (2 * (5 + 3 * r / self.rp)),
            'rho': lambda r: 1e-9 * self.rp * self.Mg / (2 * pi * r ** 4),
            'mu_r': lambda r: - 2
        },  # Soft potential. It is a toy model.

        'Power_law': {
            'rt_index': lambda r: self.gammap,
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / (self.rp * (self.gammap - 2)) * r ** (2 - self.gammap) * (
                    self.rp / self.Rmax) ** (3 - self.gammap),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg * r ** (1 - self.gammap) / self.rp ** (3 - self.gammap) * (
                    self.rp / self.Rmax) ** (3 - self.gammap),
            'd2Phi_dr2': lambda r: - 1e-9 * (self.gammap - 1) * self.G * self.Mg / self.rp ** 3 * (r / self.rp) ** (
                - self.gammap) * (self.rp / self.Rmax) ** (3 - self.gammap),
            'X2': lambda r: self.gammap - 1,
            'rho': lambda r: 1e-9 * self.Mg / (
                    4 * pi * self.rp ** 3 / (3 - self.gammap) / (self.rp / self.Rmax) ** (3 - self.gammap)) * (
                                     self.rp / r) ** self.gammap,
            'mu_r': lambda r: 2 - self.gammap
        },
        # Power-law profile. Exponent must be within [2, 3] so that it is well defined. A truncation at Rmax is needed. Serves as a toy model.

        'MIS': {
            'rt_index': lambda r: 3 - (r / self.rp) ** 3 / (1 + (r / self.rp) ** 2) / (
                    (r / self.rp) - arctan((r / self.rp))),
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / self.rp / (
                    self.Rmax / self.rp - arctan(self.Rmax / self.rp)) * (
                                     1 / (r / self.rp) * ((r / self.rp) - arctan((r / self.rp))) - log(
                                 sqrt(1 + (r / self.rp) ** 2) / sqrt(1 + (self.Rmax / self.rp) ** 2))),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / r ** 2 * (r / self.rp - arctan(r / self.rp)) / (
                    self.Rmax / self.rp - arctan(self.Rmax / self.rp)),
            'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg / r ** 3 * (r / self.rp - arctan(r / self.rp)) / (
                    self.Rmax / self.rp - arctan(self.Rmax / self.rp)) * (
                                           2 - (r / self.rp) ** 3 / (1 + (r / self.rp) ** 2) / (
                                           (r / self.rp) - arctan(r / self.rp))),
            'X2': lambda r: (r / self.rp - arctan(r / self.rp)) / ((1 + (r / self.rp) ** 2) * (
                    r / self.rp * (pi ** 2 / 4 - arctan(r / self.rp) ** 2) - 2 * arctan(r / self.rp))),
            'rho': lambda r: 1e-9 * self.Mg / (
                    4 * pi * self.rp ** 3 * (self.Rmax / self.rp - arctan(self.Rmax / self.rp))) * 1 / (
                                     1 + (r / self.rp) ** 2),
            'mu_r': lambda r: 2 * self.rp ** 2 / (r ** 2 + self.rp ** 2)
        },  # Modified Isothermal Sphere.

        'Perfect_sphere': {
            'rt_index': lambda r: 3 - 2 * (r / self.rp) ** 3 / (1 + (r / self.rp) ** 2) ** 2 / (
                    arctan(r / self.rp) - (r / self.rp) / (1 + (r / self.rp) ** 2)),
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / r / (pi / 2) * arctan(r / self.rp),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / r ** 2 * (
                    arctan(r / self.rp) - (r / self.rp) / (1 + (r / self.rp) ** 2)) / (pi / 2),
            'd2Phi_dr2': lambda r: - 2e-9 * self.G * self.Mg * (
                    arctan(r / self.rp) - (r / self.rp) / (1 + (r / self.rp) ** 2)) / (pi / 2) / r ** 3 * (
                                           1 - (r / self.rp) ** 3 / (1 + (r / self.rp) ** 2) ** 2 / (
                                           arctan(r / self.rp) - (r / self.rp) / (1 + (r / self.rp) ** 2))),
            'X2': lambda r: (arctan(r / self.rp) - (r / self.rp) / (1 + (r / self.rp) ** 2)) / (
                    (r / self.rp) * (3 * (r / self.rp) ** 2 + 4) / 2 + 3 / 2 * (r / self.rp) * (
                    (1 + (r / self.rp) ** 2) * arctan(r / self.rp)) ** 2 + (
                            3 * (r / self.rp) ** 4 + 5 * (r / self.rp) ** 2 + 2) * arctan(
                r / self.rp) - 3 * pi ** 2 / 8 * (r / self.rp) * (1 + (r / self.rp) ** 2) ** 2),
            'rho': lambda r: 1e-9 * self.Mg / (pi ** 2 * self.rp ** 3) * 1 / (1 + (r / self.rp) ** 2) ** 2,
            'mu_r': lambda r: 2 * (1 - (r / self.rp) ** 2) / (1 + (r / self.rp) ** 2)
        },  # Perfect Sphere.

        'Modified_Hubble': {
            'rt_index': lambda r: 3 - (r / self.rp) ** 3 / (1 + (r / self.rp) ** 2) ** (3 / 2) / (
                    arctanh((r / self.rp) / sqrt(1 + (r / self.rp) ** 2)) - (r / self.rp) / sqrt(
                1 + (r / self.rp) ** 2)),
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / r * arctanh(r / self.rp / sqrt(1 + (r / self.rp) ** 2)) / (
                    arctanh(self.Rpmax / self.rp / sqrt(
                        1 + (self.Rpmax / self.rp) ** 2)) - self.Rpmax / self.rp / sqrt(
                1 + (self.Rpmax / self.rp) ** 2)),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / r ** 2 * (
                    arctanh((r / self.rp) / sqrt(1 + (r / self.rp) ** 2)) - (r / self.rp) / sqrt(
                1 + (r / self.rp) ** 2)) / (arctanh(
                (self.Rmax / self.rp) / sqrt(1 + (self.Rmax / self.rp) ** 2)) - (self.Rmax / self.rp) / sqrt(
                1 + (self.Rmax / self.rp) ** 2)),
            'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg / r ** 3 * (
                    arctanh((r / self.rp) / sqrt(1 + (r / self.rp) ** 2)) - (r / self.rp) / sqrt(
                1 + (r / self.rp) ** 2)) / (arctanh(
                (self.Rmax / self.rp) / sqrt(1 + (self.Rmax / self.rp) ** 2)) - (self.Rmax / self.rp) / sqrt(
                1 + (self.Rmax / self.rp) ** 2)) * (2 - (r / self.rp) ** 3 / (1 + (r / self.rp) ** 2) ** (3 / 2) / (
                    arctanh((r / self.rp) / sqrt(1 + (r / self.rp) ** 2)) - (r / self.rp) / sqrt(
                1 + (r / self.rp) ** 2))),
            'X2': lambda r: (arctanh((r / self.rp) / sqrt(1 + (r / self.rp) ** 2)) - (r / self.rp) / sqrt(
                1 + (r / self.rp) ** 2)) / ((r / self.rp) * sqrt(1 + (r / self.rp) ** 2) + (
                    2 * (r / self.rp) ** 2 + 1) * (1 + (r / self.rp) ** 2) * log(
                ((r / self.rp) + sqrt(1 + (r / self.rp) ** 2)) / (sqrt(1 + (r / self.rp) ** 2) - (r / self.rp))) - (
                                                    r / self.rp) * (1 + (r / self.rp) ** 2) ** (3 / 2) * (
                                                    4 * log(2) + 2 * log(1 + (r / self.rp) ** 2))),
            'rho': lambda r: 1e-9 * self.Mg / (4 * pi * self.rp ** 3 * (
                    arctanh((self.Rmax / self.rp) / (sqrt(1 + (self.Rmax / self.rp) ** 2))) - (
                    self.Rmax / self.rp) / (sqrt(1 + (self.Rmax / self.rp) ** 2)))) / (
                                     1 + (r / self.rp) ** 2) ** (3 / 2),
            'mu_r': lambda r: (2 - (r / self.rp) ** 2) / (1 + (r / self.rp) ** 2)
        },  # Hubble.

        # MIS, Perfect sphere and Hubble are just a few examples that have an analytical formula for the velocity dispersion. Numerical models are available below.

        'Moore': {
            'rt_index': lambda r: 3 - (3 - self.gammap) / log(1 + (r / self.rp) ** (3 - self.gammap)) * (
                    r / self.rp) ** (3 - self.gammap) / (1 + (r / self.rp) ** (3 - self.gammap)),
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / self.rp / log(
                1 + (self.Rmax / self.rp) ** (3 - self.gammap)) * (
                                     self.rp / r * log(1 + (r / self.rp) ** (3 - self.gammap)) + beta(
                                 1 / (3 - self.gammap), (2 - self.gammap) / (3 - self.gammap)) * betainc(
                                 1 / (3 - self.gammap), (2 - self.gammap) / (3 - self.gammap),
                                 1 / (1 + (r / self.rp) ** (3 - self.gammap)))),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / r ** 2 * log(
                1 + (r / self.rp) ** (3 - self.gammap)) / log(1 + (self.Rmax / self.rp) ** (3 - self.gammap)),
            'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg / r ** 3 / log(
                1 + (self.Rmax / self.rp) ** (3 - self.gammap)) * (2 - (3 - self.gammap) / log(
                1 + (r / self.rp) ** (3 - self.gammap)) * (r / self.rp) ** (3 - self.gammap) / (
                                                                           1 + (r / self.rp) ** (
                                                                           3 - self.gammap))),
            'X2': numpy.vectorize(lambda r: (3 - self.gammap) * log(1 + (r / self.rp) ** (3 - self.gammap)) / (
                    - 2 * (r / self.rp) ** (1 + self.gammap) * (1 + (r / self.rp) ** (3 - self.gammap)) * quad(
                lambda t: t ** (-1 - (1 + self.gammap) / (3 - self.gammap)) * (1 - t) ** (
                        4 / (3 - self.gammap) - 1) * log(1 - t),
                (r / self.rp) ** (3 - self.gammap) / (1 + (r / self.rp) ** (3 - self.gammap)), 1, limit=1000,
                epsabs=1e-6, epsrel=1e-6)[0])),
            'rho': lambda r: 1e-9 * (3 - self.gammap) * self.Mg / (
                    4 * pi * self.rp ** 3 * log(1 + (self.Rmax / self.rp) ** (3 - self.gammap))) / (
                                     r / self.rp) ** self.gammap / (1 + (r / self.rp) ** (3 - self.gammap)),
            'mu_r': lambda r: (2 - self.gammap - (r / self.rp) ** (3 - self.gammap)) / (
                    1 - (r / self.rp) ** (3 - self.gammap))
        },
        # Moore / Generalized NFW. Mass diverges so it is truncated up to Rmax. It is generelized, but the user can select gammap=1.5.

        'Truncated_Einasto': {
            'rt_index': lambda r: 3 - self.gammap * (r / self.rp) ** (3 - self.gammap1) * exp(
                - (r / self.rp) ** self.gammap) / gamma((3 - self.gammap1) / self.gammap) / gammainc(
                (3 - self.gammap1) / self.gammap, (r / self.rp) ** self.gammap),
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / self.rp * (
                    self.rp / r * gammainc((3 - self.gammap1) / self.gammap,
                                           (r / self.rp) ** self.gammap) + gamma(
                (2 - self.gammap1) / self.gammap) * gammaincc((2 - self.gammap1) / self.gammap,
                                                              (r / self.rp) ** self.gammap) / gamma(
                (3 - self.gammap1) / self.gammap)),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg * gammainc((3 - self.gammap1) / self.gammap,
                                                                    (r / self.rp) ** self.gammap) / r ** 2,
            'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg * gammainc((3 - self.gammap1) / self.gammap,
                                                                        (r / self.rp) ** self.gammap) / r ** 3 * (
                                           2 - self.gammap * (r / self.rp) ** (3 - self.gammap1) * exp(
                                       -(r / self.rp) ** self.gammap) / gamma(
                                       (3 - self.gammap1) / self.gammap) / gammainc(
                                       (3 - self.gammap1) / self.gammap, (r / self.rp) ** self.gammap)),
            'X2': numpy.vectorize(lambda r: self.gammap * gammainc((3 - self.gammap1) / self.gammap,
                                                                   (r / self.rp) ** self.gammap) * exp(
                - (r / self.rp) ** self.gammap) / (2 * (r / self.rp) ** (1 + self.gammap1) * quad(
                lambda t: exp(-t) * t ** (- (1 + self.gammap1) / self.gammap - 1) * gammainc(
                    (3 - self.gammap1) / self.gammap, t), (r / self.rp) ** self.gammap, numpy.inf, limit=1000,
                epsabs=1e-6, epsrel=1e-6)[0])),
            'rho': lambda r: 1e-9 * self.Mg * self.gammap / 4 / pi / self.rp ** 3 / gamma(
                (3 - self.gammap1) / self.gammap) * (r / self.rp) ** (- self.gammap1) * exp(
                -(r / self.rp) ** self.gammap),
            'mu_r': lambda r: 2 - self.gammap1 - self.gammap * (r / self.rp) ** self.gammap
        },  # Einasto profile for gammap1=0.

        'Zhao': {
            'rt_index': lambda r: 3 - self.gammap * (r / self.rp) ** (3 - self.gammap2) / (
                    (1 + (r / self.rp) ** self.gammap) ** (
                    (self.gammap1 - self.gammap2) / self.gammap) * betainc((3 - self.gammap2) / self.gammap,
                                                                           (self.gammap1 - 3) / self.gammap,
                                                                           (r ** self.gammap / (
                                                                                   self.rp ** self.gammap + r ** self.gammap))) * beta(
                (3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap)),
            'Phi': lambda r: - 1e-3 * self.G * self.Mg / self.rp * (
                    self.rp / r * betainc((3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap, (
                    r ** self.gammap / (self.rp ** self.gammap + r ** self.gammap))) + betainc(
                (self.gammap1 - 2) / self.gammap, (2 - self.gammap2) / self.gammap,
                (self.rp ** self.gammap / (self.rp ** self.gammap + r ** self.gammap))) * beta(
                (self.gammap1 - 2) / self.gammap, (2 - self.gammap2) / self.gammap) / beta(
                (3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap)),
            'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / r ** 2 * betainc((3 - self.gammap2) / self.gammap,
                                                                            (self.gammap1 - 3) / self.gammap, (
                                                                                    r ** self.gammap / (
                                                                                    self.rp ** self.gammap + r ** self.gammap))),
            'd2Phi_dr2': lambda r: - 2e-9 * self.G * self.Mg / r ** 3 * betainc((3 - self.gammap2) / self.gammap,
                                                                                (self.gammap1 - 3) / self.gammap, (
                                                                                        r ** self.gammap / (
                                                                                        self.rp ** self.gammap + r ** self.gammap))) + 1e-9 * self.gammap * self.G * self.Mg / r ** 3 * (
                                           (r / self.rp) ** (3 - self.gammap2)) / (
                                           1 + (r / self.rp) ** self.gammap) ** (
                                           (self.gammap1 - self.gammap2) / self.gammap) / beta(
                (3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap),
            'X2': numpy.vectorize(
                lambda r: self.gammap * betainc((3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap,
                                                (r ** self.gammap / (
                                                        self.rp ** self.gammap + r ** self.gammap))) / (
                                  2 * (r / self.rp) ** (1 + self.gammap2) * (
                                  1 + (r / self.rp) ** self.gammap) ** (
                                          (self.gammap1 - self.gammap2) / self.gammap) * quad(
                              lambda t: betainc((3 - self.gammap2) / self.gammap,
                                                (self.gammap1 - 3) / self.gammap, t) * t ** (
                                                - 1 - (1 + self.gammap2) / self.gammap) * (1 - t) ** (
                                                - 1 + (self.gammap1 + 1) / self.gammap),
                              (r / self.rp) ** self.gammap / (1 + (r / self.rp) ** self.gammap), 1, limit=1000,
                              epsabs=1e-6, epsrel=1e-6)[0])),
            'rho': lambda r: 1e-9 * self.gammap * self.Mg / (4 * pi * self.rp ** 3) / (r / self.rp) ** (
                self.gammap2) / (1 + (r / self.rp) ** self.gammap) ** (
                                     (self.gammap1 - self.gammap2) / self.gammap) / beta(
                (3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap),
            'mu_r': lambda r: 2 - self.gammap2 - (self.gammap1 - self.gammap2) * (r / self.rp) ** self.gammap / (
                    1 + (r / self.rp) ** self.gammap),
        }
        # Zhao model. Generalizes the families introduced before. For example, it agrees with Hernquist, Plummer, Jaffe, perfect sphere. NFW and Moore struggle with the incomplete beta. The user should keep in mind to always insert well behaved exponents so that the special functions are well defined, that is gammap > 0, gammap1 > 3, gammap2 < 3. The family of models with gammap2=0 does not have an analytic expression for the velocity dispersion squared so it is not presented separately. The same applies to generalized Moore models with gammap -> 3 - gammap2 + epsilon, gammap1 -> 3 + epsilon.
    }

    # Future extension: Add a SMBH in the model. This shifts rt_index, Vc(r) or M(r) and X2(r) because the potential and its derivatives are shifted by the inclusion of a point-mass term.
    # Future extension2: Add a dictionary with anisotropic models and the solution for the radial velocity dispersion from Jean's equations.

    # Cluster model dictionary. NFW, Power-law, MIS and Hubble use a fixed maximum distance to estimate the potential and its derivative, to avoid divergences.
    self.cluster_model_dict = {
        # Dictionary for spherically symmetric cluster potentials. Units are [pc^2/Myr^2]. Radius should be in [pc], mass in [Msun]. Most are toy models.
        'SIS': {'phi': lambda r, M: (1.023 * self.Vcc) ** 2 * log(r / self.Rmaxc),
                'dPhi_dr': lambda r: (1.023 * self.Vcc) ** 2 / r},

        'Point_mass': {'phi': lambda r, M: - self.G * M / r, 'dPhi_dr': lambda r, M: self.G * M / r ** 2},

        'Plummer': {'phi': lambda r, M: - self.G * M / sqrt(r ** 2 + self.rpc ** 2),
                    'dPhi_dr': lambda r, M: self.G * M * r / (r ** 2 + self.rpc ** 2) ** (3 / 2)},

        'Hernquist': {'phi': lambda r, M: - self.G * M / (r + self.rpc),
                      'dPhi_dr': lambda r, M: self.G * M / (r + self.rpc) ** 2},

        'Jaffe': {'phi': lambda r, M: - self.G * M / self.rpc * log(1 + self.rpc / r),
                  'dPhi_dr': lambda r, M: self.G * M / (r * (r + self.rpc))},

        'NFW': {'phi': lambda r, M: - self.G * M / r * log(1 + r / self.rpc) / (
                log(1 + self.Rmaxc / self.rpc) - self.Rmaxc / (self.rpc + self.Rmaxc)),
                'dPhi_dr': lambda r, M: self.G * M * (log(1 + r / self.rpc) - r / (r + self.rpc)) / (
                        log(1 + self.Rmaxc / self.rpc) - self.Rmaxc / (self.rpc + self.Rmaxc)) / r ** 2},

        'Isochrone': {'phi': lambda r, M: - self.G * M / (self.rpc + sqrt(r ** 2 + self.rpc ** 2)),
                      'dPhi_dr': lambda r, M: self.G * M * r / (
                              (self.rpc + sqrt(self.rpc ** 2 + r ** 2)) ** 2 * sqrt(self.rpc ** 2 + r ** 2))},

        'Dehnen': {'phi': lambda r, M: - self.G * M / (self.rpc ** (2 - self.gammapc)) * (
                1 - (r / (r + self.rpc)) ** (2 - self.gammapc)),
                   'dPhi_dr': lambda r, M: self.G * M * r ** (1 - self.gammapc) / (r + self.rpc) ** (
                           3 - self.gammapc)},

        'Veltmann': {
            'phi': lambda r, M: - self.G * M / (r ** self.gammapc + self.rpc ** self.gammapc) ** (1 / self.gammapc),
            'dPhi_dr': lambda r, M: self.G * M * r ** (self.gammapc - 1) / (
                    r ** self.gammapc + self.rpc ** self.gammapc) ** (1 + 1 / self.gammapc)},

        'Soft': {'phi': lambda r, M: - self.G * M / r * (1 + self.rpc / r),
                 'dPhi_dr': lambda r, M: self.G * M / r ** 2 * (1 + 2 * self.rpc / r)},

        'Power_law': {
            'phi': lambda r, M: - self.G * M / (self.rpc * (self.gammapc - 2)) * r ** (2 - self.gammapc) * (
                    self.rpc / self.Rmaxc) ** (3 - self.gammapc),
            'dPhi_dr': lambda r, M: self.G * M * r ** (1 - self.gammapc) / self.rpc ** (3 - self.gammapc) * (
                    self.rpc / self.Rmaxc) ** (3 - self.gammapc)},

        'MIS': {'phi': lambda r, M: - self.G * M / self.rpc / (
                self.Rmaxc / self.rpc - arctan(self.Rmaxc / self.rpc)) * (1 / (r / self.rpc) * (
                (r / self.rpc) - arctan((r / self.rpc))) - log(
            sqrt(1 + (r / self.rpc) ** 2) / sqrt(1 + (self.Rmaxc / self.rpc) ** 2))),
                'dPhi_dr': lambda r, M: self.G * M / r ** 2 * (r / self.rpc - arctan(r / self.rpc)) / (
                        self.Rmaxc / self.rpc - arctan(self.Rmaxc / self.rpc))},

        'Perfect_sphere': {'phi': lambda r, M: - self.G * M / r / (pi / 2) * arctan(r / self.rpc),
                           'dPhi_dr': lambda r, M: self.G * M / r ** 2 * (
                                   arctan(r / self.rpc) - (r / self.rpc) / (1 + (r / self.rpc) ** 2)) / (
                                                           pi / 2)},

        'Modified_Hubble': {
            'phi': lambda r, M: - self.G * M / r * arctanh(r / self.rpc / sqrt(1 + (r / self.rpc) ** 2)) / (arctanh(
                self.Rmaxc / self.rpc / sqrt(1 + (self.Rmaxc / self.rpc) ** 2)) - self.Rmaxc / self.rpc / sqrt(
                1 + (self.Rmaxc / self.rpc) ** 2)), 'dPhi_dr': lambda r, M: self.G * M / r ** 2 * (
                    arctanh((r / self.rpc) / sqrt(1 + (r / self.rpc) ** 2)) - (r / self.rpc) / sqrt(
                1 + (r / self.rpc) ** 2)) / (arctanh(
                (self.Rmaxc / self.rpc) / sqrt(1 + (self.Rmaxc / self.rpc) ** 2)) - (self.Rmaxc / self.rpc) / sqrt(
                1 + (self.Rmaxc / self.rpc) ** 2))},

        'Moore': {
            'phi': lambda r, M: - self.G * M / self.rpc / log(1 + (self.Rmaxc / self.rpc) ** (3 - self.gammapc)) * (
                    self.rpc / r * log(1 + (r / self.rpc) ** (3 - self.gammapc)) + beta(1 / (3 - self.gammapc),
                                                                                        (2 - self.gammapc) / (
                                                                                                3 - self.gammapc)) * betainc(
                1 / (3 - self.gammapc), (2 - self.gammapc) / (3 - self.gammapc),
                1 / (1 + (r / self.rpc) ** (3 - self.gammapc)))),
            'dPhi_dr': lambda r, M: self.G * M / r ** 2 * log(1 + (r / self.rpc) ** (3 - self.gammapc)) / log(
                1 + (self.Rmaxc / self.rpc) ** (3 - self.gammapc))},

        'Truncated_Einasto': {'phi': lambda r, M: - self.G * M / self.rpc * (
                self.rpc / r * gammainc((3 - self.gammapc1) / self.gammapc,
                                        (r / self.rpc) ** self.gammapc) + gamma(
            (2 - self.gammapc1) / self.gammapc) * gammaincc(2 / self.gammapc,
                                                            (r / self.rpc) ** self.gammapc) / gamma(
            3 / self.gammapc)),
                              'dPhi_dr': lambda r, M: self.G * M * gammainc((3 - self.gammapc1) / self.gammapc, (
                                      r / self.rpc) ** self.gammapc) / r ** 2},

        'Zhao': {'phi': lambda r, M: - self.G * M / self.rpc * (
                self.rpc / r * betainc((3 - self.gammapc2) / self.gammapc, (self.gammapc1 - 3) / self.gammapc, (
                r ** self.gammapc / (self.rpc ** self.gammapc + r ** self.gammapc))) + betainc(
            (self.gammapc1 - 2) / self.gammapc, (2 - self.gammapc2) / self.gammapc,
            (self.rpc ** self.gammapc / (self.rpc ** self.gammapc + r ** self.gammapc))) * beta(
            (self.gammap1 - 2) / self.gammap, (2 - self.gammap2) / self.gammap) / beta(
            (3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap)),
                 'dPhi_dr': lambda r, M: self.G * M * r ** (1 - self.gammapc) / self.rpc ** (3 - self.gammapc)},

    }

    # The dictionary is subject to change, should a better description for BH ejections when fbh close to 0 is found.
    # Dictionary that introduces a scaling on the BH ejection rate with respect to Spitzer's paramater. Can be used as a proxy for different parametrizations as well. Can be used for describing equipartition.
    self.beta_dict = {
        'exponential': lambda S: 1 - exp(- (S / self.S0) ** self.gamma2),
        'logistic': lambda S: (S ** self.gamma2 / (S ** self.gamma2 + self.S0 ** self.gamma2)),
        'error_function': lambda S: erf(S / self.S0),
        'hyperbolic': lambda S: tanh(S / self.S0)
    }

    # Dictionary for segregation model. A derivative that evolves up until core collapse parameter eta. Does not account for its decrease with time after core collapse.
    # Future extensions:
    # 1) Include dependence on fbh. After core collapse, a cluster with more BHs in he center should have a smaller value for eta.
    # 2) Include a decrease with N, t since towards the end, the value of self.eta_min should be reached.
    self.mseg_dict_t = {
        'linear': lambda t, trh, eta: self.aseg * (self.etaf - self.eta_min) * t ** (
                self.aseg - 1) / self.tcc ** self.aseg,
        # First model evolves linearly in time but depends on core collapse. If tcc=0 (fully segregated cluster) the elapsed relaxation is preferred for the linear model.
        'relaxation': lambda t, trh, eta: eta * (self.etaf - eta) / trh,
        # Second model evolves within one relaxation.
    }

    self.mseg_dict_rlx = {
        'linear': lambda Nrlx, Nrlx_dot, eta: self.aseg * (self.etaf - self.eta_min) * Nrlx_dot * Nrlx ** (
                self.aseg - 1),
        'relaxation': lambda Nrlx, Nrlx_dot, eta: eta * (self.etaf - eta) * Nrlx_dot,
    }

    # Dictionary for balancing functions. Functions must be 0 at t=0 and for initial times, and 1 at tcc<=t. Default to step like.
    self.balance_dict = {
        'step': lambda x: numpy.heaviside(x, 1),
        'exponential': lambda x: 1 - exp(- self.gamma3 * (x + 1) ** self.gamma4)
    }
