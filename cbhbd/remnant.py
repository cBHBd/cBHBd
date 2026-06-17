import numpy as np
import warnings
import abc


class RemnantModel(abc.ABC):
    def __init__(self):
        pass

    def print(self):
        print(f"{self.__name__}:\n\n{self.get_description()}")

    @abc.abstractmethod
    def get_description(self):
        return

    @abc.abstractmethod
    def get_remnant(self, m1, m2, S1, S2):
        # Return the remnant mass, spin, and kick velocity of the merger product of a BBH with masses m1 and m2,
        # and spins S1 and S2. Masses are in solar masses, spins are dimensionless and the kick velocity is in km/s
        return


class RemnantModelVarma19Islam23(RemnantModel):
    def __init__(self):
        super().__init__()
        warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
        import surfinBH
        from BHPTNRremnant.remnant import BHPTNRSurRemnant

        self.nr_model_lowq = surfinBH.LoadFits("NRSur7dq4Remnant")
        self.nr_model_emr = BHPTNRSurRemnant()
        self.C_KM_S = 299792.458

    def get_description(self):
        return ("Model used in Chattopadhyay, Marín Pina et al. (2026) (https://arxiv.org/pdf/2604.09773). This model "
                "uses NRSur7dq4Remnant (Varma et al. 2019) for mergers with a mass ratio q <= 6 (where q = m1/m2 and "
                "m2 <= m1). This model accounts for the effects of masses and individual spin magnitudes and "
                "orientations, but accuracy is not guaranteed at higher mass ratios. Instead, for mergers with q > 6, we"
                " use the BHPTNRSurRemnant model (Islam et al. 2023), which can be extrapolated up to q ~ 1000.")

    def get_remnant(self, m1, m2, S1, S2):
        q = m1 / m2

        if q <= 6:
            theta_S1 = np.random.uniform(0, np.pi)
            phi_S1 = np.random.uniform(0, 2 * np.pi)
            theta_S2 = np.random.uniform(0, np.pi)
            phi_S2 = np.random.uniform(0, 2 * np.pi)

            S1_vec = S1 * np.array([np.sin(theta_S1) * np.cos(phi_S1),
                                    np.sin(theta_S1) * np.sin(phi_S1),
                                    np.cos(theta_S1)])
            S2_vec = S2 * np.array([np.sin(theta_S2) * np.cos(phi_S2),
                                    np.sin(theta_S2) * np.sin(phi_S2),
                                    np.cos(theta_S2)])

            warnings.filterwarnings("ignore", category=UserWarning)
            m_rem, chi_f_vec, v_kick_vec, _, _, _ = self.nr_model_lowq.all(m1 / m2, S1_vec, S2_vec)
            chi_f = np.linalg.norm(chi_f_vec)
            v_kick = np.linalg.norm(v_kick_vec)
            warnings.filterwarnings("default", category=UserWarning)
        else:
            m_rem, _, chi_f, _, v_kick, _, _, _ = self.nr_model_emr.evaluate_fit(q)
        m_rem *= (m1 + m2)
        v_kick *= self.C_KM_S

        return m_rem, chi_f, v_kick


class RemnantModelSimple(RemnantModel):
    def __init__(self):
        super().__init__()

    def get_description(self):
        return ("Model used in Antonini et al. (2023) (https://arxiv.org/pdf/2208.01081). Less accurate than the other "
                "models, but does not require external dependencies.")

    def _dotp(self, a, b):
        # Compute the dot product of two vectors
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _cross(self, a, b):
        # Compute the cross product of two vectors
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

    def get_remnant(self, m1, m2, S1, S2):
        q = m2 / m1
        eta = q / (1 + q) ** 2

        theta = np.arccos(-1. + np.random.random() * (1. + 1.))
        phi = 0. + np.random.random() * (2. * np.pi + 0.0)
        chi1 = np.array([S1 * np.cos(theta), S1 * np.sin(theta) * np.sin(phi), S1 * np.sin(theta) * np.cos(phi)])

        theta = np.arccos(-1. + np.random.random() * (1. + 1.))
        phi = 0. + np.random.random() * (2. * np.pi + 0.0)
        chi2 = np.array([S2 * np.cos(theta), S2 * np.sin(theta) * np.sin(phi), S2 * np.sin(theta) * np.cos(phi)])

        theta = np.arccos(-1. + np.random.random() * (1. + 1.))
        phi = 0. + np.random.random() * (2. * np.pi + 0.0)
        j = np.array([np.cos(theta), np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi)])

        chit = (q ** 2 * chi2 + chi1) / (1. + q) ** 2
        chit2 = chit[0] ** 2 + chit[1] ** 2 + chit[2] ** 2
        delta = (chi1 - q * chi2) / (1. + q)

        # Parallel components
        chip = self._dotp(chit, j)
        deltap = self._dotp(delta, j)

        # Perpendicular components
        chi_cross = self._cross(chit, j)
        delta_cross = self._cross(delta, j)
        chiL = np.sqrt(chi_cross[0] ** 2 + chi_cross[1] ** 2 + chi_cross[2] ** 2)
        deltaL = np.sqrt(delta_cross[0] ** 2 + delta_cross[1] ** 2 + delta_cross[2] ** 2)

        # Recoil velocity
        A = 1.2e4
        B = -0.93
        H = 6.9e3
        V11 = 3677.76
        VA = 2481.21
        VB = 1792.45
        VC = 1506.52
        C2 = 1140.
        C3 = 2481.

        vm = A * eta ** 2 * (1. - q) / (1. + q) * (1. + B * eta)
        vsL = H * eta ** 2 * deltap
        Dphi = -1. + np.random.random() * (1. + 1.)
        vsp = 16. * eta ** 2 * (
                deltaL * (V11 + 2. * VA * chip + 4. * VB * chip ** 2 + 8. * VC * chip ** 3) + 2. * chiL * deltap * (
                C2 + 2. * C3 * chip)) * Dphi
        v_kick = np.sqrt(vm ** 2 + 2. * vm * vsL * np.cos(145. * np.pi / 180.) + vsL ** 2 + vsp ** 2)

        # Compute new spin
        t0 = -2.8904
        t2 = -3.51712
        t3 = 2.5763
        s4 = -0.1229
        s5 = 0.4537
        ell = 2. * np.sqrt(3.) + t2 * eta + t3 * eta ** 2 + s4 * (1. + q) ** 4 / (1. + q ** 2) ** 2 * chit2 + (
                s5 * eta + t0 + 2.) * (1. + q) ** 2 / (1. + q ** 2) * chip

        chiv = chit + q / (1 + q) ** 2 * ell * j
        chi_f = min(1., np.sqrt(chiv[0] ** 2 + chiv[1] ** 2 + chiv[2] ** 2))
        return m1 + m2, chi_f, v_kick


class RemnantModelIslam26(RemnantModel):
    def __init__(self):
        super().__init__()
        warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
        import surfinBH
        from BHPTNRremnant.remnant import BHPTNRSurRemnant
        from gwModels.remnants import gwModel_kick_q200
        from gwModels.remnants import gwModel_kick_prec_flow
        import os

        self.nr_model_lowq = surfinBH.LoadFits("NRSur7dq4Remnant")
        self.nr_model_emr = BHPTNRSurRemnant()
        self.C_KM_S = 299792.458

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.nr_model_kick_q200 = gwModel_kick_q200
        self.nr_model_kick_prec = gwModel_kick_prec_flow(f"{dir_path}/data").sample

    def get_description(self):
        return ("This model uses Islam et al. (2026) (https://arxiv.org/pdf/2603.10170) for the kick velocity, and "
                "RemnantModelVarma19Islam23 for the remnant mass and spin.")

    def get_remnant(self, m1, m2, S1, S2):
        q = m1 / m2

        if q <= 6:
            theta_S1 = np.random.uniform(0, np.pi)
            phi_S1 = np.random.uniform(0, 2 * np.pi)
            theta_S2 = np.random.uniform(0, np.pi)
            phi_S2 = np.random.uniform(0, 2 * np.pi)

            S1_vec = S1 * np.array([np.sin(theta_S1) * np.cos(phi_S1),
                                    np.sin(theta_S1) * np.sin(phi_S1),
                                    np.cos(theta_S1)])
            S2_vec = S2 * np.array([np.sin(theta_S2) * np.cos(phi_S2),
                                    np.sin(theta_S2) * np.sin(phi_S2),
                                    np.cos(theta_S2)])

            warnings.filterwarnings("ignore", category=UserWarning)
            m_rem, chi_f_vec, _, _, _, _ = self.nr_model_lowq.all(m1 / m2, S1_vec, S2_vec)
            chi_f = np.linalg.norm(chi_f_vec)
            warnings.filterwarnings("default", category=UserWarning)
        else:
            m_rem, _, chi_f, _, _, _, _, _ = self.nr_model_emr.evaluate_fit(q)
        m_rem *= (m1 + m2)

        if S1 == 0 and S2 == 0:
            v_kick = self.nr_model_kick_q200(q, S1, S2, return_std=False)
        else:
            v_kick = self.nr_model_kick_prec(q, S1, S2, num_samples=1)[0]

        return m_rem, chi_f, v_kick


available_remnant_models = {rm.__name__: rm for rm in RemnantModel.__subclasses__()}
