import numpy as np
import warnings


class RemnantModel:
    def __init__(self):
        import surfinBH
        from BHPTNRremnant.remnant import BHPTNRSurRemnant

        self.nr_model_lowq = surfinBH.LoadFits("NRSur7dq4Remnant")
        self.nr_model_emr = BHPTNRSurRemnant()
        self.C_KM_S = 299792.458

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


class SimpleRemnantModel:
    def __init__(self):
        pass

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
        return m1 + m2, v_kick, chi_f
