import ssptools
import sys

import numpy as np
from scipy.optimize import bisect


class MergerOutcome:
    # Define all possible merger types

    InClusterInspiral = "incluster_inspiral"
    GWCaptureBS = "gw_capture_bs"
    GWCaptureBB = "gw_capture_bb"
    Ejected = "ejected"
    GWCaptureDirect = "gw_capture_direct"

    @staticmethod
    def get_outcomes():
        return [MergerOutcome.InClusterInspiral,
                MergerOutcome.GWCaptureBS,
                MergerOutcome.GWCaptureBB,
                MergerOutcome.Ejected,
                MergerOutcome.GWCaptureDirect]


def sample_power_law(xmin, xmax, alpha, size):
    # Sample from power law x^alpha with limits xmin, xmax
    # The parameter size can be an integer or an array for N-dim sampling

    r = np.random.random_sample(size)
    return (-(xmin ** (1 + alpha) * (-1 + r)) + xmax ** (1 + alpha) * r) ** (1 / (1 + alpha))


def sample_BHs(Mbh, v_esc0, mmin, mmax, alpha, FeH, SN_model, sigma):
    # Sample the original population of stellar-mass BHs. The total mass of the population is (approximately) Mbh.
    # Remove BHs with a kick velocity greater than v_esc0. The progenitors follow an IMF with slope alpha between mmin
    # and mmax (possibly with different slopes at lower masses). FeH is the iron abundance, SN_model is the supernova
    # prescription, and sigma is the parameter of the Maxwellian distribution of kick velocities

    if SN_model != "rapid" and SN_model != "delay":
        raise NotImplementedError(f"SN model {SN_model} not implemented.")

    # Load initial-final mass relation
    ifmr = ssptools.ifmr.IFMR(FeH=FeH)

    # Define limits of the IMF of the progenitor stars
    assert mmax > mmin, "mmax must be greater than mmin"
    assert mmax > ifmr.BH_mi[0], "The chosen IMF does not form BHs"
    if mmin > ifmr.BH_mi[0]:
        raise NotImplementedError(f"Can't have an IMF with a mass break M > {ifmr.BH_mi[0]} Msun")
    if mmax > ifmr.BH_mi[1]:
        raise NotImplementedError(f"Can't have a progenitor star with M > {ifmr.BH_mi[1]} Msun")

    # We iterate in case we don't sample enough BHs. This is faster than a Python loop
    i = 1
    while True:
        # We sample a number of BHs equal to nsamp, then keep only as many as needed. While the choice of nsamp is
        # arbitrary and does not affect the end result, a too small value will lead to multiple iterations, while a too
        # large value will lead to degraded performance and unnecessary memory usage. This ansatz is fairly efficient.
        nsamp = int(i * Mbh / 10)

        # Sample BH masses and kicks
        vkick_unscaled = np.linalg.norm(np.random.normal(loc=0.0, scale=sigma, size=(nsamp, 3)), axis=1)
        mproj = sample_power_law(ifmr.BH_mi[0], mmax, alpha, nsamp)
        mbh = ifmr.predict(mproj)
        fb_fun = ssptools.kicks._F12_fallback_frac(FeH, SNe_method=SN_model)
        fb = np.clip(fb_fun(mbh), 0.0, 1 - 1e-16)
        vkick = vkick_unscaled * (1 - fb)

        # Remove kicked BHs
        mbh = mbh[vkick <= v_esc0]

        # Fix the total mass in BHs to be as close as possible to Mbh
        mbh_cumsum = np.cumsum(mbh)
        idx = np.argmin(np.abs(mbh_cumsum - Mbh))
        actual_Mbh = mbh_cumsum[idx]
        if idx + 1 == len(mbh):
            i += 1
            continue
        mbh = mbh[:idx + 1]

        # Construct BH array
        spin = np.zeros_like(mbh)  # Zero initial spin
        gen = np.zeros_like(mbh, dtype=int)
        tdf = np.zeros_like(mbh)

        bhv = np.vstack([mbh, spin, tdf, gen]).T

        return actual_Mbh, bhv


def get_sync_time(cbh, fbh0, Mcl_i, M3ej, t):
    # Use the ejected BH mass to synchronise the time evolution of clusterBH and BHBdynamics

    if fbh0 * Mcl_i - M3ej >= cbh.Mbh_interp(t / 1e9):
        return t
    elif fbh0 * Mcl_i - M3ej >= cbh.Mbh0:
        return t
    elif fbh0 * Mcl_i - M3ej <= cbh.Mbh_interp(cbh.tend / 1e3):
        return cbh.tend * 1e6
    else:
        t_bisect = bisect(lambda t_param: cbh.Mbh_interp(t_param / 1e9) - (fbh0 * Mcl_i - M3ej),
                          1, cbh.tend * 1e6 - 1,
                          xtol=1e2)
        return max(t, t_bisect)


def get_mbh_params(bhv, t):
    # Get all relevant parameters of the BH population at time t

    m_d = np.where(bhv[:, 2] <= t, bhv[:, 0], np.zeros_like(bhv[:, 0]) * np.nan)
    Nbh_core = len(m_d[~np.isnan(m_d)])
    kmax = len(m_d) - np.nanargmax(np.flip(m_d)) - 1
    # The numpy flip is only needed when there are multiple
    # BHs with the same mass, otherwise we would have kmin == kmax
    kmin = np.nanargmin(m_d)
    mbhmax = m_d[kmax]
    mbhmin = m_d[kmin]

    if m_d[kmax] != m_d[np.nanargmax(m_d)]:
        raise RuntimeError(f"Error in determining {kmax=}")

    return m_d, mbhmax, mbhmin, Nbh_core, kmin, kmax


def evolve_eccentricity(a0, e0, m1, m2, f=10):
    # Evolve the eccentricity of a binary with initial SMA a0 and initial eccentricity e0 to a reference frequency f

    # f in Hz
    if e0 == 0:
        return 0
    elif e0 == 1:
        return 1
    G = 3.964e-14  # In AU^3 M_sun^-1 s^-2

    try:
        return bisect(lambda e: -((e0 / e) ** 0.631578947368421 * (
                (1 + (121 * e0 ** 2) / 304) / (1 + (121 * e ** 2) / 304)) ** 0.3784254023488473) + (
                                        a0 * (1 - e0 ** 2) * (f * np.pi) ** (2 / 3)) / (
                                        (1 + e) ** 0.7969333333333333 * (G * (m1 + m2)) ** (1 / 3)),
                      1e-300, 1 - 1e-300)
    except Exception as e:
        print(e, file=sys.stderr)
        return np.nan


def prob_esc(ma, mb, m):
    # In a binary-single resonant interaction with masses ma, mb, m;
    # return the probability that the star with mass m escapes.
    # Computed from Eq. 36 in Ginat & Perets (2021)
    temp1 = ma ** 4 * mb ** 4 / (ma + mb) ** (5 / 2)
    temp2 = ma ** 4 * m ** 4 / (ma + m) ** (5 / 2)
    temp3 = mb ** 4 * m ** 4 / (mb + m) ** (5 / 2)
    denom = (ma + mb) ** (5 / 2) * (temp1 + temp2 + temp3)

    return ma ** 4 * mb ** 4 / denom


def get_alpha_samp(m_d, Nbh_core, mbhmin, mbhmax, alpha_previous):
    # Compute the exponent, alpha, of the BH mass function, p(m) \propto m^alpha
    # This is used in the sampling of BH masses

    # Maximum and minimum values of alpha
    MIN_ALPHA = -10
    MAX_ALPHA = 10

    data = m_d[~np.isnan(m_d)]
    sum_logm = np.log(data).sum()

    def mleq(alpha):
        alpha_cont = alpha if alpha != -1 else -1 + 1e-5

        return Nbh_core / (-alpha_cont - 1) - sum_logm - Nbh_core * (
                -mbhmin ** (1 + alpha_cont) * np.log(mbhmin) + mbhmax ** (1 + alpha_cont) * np.log(mbhmax)) / (
                mbhmin ** (1 + alpha_cont) - mbhmax ** (1 + alpha_cont))

    try:
        return bisect(mleq, MIN_ALPHA, MAX_ALPHA)
    except ValueError:
        if alpha_previous is not None:
            return alpha_previous

        mleq_min = mleq(MIN_ALPHA)
        mleq_max = mleq(MAX_ALPHA)

        if np.abs(mleq_max) < np.abs(mleq_min):
            return MAX_ALPHA
        else:
            return MIN_ALPHA
