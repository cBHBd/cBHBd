import time

import astropy.cosmology
import numpy as np
import pandas as pd
import scipy
from imf.imf import Kroupa
from numpy.random import random

import cBHBd.funcs
from cBHBd.clusterbh import clusterBH  # FIXME
from cBHBd.clusterbh_old import clusterBH_old  # FIXME

from cBHBd.funcs import MergerOutcome


def run_model(tf0, Mcl_i, Z, Z_file, rho_h_i,
              r_g=8,
              output_dataframe=True,  # If True, output is a pandas dataframe, otherwise a list
              debug_mode=False,
              seed=None):
    """
    Run a cluster model using cBHBd. Returns the properties of all the BBH mergers in the cluster.

    :param tf0: Final time of the simulation [Gyr]
    :param Mcl_i: Initial mass of the cluster [Msun]
    :param Z: Metallicity
    :param Z_file: Path to file with sampled BH masses and kicks
    :param rho_h_i: Initial density within the half-mass radius [MSun/pc^3]
    :param output_dataframe: If True, output is a pandas dataframe, otherwise a list
    :param debug_mode: If True, return extra output and verbose print.
    :return: the properties of all the BBH mergers in the cluster.
    """

    # FIXME: add documentation on seed, rg, return
    try:
        return _run_model(tf0, Mcl_i, Z, Z_file, rho_h_i, r_g, debug_mode, output_dataframe, seed)
    except Exception as err:
        print("Error in model with", flush=True)
        print(f"\t Mass = {Mcl_i} M_sun", flush=True)
        print(f"\t Metallicity = {Z} ({Z_file})", flush=True)
        print(f"\t Final time = {tf0} Gyr", flush=True)
        print(f"\t Initial density = {rho_h_i :.3g} M_sun/pc^3", flush=True)
        print(f"\t Seed = {seed}", flush=True)
        print(err, flush=True)
        raise err


def _run_model(tf0, Mcl_i, Z, Z_file, rho_h_i, r_g, debug_mode, output_dataframe, seed):
    tf0 *= 1e9
    verbose = False  # FIXME
    kick = True

    clprops = []

    if debug_mode and verbose:
        print("Generating new model with")
        print(f"\t Mass = {Mcl_i:.2g} M_sun")
        print(f"\t Metallicity = {Z:.2e}")
        print(f"\t Final time = {tf0 / 1e9 :.3g} Gyr")
        print(f"\t Initial density = {rho_h_i :.3g} M_sun/pc^3", flush=True)
        print(f"\t Seed = {seed}")

    if seed is not None:
        np.random.seed(seed)

    c = 1.e4  # sol

    # Cluster evolution
    # Some other model parameters
    csi = 0.09  # Relaxation coefficient

    bhout = []
    v_esc_i = 50 * (Mcl_i / 1e5) ** (1 / 3) * (rho_h_i / 1e5) ** (1 / 6)

    # Construct the BH IMF
    mmin = 0.08 - 1e-10  # Done to work with imf library
    mmax = 150  # Using the same limits as clusterBH

    imf = Kroupa(mmin=mmin, mmax=mmax)
    mmean = imf.m_integrate(mmin, mmax)[0] / imf.integrate(mmin, mmax)[0]
    cbh = clusterBH(Mcl_i / mmean, rho_h_i, m0=mmean, Z=Z, ssp=True, kick=kick, dtout=None,
                    dense_output=True, tend=15e3, Mbh_min=0, rg=r_g)  # galactocentric distance in cbh is rg, not RG

    # FIXME: remove this
    # cbh_old = clusterBH_old(Mcl_i / mmean, rho_h_i, m0=mmean, Z=Z, ssp=True, kick=kick, dtout=None,
    #                         dense_output=True, tend=15e3, Mbh_min=0,
    #                         rg=r_g)  # galactocentric distance in cbh is rg, not RG
    # return cbh, cbh_old

    # Build array with BH properties for retained BHs
    bhv = []
    Mtot = 0
    bhdata = np.loadtxt(Z_file)
    while True:
        mprog0_i, mbh0_i, vk0_i = bhdata[np.random.randint(0, len(bhdata))]
        if Mtot + mbh0_i / 2 > cbh.Mbh[0]:
            # We add the /2 factor to account for whether to include or not the last BH
            break

        if vk0_i < v_esc_i or not kick:
            Mtot += mbh0_i
            spin0 = 0
            bhv.append([mbh0_i, spin0, 0, 1])
    bhv = np.array(bhv)

    Nbh_i = len(bhv)
    debug_data = {"a": [], "a2": [], "Nbs": 0, "Nbb": 0, "alpha_samp": []}
    if debug_mode:
        debug_data["mmax"] = np.nanmax(bhv[:, 0])
        debug_data["bhpop"] = bhv.copy()

    if debug_mode and verbose:
        # print(f"\t Number of BHs before natal kicks: {Nbh_beforekicks} ({Mtot_before:.0f} Msun)")
        print(f"\t Number of BHs after natal kicks: {Nbh_i}")
    if Nbh_i < 4:
        return bhout, None, clprops, debug_data
    mbhmean = Mtot / Nbh_i
    fbh0 = Mtot / Mcl_i
    if debug_mode and verbose:
        # print(f"\t Averaged BH mass before kicks: {Mtot_before / Nbh_i:.2f}")
        print(f"\t Averaged BH mass after kicks: {mbhmean:.2f}")
        print(f"\t BH mass fraction: {fbh0 = :.4f}")

    # compute time when balanced evolution starts (alla AG'19)
    tcc = cbh.tcc / 1e3

    #   start integration
    t = tcc * 1e9  # yr from this on
    N3ej = 0  # Ejected BHs (+ BHs destroyed in mergers)
    M3ej = 0  # Ejected BH mass

    if M3ej > 100:
        raise RuntimeError("Initial M3ej is too big")

    alpha_samp = None

    while t <= tf0:
        m_d, mbhmax, mbhmin, Nbh_core, kmin, kmax = cBHBd.funcs.get_mbh_params(bhv, t)
        alpha_samp = cBHBd.funcs.get_alpha_samp(m_d, Nbh_core, mbhmin, mbhmax, alpha_samp)
        if debug_mode:
            debug_data["alpha_samp"].append(alpha_samp)

        if Nbh_core < 4:
            if len(bhv[~np.isnan(bhv[:, 0])]) < 4:
                return _format_output(bhout, cbh, clprops, debug_data, output_dataframe)  # exit if not BHs in the core
            else:
                t += np.min(bhv[:, 2][bhv[:, 2] > 0])
                continue

        if np.isnan(mbhmax) or np.isnan(mbhmin) or kmin == kmax or kmin is None or kmax is None:
            raise ValueError(f"Error in determining m range, found {mbhmax=} ({kmax=}) and {mbhmin=} ({kmin=})")

        # use a distribution for m1
        alpha_1 = 8 + 2 * alpha_samp
        m1_t = ((mbhmax ** (1 + alpha_1) - mbhmin ** (1 + alpha_1)) * random() + mbhmin ** (1 + alpha_1)) ** (
                1 / (1 + alpha_1))

        m_d[kmin] = np.nan
        k1 = np.nanargmin(np.abs(m_d - m1_t))
        m_d[kmin] = mbhmin

        m1, S1, t1, gen1 = bhv[k1, 0], bhv[k1, 1], bhv[k1, 2], bhv[k1, 3]

        if k1 == kmin or np.isnan(m1):
            raise RuntimeError(f"Error in {k1=}, {m1=} determination")

        # use a distribution for q
        alpha_2 = 3.5 + alpha_samp

        m_d[k1] = np.nan
        qmax = np.nanmax((m_d[m_d <= m1])) / m1
        qmin = mbhmin / m1

        q_t = ((qmax ** (1 + alpha_2) - qmin ** (1 + alpha_2)) * random() + qmin ** (1 + alpha_2)) ** (
                1 / (1 + alpha_2))
        k2 = np.nanargmin(np.abs(m_d - q_t * m1))
        m_d[k1] = m1

        m2, S2, t2, gen2 = bhv[k2, 0], bhv[k2, 1], bhv[k2, 2], bhv[k2, 3]

        if k2 == k1 or np.isnan(m2):  # just avoid to select the same black hole as 1
            raise RuntimeError(f"Error in {k2=}, {m2=} determination")

        if m2 > m1:  # Force that the primary is always the most massive
            k1, k2 = k2, k1
            m1, S1, t1, gen1 = bhv[k1, 0], bhv[k1, 1], bhv[k1, 2], bhv[k1, 3]
            m2, S2, t2, gen2 = bhv[k2, 0], bhv[k2, 1], bhv[k2, 2], bhv[k2, 3]

        #      compute spins and recoil kick
        v_kick, chi_f = cBHBd.funcs.recoil(m1, m2, S1, S2)

        #      evolved cluster properties
        if t / 1e6 > cbh.tend:
            return _format_output(bhout, cbh, clprops, debug_data, output_dataframe)
        rh = cbh.rh_interp(t / 1e9)
        Mcl = cbh.M_interp(t / 1e9)
        Mbh = max(cbh.Mbh_interp(t / 1e9), 0)
        mmean = cbh.m_interp(t / 1e9)
        mst = cbh.mst_interp(t / 1e9)

        clprops.append([t, rh, Mcl, Mbh])

        lastBH = False
        if Mbh <= 0 or Mcl <= 0:  # check that cluster did not evaporate
            lastBH = True
        v_esc = 3.69e-3 * np.sqrt(Mcl / rh) * 30
        # vd = v_esc / 4.77

        #      hard radius, and some other quantities
        G_AU_MSUN_KMS = 887.1278675
        PC_TO_AU = 648000 / np.pi
        # mu = m1 * m2 / (m1 + m2)
        # ah = G_AU_MSUN_KMS * mu / vd ** 2  # in AU if sigma in Km/s and M in solar masses

        vd = np.sqrt(0.2 * G_AU_MSUN_KMS * Mcl / (rh * PC_TO_AU))
        ah = G_AU_MSUN_KMS * m1 * m2 / (2 * mmean * vd ** 2)  # in AU if sigma in Km/s and M in solar masses

        Eh = (1 / 2) * (m1 + m2) * vd ** 2  # in Msun*(km/s)^2
        # fbh = Mbh / Mcl
        # psi = 1. + fbh * (mbhmean / mmean) ** 1.25
        # trh = 2.06e5 * np.sqrt(Mcl * rh ** 3) / (
        #         psi * mmean)  # Again, Mcl, cluster mass. We miss the Coulomb logarithm, I see it is used later for core radius/half-mass radius of BHs, if you include it, clusterBH has gamma=0.02, but the constant needs to be replaced because I think that they assumed a logarithm equal to 10 before, we just multiply with 10
        # Edot = 0.0728 / 0.1 * 1.53e-7 * csi * (
        #         Mcl ** 2 / rh) / trh  # convert to M=M_sun, L=1AU, G=1. Previously, we used zeta=0.1. We now use zeta=0.0728, so we apply a prefactor.
        # # TODO: The best approach would be to have cbh.zeta for instance (same for other parameters), but we could leave that for the end, when we want to make our codes more readable.

        # Do Montecarlo of three-body encounters
        a = ah
        Ebin = Eh
        e_b = np.sqrt(random())
        # ell_b = np.sqrt(1 - e_b ** 2)
        merger_type = None

        if debug_mode:
            assert m1 <= debug_data["mmax"]

        while not lastBH:
            m_d, mbhmax, mbhmin, Nbh_core, kmin, kmax = cBHBd.funcs.get_mbh_params(bhv, t)
            alpha_samp = cBHBd.funcs.get_alpha_samp(m_d, Nbh_core, mbhmin, mbhmax, alpha_samp)
            Nbh = Nbh_i - N3ej
            if debug_mode:
                debug_data["alpha_samp"].append(alpha_samp)

            if Nbh_core < 4:
                if len(bhv[~np.isnan(bhv[:, 0])]) < 4:
                    return _format_output(bhout, cbh, clprops, debug_data,
                                          output_dataframe)  # exit if not BHs in the core
                else:
                    t += np.min(bhv[:, 2][bhv[:, 2] > 0])
                    break

            if np.isnan(mbhmax) or np.isnan(mbhmin) or kmin == kmax or kmin is None or kmax is None:
                raise ValueError(f"Error in determining m range, found {mbhmax=} ({kmax=}) and {mbhmin=} ({kmin=})")

            if N3ej + 3 >= Nbh_i:
                return _format_output(bhout, cbh, clprops, debug_data,
                                      output_dataframe)  # exit if runs out of BHs during binary hardening sequence

            # Dani
            # Compute the ratio of binary-binary interactions to binary-single interactions
            bb_bs_ratio = 0.3 * (Nbh / 1e2) ** (-1 / 3)
            pbs = 1 / (bb_bs_ratio + 1)
            assert 0 <= pbs <= 1, "Error computing pbs"
            isbinarysingle = (pbs >= random())

            if isbinarysingle or Nbh_core < 5:  # Binary-single interaction
                if debug_mode:
                    debug_data["Nbs"] += 1
                # use a distribution for m3
                alpha_3 = alpha_samp + 0.5

                m3_t = ((mbhmax ** (1 + alpha_3) - mbhmin ** (1 + alpha_3)) * random() + mbhmin ** (1 + alpha_3)) ** (
                        1 / (1 + alpha_3))

                # Do this for performance reasons
                m_d[k1] = np.nan
                m_d[k2] = np.nan
                k3 = np.nanargmin(np.abs(m_d - m3_t))
                m3 = bhv[k3, 0]
                m_d[k1] = m1
                m_d[k2] = m2

                if k3 == k1 or k3 == k2 or np.isnan(m3) or bhv[k3, 2] > t:
                    print(f"{k1=}, {k2=}, {k3=}, {m_d=}, {m3_t=}", flush=True)
                    raise ValueError(f"Error in {k3=}, {m3=} determination")

                # dE = 0.2  # fractional energy change per interaction
                # Nrs = 20  # number of resonant states per 2-1 interaction

                # Dani: changes due to Bruno's paper
                # fractional energy change per interaction
                dE = 0.2 * (1 - np.exp(-7 * m3 / (m1 + m2))) if m3 <= m1 + m2 else 0.2
                q2 = m2 / m1
                q3 = m3 / m1
                gamma = 3.75
                # number of resonant states per 2-1 interaction
                Nrs = (1 + q3 + q3 / q2) ** (gamma - 1) if q3 < q2 else (1 + q2 + q2 / q3) ** (gamma - 1)
                Nrs = int(round(Nrs))
                assert Nrs >= 1, "No intermediate resonant states"

                # Resonant encounters
                is_capture = False
                Rs = 4 * (m1 + m2) / c ** 2
                ell_cap = (Rs / a) ** (5 / 14)
                for i in range(Nrs):
                    e_b = np.sqrt(random())
                    ell_b = np.sqrt(1 - e_b ** 2)
                    if ell_b < ell_cap:
                        is_capture = True
                        merger_type = MergerOutcome.GWCaptureBS
                        break
                if is_capture:
                    break

                # TODO
                # # Exchange
                # exchange_outcomes = ["preservation", "exchange_1", "exchange_2"]
                # exchange_outcome = np.random.choice(exchange_outcomes, p=[Ppres, Pex1, max(0.0, 1 - Pex1 - Ppres)])
                #
                # if exchange_outcome != "preservation":
                #     if exchange_outcome == "exchange_1":
                #         k1, k3 = k3, k1
                #         # print("Exchange1", file=sys.stderr, flush=True)
                #     elif exchange_outcome == "exchange_2":
                #         k2, k3 = k3, k2
                #         # print("Exchange2", file=sys.stderr, flush=True)
                #
                #     if bhv[k2, 0] > bhv[k1, 0]:  # Force that the primary is always the most massive
                #         k1, k2 = k2, k1
                #
                #     m1, S1, t1, gen1 = bhv[k1, 0], bhv[k1, 1], bhv[k1, 2], bhv[k1, 3]
                #     m2, S2, t2, gen2 = bhv[k2, 0], bhv[k2, 1], bhv[k2, 2], bhv[k2, 3]

                vbin = np.sqrt(dE * Ebin * (2 / (m1 + m2)) * (m3 / (m1 + m2 + m3)))  # recoil in km/s
                q3 = m3 / (m1 + m2)
                v3 = vbin / q3

                mej_int = 0
                if v3 > v_esc:
                    mej_int += m3
                if vbin > v_esc:
                    mej_int += m1 + m2
                if mej_int > 0:
                    t3 = cBHBd.funcs.get_sync_time(cbh, fbh0, Mcl_i, M3ej + mej_int, t) - t
                    assert t3 >= 0, "Negative t3"

                    # In-cluster inspirals
                    R = (1 + 73 / 24 * e_b ** 2 + 37 / 96 * e_b ** 4)
                    t_gw = 5 * c ** 5 * a ** 4 * (1 - e_b ** 2) ** (7 / 2) / (64 * m1 * m2 * (m1 + m2) * R) * 58 / 365
                    if t_gw < t3:
                        merger_type = MergerOutcome.InClusterInspiral
                        break
                    # ell_gw = 1.3 * ((m1 * m2) ** 2 * (m1 + m2) / (c ** 5 * Edot)) ** (1 / 7) * a ** (-5 / 7)
                    # if ell_b < ell_gw:
                    #     merger_type = MergerOutcome.InClusterInspiral
                    #     break

                    # Dani
                    # Inspirals driven by direct encounters
                    # t3 = 0.2 * m1 * m2 / (2 * a) * Edot ** -1  # In units where M=M_sun, L=1AU, G=1
                    # t3 /= np.sqrt(4 * np.pi ** 2)  # Convert to yr
                    mbhmean = np.nanmean(m_d) if len(m_d) > 0 else mbhmean
                    Nst = (Mcl - Mbh) / mst
                    lnDelta = max(1, np.log(0.02 * Nst))
                    lnDelta2 = max(1, np.log(0.02 * Nbh))
                    # print(f"{mst = :.3f} \t {mmean = :.3f} ")
                    rhBH = rh * (Mbh / Mcl) ** (3 / 5) * (mbhmean / mst * lnDelta2 / lnDelta) ** (2 / 5)
                    vdBH = np.sqrt(0.2 * G_AU_MSUN_KMS * Mbh / (rhBH * PC_TO_AU))  # Breen & Heggie (2012)
                    # dseta2 = 0.0926
                    # rcBH = rhBH * Nbh ** (-2 / 3) * (104 / (dseta2 * lnDelta2)) ** (1 / 3)
                    # nBHc = Nbh / (4 / 3 * np.pi * rcBH ** 3)
                    nBH = 0.5 * Nbh / (4 / 3 * np.pi * rhBH ** 3)  # Number density of BHs within the rhBH
                    # nBHc = 9 * vdBH ** 2 / (mbhmean * 4 * np.pi * G_AU_MSUN_KMS * rcBH ** 2)
                    vc = np.sqrt(G_AU_MSUN_KMS * m1 * m2 * (m1 + m2 + mbhmean) / (mbhmean * (m1 + m2) * a))
                    vdBH_bs = np.sqrt(3 / 2) * vdBH
                    vhat = vdBH_bs / vc
                    de0 = np.sqrt(1 - ell_cap ** 2) - e_b
                    # Eq. 19 in Heggie Rasio 1996
                    Sigma = 4.29 * (mbhmean ** 2 / ((m1 + m2) * (m1 + m2 + mbhmean))) ** (1 / 3) * mbhmean * (
                            m1 + m2) * a ** 2 / (m1 * m2 * vhat ** 2) * e_b ** (2 / 3) * (1 - e_b ** 2) ** (
                                    1 / 3) * de0 ** (-2 / 3)
                    tdi = 4.163e16 / (nBH * Sigma * vdBH_bs)  # In yr
                    # rin_di = rh * np.sqrt((v_esc ** 2 / (v_esc ** 2 - vbin ** 2)) ** 2 - 1)
                    # tfric_di = 7.6e8 * (rin_di / 1.) ** 2 * (vd / 200.) * (10. / (m1 + m2))  # In yr
                    pdi = 1 - np.exp(-t3 / tdi)
                    assert 0 <= pdi <= 1, "pdi not computed properly"
                    isdirectinspiral = (pdi >= random())
                    if isdirectinspiral:
                        e_b = np.sqrt(1 - ell_cap ** 2)
                        merger_type = MergerOutcome.DirectInspiral
                        break

                # Recalculate E and SMA after interaction
                Ebin = Ebin * (1 + dE)
                a = G_AU_MSUN_KMS * m1 * m2 / (2 * Ebin)

                # Ejection of interlopers
                if v3 > v_esc:
                    N3ej += 1
                    M3ej += m3
                    bhv[k3, 0] = np.nan

                # Ejection of binaries
                if vbin > v_esc:
                    merger_type = MergerOutcome.Ejected
                    break

            else:  # Binary-binary interaction
                if debug_mode:
                    debug_data["Nbb"] += 1

                # use a distribution for m3 (same as m1)
                alpha_3 = 8 + 2 * alpha_samp
                m3_t = ((mbhmax ** (1 + alpha_3) - mbhmin ** (1 + alpha_3)) * random() + mbhmin ** (1 + alpha_3)) ** (
                        1 / (1 + alpha_3))

                assert m1 >= m2, "m2 < m1"
                m_d[k1] = np.nan
                m_d[k2] = np.nan

                kmin4b = np.nanargmin(m_d) if k2 == kmin or k1 == kmin else kmin
                mbhmin4b = m_d[kmin4b] if k2 == kmin or k1 == kmin else mbhmin

                assert kmin4b != k1, f"{m1 = }, {m2 = }"

                m_d[kmin4b] = np.nan
                k3 = np.nanargmin(np.abs(m_d - m3_t))
                m_d[kmin4b] = mbhmin4b

                m3, S3, t3, gen3 = bhv[k3, 0], bhv[k3, 1], bhv[k3, 2], bhv[k3, 3]

                if k3 == kmin or k3 == k1 or k3 == k2 or np.isnan(m3):
                    raise RuntimeError(f"Error in {k3=}, {m3=} determination  ({k1=}, {k2=}, {kmin=}, {kmin4b=})")

                # use a distribution for q (same as m2)
                alpha_4 = 3.5 + alpha_samp

                m_d[k3] = np.nan

                assert (m_d <= m3).sum() >= 1, "Not enough BHs"
                qmax = np.nanmax((m_d[m_d <= m3])) / m3
                qmin = mbhmin / m3

                q_t = ((qmax ** (3 + alpha_4) - qmin ** (3 + alpha_4)) * random() + qmin ** (3 + alpha_4)) ** (
                        1 / (1 + alpha_4))
                q_t = min(q_t, qmax)

                k4 = np.nanargmin(np.abs(m_d - q_t * m3))
                m_d[k1] = m1
                m_d[k2] = m2
                m_d[k3] = m3

                m4, S4, t4, gen4 = bhv[k4, 0], bhv[k4, 1], bhv[k4, 2], bhv[k4, 3]

                if k4 == k1 or k4 == k2 or k4 == k3 or np.isnan(m4):
                    raise RuntimeError(f"Error in {k4=}, {m4=} determination ({k1=}, {k2=}, {k3=}, {kmin=}, {kmin4b=})")

                a2 = ah
                assert a <= a2, "a > a2"

                alpha = a / a2
                if debug_mode:
                    debug_data["a"].append(a)
                    debug_data["a2"].append(a2)
                if alpha < 500:
                    # f = 3.4 * (1 + (alpha / 8.6) ** 2) ** -0.8
                    mmeanbb = (m1 + m2 + m3 + m4) / 4
                    pmerg = 0.034 * (mmeanbb / 20) ** (5 / 7) * (a / 0.1) ** (-5 / 7) * (
                            1 + (alpha / 8.6) ** 2) ** -0.83

                    if pmerg >= random():
                        # Merger in binary-binary interaction
                        merger_type = MergerOutcome.GWCaptureBB
                        break

                continue

        M3ej += m1 + m2

        if Nbh_core < 4:
            continue
        if M3ej > fbh0 * Mcl_i:
            lastBH = True

        if m1 == 0 or m2 == 0:
            raise ValueError("Error in sampling, BH masses should not be 0!")

        #      GW timescale
        R = (1 + 73 / 24 * e_b ** 2 + 37 / 96 * e_b ** 4)
        t_gw = 5 * c ** 5 * a ** 4 * (1 - e_b ** 2) ** (7 / 2) / (64 * m1 * m2 * (m1 + m2) * R) * 58 / 365

        if np.isnan(t_gw):
            raise ValueError(f"Error in {t_gw=}")

        #      hardening sequence timescale

        #     compute dynamical friction timescale
        t_sim = t
        if v_kick < v_esc and merger_type != MergerOutcome.Ejected:  # if the binary is retained then make new BH
            #    recompute hardening timescale if retained

            t = cBHBd.funcs.get_sync_time(cbh, fbh0, Mcl_i, M3ej - m1 - m2, t)

            rin = rh * np.sqrt((v_esc ** 2 / (v_esc ** 2 - v_kick ** 2)) ** 2 - 1)
            tfric = 7.6e8 * (rin / 1.) ** 2 * (vd / 200.) * (10. / (m1 + m2))

            N3ej += 1

            # set such that the tot BH mass at that time is the same as in the cluster model
            tform = t + tfric + t_gw
            bhv[k1, 0] = np.nan  # remove one
            bhv[k2, 0] = m1 + m2  # and make a new one
            bhv[k2, 1] = chi_f  # new spin
            bhv[k2, 2] = tform  # reinclude in core only after this time
            bhv[k2, 3] = max(bhv[k1, 3], bhv[k2, 3]) + 1  # increase the BH generation by one
            if debug_mode:
                debug_data["mmax"] = max(debug_data["mmax"], np.nanmax(bhv[:, 0]))

        else:  # if the binary is ejected or the end product is ejected then remove members
            M3ej += m1 + m2
            if M3ej > fbh0 * Mcl_i:
                lastBH = True
                # return bhout, cbh, clprops

            t = cBHBd.funcs.get_sync_time(cbh, fbh0, Mcl_i, M3ej, t)

            N3ej += 2

            # set such that the tot BH mass at that time is the same as in the cluster model
            bhv[k1, 0] = np.nan  # remove one
            bhv[k2, 0] = np.nan  # remove two
            if debug_mode:
                debug_data["mmax"] = max(debug_data["mmax"], np.nanmax(bhv[:, 0]))

        tmerge = t + t_gw  # time of merger

        if lastBH:
            merger_type = MergerOutcome.Ejected

        if merger_type is None:
            raise ValueError("Merger type has not been set!")

        bhout.append({"tmerge": tmerge,  # 0 merger time
                      "t_sim": t_sim,  # 1 simulation time
                      "m1": m1,  # 2 mass of component 1
                      "m2": m2,  # 3 mass of component 2
                      "e_b": e_b,  # 4 eccentricity of binary just before GW radiation takes over
                      "Mcl_i": Mcl_i,  # 5 cluster mass
                      "merger_type": merger_type,  # 6 merger type
                      "tf0": tf0,  # 7 look-back time of formation
                      "rh": rh,  # 8 half mass radius
                      "v_kick": v_kick,  # 9 recoil kick
                      "v_esc": v_esc,  # 10 escape velocity
                      "chi_f": chi_f,  # 11 chi_f final remnant spin
                      "S1": S1,  # 12 S1 spin of component 1
                      "S2": S2,  # 13 S2 spin of component 2
                      "Z": Z,  # 14 metallicity
                      "a": a,  # 15 semimajor axis
                      "gen": round(max(gen1, gen2)),  # 16 generation of merger
                      "Mbh": Mbh,  #
                      })

        if lastBH:
            return _format_output(bhout, cbh, clprops, debug_data, output_dataframe)
    return _format_output(bhout, cbh, clprops, debug_data, output_dataframe)


def _format_output(bhout, cbh, clprops, debug_data, output_dataframe):
    bhout_fmt = pd.DataFrame(bhout)
    if output_dataframe:
        if len(bhout_fmt) == 0:
            bhout_fmt = pd.DataFrame(
                columns=["tmerge", "t_sim", "m1", "m2", "e_b", "Mcl_i", "merger_type", "tf0", "rh", "v_kick", "v_esc",
                         "chi_f", "S1", "S2", "Z", "a", "gen", "Mbh"])
    else:
        bhout_fmt = bhout_fmt.to_numpy()  # FIXME: what if there are no mergers?
    return bhout_fmt, cbh, clprops, debug_data


if __name__ == "__main__":
    debug_mode = True
    tf0_tst = astropy.cosmology.Planck18.lookback_time(np.array([3, ])).value[0]

    print("---------------- TEST RUN BEGIN ----------------")

    merger_stats = {merger_type: [] for merger_type in MergerOutcome.get_outcomes()}
    merger_stats["total"] = []

    seeds = list(range(30))
    for seed in seeds:
        tst_start_time = time.time()

        # FIXME: clean this up

        # gw, _, _, _ = run_model(13, 1e6, 0.00125892541179417,
        #                         f"./data/BHs/RAPID/bh_Z0.00125892541179417.dat",
        #                         1e3, seed=seed)
        N = 8e5
        Mcl = N * 0.638
        # Mcl = N * 0.586  # Changed average mass to the one predicted by the clusterBH Kroupa
        rv = 2
        rg = 20
        rh = rv / 1.25

        rho_h_i = (Mcl / 2) / ((4 / 3) * np.pi * rh ** 3)
        Z = 0.01995262314968889
        gw, _, _, _ = run_model(13, Mcl, Z,
                                f"./data/BHs/RAPID/bh_Z{Z}.dat",
                                1e3, seed=seed)

        tst_end_time = time.time()

        print(f"Run {seed}:")
        for merger_type in MergerOutcome.get_outcomes():
            Nm_type = (gw.merger_type == merger_type).sum()
            merger_stats[merger_type].append(Nm_type)
            print(f"\t{merger_type:<20} {Nm_type}")

        print(f"\ttotal \t {len(gw)}")
        merger_stats["total"].append(len(gw))

    print("\nTotal statistics:")
    for merger_type in MergerOutcome.get_outcomes():
        merger_stats_type = np.array(merger_stats[merger_type])
        Nm_type = merger_stats_type.mean()
        Nm_type_err = merger_stats_type.std()
        print(f"\t{merger_type:<20} {Nm_type:.2f} ± {Nm_type_err:.1f}")

    Nm_total = np.array(merger_stats["total"])
    print(f"\ttotal \t {Nm_total.mean():.2f} ± {Nm_total.std():.1f}")

    print(f"\n Test run time: {tst_end_time - tst_start_time:.1f} s")
    print("---------------- TEST RUN END ----------------")
