import numpy as np
import pandas as pd
from numpy.random import random

from . import funcs
from . import cluster
from . import remnant
from .funcs import MergerOutcome

remnant_model = None


def get_end_data(bhv, cbh, t_fin_yr):
    # Now kmax includes stuff in dynamical friction
    kmax = len(bhv[:, 0]) - np.nanargmax(np.flip(bhv[:, 0])) - 1
    mIMBH = bhv[kmax, 0]
    chiIMBH = bhv[kmax, 1]
    genIMBH = bhv[kmax, 3]

    Mbhf = cbh.Mbh_interp(t_fin_yr / 1e9)
    Mclf = cbh.M_interp(t_fin_yr / 1e9)
    rhf = cbh.rh_interp(t_fin_yr / 1e9)

    aux_data = {"mIMBH": mIMBH, "chiIMBH": chiIMBH, "genIMBH": genIMBH,
                "Mbhf": Mbhf, "Mclf": Mclf, "rhf": rhf}

    return aux_data


def _run_mergers(self, **kwargs):
    t_fin0 = self.tend
    global remnant_model
    MAX_TEND = 15e3
    assert (t_fin0 <= MAX_TEND), f"Can't run a model for more than {MAX_TEND} Myr"
    t_fin_yr = t_fin0 * 1e6  # Convert time to year

    if self.verbose:
        print("Generating new model with")
        print(f"\t Mass = {self.M0:.2g} M_sun")
        print(f"\t Metallicity = {self.Z:.2e}")
        print(f"\t Final time = {t_fin0 :.3g} Myr")
        print(f"\t Initial density = {self.rhoh0 :.3g} M_sun/pc^3", flush=True)
        print(f"\t Seed = {self.seed}")

    if self.seed is not None:
        np.random.seed(self.seed)

    # Constants and conversion factors
    G_AU_MSUN_KMS = 887.1278675
    G_AU_MSUN_YR = 4 * np.pi ** 2
    PC_TO_AU = 648000 / np.pi
    C_AU_YR = 63198  # Speed of light in AU/yr
    c = 1e4  # Speed of light in units where M=M_sun, L=1AU, G=1

    # Cluster evolution
    # Some other model parameters
    bbh = []
    v_esc0 = 50 * (self.M0 / 1e5) ** (1 / 3) * (self.rhoh0 / 1e5) ** (1 / 6)

    # Construct IMBH seeds
    # bhv = []
    MIMBH = 0
    Mprog = 0  # Total mass of the progenitor stars to IMBH seeds
    NIMBH = 0  # Initial number of IMBHs
    #
    # if add_imbh_seed:
    #     assert Z_file_imbh is not None, "Z_file_imbh must be provided if add_imbh_seed is True"
    #     imbhdata = np.loadtxt(Z_file_imbh)
    #
    #     if self.Z == 0.003:
    #         NIMBH = np.random.randint(80, 110 + 1)
    #     elif self.Z == 0.007:
    #         NIMBH = np.random.randint(50, 75 + 1)
    #     else:
    #         raise NotImplementedError(f"IMBH seeding for {self.Z = } not implemented")
    #
    #     for i in range(NIMBH):
    #         mproj0, mimbh0, _ = imbhdata[np.random.randint(0, len(imbhdata))]
    #         MIMBH += mimbh0
    #         Mprog += mproj0
    #         spin0 = 0
    #         bhv.append([mimbh0, spin0, 0, 1])

    # Construct the BH IMF
    # FIXME: Mbh_min=0,  MprogIMBH=Mprog, MIMBH=MIMBH, NIMBH=NIMBH,
    self.cluster = cluster.Cluster(**kwargs)
    t_fin_yr = min(t_fin_yr, self.cluster.t.max() * 1e9)

    # Build array with BH properties for retained stellar-mass BHs
    Mtot, bhv = funcs.sample_BHs(self.cluster.Mbh[0] - MIMBH, v_esc0, self.m_breaks[-2], self.mmax, self.a_slopes[-1],
                                 self.FeH, self.SN_model, self.cluster.sigmans)
    mbhmaxmax = np.max(bhv[:, 0])

    # TODO: bhv.extend(bhvIMBH)
    Nbh0 = len(bhv)

    if self.verbose:
        print(f"\t Number of BHs after natal kicks: {Nbh0}")

    if Nbh0 < 3:  # Return if there are not enough BHs
        aux_data = get_end_data(bhv, self.cluster, t_fin_yr)
        aux_data["mbhmaxmax"] = mbhmaxmax
        return _format_output(bbh, self.output_dataframe, aux_data)

    mbhmean = Mtot / Nbh0
    fbh0 = Mtot / self.M0
    if self.verbose:
        print(f"\t Averaged BH mass after kicks: {mbhmean:.2f}")
        print(f"\t BH mass fraction: {fbh0 = :.4f}")

    # Compute time when balanced evolution starts (Antonini & Gieles 2020)
    tcc = self.cluster.tcc / 1e3

    # Start time integration
    t = tcc * 1e9  # In years from this on
    N3ej = 0  # Ejected BHs (+ BHs destroyed in mergers)
    M3ej = 0  # Ejected BH mass

    if M3ej > 100:
        raise RuntimeError("Initial M3ej is too big")

    alpha_samp = None

    while t <= t_fin_yr:
        m_d, mbhmax, mbhmin, Nbh_core, kmin, kmax = funcs.get_mbh_params(bhv, t)
        alpha_samp = funcs.get_alpha_samp(m_d, Nbh_core, mbhmin, mbhmax, alpha_samp)

        if Nbh_core < 4:
            if len(bhv[~np.isnan(bhv[:, 0])]) < 4:
                aux_data = get_end_data(bhv, self.cluster, t_fin_yr)
                aux_data["mbhmaxmax"] = mbhmaxmax
                return _format_output(bbh, self.output_dataframe, aux_data)  # Exit if there are no BHs in the core
            else:
                t += np.min(bhv[:, 2][bhv[:, 2] > 0])
                continue

        if np.isnan(mbhmax) or np.isnan(mbhmin) or kmin == kmax or kmin is None or kmax is None:
            raise ValueError(f"Error in determining m range, found {mbhmax=} ({kmax=}) and {mbhmin=} ({kmin=})")

        # Sample m_1 according to a power law p(m_1) \propto m_1^{alpha_1}
        alpha_1 = 8 + 2 * alpha_samp
        m1_t = ((mbhmax ** (1 + alpha_1) - mbhmin ** (1 + alpha_1)) * random() + mbhmin ** (1 + alpha_1)) ** (
                1 / (1 + alpha_1))
        m_d[kmin] = np.nan
        k1 = np.nanargmin(np.abs(m_d - m1_t))
        m_d[kmin] = mbhmin
        m1, S1, gen1 = bhv[k1, 0], bhv[k1, 1], bhv[k1, 3]
        if k1 == kmin or np.isnan(m1):  # Raise exception if the primary is the lightest BH
            raise RuntimeError(f"Error in {k1=}, {m1=} determination")

        # Sample q according to a power law p(q) \propto q^{alpha_2}
        alpha_2 = 3.5 + alpha_samp
        m_d[k1] = np.nan
        qmax = np.nanmax((m_d[m_d <= m1])) / m1
        qmin = mbhmin / m1
        q_t = ((qmax ** (1 + alpha_2) - qmin ** (1 + alpha_2)) * random() + qmin ** (1 + alpha_2)) ** (
                1 / (1 + alpha_2))
        k2 = np.nanargmin(np.abs(m_d - q_t * m1))
        m_d[k1] = m1
        m2, S2, gen2 = bhv[k2, 0], bhv[k2, 1], bhv[k2, 3]
        if k2 == k1 or np.isnan(m2):  # Raise exception if the primary and the secondary are the same BH
            raise RuntimeError(f"Error in {k2=}, {m2=} determination")

        assert m2 <= m1, "The primary should always be the most massive"

        # Update cluster properties
        if t > t_fin_yr:
            aux_data = get_end_data(bhv, self.cluster, t_fin_yr)
            aux_data["mbhmaxmax"] = mbhmaxmax
            return _format_output(bbh, self.output_dataframe, aux_data)
        rh = self.cluster.rh_interp(t / 1e9)
        Mcl = self.cluster.M_interp(t / 1e9)
        Mbh = max(self.cluster.Mbh_interp(t / 1e9), 0)
        mmean = self.cluster.m_interp(t / 1e9)
        mst = self.cluster.mst_interp(t / 1e9)
        Edot = self.cluster.Edot_rlx_interp(t / 1e9)
        Edot *= PC_TO_AU ** 2 * 1e-18  # Convert to AU^2 Msun / yr^3

        lastBH = False
        if Mbh <= 0 or Mcl <= 0:  # Finish evolution if cluster evaporates
            lastBH = True
        v_esc = 3.69e-3 * np.sqrt(Mcl / rh) * 30

        # Velocity dispersion, SMA, and binding energy at the hard-soft boundary
        vd = np.sqrt(0.2 * G_AU_MSUN_KMS * Mcl / (rh * PC_TO_AU))
        ah = G_AU_MSUN_KMS * m1 * m2 / (2 * mmean * vd ** 2)  # In AU if sigma in Km/s and M in solar masses
        Eh = (1 / 2) * (m1 + m2) * vd ** 2  # In Msun*(km/s)^2

        # Form a new binary and run few-body interactions
        a = ah
        Ebin = Eh
        e = np.sqrt(random())
        merger_type = None

        while not lastBH:
            m_d, mbhmax, mbhmin, Nbh_core, kmin, kmax = funcs.get_mbh_params(bhv, t)
            alpha_samp = funcs.get_alpha_samp(m_d, Nbh_core, mbhmin, mbhmax, alpha_samp)
            Nbh = Nbh0 - N3ej

            if Nbh_core < 4:
                if len(bhv[~np.isnan(bhv[:, 0])]) < 4:
                    aux_data = get_end_data(bhv, self.cluster, t_fin_yr)
                    aux_data["mbhmaxmax"] = mbhmaxmax
                    return _format_output(bbh, self.output_dataframe, aux_data)  # Exit if there are no BHs in the core
                else:
                    t += np.min(bhv[:, 2][bhv[:, 2] > 0])
                    break

            if np.isnan(mbhmax) or np.isnan(mbhmin) or kmin == kmax or kmin is None or kmax is None:
                raise ValueError(f"Error in determining m range, found {mbhmax=} ({kmax=}) and {mbhmin=} ({kmin=})")

            if N3ej + 3 >= Nbh0:
                aux_data = get_end_data(bhv, self.cluster, t_fin_yr)
                aux_data["mbhmaxmax"] = mbhmaxmax
                return _format_output(bbh, self.output_dataframe, aux_data)  # Exit if there are no BHs remaining

            # BINARY-SINGLE INTERACTION
            # Sample m_3 according to a power law p(m_3) \propto m_3^{alpha_3}
            alpha_3 = alpha_samp + 0.5
            m3_t = ((mbhmax ** (1 + alpha_3) - mbhmin ** (1 + alpha_3)) * random() + mbhmin ** (1 + alpha_3)) ** (
                    1 / (1 + alpha_3))
            m_d[k1] = np.nan  # Do this for performance reasons
            m_d[k2] = np.nan
            k3 = np.nanargmin(np.abs(m_d - m3_t))
            m3 = bhv[k3, 0]
            m_d[k1] = m1
            m_d[k2] = m2
            if k3 == k1 or k3 == k2 or np.isnan(m3) or bhv[k3, 2] > t:
                print(f"{k1=}, {k2=}, {k3=}, {m_d=}, {m3_t=}", flush=True)
                raise ValueError(f"Error in {k3=}, {m3=} determination")

            # Fractional energy change per interaction
            dE = 0.2 * (1 - np.exp(-7 * m3 / (m1 + m2))) if m3 <= m1 + m2 else 0.2
            q2 = m2 / m1
            q3 = m3 / m1
            gamma = 3.75
            # Number of resonant states per binary-single interaction
            Nrs = (1 + q3 + q3 / q2) ** (gamma - 1) if q3 < q2 else (1 + q2 + q2 / q3) ** (gamma - 1)
            Nrs = int(round(Nrs))
            assert Nrs >= 1, "No intermediate resonant states"

            # Check for captures in binary-single resonant encounters
            is_capture = False
            Rs = 4 * (m1 + m2) / c ** 2
            ell_cap = (Rs / a) ** (5 / 14)
            for i in range(Nrs):
                e = np.sqrt(random())
                ell_b = np.sqrt(1 - e ** 2)
                if ell_b < ell_cap:
                    is_capture = True
                    merger_type = MergerOutcome.GWCaptureBS
                    break
            if is_capture:
                break

            # Exchanges
            Ppres = funcs.prob_esc(m1, m2, m3)
            Pex2 = funcs.prob_esc(m1, m3, m2)
            exchange_outcomes = ["preservation", "exchange_1", "exchange_2"]
            exchange_outcome = np.random.choice(exchange_outcomes, p=[Ppres, max(0.0, 1 - Pex2 - Ppres), Pex2])
            if exchange_outcome != "preservation":
                if exchange_outcome == "exchange_1":
                    k1, k3 = k3, k1
                elif exchange_outcome == "exchange_2":
                    k2, k3 = k3, k2
                if bhv[k2, 0] > bhv[k1, 0]:  # Force that the primary is always the most massive
                    k1, k2 = k2, k1
                m1, S1, gen1 = bhv[k1, 0], bhv[k1, 1], bhv[k1, 3]
                m2, S2, gen2 = bhv[k2, 0], bhv[k2, 1], bhv[k2, 3]
                m3, S3, gen3 = bhv[k3, 0], bhv[k3, 1], bhv[k3, 3]

            # Compute the recoil of the binary and interloper
            vbin = np.sqrt(dE * Ebin * (2 / (m1 + m2)) * (m3 / (m1 + m2 + m3)))  # In km/s
            q3 = m3 / (m1 + m2)
            v3 = vbin / q3  # In km/s

            # Recalculate energy and SMA after interaction
            Ebin = Ebin * (1 + dE)
            a = G_AU_MSUN_KMS * m1 * m2 / (2 * Ebin)

            # Ejection of interlopers
            if v3 > v_esc:
                N3ej += 1
                M3ej += m3
                bhv[k3, 0] = np.nan
                m_d, mbhmax, mbhmin, Nbh_core, kmin, kmax = funcs.get_mbh_params(bhv, t)

            # Ejection of binaries
            if vbin > v_esc:
                merger_type = MergerOutcome.Ejected
                break

            t3 = 0.2 * G_AU_MSUN_YR * m1 * m2 / (2 * a) * Edot ** -1  # In yr

            # BINARY-BINARY INTERACTION
            if Nbh_core >= 5:  # We need two binaries, plus to form the second one we need at least three BHs
                # Sample m_3 (same as m_1)
                alpha_3 = 8 + 2 * alpha_samp
                m3_t = ((mbhmax ** (1 + alpha_3) - mbhmin ** (1 + alpha_3)) * random() + mbhmin ** (1 + alpha_3)) ** (
                        1 / (1 + alpha_3))
                m_d[k1] = np.nan
                m_d[k2] = np.nan
                kmin4b = np.nanargmin(m_d) if k2 == kmin or k1 == kmin else kmin
                mbhmin4b = m_d[kmin4b] if k2 == kmin or k1 == kmin else mbhmin
                m_d[kmin4b] = np.nan
                k3 = np.nanargmin(np.abs(m_d - m3_t))
                m_d[kmin4b] = mbhmin4b
                m3, S3, gen3 = bhv[k3, 0], bhv[k3, 1], bhv[k3, 3]
                if k3 == kmin or k3 == k1 or k3 == k2 or np.isnan(m3):  # Check that we don't select the same BH twice
                    raise RuntimeError(f"Error in {k3=}, {m3=} determination  ({k1=}, {k2=}, {kmin=}, {kmin4b=})")

                # Sample m_4 (from q, same as m_2)
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
                m4, S4, gen4 = bhv[k4, 0], bhv[k4, 1], bhv[k4, 3]
                if k4 == k1 or k4 == k2 or k4 == k3 or np.isnan(m4):  # Check that we don't select the same BH twice
                    raise RuntimeError(f"Error in {k4=}, {m4=} determination ({k1=}, {k2=}, {k3=}, {kmin=}, {kmin4b=})")

                # Form the second BBH at the hard-soft boundary
                a2 = G_AU_MSUN_KMS * m3 * m4 / (2 * mmean * vd ** 2)  # In AU if sigma in Km/s and M in solar masses
                a_r = a2 / a if a2 >= a else a / a2
                a_min = min(a, a2)
                mmeanbb = (m1 + m2 + m3 + m4) / 4
                pcapbb = 0.034 * (mmeanbb / 20) ** (5 / 7) * (a_min / 0.1) ** (-5 / 7) * (
                        1 + (a_r / 8.6) ** 2) ** -0.83 if a_r < 500 else 0  # (Marín Pina et al. 2025)
                Nbb = 0.3 * (Nbh_core / 1e2) ** (-1 / 3)
                pbb = 1 - np.exp(-pcapbb * Nbb)

                # Check for captures in binary-binary encounters
                if pbb >= random():
                    if a2 < a:  # Handle the edge case that the second BBH is smaller
                        k1, m1, S1, gen1 = k3, m3, S3, gen3
                        k2, m2, S2, gen2 = k4, m4, S4, gen4
                        a = a2
                        Rs = 4 * (m1 + m2) / c ** 2
                        ell_cap = (Rs / a) ** (5 / 14)

                    e = np.sqrt(1 - ell_cap ** 2)
                    merger_type = MergerOutcome.GWCaptureBB
                    break

            # IN-CLUSTER INSPIRALS
            R = (1 + 73 / 24 * e ** 2 + 37 / 96 * e ** 4)
            t_gw = 5 * C_AU_YR ** 5 * a ** 4 * (1 - e ** 2) ** (7 / 2) / (
                    64 * G_AU_MSUN_YR ** 3 * m1 * m2 * (m1 + m2) * R)

            if t_gw < t3:
                merger_type = MergerOutcome.InClusterInspiral
                break

            # GW CAPTURES DRIVEN BY DIRECT ENCOUNTERS
            mbhmean = np.nanmean(m_d) if len(m_d) > 0 else mbhmean
            Nst = (Mcl - Mbh) / mst
            lnDelta = max(1, np.log(0.02 * Nst))
            lnDelta2 = max(1, np.log(0.02 * Nbh))
            rhBH = rh * (Mbh / Mcl) ** (3 / 5) * (mbhmean / mst * lnDelta2 / lnDelta) ** (2 / 5) if Nbh_core >= 40 \
                else rh * (mbhmean / mst) * Mbh / Mcl
            vdBH = np.sqrt(0.2 * G_AU_MSUN_KMS * Mbh / (rhBH * PC_TO_AU))  # Breen & Heggie (2012)
            nBH = 0.5 * Nbh / (4 / 3 * np.pi * rhBH ** 3)  # Number density of BHs within the rhBH
            vc = np.sqrt(G_AU_MSUN_KMS * m1 * m2 * (m1 + m2 + mbhmean) / (mbhmean * (m1 + m2) * a))
            vdBH_bs = np.sqrt(3 / 2) * vdBH
            vhat = vdBH_bs / vc
            de0 = np.sqrt(1 - ell_cap ** 2) - e
            # Eq. 19 in Heggie & Rasio (1996)
            Sigma = 4.29 * (mbhmean ** 2 / ((m1 + m2) * (m1 + m2 + mbhmean))) ** (1 / 3) * mbhmean * (
                    m1 + m2) * a ** 2 / (m1 * m2 * vhat ** 2) * e ** (2 / 3) * (1 - e ** 2) ** (
                            1 / 3) * de0 ** (-2 / 3)
            tdi = 4.163e16 / (nBH * Sigma * vdBH_bs)  # In yr
            pdi = 1 - np.exp(-t3 / tdi)
            assert 0 <= pdi <= 1, "pdi not computed properly"
            isdirectcapture = (pdi >= random())
            if isdirectcapture:
                e = np.sqrt(1 - ell_cap ** 2)
                merger_type = MergerOutcome.GWCaptureDirect
                break

        M3ej += m1 + m2

        if Nbh_core < 4:
            continue
        if M3ej > fbh0 * self.M0:
            lastBH = True

        if m1 == 0 or m2 == 0:
            raise ValueError("Error in sampling, BH masses should not be 0")

        # Compute GW timescale
        R = (1 + 73 / 24 * e ** 2 + 37 / 96 * e ** 4)
        t_gw = 5 * C_AU_YR ** 5 * a ** 4 * (1 - e ** 2) ** (7 / 2) / (
                64 * G_AU_MSUN_YR ** 3 * m1 * m2 * (m1 + m2) * R)

        assert not np.isnan(t_gw), f"Error in {t_gw = }"

        # Compute GW recoil kick and spins
        load_remnant_model = remnant_model is None
        load_remnant_model |= self.simple_remnant_model and isinstance(remnant_model, remnant.RemnantModel)
        load_remnant_model |= not self.simple_remnant_model and isinstance(remnant_model, remnant.SimpleRemnantModel)
        if load_remnant_model:
            remnant_model = remnant.SimpleRemnantModel() if self.simple_remnant_model else remnant.RemnantModel()
        m_rem, chi_f, v_kick = remnant_model.get_remnant(m1, m2, S1, S2)
        mbhmaxmax = max(mbhmaxmax, m_rem)

        # TODO: Give m_rem in output

        t_sim = t
        if v_kick < v_esc and merger_type != MergerOutcome.Ejected:  # If the binary is retained, form a merger product BH
            # Recompute hardening timescale if retained
            t = funcs.get_sync_time(self.cluster, fbh0, self.M0, M3ej - m1 - m2, t)

            rin = rh * np.sqrt((v_esc ** 2 / (v_esc ** 2 - v_kick ** 2)) ** 2 - 1)
            tfric = 7.6e8 * (rin / 1.) ** 2 * (vd / 200.) * (10. / (m1 + m2))

            N3ej += 1

            # Set such that the total BH mass at that time is the same as in the cluster model
            t_form = t + tfric + t_gw
            bhv[k1, 0] = np.nan  # Remove one BH
            bhv[k2, 0] = m_rem  # Make a new BH (merger product)
            bhv[k2, 1] = chi_f  # New spin
            bhv[k2, 2] = t_form  # Reinclude merger product in core only after dynamical friction
            bhv[k2, 3] = max(bhv[k1, 3], bhv[k2, 3]) + 1  # Increase the BH generation by one

        else:  # If the binary is ejected or the end product is ejected then remove members
            M3ej += m1 + m2
            if M3ej > fbh0 * self.M0:
                lastBH = True

            t = funcs.get_sync_time(self.cluster, fbh0, self.M0, M3ej, t)

            N3ej += 2

            # Set such that the total BH mass at that time is the same as in the cluster model
            bhv[k1, 0] = np.nan  # Remove one BH
            bhv[k2, 0] = np.nan  # Remove other BH

        t_merge = t + t_gw  # Time of merger

        if lastBH:
            merger_type = MergerOutcome.Ejected

        if merger_type is None:
            raise ValueError("Merger type has not been set!")

        bbh.append({"t_merge": t_merge,  # 0 Merger time [yr]
                    "t_sim": t_sim,  # 1 Simulation time [yr]
                    "m1": m1,  # 2 Mass of primary BH [Msun]
                    "m2": m2,  # 3 Mass of secondary BH [Msun]
                    "e": e,  # 4 Eccentricity of binary just before GW radiation takes over
                    "merger_type": merger_type,  # 6 Merger type [cbhbd.funcs.MergerOutcome]
                    "rh": rh,  # 8 Half-mass radius at t_sim [pc]
                    "v_kick": v_kick,  # 9 Recoil kick [km/s]
                    "v_esc": v_esc,  # 10 Escape velocity of the cluster at t_sim [km/s]
                    "chi_f": chi_f,  # 11 Final remnant spin
                    "S1": S1,  # 12 Spin of primary BH
                    "S2": S2,  # 13 Spin of secondary BH
                    "a": a,  # 15 Semimajor axis before GW radiation takes over [AU]
                    "gen": round(max(gen1, gen2)),  # 16 Generation of merger
                    "Mbh": Mbh,  # 17 Total mass in BHs in the cluster at t_sim [Msun]
                    })

        if lastBH:
            aux_data = get_end_data(bhv, self.cluster, t_fin_yr)
            aux_data["mbhmaxmax"] = mbhmaxmax
            return _format_output(bbh, self.output_dataframe, aux_data)
    aux_data = get_end_data(bhv, self.cluster, t_fin_yr)
    aux_data["mbhmaxmax"] = mbhmaxmax
    return _format_output(bbh, self.output_dataframe, aux_data)


def _format_output(bbh, output_dataframe, aux_data):
    bbh = pd.DataFrame(bbh)
    colnames = ["t_merge", "t_sim", "m1", "m2", "e", "merger_type", "rh", "v_kick", "v_esc",
                "chi_f", "S1", "S2", "a", "gen", "Mbh"]
    assert len(bbh) == 0 or (bbh.columns.values == colnames).all(), f"Wrong column names in output"
    if len(bbh) == 0:
        bbh = pd.DataFrame(columns=colnames)

    retval = bbh if output_dataframe else bbh.to_numpy()
    return retval, aux_data

# if __name__ == "__main__":
#     verbose = True
#
#     print("---------------- TEST RUN BEGIN ----------------")
#
#     merger_stats = {merger_type: [] for merger_type in MergerOutcome.get_outcomes()}
#     merger_stats["total"] = []
#
#     delta_t_arr = []
#     seeds = list(range(30))
#     for seed in seeds:
#         tst_start_time = time.time()
#
#         t_end = 13
#         Mcl = 8e5
#         rg = 8
#         rh = 1
#         rho_h_i = (Mcl / 2) / ((4 / 3) * np.pi * rh ** 3)
#         Z = 0.003
#         bbh, _ = self.run_model(t_end, Mcl, Z, rho_h_i, rg=rg, seed=seed, verbose=verbose)
#
#         tst_end_time = time.time()
#
#         delta_t_arr.append(tst_end_time - tst_start_time)
#         print(f"Run {seed}:")
#         for merger_type in MergerOutcome.get_outcomes():
#             Nm_type = (bbh.merger_type == merger_type).sum()
#             merger_stats[merger_type].append(Nm_type)
#             print(f"\t{merger_type:<20} {Nm_type}")
#
#         print(f"\ttotal \t {len(bbh)}")
#         merger_stats["total"].append(len(bbh))
#
#     print("\nTotal statistics:")
#     for merger_type in MergerOutcome.get_outcomes():
#         merger_stats_type = np.array(merger_stats[merger_type])
#         Nm_type = merger_stats_type.mean()
#         Nm_type_err = merger_stats_type.std()
#         print(f"\t{merger_type:<20} {Nm_type:.2f} ± {Nm_type_err:.1f}")
#
#     Nm_total = np.array(merger_stats["total"])
#     print(f"\ttotal \t {Nm_total.mean():.2f} ± {Nm_total.std():.1f}")
#
#     delta_t_arr = np.array(delta_t_arr)
#     print(f"\n Run time: ({delta_t_arr.mean():.4g} ± {delta_t_arr.std():.4g}) s per model")
#     print("---------------- TEST RUN END ----------------")
