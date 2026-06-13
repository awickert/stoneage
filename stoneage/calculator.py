"""
Core exposure age and production rate calculator.
Port of get_ages_v3.m by Greg Balco (BGC).
"""

import numpy as np
import scipy.optimize

from .constants import Constants, make_consts
from .atmosphere import ERA40atm, antatm
from .scaling import stone2000, get_LmSF, get_LSDnSF
from .corrections import thickness, Lsp
from .cutoff_rigidity import get_DipRc


def get_ages(in_data: dict, control: dict = None, consts: Constants = None) -> dict:
    """
    Compute exposure ages or calibration production rates.

    Port of get_ages_v3.m (Balco, BGC, 2016).

    Parameters
    ----------
    in_data : dict
        Validated input structure (output of parse_v3_input).
        Sub-dicts ``s`` (samples), ``n`` (nuclide measurements),
        and optionally ``c`` (calibration ages).
    control : dict, optional
        ``cal`` : 0 = compute exposure ages (default), 1 = compute production rates
        ``result_type`` : ``'short'`` (NTD only, default) or ``'long'`` (all SFs)
    consts : Constants, optional
        Physical constants; loaded from defaults if omitted.

    Returns
    -------
    dict  -- in_data with age/production-rate fields added to ``n``.
    """
    if consts is None:
        consts = make_consts()
    if control is None:
        control = {}
    cal         = control.get("cal", 0)
    result_type = control.get("result_type", "short")

    in_data["version_ages"]  = "3.0.2"
    in_data["version_muons"] = consts.version_muons
    in_data["flags"]         = []

    s = in_data["s"]
    n = in_data["n"]

    num_n = len(n["index"])

    # -----------------------------------------------------------------------
    # 1. Atmospheric pressure
    # -----------------------------------------------------------------------
    pressure = np.zeros(len(s["lat"]))
    std_idx = [i for i, aa in enumerate(s["aa"]) if aa == "std"]
    ant_idx = [i for i, aa in enumerate(s["aa"]) if aa == "ant"]
    pre_idx = [i for i, aa in enumerate(s["aa"]) if aa == "pre"]

    if std_idx:
        pressure[std_idx] = ERA40atm(
            np.array(s["lat"])[std_idx],
            np.array(s["long"])[std_idx],
            np.array(s["elv"])[std_idx],
        )
    if ant_idx:
        pressure[ant_idx] = antatm(np.array(s["elv"])[ant_idx])
    if pre_idx:
        pressure[pre_idx] = np.array(s["pressure"])[pre_idx]
    s["pressure"] = pressure

    # -----------------------------------------------------------------------
    # 2. Nuclide-index mapping into consts arrays
    # -----------------------------------------------------------------------
    nindex = np.array([consts.nuclides.index(nuc) for nuc in n["nuclide"]])
    n["nindex"] = nindex

    ni = n["index"]  # sample index for each nuclide measurement (0-based)
    N_arr   = np.array(n["N"])
    delN_arr = np.array(n["delN"])

    # -----------------------------------------------------------------------
    # 3. Attenuation length, thickness, St scaling
    # -----------------------------------------------------------------------
    Lsp_val = Lsp()
    Lsp_vec = np.full(num_n, Lsp_val)
    n["Lsp"] = Lsp_vec

    sf_thick = thickness(
        np.array(s["thick"])[ni],
        Lsp_vec,
        np.array(s["rho"])[ni],
    )
    n["sf_thick"] = sf_thick

    # Fsp=1: spallation-only scaling factor; muon production is added separately below.
    sf_St = stone2000(np.array(s["lat"])[ni], pressure[ni], Fsp=1.0)
    n["sf_St"] = sf_St

    # -----------------------------------------------------------------------
    # 4. Decay constants and muon production
    # -----------------------------------------------------------------------
    n["l"]    = consts.l[nindex]
    n["dell"] = consts.dell[nindex]

    Pmu0    = consts.Pmu0[nindex]
    L_mu   = consts.L_mu_atm[nindex]
    P_mu   = Pmu0 * np.exp((1013.25 - pressure[ni]) / L_mu)
    n["P_mu"] = P_mu

    Lmu = 1.0 / (consts.Lmu_a + consts.Lmu_b * pressure[ni])
    n["Lmu"] = Lmu

    E_arr      = np.array(s["E"])[ni]
    rho_arr    = np.array(s["rho"])[ni]
    othercorr  = np.array(s["othercorr"])[ni]

    A_sp = n["l"] + E_arr * rho_arr / Lsp_vec
    A_mu = n["l"] + E_arr * rho_arr / Lmu
    n["A_sp"] = A_sp
    n["A_mu"] = A_mu

    # -----------------------------------------------------------------------
    # 5a. NTD exposure ages (St scaling)
    # -----------------------------------------------------------------------
    if cal == 0:
        Psp_St  = sf_St * othercorr * sf_thick * consts.refP_St[nindex]
        delPsp_St = sf_St * othercorr * sf_thick * consts.delrefP_St[nindex]
        P_St    = Psp_St + P_mu
        n["Psp_St"]   = Psp_St
        n["delPsp_St"] = delPsp_St
        n["P_St"]     = P_St

        n["Nnorm_St"]        = N_arr / P_St
        n["delNnorm_St"]     = delN_arr / P_St
        n["delNnorm_ext_St"] = np.sqrt(
            (delN_arr / P_St)**2
            + (delPsp_St * N_arr / P_St**2)**2
        )

        t_St = np.zeros(num_n)
        opts = {"xtol": 1e-4}

        is_simple = A_sp == 0
        for b in np.where(is_simple)[0]:
            t_St[b] = N_arr[b] / P_St[b]

        for b in np.where(~is_simple)[0]:
            Pv = np.array([Psp_St[b], P_mu[b]])
            Av = np.array([A_sp[b], A_mu[b]])
            sat = np.sum(Pv / Av)
            if N_arr[b] >= sat:
                t_St[b] = 0
                in_data["flags"].append(
                    f"Sample {s['sample_name'][ni[b]]} -- {n['nuclide'][b]} appears saturated (St)."
                )
            elif E_arr[b] == 0:
                t_St[b] = (-1.0 / A_sp[b]) * np.log(
                    1.0 - N_arr[b] * A_sp[b] / P_St[b]
                )
            else:
                # Bounds for Brent root-finding
                one_term_sat = P_St[b] / A_sp[b]
                if N_arr[b] >= one_term_sat:
                    ub = np.inf
                else:
                    ub = (-1.0 / A_sp[b]) * np.log(
                        1.0 - N_arr[b] * A_sp[b] / P_St[b]
                    )
                lb = (-1.0 / A_mu[b]) * np.log(
                    1.0 - N_arr[b] * A_mu[b] / P_St[b]
                )

                def fun(t, b=b, Pv=Pv, Av=Av):
                    return N_arr[b] - np.sum((Pv / Av) * (1.0 - np.exp(-t * Av)))

                t_St[b] = scipy.optimize.brentq(
                    fun, lb * 0.99, min(ub * 1.01, 1e9), **opts
                )
        n["t_St"] = t_St

    # -----------------------------------------------------------------------
    # 5b. NTD production rates (St, calibration mode)
    # -----------------------------------------------------------------------
    elif cal == 1:
        c = in_data["c"]
        # Adjust calibration ages from yr-before-1950 to yr-before-collection.
        # c["truet"] is sample-indexed (position i = sample i), so use s["yr"][i]
        # directly — not c["index"][i], which maps calibration-line order to sample
        # order and would give the wrong year if lines appear out of sample order.
        for i in range(len(c["truet"])):
            if c["truet"][i] > 0:
                c["truet"][i] += s["yr"][i] - 1950
            if c["mint"][i] > 0:
                c["mint"][i] += s["yr"][i] - 1950
            if c["maxt"][i] < np.inf:
                c["maxt"][i] += s["yr"][i] - 1950

        calc_P_St   = np.zeros(num_n)
        calc_minP   = np.zeros(num_n)
        calc_maxP   = np.zeros(num_n)
        Nmu         = np.zeros(num_n)
        Nsp         = np.zeros(num_n)

        is_simple = A_sp == 0

        for b in range(num_n):
            si = ni[b]
            truet = c["truet"][si]
            mint  = c["mint"][si]
            maxt  = c["maxt"][si]

            if truet > 0:
                Nmu[b] = (P_mu[b] * truet if is_simple[b]
                          else (P_mu[b] / A_mu[b]) * (1 - np.exp(-A_mu[b] * truet)))
                Nsp[b] = N_arr[b] - Nmu[b]
                if is_simple[b]:
                    calc_P_St[b] = Nsp[b] / truet
                else:
                    calc_P_St[b] = (Nsp[b] * A_sp[b]) / (1 - np.exp(-A_sp[b] * truet))
                calc_P_St[b] /= sf_St[b] * othercorr[b] * sf_thick[b]

            if mint > 0:
                Nmu_min = (P_mu[b] * mint if is_simple[b]
                           else (P_mu[b] / A_mu[b]) * (1 - np.exp(-A_mu[b] * mint)))
                Nsp_min = N_arr[b] - Nmu_min
                if is_simple[b]:
                    calc_maxP[b] = Nsp_min / mint
                else:
                    calc_maxP[b] = (Nsp_min * A_sp[b]) / (1 - np.exp(-A_sp[b] * mint))
                calc_maxP[b] /= sf_St[b] * othercorr[b] * sf_thick[b]

            if maxt < np.inf:
                Nmu_max = (P_mu[b] * maxt if is_simple[b]
                           else (P_mu[b] / A_mu[b]) * (1 - np.exp(-A_mu[b] * maxt)))
                Nsp_max = N_arr[b] - Nmu_max
                if is_simple[b]:
                    calc_minP[b] = Nsp_max / maxt
                else:
                    calc_minP[b] = (Nsp_max * A_sp[b]) / (1 - np.exp(-A_sp[b] * maxt))
                calc_minP[b] /= sf_St[b] * othercorr[b] * sf_thick[b]

        n["calc_P_St"]   = calc_P_St
        n["calc_maxP_St"] = calc_maxP
        n["calc_minP_St"] = calc_minP
        n["Nmu"] = Nmu
        n["Nsp"] = Nsp

    # -----------------------------------------------------------------------
    # 6. Time-dependent scaling (Lm and LSDn) -- only if result_type='long'
    # -----------------------------------------------------------------------
    sfa = ["St"]
    if result_type == "long":
        sfa = ["St", "Lm", "LSDn"]

        # Build time vector for Rc integration
        rin = {
            "lat":  np.array(s["lat"])[ni],
            "long": np.array(s["long"])[ni],
            "yr":   np.array(s["yr"])[ni],
        }
        if cal == 0:
            req_t = n["t_St"] * 1.5
            # Saturated cases: use 8 half-lives
            is_sat = n["t_St"] == 0
            eff_hl = -np.log(0.5) / np.where(A_sp > 0, A_sp, 1)
            req_t[is_sat] = 8 * eff_hl[is_sat]
            # C-14: cap at 100 ka
            is_c14 = nindex == 4
            req_t[is_c14] = 1e5
        else:
            c = in_data["c"]
            req_t = np.array([c["truet"][ni[b]] for b in range(num_n)], dtype=float)
            for b in range(num_n):
                si = ni[b]
                if not np.isinf(c["maxt"][si]):
                    req_t[b] = c["maxt"][si]
                elif c["mint"][si] > 0:
                    req_t[b] = c["mint"][si]
        rin["t"] = req_t

        sfdata = get_DipRc(rin)
        sfdata["nuclide"]  = n["nuclide"]
        sfdata["pressure"] = pressure[ni]

        sfdata = get_LmSF(sfdata)
        sfdata = get_LSDnSF(sfdata)

        in_data["f"] = sfdata

        # For each TD scheme compute cumulated N
        for sf in ["Lm", "LSDn"]:
            refP_key   = f"refP_{sf}"
            refP_arr   = getattr(consts, refP_key)

            cum_N_sp = []
            cum_N_mu = []
            cum_N    = []

            for b in range(num_n):
                P0 = refP_arr[nindex[b]] * othercorr[b] * sf_thick[b]
                pp = P0 * sfdata[sf][b]
                dt = sfdata["tmax"][b] - sfdata["tmin"][b]

                if A_sp[b] == 0:
                    int_sp = pp * dt
                    int_mu = P_mu[b] * dt
                else:
                    tmax_b = sfdata["tmax"][b]
                    tmin_b = sfdata["tmin"][b]
                    int_sp = -(pp / A_sp[b]) * (np.exp(-tmax_b * A_sp[b]) - np.exp(-tmin_b * A_sp[b]))
                    int_mu = -(P_mu[b] / A_mu[b]) * (np.exp(-tmax_b * A_mu[b]) - np.exp(-tmin_b * A_mu[b]))

                cum_sp = np.cumsum(int_sp)
                cum_mu = np.cumsum(int_mu)
                cum_N_sp.append(cum_sp)
                cum_N_mu.append(cum_mu)
                cum_N.append(cum_sp + cum_mu)

            sfdata[f"cum_N_sp_{sf}"] = cum_N_sp
            sfdata[f"cum_N_mu_{sf}"] = cum_N_mu
            sfdata[f"cum_N_{sf}"]    = cum_N

        if cal == 0:
            for sf in ["Lm", "LSDn"]:
                t_td = np.zeros(num_n)
                for b in range(num_n):
                    max_N = sfdata[f"cum_N_{sf}"][b][-1]
                    if N_arr[b] > max_N:
                        t_td[b] = 0
                        in_data["flags"].append(
                            f"Sample {s['sample_name'][ni[b]]} -- {n['nuclide'][b]} saturated ({sf})."
                        )
                    else:
                        t_td[b] = np.interp(
                            N_arr[b],
                            sfdata[f"cum_N_{sf}"][b],
                            sfdata["tmax"][b],
                        )
                n[f"t_{sf}"] = t_td

        elif cal == 1:
            c = in_data["c"]
            for sf in ["Lm", "LSDn"]:
                refP_arr = getattr(consts, f"refP_{sf}")
                calc_P   = np.zeros(num_n)
                calc_min = np.zeros(num_n)
                calc_max = np.zeros(num_n)

                for b in range(num_n):
                    si = ni[b]
                    cum_mu  = sfdata[f"cum_N_mu_{sf}"][b]
                    cum_sf  = sfdata[f"cum_N_sp_{sf}"][b] / refP_arr[nindex[b]]
                    tmax_b  = sfdata["tmax"][b]

                    truet = c["truet"][si]
                    mint  = c["mint"][si]
                    maxt  = c["maxt"][si]

                    if truet > 0:
                        Nsp_t = N_arr[b] - cum_mu[-1]
                        calc_P[b] = Nsp_t / cum_sf[-1]

                    if mint > 0:
                        Nsp_t = N_arr[b] - np.interp(mint, tmax_b, cum_mu)
                        calc_max[b] = Nsp_t / np.interp(mint, tmax_b, cum_sf)

                    if maxt < np.inf:
                        Nsp_t = N_arr[b] - cum_mu[-1]
                        calc_min[b] = Nsp_t / cum_sf[-1]

                n[f"calc_P_{sf}"]    = calc_P
                n[f"calc_minP_{sf}"] = calc_min
                n[f"calc_maxP_{sf}"] = calc_max

    # -----------------------------------------------------------------------
    # 7. Uncertainty propagation
    # -----------------------------------------------------------------------
    if cal == 0:
        for sf in sfa:
            refP_arr  = getattr(consts, f"refP_{sf}")
            drefP_arr = getattr(consts, f"delrefP_{sf}")
            fdelP = drefP_arr[nindex] / refP_arr[nindex]

            tt = n[f"t_{sf}"]
            ok     = tt > 0
            case1  = ok & (A_sp > 0)
            case2  = ok & (A_sp == 0)

            FP    = np.zeros(num_n)
            FP[case1] = N_arr[case1] * A_sp[case1] / (1 - np.exp(-A_sp[case1] * tt[case1]))
            FP[case2] = N_arr[case2] / tt[case2]

            delFP  = np.zeros(num_n)
            dtdN   = np.zeros(num_n)
            dtdP   = np.zeros(num_n)

            delFP[ok]  = fdelP[ok] * FP[ok]
            dtdN[ok]   = 1.0 / (FP[ok] - N_arr[ok] * A_sp[ok])
            dtdP[ok]   = -N_arr[ok] / (FP[ok]**2 - N_arr[ok] * A_sp[ok] * FP[ok])

            delt_ext = np.zeros(num_n)
            delt_int = np.zeros(num_n)
            delt_ext[ok] = np.sqrt(dtdN[ok]**2 * delN_arr[ok]**2 + dtdP[ok]**2 * delFP[ok]**2)
            delt_int[ok] = np.sqrt(dtdN[ok]**2 * delN_arr[ok]**2)

            n[f"delt_ext_{sf}"] = delt_ext
            n[f"delt_int_{sf}"] = delt_int

            if sf != "St":
                Nnorm = np.zeros(num_n)
                dNnorm = np.zeros(num_n)
                dNnorm_ext = np.zeros(num_n)
                Nnorm[ok]      = N_arr[ok] / FP[ok]
                dNnorm[ok]     = delN_arr[ok] / FP[ok]
                dNnorm_ext[ok] = np.sqrt(
                    (delN_arr[ok] / FP[ok])**2
                    + (fdelP[ok] * N_arr[ok] / FP[ok])**2
                )
                n[f"Nnorm_{sf}"]        = Nnorm
                n[f"delNnorm_{sf}"]     = dNnorm
                n[f"delNnorm_ext_{sf}"] = dNnorm_ext

    else:
        c = in_data["c"]
        for sf in sfa:
            calc_delP    = np.zeros(num_n)
            calc_delmaxP = np.zeros(num_n)
            calc_delminP = np.zeros(num_n)

            for b in range(num_n):
                si = ni[b]
                truet = c["truet"][si]
                mint  = c["mint"][si]
                maxt  = c["maxt"][si]
                P_b   = n[f"calc_P_{sf}"][b] if f"calc_P_{sf}" in n else 0.0

                if truet > 0 and P_b > 0:
                    if A_sp[b] > 0:
                        SFeff  = (Nsp[b] * A_sp[b]) / (P_b * othercorr[b] * sf_thick[b] * (1 - np.exp(-A_sp[b] * truet)))
                        dPdNsp = A_sp[b] / (SFeff * othercorr[b] * sf_thick[b] * (1 - np.exp(-A_sp[b] * truet)))
                        dPdt   = (Nsp[b] * A_sp[b]**2 * np.exp(-A_sp[b] * truet)
                                  / (SFeff * sf_thick[b] * othercorr[b] * (1 - np.exp(-A_sp[b] * truet))**2))
                    else:
                        SFeff  = (Nsp[b] * truet) / (P_b * othercorr[b] * sf_thick[b])
                        dPdNsp = truet / (SFeff * othercorr[b] * sf_thick[b])
                        dPdt   = Nsp[b] / (SFeff * othercorr[b] * sf_thick[b])
                    calc_delP[b] = np.sqrt(
                        (delN_arr[b] * dPdNsp)**2 + (c["dtruet"][si] * dPdt)**2
                    )

            n[f"calc_delP_{sf}"]    = calc_delP
            n[f"calc_delmaxP_{sf}"] = calc_delmaxP
            n[f"calc_delminP_{sf}"] = calc_delminP

    return in_data
