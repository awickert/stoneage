"""
Summary statistics for a set of exposure ages.
Port of summarize_ages.m (and its inner sumstats function) by Greg Balco (BGC).
"""

import numpy as np
import scipy.special
from .constants import make_consts


def _chi2_right_tail(x, dof):
    """Right-tailed chi-squared CDF probability. Port of chisquarecdf_rt.m."""
    if dof <= 0:
        return 1.0
    return 1.0 - scipy.special.gammainc(dof / 2.0, x / 2.0)


def _ewmean(values, errors):
    """Error-weighted mean and standard error. Port of ewmean.m."""
    w = 1.0 / errors**2
    mean  = np.sum(w * values) / np.sum(w)
    stdev = 1.0 / np.sqrt(np.sum(w))
    return mean, stdev


def _sumstats(t, dt_int, dt_ext):
    """
    Summary statistics with outlier pruning.
    Inner function from summarize_ages.m.
    """
    ok = t > 0
    t     = t[ok]
    dt_int = dt_int[ok]
    dt_ext = dt_ext[ok]

    if len(t) == 0:
        return None

    result = {
        "mean": float(np.mean(t)),
        "sd":   float(np.std(t, ddof=1)) if len(t) > 1 else 0.0,
    }
    if len(t) >= 2:
        result["ewmean"], result["ewse"] = _ewmean(t, dt_int)
    else:
        result["ewmean"] = float(t[0])
        result["ewse"]   = float(dt_int[0])

    result["chi2"] = float(np.sum(((t - result["ewmean"]) / dt_int)**2))
    result["dof"]  = len(t) - 1
    result["pchi2"] = _chi2_right_tail(result["chi2"], result["dof"])

    # Outlier pruning
    strip_ok = np.arange(len(t))
    n_pruned = 0
    max_prune = len(t) // 2

    while n_pruned <= max_prune and len(strip_ok) >= 3:
        ts = t[strip_ok]; ds = dt_int[strip_ok]
        strip_mean = np.mean(ts)
        all_chi2   = ((ts - strip_mean) / ds)**2
        chi2_tot   = np.sum(all_chi2)
        p          = _chi2_right_tail(chi2_tot, len(strip_ok) - 1)
        if p > 0.01:
            break
        worst = strip_ok[np.argmax(all_chi2)]
        strip_ok = strip_ok[strip_ok != worst]
        n_pruned += 1

    strip_t    = t[strip_ok]
    strip_dt   = dt_int[strip_ok]
    strip_dext = dt_ext[strip_ok]

    if len(strip_ok) >= 2:
        strip_ewmean, strip_ewse = _ewmean(strip_t, strip_dt)
        strip_chi2 = float(np.sum(((strip_t - strip_ewmean) / strip_dt)**2))
        strip_p    = _chi2_right_tail(strip_chi2, len(strip_ok) - 1)
    else:
        strip_ewmean = float(strip_t[0]) if len(strip_ok) else np.nan
        strip_ewse   = float(strip_dt[0]) if len(strip_ok) else np.nan
        strip_p      = 0.0

    strip_mean = float(np.mean(strip_t))
    strip_sd   = float(np.std(strip_t, ddof=1)) if len(strip_ok) > 1 else 0.0

    result["strip"] = {
        "ok":      strip_ok,
        "n_pruned": n_pruned,
        "mean":    strip_mean,
        "sd":      strip_sd,
        "ewmean":  strip_ewmean,
        "ewse":    strip_ewse,
        "pchi2":   strip_p,
    }

    if strip_p > 0.05:
        result["sumval"]      = strip_ewmean
        result["sumdel_int"]  = strip_ewse
        result["method"]      = "error-weighted mean"
    else:
        result["sumval"]      = strip_mean
        result["sumdel_int"]  = strip_sd
        result["method"]      = "arithmetic mean + SD"

    # External uncertainty: propagate production rate uncertainty linearly
    if len(strip_ok) > 0 and np.any(strip_dext > strip_dt):
        rel_del_P = np.mean(
            np.sqrt(np.maximum(strip_dext**2 - strip_dt**2, 0)) / strip_t
        )
    else:
        rel_del_P = 0.0
    result["sumdel_ext"] = float(
        np.sqrt(result["sumdel_int"]**2 + (rel_del_P * result["sumval"])**2)
    )

    return result


def summarize_ages(ages: dict, consts=None) -> dict:
    """
    Compute summary statistics from get_ages output.

    Port of summarize_ages.m (Balco, BGC, 2016).

    Parameters
    ----------
    ages : dict
        Output of get_ages() with n['t_St'] present.
    consts : Constants, optional

    Returns
    -------
    dict with summary statistics per nuclide and scaling scheme.
    """
    if consts is None:
        consts = make_consts()

    n = ages["n"]
    if "t_St" not in n:
        raise ValueError("Pass exposure-age results, not calibration results.")

    sfs_available = [sf for sf in ["St", "Lm", "LSDn"] if f"t_{sf}" in n]

    out = {}

    # Summary over all nuclides
    for sf in sfs_available:
        out.setdefault("all", {})[sf] = _sumstats(
            n[f"t_{sf}"],
            n[f"delt_int_{sf}"],
            n[f"delt_ext_{sf}"],
        )

    # Summary per nuclide
    nindex = n.get("nindex")
    if nindex is not None:
        for ni, nuc in enumerate(consts.nuclides):
            use = np.where(nindex == ni)[0]
            if len(use) == 0:
                continue
            for sf in sfs_available:
                out.setdefault(nuc, {})[sf] = _sumstats(
                    n[f"t_{sf}"][use],
                    n[f"delt_int_{sf}"][use],
                    n[f"delt_ext_{sf}"][use],
                )

    return out
