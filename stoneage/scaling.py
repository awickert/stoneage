"""
Scaling factor functions: Stone (2000) / St, time-dependent Lm, and LSDn.
Port of stone2000.m, get_LmSF.m, get_LSDnSF.m by Greg Balco (BGC).
"""

import numpy as np
import scipy.io
import scipy.interpolate
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"
_lm_cache = None
_lsdn_cache = {}


def stone2000(lat, P, Fsp=0.978):
    """
    Geographic scaling factor for cosmogenic-nuclide production.

    Port of stone2000.m (Balco, 2001).  Implements Stone (2000), JGR 105:B10.

    Parameters
    ----------
    lat  : array-like, latitude (decimal degrees; south negative)
    P    : array-like, atmospheric pressure (hPa); same shape as lat
    Fsp  : float, fraction of SLHL production due to spallation (default 0.978)

    Returns
    -------
    scaling_factor : ndarray
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    P   = np.atleast_1d(np.asarray(P,   dtype=float))

    if lat.shape != P.shape:
        raise ValueError("lat and P must have the same shape")

    # Polynomial coefficients for spallogenic production at index latitudes
    # (Table 1 of Stone 2000)
    a = np.array([31.8518, 34.3699, 40.3153, 42.0983, 56.7733, 69.0720, 71.8733])
    b = np.array([250.3193, 258.4759, 308.9894, 512.6857, 649.1343, 832.4566, 863.1927])
    c = np.array([-0.083393, -0.089807, -0.106248, -0.120551, -0.160859, -0.199252, -0.207069])
    d = np.array([7.4260e-5, 7.9457e-5, 9.4508e-5, 1.1752e-4, 1.5463e-4, 1.9391e-4, 2.0127e-4])
    e = np.array([-2.2397e-8, -2.3697e-8, -2.8234e-8, -3.8809e-8, -5.0330e-8, -6.3653e-8, -6.6043e-8])
    ilats = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    # Compute spallogenic scaling at each index latitude for the given pressures
    lat_vals = np.vstack([
        a[i] + b[i] * np.exp(P / (-150.0)) + c[i] * P + d[i] * P**2 + e[i] * P**3
        for i in range(7)
    ]).T  # shape (n, 7)

    # Muon production at index latitudes
    mk = np.array([0.587, 0.600, 0.678, 0.833, 0.933, 1.000, 1.000])
    mu_vals = np.outer(np.ones(len(P)), mk) * np.exp((1013.25 - P[:, None]) / 242.0)
    # shape (n, 7)

    # Use absolute latitude; cap at 60
    abs_lat = np.clip(np.abs(lat), 0.0, 60.0)

    S = np.array([
        np.interp(abs_lat[i], ilats, lat_vals[i])
        for i in range(len(lat))
    ])
    M = np.array([
        np.interp(abs_lat[i], ilats, mu_vals[i])
        for i in range(len(lat))
    ])

    Fm = 1.0 - Fsp
    return S * Fsp + M * Fm


# ---------------------------------------------------------------------------
# Time-dependent scaling: Lm
# ---------------------------------------------------------------------------

def _load_lm():
    global _lm_cache
    if _lm_cache is None:
        grid = scipy.io.loadmat(_DATA_DIR / "LSDn2015_grid.mat")
        lm   = scipy.io.loadmat(_DATA_DIR / "Lm2015_all.mat")
        sfgrid = grid["sfgrid"]
        Rc2d = sfgrid["Rc"][0, 0]
        p2d  = sfgrid["p"][0, 0]
        # Rc varies along axis=1 (columns), p varies along axis=0 (rows)
        Rc_1d = Rc2d[0, :]   # unique Rc values
        p_1d  = p2d[:, 0]    # unique p values
        _lm_cache = {
            "Rc": Rc_1d.astype(float),
            "p":  p_1d.astype(float),
            "sf": lm["sf"].astype(float),
        }
    return _lm_cache


def _load_lsdn(nuclide: str):
    if nuclide not in _lsdn_cache:
        grid = scipy.io.loadmat(_DATA_DIR / "LSDn2015_grid.mat")
        sfgrid = grid["sfgrid"]
        Rc2d = sfgrid["Rc"][0, 0]
        p2d  = sfgrid["p"][0, 0]
        Rc = Rc2d[0, :].astype(float)   # unique Rc values (along columns)
        p  = p2d[:, 0].astype(float)    # unique p values (along rows)

        fname_map = {
            "N3quartz":  "LSDn2015_N3quartz.mat",
            "N3olivine": "LSDn2015_N3pxol.mat",
            "N3pyroxene":"LSDn2015_N3pxol.mat",
            "N10quartz": "LSDn2015_N10quartz.mat",
            "N10pyroxene":"LSDn2015_N10quartz.mat",
            "N14quartz": "LSDn2015_N14quartz.mat",
            "N21quartz": "LSDn2015_N21quartz.mat",
            "N26quartz": "LSDn2015_N26quartz.mat",
        }
        mat = scipy.io.loadmat(_DATA_DIR / fname_map[nuclide])
        key = f"sf{nuclide}" if nuclide not in ("N10pyroxene",) else "sfN10quartz"
        d = mat[key]
        _lsdn_cache[nuclide] = {"Rc": Rc, "p": p, "b": d["b"][0, 0], "m": d["m"][0, 0]}
    return _lsdn_cache[nuclide]


def get_LmSF(sfdata: dict) -> dict:
    """
    Interpolate Lm (time-variable Lal/Stone) scaling factors.

    Port of get_LmSF.m (Balco).

    Parameters
    ----------
    sfdata : dict with keys
        tmax      : list of 1-D arrays, upper time-step bounds (yr before collection)
        Rc        : list of 1-D arrays, cutoff rigidity (GV)
        pressure  : 1-D array or list of 1-D arrays, atmospheric pressure (hPa)

    Returns
    -------
    sfdata with key ``Lm`` added: list of 1-D scaling-factor arrays.
    """
    lm = _load_lm()
    interp = scipy.interpolate.RegularGridInterpolator(
        (lm["p"], lm["Rc"]), lm["sf"],
        method="linear", bounds_error=False, fill_value=None,
    )

    sfdata["Lm"] = []
    for a in range(len(sfdata["tmax"])):
        if isinstance(sfdata["pressure"], (list, tuple)):
            p_vec = np.asarray(sfdata["pressure"][a], dtype=float)
        else:
            p_vec = np.zeros(len(sfdata["tmax"][a])) + float(sfdata["pressure"][a])

        Rc_vec = sfdata["Rc"][a].copy().astype(float)
        Rc_vec = np.clip(Rc_vec, lm["Rc"].min(), 20.0)
        p_vec  = np.clip(p_vec,  lm["p"].min(),  lm["p"].max())

        pts = np.column_stack([p_vec, Rc_vec])  # (n_steps, 2) with (p, Rc)
        sfdata["Lm"].append(interp(pts))
    return sfdata


def lm_sf_at(p: float, Rc: float) -> float:
    """
    Lm scaling factor at a single (pressure, cutoff rigidity) point.

    Wraps the Lm2015 grid interpolator for use outside the time-step loop.

    Parameters
    ----------
    p   : atmospheric pressure (hPa)
    Rc  : geomagnetic cutoff rigidity (GV)

    Returns
    -------
    Dimensionless Lm scaling factor.
    """
    lm  = _load_lm()
    il  = scipy.interpolate.RegularGridInterpolator(
        (lm["p"], lm["Rc"]), lm["sf"],
        method="linear", bounds_error=False, fill_value=None,
    )
    p_c  = float(np.clip(p,  lm["p"].min(),  lm["p"].max()))
    Rc_c = float(np.clip(Rc, lm["Rc"].min(), 20.0))
    return float(il(np.array([[p_c, Rc_c]]))[0])


def lsdn_sf_at(p: float, Rc: float, S: float,
               nuclide: str = "N10quartz") -> float:
    """
    LSDn scaling factor at a single (pressure, cutoff rigidity, solar
    modulation) point.

    SF = b(p, Rc) + S · m(p, Rc)

    where b and m are interpolated from the LSDn2015 grid and S is the
    solar modulation parameter (≈ 460 sfu for the long-term mean,
    ≈ 620 sfu for the modern solar cycle).

    Parameters
    ----------
    p       : atmospheric pressure (hPa)
    Rc      : geomagnetic cutoff rigidity (GV)
    S       : solar modulation parameter (sfu)
    nuclide : nuclide code, e.g. "N10quartz"

    Returns
    -------
    float — LSDn scaling factor in the same (absolute) units as the grid,
    such that P_ref_LSDn × lsdn_sf_at(…) gives at/g/yr.
    """
    d    = _load_lsdn(nuclide)
    p_c  = float(np.clip(p,  d["p"].min(),  d["p"].max()))
    Rc_c = float(np.clip(Rc, d["Rc"].min(), 20.0))
    pts  = np.array([[p_c, Rc_c]])
    ib   = scipy.interpolate.RegularGridInterpolator(
        (d["p"], d["Rc"]), d["b"],
        method="linear", bounds_error=False, fill_value=None,
    )
    im   = scipy.interpolate.RegularGridInterpolator(
        (d["p"], d["Rc"]), d["m"],
        method="linear", bounds_error=False, fill_value=None,
    )
    return float(ib(pts)[0]) + S * float(im(pts)[0])


def get_LSDnSF(sfdata: dict) -> dict:
    """
    Interpolate LSDn (Lifton/Sato/Dunai) scaling factors.

    Port of get_LSDnSF.m (Balco).

    Parameters
    ----------
    sfdata : dict with keys
        tmax     : list of 1-D arrays
        Rc       : list of 1-D arrays, cutoff rigidity (GV)
        S        : list of 1-D arrays, solar modulation parameter
        nuclide  : list of nuclide-code strings
        pressure : 1-D array or list of 1-D arrays (hPa)

    Returns
    -------
    sfdata with key ``LSDn`` added: list of 1-D scaling-factor arrays.
    """
    sfdata["LSDn"] = []
    for a in range(len(sfdata["tmax"])):
        nuc = sfdata["nuclide"][a]
        d = _load_lsdn(nuc)

        if isinstance(sfdata["pressure"], (list, tuple)):
            p_vec = np.asarray(sfdata["pressure"][a], dtype=float)
        else:
            p_vec = np.zeros(len(sfdata["tmax"][a])) + float(sfdata["pressure"][a])

        Rc_vec = sfdata["Rc"][a].copy().astype(float)
        Rc_vec[Rc_vec > 20] = 20.0

        interp_b = scipy.interpolate.RegularGridInterpolator(
            (d["p"], d["Rc"]), d["b"],
            method="linear", bounds_error=False, fill_value=None,
        )
        interp_m = scipy.interpolate.RegularGridInterpolator(
            (d["p"], d["Rc"]), d["m"],
            method="linear", bounds_error=False, fill_value=None,
        )
        p_clipped  = np.clip(p_vec,  d["p"].min(),  d["p"].max())
        Rc_clipped = np.clip(Rc_vec, d["Rc"].min(), 20.0)
        pts = np.column_stack([p_clipped, Rc_clipped])
        sf = interp_b(pts) + sfdata["S"][a] * interp_m(pts)
        sfdata["LSDn"].append(sf)
    return sfdata
