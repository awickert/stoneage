"""
Paleomagnetic cutoff rigidity time series.
Port of get_DipRc.m and angdistr.m by Greg Balco (BGC).
"""

import numpy as np
import scipy.io
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"


def angdistr(lat1, long1, lat2, long2):
    """
    Angular distance between points on a sphere (all args in radians).

    Port of angdistr.m (Balco, 2015).
    """
    return np.arccos(
        cos(lat1) * np.cos(lat2) * (np.cos(long1) * np.cos(long2) + np.sin(long1) * np.sin(long2))
        + np.sin(lat1) * np.sin(lat2)
    )


# Avoid shadowing numpy cos
_cos = np.cos
_sin = np.sin
_arccos = np.arccos


def _angdistr(lat1, long1, lat2, long2):
    return _arccos(
        _cos(lat1) * _cos(lat2) * (_cos(long1) * _cos(long2) + _sin(long1) * _sin(long2))
        + _sin(lat1) * _sin(lat2)
    )


_pavon_cache = None
_lifton_cache = None


def _load_pavon():
    global _pavon_cache
    if _pavon_cache is None:
        mat = scipy.io.loadmat(_DATA_DIR / "PavonDip.mat")
        _pavon_cache = mat["m"]
    return _pavon_cache


def _load_lifton():
    global _lifton_cache
    if _lifton_cache is None:
        mat = scipy.io.loadmat(_DATA_DIR / "LiftonS2015.mat")
        _lifton_cache = mat
    return _lifton_cache


def get_DipRc(in_data: dict) -> dict:
    """
    Cutoff rigidity time series from dipole approximation + Pavon-Carrasco.

    Port of get_DipRc.m (Balco, 2015).

    Parameters
    ----------
    in_data : dict with keys
        lat  : 1-D array, decimal degrees
        long : 1-D array, decimal degrees (west negative OK)
        t    : 1-D array, required record length (yr before collection)
        yr   : 1-D array, year of collection

    Returns
    -------
    dict with keys
        tmin : list of 1-D arrays, lower time-step bounds
        tmax : list of 1-D arrays, upper time-step bounds
        Rc   : list of 1-D arrays, cutoff rigidity (GV)
        S    : list of 1-D arrays, solar modulation parameter
    """
    lat  = np.atleast_1d(np.asarray(in_data["lat"],  dtype=float))
    long = np.atleast_1d(np.asarray(in_data["long"], dtype=float))
    t    = np.atleast_1d(np.asarray(in_data["t"],    dtype=float))
    yr   = np.atleast_1d(np.asarray(in_data["yr"],   dtype=float))

    # Rectify longitudes to 0-360
    long = long.copy()
    long[long < 0] += 360.0

    latr  = np.radians(lat)
    longr = np.radians(long)

    m = _load_pavon()
    sf = _load_lifton()

    # Polynomial (Lifton-Sato-Dunai 2014, Eq. 2) for geocentric dipole Rc
    pRc = np.array([-448.004, 1189.18, -1152.15, 522.061, -103.241, 6.89901, 0.0])

    SPhi    = sf["SPhi"].squeeze()
    SPhiInf = float(sf["SPhiInf"].squeeze())

    # Extract scalar attributes from the MATLAB struct array m
    m0 = m[0, 0]
    latr_pp  = m0["latr_pp"].squeeze()
    lonr_pp  = m0["lonr_pp"].squeeze()
    MM01     = m0["MM01"].squeeze()
    tzero    = float(m0["tzero"].squeeze())
    t1min    = m0["t1min"].squeeze()
    t1max    = m0["t1max"].squeeze()
    t2min    = m0["t2min"].squeeze().copy()
    t2max    = m0["t2max"].squeeze()
    t2       = m0["t2"].squeeze()
    MM02     = m0["MM02"].squeeze()

    out = {"tmin": [], "tmax": [], "Rc": [], "S": []}

    for a in range(len(lat)):
        gmlat = np.abs(np.pi / 2 - _angdistr(latr[a], longr[a], latr_pp, lonr_pp))
        dipRc = MM01 * np.polyval(pRc, np.cos(gmlat))

        temptmin = t1min + (yr[a] - tzero)
        temptmax = t1max + (yr[a] - tzero)

        # Must reset the boundary of the long record joining point
        t2min[0] = temptmax[-1]

        if yr[a] > tzero:
            temptmin[0] = 0.0
            tempS = SPhi.copy()
        elif yr[a] < tzero:
            inpast = temptmax > 0
            temptmin = temptmin[inpast]
            temptmax = temptmax[inpast]
            dipRc    = dipRc[inpast]
            tempS    = SPhi[inpast]
            temptmin[0] = 0.0
        else:
            tempS = SPhi.copy()

        req_t = t[a]

        if req_t <= t2min[0]:
            ok = temptmin < req_t
            tmin_out = temptmin[ok]
            tmax_out = temptmax[ok].copy()
            tmax_out[-1] = req_t
            Rc_out  = dipRc[ok]
            S_out   = tempS[ok]
        elif req_t <= t2[-1]:
            ok = t2min < req_t
            tmin_out = np.concatenate([temptmin, t2min[ok]])
            tmax_out = np.concatenate([temptmax, t2max[ok]])
            tmax_out[-1] = req_t
            S_out  = np.concatenate([tempS, np.full(ok.sum(), SPhiInf)])
            Rc_out = np.concatenate([
                dipRc,
                MM02[ok] * np.polyval(pRc, np.cos(np.abs(latr[a]))),
            ])
        else:
            tmin_out = np.concatenate([temptmin, t2min])
            tmax_out = np.concatenate([temptmax, t2max])
            tmax_out[-1] = req_t
            S_out  = np.concatenate([tempS, np.full(len(MM02), SPhiInf)])
            Rc_out = np.concatenate([
                dipRc,
                MM02 * np.polyval(pRc, np.cos(np.abs(latr[a]))),
            ])

        out["tmin"].append(tmin_out)
        out["tmax"].append(tmax_out)
        out["Rc"].append(np.abs(Rc_out))
        out["S"].append(S_out)

    return out
