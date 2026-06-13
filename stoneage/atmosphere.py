"""
Atmospheric pressure from elevation.
Port of ERA40atm.m and antatm.m by Greg Balco (BGC).
"""

import numpy as np
import scipy.io
import scipy.interpolate
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"
_era40_cache = None


def _load_era40():
    global _era40_cache
    if _era40_cache is None:
        mat = scipy.io.loadmat(_DATA_DIR / "ERA40.mat")
        _era40_cache = {
            "lon": mat["ERA40lon"].squeeze(),
            "lat": mat["ERA40lat"].squeeze(),
            "meanP": mat["meanP"],
            "meanT": mat["meanT"],
        }
    return _era40_cache


def ERA40atm(site_lat, site_lon, site_elv):
    """
    Atmospheric pressure from ERA-40 reanalysis + standard atmosphere.

    Port of ERA40atm.m (Balco / Lifton).

    Parameters
    ----------
    site_lat : array-like, decimal degrees, south negative
    site_lon : array-like, decimal degrees, west negative
    site_elv : array-like, metres

    Returns
    -------
    pressure : ndarray, hPa
    """
    site_lat = np.atleast_1d(np.asarray(site_lat, dtype=float))
    site_lon = np.atleast_1d(np.asarray(site_lon, dtype=float))
    site_elv = np.atleast_1d(np.asarray(site_elv, dtype=float))

    # Convert negative longitudes to 0-360
    site_lon = site_lon.copy()
    site_lon[site_lon < 0] += 360.0

    era = _load_era40()
    lon = era["lon"]
    lat = era["lat"]
    meanP = era["meanP"]
    meanT = era["meanT"]

    # interp2 equivalent: RegularGridInterpolator on (lat, lon) grid
    interp_P = scipy.interpolate.RegularGridInterpolator(
        (lat, lon), meanP, method="linear", bounds_error=False, fill_value=None
    )
    interp_T = scipy.interpolate.RegularGridInterpolator(
        (lat, lon), meanT, method="linear", bounds_error=False, fill_value=None
    )

    pts = np.column_stack([site_lat, site_lon])
    site_slp = interp_P(pts)
    site_T = interp_T(pts)

    # Lifton lapse-rate fit to COSPAR CIRA-86 (<10 km)
    lr = np.array([
        -6.1517e-3, -3.1831e-6, -1.5014e-7,
        1.8097e-9,   1.1791e-10, -6.5359e-14, -9.5209e-15,
    ])
    dtdz = (
        lr[0]
        + lr[1] * site_lat
        + lr[2] * site_lat**2
        + lr[3] * site_lat**3
        + lr[4] * site_lat**4
        + lr[5] * site_lat**5
        + lr[6] * site_lat**6
    )
    dtdz = -dtdz  # positive lapse rate sign convention

    gmr = -0.03417
    pressure = site_slp * np.exp(
        (gmr / dtdz) * (np.log(site_T) - np.log(site_T - site_elv * dtdz))
    )
    return pressure


def antatm(z):
    """
    Antarctic atmospheric pressure from elevation.

    Port of antatm.m (Balco).  Radok, Allison & Wendler (1996).

    Parameters
    ----------
    z : array-like, metres

    Returns
    -------
    pressure : ndarray, hPa
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return 989.1 * np.exp(z / (-7588.0))
