"""
Sample corrections: thickness and spallation attenuation length.
Port of thickness.m and Lsp.m by Greg Balco (BGC).
"""

import numpy as np


def Lsp() -> float:
    """Spallogenic attenuation length (g/cm^2)."""
    return 160.0


def thickness(zmax, L, rho):
    """
    Thickness correction for spallogenic cosmogenic-nuclide production.

    Port of thickness.m (Balco).

    Parameters
    ----------
    zmax : array-like, sample thickness (cm)
    L    : array-like, effective attenuation length (g/cm^2)
    rho  : array-like, bulk density (g/cm^3)

    Returns
    -------
    correction : ndarray, dimensionless (1.0 for zero-thickness sample)
    """
    zmax = np.atleast_1d(np.asarray(zmax, dtype=float))
    L    = np.atleast_1d(np.asarray(L,    dtype=float))
    rho  = np.atleast_1d(np.asarray(rho,  dtype=float))

    with np.errstate(divide="ignore", invalid="ignore"):
        out = (L / (rho * zmax)) * (1.0 - np.exp(-rho * zmax / L))
    out[zmax == 0] = 1.0
    return out
