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

    Integrates the exponential production-depth profile over the sample
    thickness, normalising to the surface production rate:

        tc = (L / (rho × zmax)) × (1 − exp(−rho × zmax / L))

    Port of thickness.m (Balco).

    Parameters
    ----------
    zmax : array-like
        Sample thickness in cm.  Pass 0 to get tc = 1.0 (no correction).
    L    : array-like
        Effective spallogenic attenuation length in g/cm².
        Use Lsp() = 160 g/cm² for the standard value.
    rho  : array-like
        Bulk rock density in g/cm³.

    All three arguments are broadcast to a common shape.

    Returns
    -------
    correction : ndarray, dimensionless, in (0, 1].
        Multiply the surface spallogenic production rate by this factor to
        obtain the production rate averaged over the sample thickness.
        A 2-cm sample at rho=2.65 gives tc ≈ 0.98.
    """
    zmax = np.atleast_1d(np.asarray(zmax, dtype=float))
    L    = np.atleast_1d(np.asarray(L,    dtype=float))
    rho  = np.atleast_1d(np.asarray(rho,  dtype=float))

    with np.errstate(divide="ignore", invalid="ignore"):
        out = (L / (rho * zmax)) * (1.0 - np.exp(-rho * zmax / L))
    out[zmax == 0] = 1.0
    return out
