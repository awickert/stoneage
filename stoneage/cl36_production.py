"""
³⁶Cl surface production rate calculator.

Computes total production and per-pathway breakdown for one sample at a time,
following the physics outlined in docs/design_cl36.md.

Physics
-------
Spallation (Ca, K, Ti, Fe)
    Marrero et al. (2016) Stone-2000 reference rates, scaled by the provided
    spallation scaling factor.

Muon production on Ca
    Heisinger et al. (2002) combined slow + fast muons; simplified for v1
    (Ca only, single Pmu0 parameter).

Thermal/epithermal neutron capture on ³⁵Cl
    Gosse & Phillips (2001) Appendix A approach at z=0.
    Fast-neutron flux: P_f = PHI_N_SLHL × SF × Σ_i X_i × Y_fast_i
    Effective flux:   Φ_eff = P_f / (2 × ξΣ_s)
    Production:       P_nth = Φ_eff × (σ_th + I_eff/2)_35Cl × N_35Cl × SF_thick

    ⚠ The PHI_N_SLHL calibration (4000 n/cm²/yr) reproduces Φ_eff ≈ 750 n/cm²/yr
    for standard granite to within ~10 %.  See tests/test_cl36.py::test_granite_phi_th
    for the formal validation check before relying on P_nth in published work.

All three contributions are returned in the Cl36Production dataclass so callers
can inspect, report, or re-weight them without re-running the calculation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .cl36_nuclear import Cl36Nuclear, PHI_N_SLHL

_NA = 6.02214076e23   # Avogadro's number


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclass
class Cl36Production:
    """
    Per-pathway ³⁶Cl surface production rate breakdown (at g⁻¹ yr⁻¹).

    All values are for the *surface* of the sample, *after* the thickness
    and topographic shielding corrections have been applied.

    Attributes
    ----------
    P_sp_Ca, P_sp_K, P_sp_Ti, P_sp_Fe
        Spallation contributions from each element.
    P_sp
        Total spallation  (sum of above).
    P_mu_Ca
        Muon production on Ca (combined slow + fast; v1 approximation).
    P_mu
        Total muon production (= P_mu_Ca in v1).
    P_nth
        Thermal + epithermal neutron capture on ³⁵Cl.
    P_total
        Grand total  (P_sp + P_mu + P_nth).
    phi_eff
        Effective low-energy neutron flux at sample surface (n cm⁻² yr⁻¹),
        from which P_nth is derived.  Useful for cross-checking and reporting.
    """
    P_sp_Ca: float
    P_sp_K:  float
    P_sp_Ti: float
    P_sp_Fe: float
    P_sp:    float
    P_mu_Ca: float
    P_mu:    float
    P_nth:   float
    P_total: float
    phi_eff: float


# ── Public conversion helper ──────────────────────────────────────────────────

def bulk_chem_gg(
    oxide_pct: dict,
    Cl_ppm:  float,
    B_ppm:   float = 0.0,
    Gd_ppm:  float = 0.0,
    Sm_ppm:  float = 0.0,
    nuclear: Cl36Nuclear | None = None,
) -> dict:
    """
    Convert oxide weight-percent values to element mass fractions (g/g).

    Also splits Cl into ³⁵Cl/³⁷Cl and B into ¹⁰B/¹¹B at natural abundances,
    and converts LOI to H (plus the O already accounted for in the oxides).

    Parameters
    ----------
    oxide_pct : dict
        Oxide name → weight percent.  Recognised keys:
        SiO2, Al2O3, Fe2O3, MgO, CaO, Na2O, K2O, TiO2, P2O5, MnO, Cr2O3,
        LOI (treated as H₂O).
    Cl_ppm    : total Cl concentration (µg Cl / g sample).
    B_ppm     : boron (ppm).  Defaults to 0.
    Gd_ppm    : gadolinium (ppm).  Defaults to 0.
    Sm_ppm    : samarium (ppm).  Defaults to 0.
    nuclear   : Cl36Nuclear; uses defaults if None.

    Returns
    -------
    dict  element symbol → mass fraction (g/g).
    Keys present: Si, Al, Fe, Mg, Ca, Na, K, Ti, Mn, Cr, P, O, H,
                  Cl35, Cl37, B10, B11, Gd, Sm.
    """
    if nuclear is None:
        nuclear = Cl36Nuclear()

    oxf = nuclear.oxide_to_element

    # ── Oxide → element ───────────────────────────────────────────────────────
    chem: dict[str, float] = {}
    oxide_map = {
        "Si": ("SiO2",  1), "Al": ("Al2O3", 2), "Fe": ("Fe2O3", 2),
        "Mg": ("MgO",   1), "Ca": ("CaO",   1), "Na": ("Na2O",  2),
        "K":  ("K2O",   2), "Ti": ("TiO2",  1), "Mn": ("MnO",   1),
        "Cr": ("Cr2O3", 2), "P":  ("P2O5",  2),
    }
    for elem, (oxide, _n_elem) in oxide_map.items():
        chem[elem] = oxide_pct.get(oxide, 0.0) / 100.0 * oxf[oxide]

    # ── H₂O from LOI → H and structural O ───────────────────────────────────
    loi_frac  = oxide_pct.get("LOI", 0.0) / 100.0
    h2o_frac  = loi_frac            # conservative: treat all LOI as H₂O
    chem["H"] = h2o_frac * (2 * 1.008 / 18.015)

    # ── Oxygen: mineral-bound + from LOI ─────────────────────────────────────
    o_mineral = sum(
        oxide_pct.get(ox, 0.0) / 100.0 * (1.0 - oxf[ox])
        for ox in oxf
    )
    o_water = h2o_frac * (15.999 / 18.015)
    chem["O"] = o_mineral + o_water

    # ── Trace elements (ppm → g/g) ────────────────────────────────────────────
    Cl_gg  = Cl_ppm  * 1e-6
    B_gg   = B_ppm   * 1e-6
    Gd_gg  = Gd_ppm  * 1e-6
    Sm_gg  = Sm_ppm  * 1e-6

    # ── Isotope splitting ─────────────────────────────────────────────────────
    chem["Cl35"] = Cl_gg * nuclear.Cl35_abundance
    chem["Cl37"] = Cl_gg * nuclear.Cl37_abundance
    chem["B10"]  = B_gg  * nuclear.B10_abundance
    chem["B11"]  = B_gg  * nuclear.B11_abundance
    chem["Gd"]   = Gd_gg
    chem["Sm"]   = Sm_gg

    return chem


# ── Internal helpers ──────────────────────────────────────────────────────────

def _xi(A: float) -> float:
    """
    Average logarithmic energy decrement per elastic collision for mass A.

    ξ = 1 + [(A-1)²/(2A)] × ln[(A-1)/(A+1)]
    For A=1 (H), ξ = 1 exactly.
    """
    if A <= 1.0:
        return 1.0
    return 1.0 + ((A - 1.0)**2 / (2.0 * A)) * np.log((A - 1.0) / (A + 1.0))


def _macro_xsections(
    chem_gg: dict,
    nuclear: Cl36Nuclear,
) -> tuple[float, float, float]:
    """
    Macroscopic cross sections (cm² g⁻¹) for the bulk rock composition.

    Returns
    -------
    Sigma_a   : macroscopic thermal absorption cross section  [cm²/g]
    xiSigma_s : macroscopic slowing-down cross section        [cm²/g]
    Sigma_I   : macroscopic epithermal resonance integral     [cm²/g]
    """
    Sigma_a   = 0.0
    xiSigma_s = 0.0
    Sigma_I   = 0.0

    for elem, X in chem_gg.items():
        if X == 0.0 or elem not in nuclear.atomic_mass:
            continue
        A   = nuclear.atomic_mass[elem]
        n_at = X * _NA / A   # atoms/g of rock

        if elem in nuclear.sigma_th:
            Sigma_a   += n_at * nuclear.sigma_th[elem]  * 1e-24
        if elem in nuclear.sigma_sc:
            xiSigma_s += n_at * nuclear.sigma_sc[elem]  * 1e-24 * _xi(A)
        if elem in nuclear.I_eff:
            Sigma_I   += n_at * nuclear.I_eff[elem]     * 1e-24

    return Sigma_a, xiSigma_s, Sigma_I


# ── Neutron flux (public — inspectable) ──────────────────────────────────────

def neutron_fluxes(
    chem_gg: dict,
    SF_sp:   float,
    nuclear: Cl36Nuclear | None = None,
) -> tuple[float, float]:
    """
    Effective low-energy neutron flux at z=0 (n cm⁻² yr⁻¹).

    Follows Gosse & Phillips (2001) Appendix A, restricted to the rock surface.

    Parameters
    ----------
    chem_gg : dict
        Element mass fractions from bulk_chem_gg().
    SF_sp   : Stone-2000 (or equivalent) spallation scaling factor.
    nuclear : Cl36Nuclear; uses defaults if None.

    Returns
    -------
    (phi_sp, phi_eth)
        phi_sp  : fast-spallation neutron flux feeding the epithermal cascade
        phi_eth : effective epithermal/thermal flux at z=0

    Notes
    -----
    The fast-neutron production rate is:
        P_f = PHI_N_SLHL × SF_sp × Σ_i  X_i × Y_fast_i   [n/g/yr]

    The effective low-energy flux at the surface is:
        phi_eff = P_f / (2 × ξΣ_s)                        [n/cm²/yr]

    This combines epithermal and thermal capture into one effective flux;
    production from ³⁵Cl capture is computed as:
        P_nth = phi_eff × (σ_th(³⁵Cl) + I_eff(³⁵Cl)/2) × N_35Cl

    ⚠ PHI_N_SLHL (= 4000 n/cm²/yr) reproduces φ_eff ≈ 750 n/cm²/yr for
    standard granite to within ~10 %.  Validate with test_granite_phi_th
    before relying on P_nth in published work.
    """
    if nuclear is None:
        nuclear = Cl36Nuclear()

    # Fast-neutron production rate (n/g/yr)
    P_f = 0.0
    for elem, X in chem_gg.items():
        if X == 0.0 or elem not in nuclear.Y_fast:
            continue
        P_f += X * nuclear.Y_fast[elem]
    P_f *= PHI_N_SLHL * SF_sp

    # Slowing-down cross section
    _, xiSigma_s, _ = _macro_xsections(chem_gg, nuclear)

    phi_eff = P_f / (2.0 * xiSigma_s) if xiSigma_s > 0.0 else 0.0
    return P_f, phi_eff


# ── Main production function ──────────────────────────────────────────────────

def cl36_production(
    chem:      dict,
    SF_sp:     float,
    SF_mu:     float,
    SF_thick:  float,
    nuclear:   Cl36Nuclear | None = None,
) -> Cl36Production:
    """
    Total ³⁶Cl surface production rate and per-pathway breakdown.

    Parameters
    ----------
    chem : dict
        Oxide weight-percent values *plus* trace-element concentrations:

        Required oxide keys (wt %):
            SiO2, Al2O3, Fe2O3, MgO, CaO, Na2O, K2O, TiO2, P2O5, MnO,
            Cr2O3, LOI
        Required scalar key:
            Cl_ppm  (µg Cl / g sample, i.e. ppm by mass)
        Optional scalar keys (default 0 if absent):
            B_ppm, Gd_ppm, Sm_ppm

    SF_sp    : Stone-2000 (or Lm/LSDn) spallation scaling factor.
    SF_mu    : Muon scaling factor (stone2000 with Fsp=0).
    SF_thick : Thickness correction factor (from corrections.thickness()).
    nuclear  : Cl36Nuclear; uses module defaults if None.

    Returns
    -------
    Cl36Production
        Per-pathway breakdown plus grand total.  All rates are for the
        sample surface, after thickness and shielding corrections.

    Examples
    --------
    >>> from stoneage.cl36_production import cl36_production
    >>> chem = dict(CaO=9.68, K2O=0.13, TiO2=0.98, SiO2=46.24,
    ...             Al2O3=15.85, Fe2O3=10.85, MgO=8.88, Na2O=1.76,
    ...             MnO=0.13, Cr2O3=0.047, P2O5=0.04, LOI=5.1,
    ...             Cl_ppm=2.82)
    >>> p = cl36_production(chem, SF_sp=1.176, SF_mu=1.067, SF_thick=0.982)
    >>> round(p.P_sp, 2)
    4.18
    """
    if nuclear is None:
        nuclear = Cl36Nuclear()

    Cl_ppm  = float(chem.get("Cl_ppm",  0.0))
    B_ppm   = float(chem.get("B_ppm",   0.0))
    Gd_ppm  = float(chem.get("Gd_ppm",  0.0))
    Sm_ppm  = float(chem.get("Sm_ppm",  0.0))

    chem_gg = bulk_chem_gg(chem, Cl_ppm, B_ppm, Gd_ppm, Sm_ppm, nuclear)

    # ── Spallation ────────────────────────────────────────────────────────────
    X_Ca = chem_gg.get("Ca", 0.0)
    X_K  = chem_gg.get("K",  0.0)
    X_Ti = chem_gg.get("Ti", 0.0)
    X_Fe = chem_gg.get("Fe", 0.0)

    P_sp_Ca = nuclear.P_sp_Ca * X_Ca * SF_sp * SF_thick
    P_sp_K  = nuclear.P_sp_K  * X_K  * SF_sp * SF_thick
    P_sp_Ti = nuclear.P_sp_Ti * X_Ti * SF_sp * SF_thick
    P_sp_Fe = nuclear.P_sp_Fe * X_Fe * SF_sp * SF_thick
    P_sp    = P_sp_Ca + P_sp_K + P_sp_Ti + P_sp_Fe

    # ── Muon production (Ca; v1 simplified) ───────────────────────────────────
    P_mu_Ca = nuclear.Pmu0_Ca * X_Ca * SF_mu * SF_thick
    P_mu    = P_mu_Ca

    # ── Thermal/epithermal neutron capture on ³⁵Cl ───────────────────────────
    _P_f, phi_eff = neutron_fluxes(chem_gg, SF_sp, nuclear)

    # ³⁵Cl atom density (atoms/g rock)
    N_35Cl = chem_gg.get("Cl35", 0.0) * _NA / nuclear.atomic_mass["Cl35"]

    # Combined thermal + epithermal capture cross section for ³⁵Cl
    sigma_eff_35Cl = (nuclear.sigma_th["Cl35"] + nuclear.I_eff["Cl35"] / 2.0) * 1e-24  # cm²

    P_nth = phi_eff * sigma_eff_35Cl * N_35Cl * SF_thick

    if P_nth > 0.05 * P_sp:
        warnings.warn(
            f"P_nth ({P_nth:.3f}) is > 5 % of P_sp ({P_sp:.3f}).  "
            "The thermal neutron model uses PHI_N_SLHL = 4000 n/cm²/yr; "
            "validate with test_granite_phi_th before publishing P_nth-sensitive ages.",
            stacklevel=2,
        )

    P_total = P_sp + P_mu + P_nth

    return Cl36Production(
        P_sp_Ca=P_sp_Ca,
        P_sp_K=P_sp_K,
        P_sp_Ti=P_sp_Ti,
        P_sp_Fe=P_sp_Fe,
        P_sp=P_sp,
        P_mu_Ca=P_mu_Ca,
        P_mu=P_mu,
        P_nth=P_nth,
        P_total=P_total,
        phi_eff=phi_eff,
    )
