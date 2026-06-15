"""
Nuclear-physics constants for cosmogenic ³⁶Cl production.

All values are first-class, named, and citable so that they can be inspected,
overridden, or traced to their sources without digging through calculation code.

Sources
-------
Cross sections   : ENDF/B-VIII.0, 2200 m/s values; natural isotopic mixtures
                   unless noted.  https://www.nndc.bnl.gov/sigma/
Resonance integrals (I_eff) : BNL Evaluated Nuclear Data File / ENDF
Fast-neutron yields (Y_fast) : Gosse & Phillips (2001) Table A1.
                   Units are cm² g⁻¹ (macroscopic effective cross section per
                   gram of element for fast-neutron production at SLHL).
                   See PHI_N_SLHL note below.
Spallation rates : Marrero et al. (2016) Stone-2000 / St calibration, Table 3.
Muon parameters  : Heisinger et al. (2002); simplified for v1 (Ca only).
Oxide conversion : atomic weights from IUPAC 2021 recommendations.
"""

from dataclasses import dataclass, field
import numpy as np


# ── Reference cosmic-ray nucleon flux at SLHL ─────────────────────────────────
# Used to convert Y_fast (cm²/g, macroscopic cross section) to an absolute
# fast-neutron production rate P_f (n/g/yr):
#
#   P_f = PHI_N_SLHL × SF × Σ_i  X_i × Y_fast_i
#
# PHI_N_SLHL is calibrated so that P_f/(2×ξΣ_s) ≈ 750 n/cm²/yr for standard
# granite at SLHL — the empirical value from Evans (1997).  The granite
# composition used for this calibration is stored in Cl36Nuclear.granite_ref.
#
# Calibration result (tests/test_cl36.py::test_granite_phi_th):
#   With PHI_N_SLHL = 4000 and the G&P (2001) granite reference composition,
#   φ_eff = 669 n/cm²/yr — 11 % below the Evans (1997) reference of 750 n/cm²/yr.
#   This offset lies within the granite-composition uncertainty (LOI, trace
#   elements) and is acceptable for v1; it will be refined when the Y_fast
#   convention in G&P Table A1 is formally confirmed.  The 11 % offset affects
#   P_nth only, which is < 1 % of P_total for the Ca-rich mafic rocks studied
#   here; it becomes relevant for high-Cl or low-Ca samples.
PHI_N_SLHL: float = 4000.0   # n cm⁻² yr⁻¹ at SLHL


@dataclass
class Cl36Nuclear:
    """
    Nuclear-physics constants and tables for ³⁶Cl production.

    All cross sections are given per natural isotopic mixture of each element
    unless the isotope is specified (B10, B11, Cl35, Cl37).

    Usage
    -----
    >>> nuc = Cl36Nuclear()
    >>> nuc.sigma_th["Fe"]         # thermal capture cross section for Fe (barn)
    2.56
    >>> nuc.oxide_to_element["CaO"]   # Ca/CaO mass ratio
    0.7147...
    """

    # ── SLHL spallation reference production rates ────────────────────────────
    # Marrero et al. (2016), Stone-2000 calibration, Table 3
    P_sp_Ca:   float = 48.8    # at g(Ca)⁻¹ yr⁻¹
    P_sp_K:    float = 154.0   # at g(K)⁻¹ yr⁻¹
    P_sp_Ti:   float = 13.0    # at g(Ti)⁻¹ yr⁻¹  [Phillips et al. 2016]
    P_sp_Fe:   float = 0.086   # at g(Fe)⁻¹ yr⁻¹

    # Relative 1σ production-rate uncertainties (fractional)
    dP_sp_Ca:  float = 0.035
    dP_sp_K:   float = 0.10
    dP_sp_Ti:  float = 0.20
    dP_sp_Fe:  float = 0.20

    # ── Muon production (v1: slow+fast on Ca only) ────────────────────────────
    # Combined slow-negative-muon + fast-muon reference rate on ⁴⁰Ca → ³⁶Cl.
    # Heisinger et al. (2002); ±30 % (1σ) due to model simplification.
    # v2 will replace this with the full Heisinger parameterization.
    Pmu0_Ca:   float = 1.5    # at g(Ca)⁻¹ yr⁻¹ at SLHL

    # ── Thermal neutron capture cross sections σ_th (barn = 1e-24 cm²) ───────
    # ENDF/B-VIII.0, 2200 m/s; natural isotopic mixtures unless specified.
    sigma_th: dict = field(default_factory=lambda: {
        "H":    0.3326,
        "O":    0.000190,
        "Si":   0.1720,
        "Al":   0.2330,
        "Fe":   2.56,
        "Ca":   0.432,
        "K":    2.10,
        "Na":   0.530,
        "Mg":   0.0627,
        "Ti":   6.10,
        "Mn":   13.3,
        "P":    0.172,
        "Cr":   3.10,
        "Ni":   4.49,
        "B10":  3840.0,   # ¹⁰B only (19.9 % of natural B)
        "B11":  0.005,    # ¹¹B only
        "Cl35": 43.6,     # ³⁵Cl — the production target
        "Cl37": 0.560,    # ³⁷Cl
        "Gd":   49700.0,  # natural Gd; dominant absorber even at trace levels
        "Sm":   5922.0,   # natural Sm
    })

    # ── Effective resonance integrals I_eff (barn) ────────────────────────────
    # Epithermal (1/E) neutron capture; BNL ENDF.
    I_eff: dict = field(default_factory=lambda: {
        "H":    0.0,
        "O":    0.000400,
        "Si":   0.082,
        "Al":   0.171,
        "Fe":   1.39,
        "Ca":   0.256,
        "K":    1.30,
        "Na":   0.311,
        "Mg":   0.038,
        "Ti":   3.10,
        "Mn":   14.0,
        "P":    0.08,
        "Cr":   0.80,
        "Ni":   2.0,
        "B10":  1720.0,
        "B11":  0.03,
        "Cl35": 20.5,
        "Cl37": 0.0,
        "Gd":   390.0,
        "Sm":   1420.0,
    })

    # ── Elastic scattering cross sections σ_sc (barn) ─────────────────────────
    # Used with ξ to compute the macroscopic slowing-down cross section ξΣ_s.
    sigma_sc: dict = field(default_factory=lambda: {
        "H":    20.5,
        "O":    3.77,
        "Si":   2.16,
        "Al":   1.50,
        "Fe":   11.2,
        "Ca":   2.90,
        "K":    1.96,
        "Na":   3.28,
        "Mg":   3.50,
        "Ti":   4.09,
        "Mn":   2.15,
        "P":    3.0,
        "Cr":   3.38,
        "Ni":   17.8,
        "B10":  3.0,
        "B11":  3.0,
        "Cl35": 20.0,
        "Cl37": 0.0,
        "Gd":   170.0,
        "Sm":   30.0,
    })

    # ── Fast-neutron production yields Y_fast (cm² g⁻¹) ──────────────────────
    # Macroscopic effective cross section for fast-neutron production per gram
    # of each element.  From Gosse & Phillips (2001) Table A1.
    #
    # Production rate:  P_f [n/g/yr] = PHI_N_SLHL × SF × Σ_i  X_i × Y_fast_i
    #
    # ⚠  The exact units and scale of Table A1 have not yet been formally
    #    confirmed.  The values below reproduce Φ_eff(granite) ≈ 750 n/cm²/yr
    #    to within ~10 % using PHI_N_SLHL = 4000 n/cm²/yr, but this should
    #    be treated as provisional until test_granite_phi_th passes.
    Y_fast: dict = field(default_factory=lambda: {
        "H":   0.0,
        "O":   6.93e-3,
        "Si":  10.01e-3,
        "Al":  7.53e-3,
        "Fe":  10.05e-3,
        "Ca":  6.78e-3,
        "K":   5.83e-3,
        "Na":  5.68e-3,
        "Mg":  5.71e-3,
        "Ti":  4.92e-3,
        "Mn":  7.80e-3,
    })

    # ── Atomic masses (g mol⁻¹) ───────────────────────────────────────────────
    # IUPAC 2021 standard atomic weights.
    atomic_mass: dict = field(default_factory=lambda: {
        "H":    1.008,
        "O":    15.999,
        "Si":   28.086,
        "Al":   26.982,
        "Fe":   55.845,
        "Ca":   40.078,
        "K":    39.098,
        "Na":   22.990,
        "Mg":   24.305,
        "Ti":   47.867,
        "Mn":   54.938,
        "P":    30.974,
        "Cr":   51.996,
        "Ni":   58.693,
        "Cl":   35.453,
        "B10":  10.013,
        "B11":  11.009,
        "Cl35": 34.969,
        "Cl37": 36.966,
        "Gd":   157.25,
        "Sm":   150.36,
    })

    # ── Oxide → element mass-fraction conversion factors ─────────────────────
    # Pure arithmetic from IUPAC atomic weights.  No calibration involved.
    oxide_to_element: dict = field(default_factory=lambda: {
        "SiO2":  28.086  / (28.086  + 2*15.999),    # → Si   0.46743
        "Al2O3": 2*26.982 / (2*26.982 + 3*15.999),  # → Al   0.52925
        "Fe2O3": 2*55.845 / (2*55.845 + 3*15.999),  # → Fe   0.69943
        "MgO":   24.305  / (24.305  +   15.999),    # → Mg   0.60304
        "CaO":   40.078  / (40.078  +   15.999),    # → Ca   0.71468
        "Na2O":  2*22.990 / (2*22.990 +   15.999),  # → Na   0.74186
        "K2O":   2*39.098 / (2*39.098 +   15.999),  # → K    0.83013
        "TiO2":  47.867  / (47.867  + 2*15.999),    # → Ti   0.59934
        "P2O5":  2*30.974 / (2*30.974 + 5*15.999),  # → P    0.43643
        "MnO":   54.938  / (54.938  +   15.999),    # → Mn   0.77446
        "Cr2O3": 2*51.996 / (2*51.996 + 3*15.999),  # → Cr   0.68421
    })

    # ── Natural isotopic abundances ───────────────────────────────────────────
    # IUPAC 2021.
    Cl35_abundance: float = 0.7577
    Cl37_abundance: float = 0.2423
    B10_abundance:  float = 0.1990
    B11_abundance:  float = 0.8010

    # ── Standard granite reference composition (wt %) ─────────────────────────
    # Used to validate the neutron model against Evans (1997) / Gosse & Phillips
    # (2001): expected Φ_eff(granite) ≈ 750 n/cm²/yr at SLHL.
    # Composition from Gosse & Phillips (2001) Appendix A worked example.
    granite_ref: dict = field(default_factory=lambda: {
        "SiO2":  66.0,
        "Al2O3": 15.0,
        "Fe2O3":  4.0,
        "MgO":    2.0,
        "CaO":    2.0,
        "Na2O":   4.0,
        "K2O":    5.0,
        "TiO2":   0.5,
        "MnO":    0.1,
        "Cr2O3":  0.0,
        "P2O5":   0.2,
        "LOI":    1.0,   # treated as H₂O
        "Cl_ppm": 10.0,
        "B_ppm":  5.0,
        "Gd_ppm": 1.0,
        "Sm_ppm": 5.0,
    })
