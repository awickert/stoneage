"""
Physical constants and calibrated production rates for the v3 calculator.
Port of make_consts_v3.m by Greg Balco (BGC).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Constants:
    version: str = "3.0.2"

    # ---- Nuclide / mineral identifiers ----
    nuclides: List[str] = field(default_factory=lambda: [
        "N3quartz", "N3olivine", "N3pyroxene",
        "N10quartz", "N14quartz", "N21quartz",
        "N26quartz", "N10pyroxene",
    ])
    proper_name: List[str] = field(default_factory=lambda: [
        "He-3 (qtz)", "He-3 (ol)", "He-3 (px)",
        "Be-10 (qtz)", "C-14 (qtz)", "Ne-21 (qtz)",
        "Al-26 (qtz)", "Be-10 (px)",
    ])

    # ---- Scaling schemes ----
    sschemes: List[str] = field(default_factory=lambda: ["St", "Lm", "LSDn"])

    # ---- Be-10 standards ----
    be_stds_names: List[str] = field(default_factory=lambda: [
        "07KNSTD", "KNSTD", "NIST_Certified", "LLNL31000",
        "LLNL10000", "LLNL3000", "LLNL1000", "LLNL300",
        "NIST_30000", "NIST_30200", "NIST_30300", "NIST_30600",
        "NIST_27900", "S555", "S2007", "BEST433",
        "BEST433N", "S555N", "S2007N", "STD11", "0",
        "NIST_30500", "SMDBe12",
    ])
    be_stds_cfs: np.ndarray = field(default_factory=lambda: np.array([
        1.0000, 0.9042, 1.0425, 0.8761, 0.9042, 0.8644, 0.9313,
        0.8562, 0.9313, 0.9251, 0.9221, 0.9130, 1.0000, 0.9124,
        0.9124, 0.9124, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000,
        0.9124, 1.0000,
    ]))

    # ---- Al-26 standards ----
    al_stds_names: List[str] = field(default_factory=lambda: [
        "KNSTD", "ZAL94", "SMAL11", "0", "ZAL94N", "ASTER", "Z92-0222", "SMDAl11",
    ])
    al_stds_cfs: np.ndarray = field(default_factory=lambda: np.array([
        1.0000, 0.9134, 1.021, 0.0, 1.0, 1.021, 1.0, 1.018,
    ]))

    # ---- He-3 standards ----
    he_stds_names: List[str] = field(default_factory=lambda: ["CRONUS-P", "NONE"])
    he_stds_ref: np.ndarray = field(default_factory=lambda: np.array([5.02e9, 0.0]))

    # ---- Ne-21 standards ----
    ne_stds_names: List[str] = field(default_factory=lambda: ["CRONUS-A", "CREU-1", "NONE"])
    ne_stds_ref: np.ndarray = field(default_factory=lambda: np.array([320e6, 348e6, 0.0]))

    # ---- C-14 standards ----
    c_stds_names: List[str] = field(default_factory=lambda: ["CRONUS-A", "NONE"])
    c_stds_ref: np.ndarray = field(default_factory=lambda: np.array([690000.0, 0.0]))

    # ---- Allowed minerals per nuclide ----
    ok_mins_10: List[str] = field(default_factory=lambda: ["quartz", "pyroxene"])
    ok_mins_26: List[str] = field(default_factory=lambda: ["quartz"])
    ok_mins_3:  List[str] = field(default_factory=lambda: ["quartz", "pyroxene", "olivine"])
    ok_mins_21: List[str] = field(default_factory=lambda: ["quartz"])
    ok_mins_14: List[str] = field(default_factory=lambda: ["quartz"])

    # ---- Cl-36 AMS standards ----
    # Correction factors normalise measured ³⁶Cl/Cl ratios to a common scale.
    # "KNSTD" (ratio = 1.0) means the user has already corrected to absolute
    # atoms/g; "0" means the measurement was flagged as invalid.
    cl36_stds_names: List[str] = field(default_factory=lambda: [
        "KNSTD", "0",
    ])
    cl36_stds_cfs: np.ndarray = field(default_factory=lambda: np.array([
        1.0, 0.0,
    ]))

    default_yr: int = 2010

    # ---- Decay constants ----
    # Be-10: Chmeleff/Korschinek half-life 1.387 Ma
    l10:    float = field(default_factory=lambda: np.log(2) / 1.387e6)
    dell10: float = field(default_factory=lambda: (np.log(2) / 1.387e6**2) * 0.012e6)

    # Cl-36: Holden & Hoffman (2000) half-life 301 ka
    l36:    float = field(default_factory=lambda: np.log(2) / 301e3)
    dell36: float = field(default_factory=lambda: np.log(2) / 301e3 * 0.003)

    # Al-26: Nishiizumi (2004) value
    l26:    float = 9.83e-7
    dell26: float = 2.5e-8

    # C-14
    l14:    float = field(default_factory=lambda: np.log(2) / 5730)
    dell14: float = field(default_factory=lambda: np.log(2) / 5730 * 0.005)

    # Decay constant vector (8 elements, nuclide order)
    # Order: N3q, N3ol, N3px, N10q, N14q, N21q, N26q, N10px
    @property
    def l(self) -> np.ndarray:
        return np.array([0, 0, 0, self.l10, self.l14, 0, self.l26, self.l10])

    @property
    def dell(self) -> np.ndarray:
        return np.array([0, 0, 0, self.dell10, self.dell14, 0, self.dell26, self.l10])

    # ---- Simplified muon scheme (model 1A, alpha=1; revised 20200826) ----
    version_muons: str = "1A, alpha = 1"
    # 8-element vectors (same nuclide order)
    Pmu0: np.ndarray = field(default_factory=lambda: np.array([
        0.0, 0.0, 0.0, 0.07345, 3.06690, 0.20281, 0.67644, 0.05983,
    ]))
    L_mu_atm: np.ndarray = field(default_factory=lambda: np.array([
        1.0, 1.0, 1.0, 299.2, 267.8, 482.2, 288.0, 311.0,
    ]))
    # Lmu effective near-surface attenuation length coefficients
    Lmu_a: float = 0.01036
    Lmu_b: float = -9.697e-6

    # ---- Reference production rates: St scaling ----
    # Order: N3q, N3ol, N3px, N10q, N14q, N21q, N26q, N10px
    refP_St: np.ndarray = field(default_factory=lambda: np.array([
        115.7,  119.6,  119.6,  4.086, 12.1,  16.896, 28.535, 3.72,
    ]))
    delrefP_St: np.ndarray = field(default_factory=lambda: np.array([
        10.0,
        119.6 * 0.11,  119.6 * 0.11,
        4.086 * 0.079,
        12.1  * 0.05,
        1.033,
        28.535 * 0.104,
        3.72  * 0.06,
    ]))

    # ---- Reference production rates: Lm scaling ----
    refP_Lm: np.ndarray = field(default_factory=lambda: np.array([
        115.7,  119.6,  119.6,  4.208, 12.1,  16.243, 29.416, 3.72,
    ]))
    delrefP_Lm: np.ndarray = field(default_factory=lambda: np.array([
        10.0,
        119.6 * 0.11,  119.6 * 0.11,
        4.208 * 0.075,
        12.1  * 0.05,
        0.994,
        29.416 * 0.094,
        3.72  * 0.06,
    ]))

    # ---- Reference production rates: LSDn scaling (nondimensional factors) ----
    refP_LSDn: np.ndarray = field(default_factory=lambda: np.array([
        1.0,   1.323,  1.323,  0.849, 0.725, 1.272, 0.819, 0.680,
    ]))
    delrefP_LSDn: np.ndarray = field(default_factory=lambda: np.array([
        0.1,
        1.323 * 0.11,  1.323 * 0.11,
        0.849 * 0.059,
        0.725 * 0.05,
        0.079,
        0.819 * 0.086,
        0.680 * 0.06,
    ]))


def make_consts() -> Constants:
    """Return the v3 constants structure."""
    return Constants()
