# Design: ³⁶Cl support for stoneage

## Goals

1. Add ³⁶Cl exposure-age and production-rate calculation at the same level of
   completeness as the existing ¹⁰Be/²⁶Al/²¹Ne/³He/¹⁴C implementation.
2. **Do not change the interface for any existing nuclide.**  A user running
   ¹⁰Be samples must not need to touch any new code or files.
3. Keep nuclear-physics tables first-class, inspectable objects — not magic
   numbers buried in functions.
4. Follow the existing code style: dataclasses for constants, module-level
   functions for computation, no external dependencies beyond NumPy/SciPy.

---

## What makes ³⁶Cl different

| Property | ¹⁰Be / ²⁶Al / etc. | ³⁶Cl |
|---|---|---|
| Production pathways | spallation on one target mineral | spallation on Ca, K, Ti, Fe; muon capture on Ca, K; thermal + epithermal neutrons on ³⁵Cl |
| Input per sample | N, δN, standard | same + full bulk geochemistry (all major oxides, Cl, B, Gd, Sm) |
| Reference production rate | one scalar P_ref | one scalar per target element (P_Ca, P_K, P_Ti, P_Fe) |
| Production rate scaling | same Stone/Lm/LSDn SFs | same SFs for spallation; separate (shallower) attenuation for muons and neutrons |
| Nuclide-index in constants | single slot per nuclide | needs per-element slots |

The bulk-chemistry requirement is the main structural difference.  Everything
else — atmospheric scaling, cutoff rigidity, thickness/shielding corrections,
age inversion — is shared infrastructure that already exists.

---

## New files

```
stoneage/
    cl36_nuclear.py      # Nuclear-physics constants and cross-section tables
    cl36_production.py   # Per-sample production-rate calculation
docs/
    design_cl36.md       # This file
tests/
    test_cl36.py         # Validation against CRONUS online calculator
```

### Changes to existing files

| File | Change |
|---|---|
| `constants.py` | Add `Cl36Constants` dataclass; add λ₃₆ and δλ₃₆ to the scalar block |
| `input_parser.py` | Accept optional `chem_csv` path; join geochemistry to sample table on `sample_name`; remove `raise ValueError("Cl-36 not yet supported")` |
| `calculator.py` | Call `cl36_production()` when a ³⁶Cl nuclide measurement is present |
| `summarize.py` | Add ³⁶Cl to the report formatter |

---

## `cl36_nuclear.py`

Holds all nuclear-physics tables as a dataclass so they can be inspected,
overridden, or cited independently of any calculation.

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Cl36Nuclear:
    """
    Nuclear-physics constants for cosmogenic ³⁶Cl production.

    All cross sections from ENDF/B-VIII.0 unless noted otherwise.
    Fast-neutron yields Y_fast from Gosse & Phillips (2001) Table A1.
    Oxide conversion factors are exact from atomic weights (IUPAC 2021).
    """

    # ── Reference spallation production rates at SLHL ────────────────────────
    # Marrero et al. (2016), Stone-2000 calibration, Table 3
    P_sp_Ca:  float = 48.8    # at g(Ca)⁻¹ yr⁻¹   (1σ ≈ 3.5 %)
    P_sp_K:   float = 154.0   # at g(K)⁻¹ yr⁻¹    (1σ ≈ 10 %)
    P_sp_Ti:  float = 13.0    # at g(Ti)⁻¹ yr⁻¹   (Phillips et al. 2016)
    P_sp_Fe:  float = 0.086   # at g(Fe)⁻¹ yr⁻¹   (Marrero et al. 2016)

    dP_sp_Ca: float = 0.035   # relative 1σ (fractional)
    dP_sp_K:  float = 0.10
    dP_sp_Ti: float = 0.20
    dP_sp_Fe: float = 0.20

    # ── Muon production on Ca (Heisinger et al. 2002) ────────────────────────
    # Combined slow-negative-muon + fast-muon on ⁴⁰Ca → ³⁶Cl at SLHL
    Pmu0_Ca: float = 1.5      # at g(Ca)⁻¹ yr⁻¹  (1σ ~30 %; see note below)
    # NOTE: Pmu0_Ca is intentionally approximate.  A full Heisinger et al.
    # model (f_neg, f_D, φ_μ⁻, fast-muon flux) should replace this in v2.
    # Muon attenuation length for slow muons (g cm⁻²)
    L_mu_slow: float = 1500.0

    # ── Thermal neutron capture cross sections σ_th (barn = 1e-24 cm²) ──────
    # ENDF/B-VIII.0 2200 m/s values; natural isotopic mixtures where relevant
    sigma_th: dict = field(default_factory=lambda: {
        'H':   0.3326,
        'O':   0.00019,
        'Si':  0.172,
        'Al':  0.233,
        'Fe':  2.56,
        'Ca':  0.432,
        'K':   2.10,
        'Na':  0.530,
        'Mg':  0.0627,
        'Ti':  6.10,
        'Mn':  13.3,
        'P':   0.172,
        'Cr':  3.10,
        'Ni':  4.49,
        'B10': 3840.,   # ¹⁰B; natural B is 19.9 % ¹⁰B
        'B11': 0.005,
        'Cl35': 43.6,   # target nuclide
        'Cl37': 0.56,
        'Gd':  49700.,  # natural Gd; even trace amounts matter
        'Sm':  5922.,   # natural Sm
    })

    # ── Effective resonance integrals I_eff (barn) ───────────────────────────
    # Epithermal capture; BNL Evaluated Nuclear Data File
    I_eff: dict = field(default_factory=lambda: {
        'H':    0.0,
        'O':    0.00040,
        'Si':   0.082,
        'Al':   0.171,
        'Fe':   1.39,
        'Ca':   0.256,
        'K':    1.30,
        'Na':   0.311,
        'Mg':   0.038,
        'Ti':   3.10,
        'Mn':   14.0,
        'P':    0.08,
        'Cr':   0.80,
        'Ni':   2.0,
        'Cl35': 20.5,
        'Cl37': 0.0,
        'B10':  1720.,
        'B11':  0.03,
        'Gd':   390.,
        'Sm':   1420.,
    })

    # ── Elastic scattering cross sections σ_sc (barn) ────────────────────────
    sigma_sc: dict = field(default_factory=lambda: {
        'H': 20.5, 'O': 3.77, 'Si': 2.16, 'Al': 1.50,
        'Fe': 11.2, 'Ca': 2.90, 'K': 1.96, 'Na': 3.28,
        'Mg': 3.50, 'Ti': 4.09, 'Mn': 2.15, 'P': 3.0,
        'Cr': 3.38, 'Ni': 17.8, 'Cl35': 20.0, 'Cl37': 0.0,
        'B10': 3.0, 'B11': 3.0, 'Gd': 170., 'Sm': 30.,
    })

    # ── Atomic masses (g mol⁻¹) for the relevant elements ───────────────────
    atomic_mass: dict = field(default_factory=lambda: {
        'H': 1.008, 'O': 15.999, 'Si': 28.086, 'Al': 26.982,
        'Fe': 55.845, 'Ca': 40.078, 'K': 39.098, 'Na': 22.990,
        'Mg': 24.305, 'Ti': 47.867, 'Mn': 54.938, 'P': 30.974,
        'Cr': 51.996, 'Ni': 58.693, 'Cl': 35.453,
        'B10': 10.013, 'B11': 11.009,
        'Cl35': 34.969, 'Cl37': 36.966,
        'Gd': 157.25, 'Sm': 150.36,
    })

    # ── Fast-neutron production yields Y_fast (n atom⁻¹ yr⁻¹ at SLHL) ──────
    # Gosse & Phillips (2001) Table A1.
    # OPEN QUESTION: confirm exact units and scale vs. published source
    # before committing these values to any calculation.
    Y_fast: dict = field(default_factory=lambda: {
        'H': 0.0,  'O': 6.93e-3, 'Si': 10.01e-3, 'Al': 7.53e-3,
        'Fe': 10.05e-3, 'Ca': 6.78e-3, 'K': 5.83e-3, 'Na': 5.68e-3,
        'Mg': 5.71e-3, 'Ti': 4.92e-3, 'Mn': 7.80e-3,
    })

    # ── Oxide → element mass-fraction conversion factors ─────────────────────
    # Pure arithmetic from IUPAC atomic weights; no calibration involved.
    oxide_to_element: dict = field(default_factory=lambda: {
        'SiO2':  28.086  / (28.086  + 2*15.999),   # → Si  0.4674
        'Al2O3': 2*26.982 / (2*26.982 + 3*15.999), # → Al  0.5293
        'Fe2O3': 2*55.845 / (2*55.845 + 3*15.999), # → Fe  0.6994
        'MgO':   24.305  / (24.305  +   15.999),   # → Mg  0.6031
        'CaO':   40.078  / (40.078  +   15.999),   # → Ca  0.7147
        'Na2O':  2*22.990 / (2*22.990 +   15.999), # → Na  0.7419
        'K2O':   2*39.098 / (2*39.098 +   15.999), # → K   0.8302
        'TiO2':  47.867  / (47.867  + 2*15.999),   # → Ti  0.5993
        'P2O5':  2*30.974 / (2*30.974 + 5*15.999), # → P   0.4364
        'MnO':   54.938  / (54.938  +   15.999),   # → Mn  0.7745
        'Cr2O3': 2*51.996 / (2*51.996 + 3*15.999), # → Cr  0.6842
    })

    # ── Natural isotopic abundances ──────────────────────────────────────────
    Cl35_abundance: float = 0.7577
    Cl37_abundance: float = 0.2423
    B10_abundance:  float = 0.1990
    B11_abundance:  float = 0.8010
```

---

## `cl36_production.py`

One public function, returning a named breakdown so callers can inspect each
term rather than receiving only a total.

```python
def cl36_production(
    chem: dict,
    SF_sp: float,
    SF_mu: float,
    SF_thick: float,
    nuclear: Cl36Nuclear | None = None,
) -> Cl36Production:
    """
    Total ³⁶Cl surface production rate and per-pathway breakdown.

    Parameters
    ----------
    chem     : dict mapping oxide name (str) → value (float, wt %)
               Required keys: SiO2, Al2O3, Fe2O3, MgO, CaO, Na2O, K2O,
               TiO2, P2O5, MnO, Cr2O3, LOI, Cl_ppm
               Optional keys: B_ppm, Gd_ppm, Sm_ppm (default 0 if absent)
    SF_sp    : Stone-2000 (or Lm/LSDn) spallation scaling factor
    SF_mu    : muon scaling factor (from stone2000 with Fsp=0)
    SF_thick : thickness correction factor
    nuclear  : Cl36Nuclear instance; uses defaults if None

    Returns
    -------
    Cl36Production dataclass with fields:
        P_sp_Ca, P_sp_K, P_sp_Ti, P_sp_Fe  (at/g/yr each)
        P_sp                                (sum of above)
        P_mu_Ca, P_mu_K                     (at/g/yr each)
        P_mu                                (sum of above)
        P_nth                               (thermal+epithermal on ³⁵Cl)
        P_total                             (grand total)
        phi_th, phi_eth                     (neutron fluxes, n/cm²/yr)
    """
```

```python
@dataclass
class Cl36Production:
    P_sp_Ca: float
    P_sp_K:  float
    P_sp_Ti: float
    P_sp_Fe: float
    P_sp:    float
    P_mu_Ca: float
    P_mu_K:  float
    P_mu:    float
    P_nth:   float
    P_total: float
    phi_th:  float   # thermal neutron flux at surface (n/cm²/yr)
    phi_eth: float   # epithermal neutron flux at surface (n/cm²/yr)
```

Keeping the intermediate fluxes in the return value means a user can ask
"what was the thermal neutron flux in my sample?" without re-running anything.

---

## Input format

### Main sample CSV — unchanged

The existing v3 CSV format gains two new optional nuclide columns.
All other nuclides continue to work with no changes.

```
sample_name, lat, lon, elv, thick, rho, shield, E, yr, aa, \
    N36, dN36, std_36
```

`std_36` follows the same pattern as `std_10` (standard name string).

### Geochemistry CSV — new, optional

A separate file, one row per sample:

```
sample_name, SiO2, Al2O3, Fe2O3, MgO, CaO, Na2O, K2O, TiO2, \
    P2O5, MnO, Cr2O3, LOI, Cl_ppm, B_ppm, Gd_ppm, Sm_ppm
```

`B_ppm`, `Gd_ppm`, `Sm_ppm` are optional (default 0); all oxide values in
wt %.  The `LOI` column is treated as H₂O for the neutron model.

Joined to the sample table at parse time on `sample_name`.  Samples present
in the main CSV but absent from the geochemistry CSV are valid for non-³⁶Cl
nuclides and raise an error only if they carry a ³⁶Cl measurement.

### `parse_v3_input()` signature change

```python
def parse_v3_input(
    text: str,
    chem_text: str | None = None,   # <-- new, optional
) -> dict:
```

---

## Changes to `constants.py`

Add to the `Constants` dataclass:

```python
# ³⁶Cl decay constant — Holden & Hoffman (2000)
l36:    float = field(default_factory=lambda: np.log(2) / 301e3)
dell36: float = field(default_factory=lambda: np.log(2) / 301e3 * 0.003)
```

The `l` and `dell` property vectors gain a ³⁶Cl slot only if the nuclide
list is extended.  Option: keep ³⁶Cl decay entirely in `cl36_production.py`
to avoid touching the existing vector layout.  **Decision needed.**

---

## Thermal neutron model — implementation plan

The model follows Gosse & Phillips (2001) Appendix A, restricted to z = 0
(surface sample) for v1; depth-dependent profiles added in v2.

**Key unit hazard (flagged during prototype work):** the fast-neutron yield
table Y_fast in G&P (2001) Table A1 has units that are easy to misread.
The first implementation task is to reproduce G&P's worked example for
standard granite and confirm Φ_th(0) ≈ 750 n/cm²/yr before integrating with
the rest of the calculator.  This worked example becomes a unit test.

Order of implementation within `cl36_production.py`:

1. Oxide → element mass fractions (trivial; test against known values)
2. Isotope splitting (Cl→³⁵Cl/³⁷Cl, B→¹⁰B/¹¹B, LOI→H)
3. Macroscopic cross sections: Σ_a, ξΣ_s, Σ_I (from nuclear tables)
4. Fast-neutron production rate P_f
5. Epithermal flux Φ_eth(0) and thermal flux Φ_th(0)
6. ³⁶Cl production from neutron capture
7. Spallation on Ca, K, Ti, Fe
8. Muon production on Ca (simplified v1; full Heisinger in v2)

---

## Validation plan

Before any PR is merged, the Python output must agree with the CRONUS online
calculator (`stoneage.ice-d.org/math/Cl36/`) to within 2 % on production
rates and ages for a set of test samples spanning:

- Low-Ca mafic rock with significant K contribution (Lester River L3/L4)
- High-Ca mafic rock, Ca-dominated (Gooseberry GB3)
- High-Cl sample where thermal neutron term is non-negligible

These samples have measured ³⁶Cl concentrations, full LF200 bulk chemistry,
and approximate known ages, making them useful end-to-end tests as well as
unit-test fixtures.

---

## Open questions / decisions needed

| # | Question | Options | Notes |
|---|---|---|---|
| 1 | Y_fast units | Confirm against G&P (2001) Table A1 before any use | Prototype hit a ~10⁴ scale error here |
| 2 | Muon model depth | v1: surface only (Pmu0_Ca simplified); v2: full Heisinger | Affects samples above ~500 m significantly |
| 3 | λ₃₆ placement | Add to `Constants.l` vector, or keep in `cl36_production.py` only | Avoid breaking existing vector indexing |
| 4 | Lm / LSDn scaling for ³⁶Cl | Use same SF grid as other nuclides? | Yes — spallation SFs are nuclide-independent; only the reference rates differ |
| 5 | Gd / Sm defaults | Default 0 (ignore) or use crustal average? | Matters for high-Gd/Sm lithologies; low risk to default 0 and require explicit input |
| 6 | Calibration dataset | Which samples to include as bundled fixtures? | G&P (2001) granite example + Marrero et al. (2016) calibration sites |

---

## Out of scope for v1

- Depth profiles (z > 0 neutron flux model)
- Full Heisinger et al. (2002) muon parameterization (fast + slow, all
  elements)
- Lm and LSDn time-varying scaling for ³⁶Cl (spallation SFs already work;
  only the reference-rate calibration differs and that is a constants change)
- ³⁶Cl from ⁴⁰Ar (atmospheric ³⁶Cl ingrowth — niche use case)
- Erosion-corrected ages involving Cl-concentration-dependent production
  (the production rate itself changes with depth through the neutron term)
