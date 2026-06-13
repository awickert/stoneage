# stoneage

Python port of Greg Balco's v3 cosmogenic nuclide exposure age and production rate calibration calculator, originally implemented in MATLAB and served at [stoneage.ice-d.org](https://stoneage.ice-d.org/math/v3/) and [hess.ess.washington.edu](https://hess.ess.washington.edu/math/v3/).

[![Tests](https://github.com/awickert/stoneage/actions/workflows/tests.yml/badge.svg)](https://github.com/awickert/stoneage/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/awickert/stoneage/branch/main/graph/badge.svg)](https://codecov.io/gh/awickert/stoneage)

---

## What it does

Given a set of samples with measured cosmogenic nuclide concentrations and site geometry, `stoneage` computes either:

- **Exposure ages** — how long a rock surface has been exposed to cosmic rays, or
- **Production rate calibration** — what reference production rate is implied by samples of independently known age.

Both modes support three scaling schemes (St, Lm, LSDn) and five nuclide–mineral pairs.

---

## Installation

```bash
git clone https://github.com/awickert/stoneage.git
cd stoneage
pip install .
```

**Dependencies:** `numpy`, `scipy`.
The bundled `.mat` data files (ERA-40 reanalysis, LSDn scaling grids, paleomagnetic reconstruction) are included in the package and require no separate download.

---

## Input format

Input follows the v3 text format documented at  
https://hess.ess.washington.edu/math/docs/v3/v3_input_explained.html

Each sample occupies two or three semicolon-terminated lines:

```
# Sample line — 10 whitespace-separated fields
name  lat  lon  elevation  flag  thickness  density  shielding  erosion  year;

# Nuclide line — fields vary by nuclide (see table below)
name  nuclide  mineral  concentration  uncertainty  standard;

# Optional: independent age line for calibration mode
name  true_t  site_name  age_BP  d_age_BP;
```

**Elevation flag:** `std` (ERA-40 atmosphere), `ant` (Antarctic formula), or `pre` (supply pressure in hPa directly).

**Units:** elevation in m, thickness in cm, density in g/cm³, erosion in cm/yr, age in yr BP (before 1950).

### Supported nuclide–mineral pairs

| Nuclide | Minerals | Standard field | Extra fields |
|---------|----------|----------------|--------------|
| `Be-10` | `quartz`, `pyroxene` | `07KNSTD`, `KNSTD`, … | — |
| `Al-26` | `quartz` | `KNSTD`, `ZAL94`, … | — |
| `He-3`  | `quartz`, `olivine`, `pyroxene` | `CRONUS-P`, `NONE` | lab measurement of standard |
| `Ne-21` | `quartz` | `CRONUS-A`, `CREU-1`, `NONE` | lab measurement of standard |
| `C-14`  | `quartz` | *(none — no standardisation)* | — |

For He-3 and Ne-21 with a named standard, a seventh field gives the concentration of that standard as measured in your lab (atoms/g); use `0` with `NONE`.

---

## Exposure age calculation

```python
from stoneage import parse_v3_input, get_ages, summarize_ages

# Sample PH-1: Martha's Vineyard, MA (from the v3 format documentation)
text = """
PH-1  41.3567  -70.7348  91  std  4.5  2.65  1  0.00008  1999;
PH-1  Be-10  quartz  123453  3717  KNSTD;
"""

data   = parse_v3_input(text)
result = get_ages(data, control={"result_type": "long"})  # all three SFs

n = result["n"]
for sf in ["St", "Lm", "LSDn"]:
    t  = n[f"t_{sf}"][0]
    di = n[f"delt_int_{sf}"][0]
    de = n[f"delt_ext_{sf}"][0]
    print(f"{sf:5s}  {t:,.0f} ± {di:,.0f} yr (int)  ± {de:,.0f} yr (ext)")
```

```
St     28,196 ± 871 yr (int)  ± 2,446 yr (ext)
Lm     26,948 ± 832 yr (int)  ± 2,232 yr (ext)
LSDn   28,857 ± 892 yr (int)  ± 1,963 yr (ext)
```

`int` = analytical (measurement) uncertainty only.  
`ext` = adds production rate calibration uncertainty in quadrature.

### Scaling schemes

| Key | Name | Reference |
|-----|------|-----------|
| `St` | Lal (1991) / Stone (2000) — non-time-dependent | Stone (2000), *JGR* 105 |
| `Lm` | Time-variable Lal/Stone | Lifton et al. (2014), *EPSL* |
| `LSDn` | Lifton–Sato–Dunai, fully time-dependent | Lifton et al. (2014), *EPSL* |

Omit `"result_type": "long"` (or pass `"short"`) to compute only the St scheme, which is faster and requires no internet access to precomputed scaling grids.

### Multiple samples and nuclides

Multiple samples and multiple nuclide measurements per sample are fully supported. Lines are linked by sample name, so order does not matter.

```python
text = """
S1  44.5  -73.4  1000  std  3  2.65  1.0  0  2010;
S2  61.0  15.0   200   std  1  2.70  0.95 0  2015;
S1  Be-10  quartz  200000  4000  07KNSTD;
S1  Al-26  quartz  780000  31200  KNSTD;
S2  Be-10  quartz  500000  10000  07KNSTD;
"""
result = get_ages(parse_v3_input(text))
```

### Summary statistics

```python
summary = summarize_ages(result)
s = summary["all"]["St"]
print(f"Summary age: {s['sumval']:,.0f} ± {s['sumdel_int']:,.0f} yr (int)"
      f"  ± {s['sumdel_ext']:,.0f} yr (ext)")
print(f"Method: {s['method']}")
```

`summarize_ages` applies error-weighted averaging when ages are statistically consistent (χ² p > 0.05), or arithmetic mean + SD otherwise. It includes an iterative outlier-pruning scheme.

---

## Production rate calibration

To derive a local reference production rate from samples of independently known age, add a `true_t` line for each sample and pass `control={"cal": 1}`.

**Independent age formats:**

```
# Exact age (2 fields after the site name)
name  true_t  SITENAME  age_BP  d_age_BP;

# Bracketed age (4 fields: min ± dmin, max ± dmax; use 0 or Inf for absent bounds)
name  true_t  SITENAME  min_age_BP  d_min  max_age_BP  d_max;
```

Ages are in calendar years BP (before 1950).

```python
import numpy as np
from stoneage import parse_v3_input, get_ages, make_consts

text = """
SKY-03  66.0  -40.0  200  std  2  2.65  1  0  2006;
SKY-03  Be-10  quartz  35000  700  07KNSTD;
SKY-03  true_t  FEAR  11700  300;
"""

data   = parse_v3_input(text)
result = get_ages(data, control={"cal": 1, "result_type": "long"})

n      = result["n"]
consts = make_consts()
idx    = consts.nuclides.index("N10quartz")

print("Implied reference production rates (SLHL):")
print(f"  St   : {n['calc_P_St'][0]:.4f} at/g/yr  (global default: {consts.refP_St[idx]:.3f})")
print(f"  Lm   : {n['calc_P_Lm'][0]:.4f} at/g/yr  (global default: {consts.refP_Lm[idx]:.3f})")
print(f"  LSDn : {n['calc_P_LSDn'][0]:.4f}          (global default: {consts.refP_LSDn[idx]:.3f}, dimensionless)")
```

```
Implied reference production rates (SLHL):
  St   : 2.2794 at/g/yr  (global default: 4.086)
  Lm   : 2.2795 at/g/yr  (global default: 4.208)
  LSDn : 0.4429          (global default: 0.849, dimensionless)
```

> **Note on LSDn units.** St and Lm report dimensional production rates in atoms g⁻¹ yr⁻¹. LSDn reports a dimensionless correction factor (global calibration ≈ 0.849 for Be-10 in quartz). Ratios of the returned value to the global default indicate how your site compares to the global calibration dataset.

For a real calibration dataset with multiple samples, take the error-weighted mean of `calc_P_St` (or `_Lm`, `_LSDn`) across samples, weighted by `calc_delP_St`.

### Minimum and maximum age constraints

When only age bounds are available (e.g. from stratigraphic bracketing):

```python
text = """
S1  55.0  -3.0  300  std  3  2.65  1  0  2010;
S1  Be-10  quartz  40000  800  07KNSTD;
S1  true_t  SITE  7950  45  Inf  0;   # minimum age only
"""
result = get_ages(parse_v3_input(text), control={"cal": 1})
n = result["n"]
# Minimum age → maximum production rate
print(f"Max P_St: {n['calc_maxP_St'][0]:.4f} at/g/yr")
```

---

## Validation

Ages computed by `stoneage` match the online MATLAB calculator  
(stoneage.ice-d.org) to the nearest year for all tested nuclides and  
scaling schemes. Online validation tests can be run with:

```bash
pytest -m network tests/test_online_validation.py
```

---

## Running the tests

```bash
# Offline tests only (no internet required)
pytest tests/test_functional.py tests/test_physics.py \
       tests/test_edge_cases.py tests/test_coverage_gaps.py

# Include online validation against the live MATLAB calculator
pytest -m network tests/test_online_validation.py

# Skip online tests (e.g. in CI without outbound access)
pytest -m "not network"

# With coverage report
pytest --cov=stoneage --cov-report=term-missing -m "not network"
```

---

## Attribution

The algorithms, physical constants, scaling grids, and paleomagnetic data  
bundled in this package are the work of Greg Balco (Berkeley Geochronology  
Center) and collaborators. If you use `stoneage` in published research,  
please cite the original calculator:

> Balco, G. (2016). *Production rate calculations for cosmic-ray-muon-produced  
> ¹⁰Be and ²⁶Al benchmarked against geological calibration data.* Quaternary  
> Geochronology, 39, 150–173. https://doi.org/10.1016/j.quageo.2017.02.001

and the scaling scheme references appropriate to your chosen method:

- **St:** Stone, J. (2000). Air pressure and cosmogenic isotope production. *JGR*, 105(B10), 23753–23759.
- **Lm / LSDn:** Lifton, N., Sato, T., & Dunai, T. J. (2014). Scaling in situ cosmogenic nuclide production rates using analytical approximations to atmospheric cosmic-ray fluxes. *EPSL*, 386, 149–160.

### License

The Python source code in this repository is released under the  
[GNU General Public License v2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html),  
consistent with the older CRONUS-Earth MATLAB files on which parts of the  
implementation are based. The v3-specific MATLAB files and data grids  
(sourced from stoneage.ice-d.org) carry a "not licensed for use or  
distribution" notice from Greg Balco / Berkeley Geochronology Center;  
permission to include them here is being sought from the original author.
