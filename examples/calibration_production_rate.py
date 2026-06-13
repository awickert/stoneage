"""
Example: deriving a local 10Be reference production rate from samples of
known age, then applying that rate to date nearby undated samples.

Background
----------
The v3 calculator has two modes:

  Age mode (default):   supply concentrations → get exposure ages
  Calibration mode:     supply concentrations + known ages → get implied
                        SLHL reference production rate (P_ref)

This example shows the full workflow:

  Step 1. Compute P_ref from a set of calibration samples.
  Step 2. Use that P_ref to date new samples from the same region.

Calibration input format
------------------------
Add a  true_t  line alongside each sample and nuclide line.

Exact age (age ± uncertainty, both in calendar years BP = before 1950):

    NAME  true_t  SITENAME  age_BP  d_age;

Age bounds only (use 0 / Inf for absent bounds):

    NAME  true_t  SITENAME  min_BP  d_min  max_BP  d_max;

The age shown in the output is "years before sample collection", not
"years before 1950": the calculator adds (collection_year − 1950) to
convert. E.g. age_BP = 13 500 yr, collected 2015 → output shows 13 565 yr.

Online calculator note
----------------------
Balco's online calibration form (stoneage.ice-d.org/math/v3/v3_cal_in.html)
accepts only ONE nuclide per submission. This Python library has no such
restriction; multiple nuclides on the same calibration run are supported.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stoneage import parse_v3_input, get_ages, make_consts

# ---------------------------------------------------------------------------
# Step 1A. Calibration samples with exact independent ages
#
# Five glacially polished boulders, Scottish Highlands.
# Ages from radiocarbon dating of basal organic material (calendar yr BP).
# Concentrations are synthetic but self-consistent (see note in docstring).
# ---------------------------------------------------------------------------

# Field order: name, lat, lon, elv(m), thick(cm), yr_collected,
#              age_BP, d_age(yr), N10(atoms/g), delN10(atoms/g)
cal_samples = [
    ("SCT-01", 56.8, -4.9, 600, 3.0, 2015, 13500, 200,  97500, 2925),
    ("SCT-02", 56.9, -4.7, 580, 2.5, 2015, 14200, 250, 102100, 3063),
    ("SCT-03", 56.7, -5.1, 620, 4.0, 2015, 12800, 180,  92800, 2784),
    ("SCT-04", 56.8, -4.8, 610, 3.5, 2015, 15100, 300, 111000, 3330),
    ("SCT-05", 56.9, -5.0, 595, 2.0, 2015, 13900, 220,  99800, 2994),
]

# ---------------------------------------------------------------------------
# Step 1B. One sample with only a minimum age (e.g. organic material
# immediately above the till gives only a minimum deglaciation age).
# ---------------------------------------------------------------------------

min_age_sample = ("SCT-06", 56.8, -5.0, 605, 3.0, 2015, 12500, 150, "Inf", 0)
# true_t format for min-only: age_BP d_age Inf 0
# → calc_maxP is populated; calc_P and calc_minP remain zero.

rho     = 2.65   # g/cm³
shield  = 1.0
erosion = 0.0
std     = "07KNSTD"
site    = "SCT"

# ---------------------------------------------------------------------------
# Build the v3 input text block
# ---------------------------------------------------------------------------

lines = []
for name, lat, lon, elv, thick, yr, age_bp, d_age, N, delN in cal_samples:
    lines.append(f"{name} {lat} {lon} {elv} std {thick} {rho} {shield} {erosion} {yr};")
    lines.append(f"{name} Be-10 quartz {N} {delN} {std};")
    lines.append(f"{name} true_t {site} {age_bp} {d_age};")

# Append the minimum-age sample
name, lat, lon, elv, thick, yr, min_bp, d_min, max_bp, d_max = min_age_sample
lines.append(f"{name} {lat} {lon} {elv} std {thick} {rho} {shield} {erosion} {yr};")
lines.append(f"{name} Be-10 quartz 90000 2700 {std};")
lines.append(f"{name} true_t {site} {min_bp} {d_min} {max_bp} {d_max};")

cal_text = "\n".join(lines)

# ---------------------------------------------------------------------------
# Parse and run calibration mode
# ---------------------------------------------------------------------------

data   = parse_v3_input(cal_text)
result = get_ages(data, control={"cal": 1, "result_type": "long"})

n      = result["n"]
c      = result["c"]
s      = result["s"]
consts = make_consts()
idx    = consts.nuclides.index("N10quartz")

# ---------------------------------------------------------------------------
# Display per-sample results
# ---------------------------------------------------------------------------

print("=" * 76)
print("  Per-sample implied reference production rates")
print("  (ages shown are years before sample collection, not before 1950)")
print("=" * 76)
print(f"  {'Sample':8s}  {'Age (yr)':>9s}  {'Constraint':10s}  "
      f"{'P_St':>7s}  {'P_Lm':>7s}  {'P_LSDn':>8s}")
print("  " + "-" * 62)

for b in range(len(n["index"])):
    si       = n["index"][b]
    sname    = s["sample_name"][si]
    truet    = c["truet"][si]
    mint     = c["mint"][si]
    maxt     = c["maxt"][si]

    if truet > 0:
        age_str  = f"{truet:>9.0f}"
        kind     = "exact"
        P_St     = n["calc_P_St"][b]
        P_Lm     = n["calc_P_Lm"][b]
        P_LSDn   = n["calc_P_LSDn"][b]
    elif mint > 0 and np.isinf(maxt):
        age_str  = f">{mint:>8.0f}"
        kind     = "min only"
        P_St     = n["calc_maxP_St"][b]   # min age → max production rate
        P_Lm     = n["calc_maxP_Lm"][b]
        P_LSDn   = n["calc_maxP_LSDn"][b]
    else:
        age_str  = f"{mint:.0f}–{maxt:.0f}"
        kind     = "bracket"
        P_St     = n["calc_P_St"][b]
        P_Lm     = n["calc_P_Lm"][b]
        P_LSDn   = n["calc_P_LSDn"][b]

    print(f"  {sname:8s}  {age_str}  {kind:10s}  "
          f"{P_St:>7.4f}  {P_Lm:>7.4f}  {P_LSDn:>8.5f}")

print("  " + "-" * 62)
print(f"  {'Global default':19s}  "
      f"{consts.refP_St[idx]:>7.4f}  "
      f"{consts.refP_Lm[idx]:>7.4f}  "
      f"{consts.refP_LSDn[idx]:>8.5f}")
print()
print("  LSDn column is a dimensionless correction factor, not at/g/yr.")
print("  All values are reference production rates at sea level, high latitude.")

# ---------------------------------------------------------------------------
# Error-weighted mean from exact-age samples only
# (min-age samples give an upper bound on P, not a point estimate)
# ---------------------------------------------------------------------------

def ewmean(values, errors):
    w     = 1.0 / errors**2
    mean  = np.sum(w * values) / np.sum(w)
    stdev = 1.0 / np.sqrt(np.sum(w))
    return mean, stdev

print()
print("=" * 76)
print("  Error-weighted mean (exact-age samples only)")
print("=" * 76)

calibrated = {}
for sf, ref_key in [("St", "refP_St"), ("Lm", "refP_Lm"), ("LSDn", "refP_LSDn")]:
    P_vals  = n[f"calc_P_{sf}"]
    dP_vals = n[f"calc_delP_{sf}"]
    ref_val = getattr(consts, ref_key)[idx]

    ok = (P_vals > 0) & (dP_vals > 0)
    if ok.sum() == 0:
        print(f"  {sf}: no valid exact-age uncertainties — run with result_type='long'")
        continue

    P_mean, P_err = ewmean(P_vals[ok], dP_vals[ok])
    calibrated[sf] = (P_mean, P_err)

    unit  = "(dimensionless)" if sf == "LSDn" else "at/g/yr"
    ratio = P_mean / ref_val
    print(f"  {sf:5s}: {P_mean:.4f} ± {P_err:.4f} {unit}")
    print(f"         global default {ref_val:.4f};  ratio = {ratio:.3f}")

# ---------------------------------------------------------------------------
# Step 2. Apply the locally calibrated production rate to date new samples
#
# Modify the constants object and pass it to get_ages().
# Only exact-age samples (calc_P_St) contribute to the calibration.
# ---------------------------------------------------------------------------

print()
print("=" * 76)
print("  Step 2: dating new samples with the local production rate")
print("=" * 76)

new_text = """
NEW-A  56.8  -4.9  590  std  3  2.65  1  0  2020;
NEW-A  Be-10  quartz  85000  2550  07KNSTD;
NEW-B  56.7  -5.0  615  std  2  2.65  1  0  2020;
NEW-B  Be-10  quartz  110000  3300  07KNSTD;
"""

# Build a modified constants object with the locally calibrated St rate
local_consts = make_consts()
if "St" in calibrated:
    P_local, dP_local = calibrated["St"]
    local_consts.refP_St[idx]    = P_local
    local_consts.delrefP_St[idx] = dP_local

# parse_v3_input must be called separately for each run: get_ages mutates
# its input dict in place and returns the same object, so reusing one dict
# would make both variables alias the same result.
new_global = get_ages(parse_v3_input(new_text))
new_local  = get_ages(parse_v3_input(new_text), consts=local_consts)

print(f"  {'Sample':8s}  {'Age (global)':>14s}  {'Age (local cal.)':>16s}  {'Shift':>7s}")
print("  " + "-" * 54)
for b in range(len(new_global["n"]["index"])):
    si    = new_global["n"]["index"][b]
    name  = new_global["s"]["sample_name"][si]
    t_g   = new_global["n"]["t_St"][b]
    t_l   = new_local["n"]["t_St"][b]
    print(f"  {name:8s}  {t_g:>12,.0f} yr  {t_l:>14,.0f} yr  {t_l-t_g:>+6,.0f} yr")
print()
if "St" in calibrated:
    ratio = calibrated["St"][0] / consts.refP_St[idx]
    print(f"  Local P_St is {ratio:.3f}× the global default → ages shift by "
          f"~{100*(1/ratio - 1):+.1f}%.")
