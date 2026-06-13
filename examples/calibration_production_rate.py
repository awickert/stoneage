"""
Example: deriving a local 10Be reference production rate from samples of known age.

Scenario
--------
Five glacially polished boulders in the Scottish Highlands were sampled.
Each has an independently constrained deglaciation age from radiocarbon
dating of overlying organic material (reported as calendar years BP, i.e.
years before 1950). Their 10Be concentrations were measured in the lab.

We want to know what reference production rate (atoms/g/yr at sea level,
high latitude — SLHL) is implied by these measurements, using each of the
three scaling schemes (St, Lm, LSDn).

Calibration mode in v3 format
------------------------------
Add a  true_t  line for each sample alongside the usual sample and nuclide
lines. The true_t line carries the independently known age:

    SAMPLE  true_t  SITENAME  age_BP  d_age_BP ;         ← exact age ± uncertainty

For min/max constraints instead of an exact age, use four fields:
    SAMPLE  true_t  SITENAME  min_age  d_min  max_age  d_max ;
Use 0 for an absent bound and Inf for an absent maximum.

Calling the calculator
----------------------
Pass  control={"cal": 1}  to get_ages().
For all three scaling schemes, also pass  "result_type": "long".

Output
------
Each sample yields  calc_P_St,  calc_P_Lm,  calc_P_LSDn  —
the implied SLHL reference production rate.
The error-weighted mean across samples is the calibrated site value.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stoneage import parse_v3_input, get_ages, make_consts

# ---------------------------------------------------------------------------
# "Known" ages (calendar years BP = before 1950) from radiocarbon dating,
# and measured 10Be concentrations (07KNSTD) with 3 % analytical uncertainty.
#
# These samples are synthetic but self-consistent: concentrations were
# computed from  P_site * t  where P_site ≈ 7.30 at/g/yr at this location,
# then perturbed by ±2–4 % to mimic real scatter.
# ---------------------------------------------------------------------------

samples = [
    # name       lat     lon     elv  thick  yr_collected  age_BP  d_age  N10      delN
    ("SCT-01", 56.8,  -4.9,  600,   3.0,  2015,         13500,   200,  97500,   2925),
    ("SCT-02", 56.9,  -4.7,  580,   2.5,  2015,         14200,   250, 102100,   3063),
    ("SCT-03", 56.7,  -5.1,  620,   4.0,  2015,         12800,   180,  92800,   2784),
    ("SCT-04", 56.8,  -4.8,  610,   3.5,  2015,         15100,   300, 111000,   3330),
    ("SCT-05", 56.9,  -5.0,  595,   2.0,  2015,         13900,   220,  99800,   2994),
]

rho      = 2.65   # g/cm³ (granite)
shield   = 1.0    # no topographic shielding
erosion  = 0.0    # cm/yr
std      = "07KNSTD"
sitename = "SCT"

# ---------------------------------------------------------------------------
# Build the v3 input text block
# ---------------------------------------------------------------------------
lines = []
for name, lat, lon, elv, thick, yr, age_bp, d_age, N, delN in samples:
    lines.append(
        f"{name}  {lat}  {lon}  {elv}  std  {thick}  {rho}  {shield}  {erosion}  {yr};"
    )
    lines.append(f"{name}  Be-10  quartz  {N}  {delN}  {std};")
    lines.append(f"{name}  true_t  {sitename}  {age_bp}  {d_age};")

input_text = "\n".join(lines)

print("Input block:")
print("-" * 60)
print(input_text)
print("-" * 60)

# ---------------------------------------------------------------------------
# Parse and run in calibration mode (all three scaling schemes)
# ---------------------------------------------------------------------------
data   = parse_v3_input(input_text)
result = get_ages(data, control={"cal": 1, "result_type": "long"})

n = result["n"]
c = result["c"]
s = result["s"]
consts = make_consts()
idx_be10 = consts.nuclides.index("N10quartz")

print()
print("=" * 72)
print("  Per-sample implied reference production rates (at/g/yr at SLHL)")
print("=" * 72)
print(f"  {'Sample':8s}  {'Age (yr)':>9s}  {'P_St':>7s}  {'P_Lm':>7s}  {'P_LSDn':>8s}")
print("  " + "-" * 52)

for b in range(len(n["index"])):
    si       = n["index"][b]
    sname    = s["sample_name"][si]
    true_age = c["truet"][si]
    P_St     = n["calc_P_St"][b]
    P_Lm     = n["calc_P_Lm"][b]
    # LSDn is a nondimensional correction factor (ref value ~0.849 for Be-10)
    P_LSDn   = n["calc_P_LSDn"][b]
    print(f"  {sname:8s}  {true_age:>9.0f}  {P_St:>7.4f}  {P_Lm:>7.4f}  {P_LSDn:>8.5f}")

print("  " + "-" * 52)
print(f"  {'Ref. value':17s}  {consts.refP_St[idx_be10]:>7.4f}  "
      f"{consts.refP_Lm[idx_be10]:>7.4f}  {consts.refP_LSDn[idx_be10]:>8.5f}")
print()
print("  (Ref. value = default SLHL production rate built into the v3 constants)")

# ---------------------------------------------------------------------------
# Error-weighted mean across samples for each scaling scheme
# ---------------------------------------------------------------------------

def ewmean(values, errors):
    w = 1.0 / errors**2
    mean  = np.sum(w * values) / np.sum(w)
    stdev = 1.0 / np.sqrt(np.sum(w))
    return mean, stdev

print()
print("=" * 72)
print("  Error-weighted mean reference production rate")
print("=" * 72)

for sf, ref_key in [("St",   "refP_St"),
                    ("Lm",   "refP_Lm"),
                    ("LSDn", "refP_LSDn")]:
    P_key    = f"calc_P_{sf}"
    dP_key   = f"calc_delP_{sf}"
    ref_val  = getattr(consts, ref_key)[idx_be10]

    P_vals  = n[P_key]
    dP_vals = n[dP_key]

    ok = dP_vals > 0
    if ok.sum() == 0:
        print(f"  {sf:6s}: no valid uncertainties computed")
        continue

    P_mean, P_err = ewmean(P_vals[ok], dP_vals[ok])

    if sf == "LSDn":
        unit = "(dimensionless)"
    else:
        unit = "at/g/yr"

    print(f"  {sf:6s}: {P_mean:.4f} ± {P_err:.4f} {unit}")
    print(f"          (default v3 reference: {ref_val:.4f}; "
          f"ratio = {P_mean/ref_val:.3f})")

print()
print("  Note: ratio near 1.0 indicates good agreement with the global")
print("  calibration; values outside ~0.8–1.2 suggest the site is unusual")
print("  or that the independent ages require revision.")
