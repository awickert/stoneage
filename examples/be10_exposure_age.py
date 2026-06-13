"""
Example: 10Be exposure age calculation using the stoneage library.

Sample PH-1 is from Putnam et al. (2010), collected on Martha's Vineyard,
Massachusetts. It is one of the example samples in the v3 input format
documentation at:
  https://hess.ess.washington.edu/math/docs/v3/v3_input_explained.html

The sample line format (10 whitespace-separated fields, terminated with ;):
  name  lat  lon  elevation  flag  thickness  density  shielding  erosion  year

The nuclide line format (6 fields for Be-10):
  name  Be-10  mineral  concentration  uncertainty  standard
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stoneage import parse_v3_input, get_ages, summarize_ages

# ---------------------------------------------------------------------------
# Input data in v3 text format
# ---------------------------------------------------------------------------
# Sample PH-1:
#   Location:   Martha's Vineyard, Massachusetts, USA
#   Lat/lon:    41.3567 N, 70.7348 W
#   Elevation:  91 m (standard atmosphere)
#   Thickness:  4.5 cm
#   Density:    2.65 g/cm³
#   Shielding:  1.0 (no topographic shielding)
#   Erosion:    0.00008 cm/yr (very slow)
#   Collected:  1999
#
# Be-10 measurement:
#   Concentration: 123 453 atoms/g (KNSTD standard)
#   Uncertainty:     3 717 atoms/g

input_text = """
PH-1  41.3567  -70.7348  91  std  4.5  2.65  1  0.00008  1999;
PH-1  Be-10  quartz  123453  3717  KNSTD;
"""

# ---------------------------------------------------------------------------
# Parse and calculate
# ---------------------------------------------------------------------------
data   = parse_v3_input(input_text)
result = get_ages(data, control={"result_type": "long"})  # all three SFs

n = result["n"]

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
print("=" * 60)
print("  10Be exposure age — PH-1 (Martha's Vineyard, MA)")
print("=" * 60)
print(f"  Latitude:     {result['s']['lat'][0]:.4f} °N")
print(f"  Longitude:    {result['s']['long'][0]:.4f} °W")
print(f"  Elevation:    91 m  (ERA-40 pressure: {result['s']['pressure'][0]:.1f} hPa)")
print(f"  10Be:         {data['n']['N'][0]:,.0f} ± {data['n']['delN'][0]:,.0f} atoms/g")
print()
print(f"  Stone(2000) scaling factor : {n['sf_St'][0]:.4f}")
print(f"  Thickness correction       : {n['sf_thick'][0]:.4f}")
print(f"  Muon production rate       : {n['P_mu'][0]:.4f} atoms/g/yr")
print(f"  Total spallogenic rate (St): {n['Psp_St'][0]:.4f} atoms/g/yr")
print()
print("  Exposure ages")
print("  " + "-" * 50)
print(f"  {'Scaling':8s}  {'Age (yr)':>10s}  {'±int (yr)':>10s}  {'±ext (yr)':>10s}")
print("  " + "-" * 50)
for sf in ["St", "Lm", "LSDn"]:
    t  = n[f"t_{sf}"][0]
    di = n[f"delt_int_{sf}"][0]
    de = n[f"delt_ext_{sf}"][0]
    print(f"  {sf:8s}  {t:>10,.0f}  {di:>10,.0f}  {de:>10,.0f}")
print("  " + "-" * 50)
print()
print("  int = analytical (measurement) uncertainty only")
print("  ext = includes production rate uncertainty")

if result["flags"]:
    print()
    print("  Flags:", "; ".join(result["flags"]))
