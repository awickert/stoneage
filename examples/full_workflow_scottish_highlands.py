"""
Full workflow example: Scottish Highlands Be-10 exposure ages.

Demonstrates the complete ProductionRateWorkflow pipeline:

  1. Load sample locations and measurements from a CSV
  2. Fetch nearby calibration data from ICE-D
  3. Derive a local SLHL reference production rate (P_ref)
  4. Compute local production rates per sample
  5. Report calibration quality and corrections checklist
  6. Date all samples using the local production rate

The two target samples (NEW-A, NEW-B) are synthetic boulders with
geologically plausible concentrations for a Late Weichselian site in the
western Scottish Highlands (~13–18 ka deglaciation).  All other parameters
(elevation, geometry) are realistic for the region.

Requires an internet connection to fetch calibration data from ICE-D.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stoneage.workflow import ProductionRateWorkflow

# ---------------------------------------------------------------------------
# Step 1. Load samples from CSV
# ---------------------------------------------------------------------------
# The CSV has columns: name, lat, lon, elevation, elv_flag, thickness, density,
# shielding, erosion, year, nuclide, mineral, concentration, uncertainty, standard

csv_path = Path(__file__).parent / "samples_scottish_highlands.csv"
wf = ProductionRateWorkflow.from_csv(str(csv_path))

print("Samples loaded:")
for t in wf.targets:
    print(f"  {t.name}: {t.lat:.3f}°N, {t.lon:.3f}°E, {t.elv:.0f} m")

# ---------------------------------------------------------------------------
# Step 2. Find calibration data from ICE-D
#
# Searches the complete Be-10/quartz dataset (dataset 20) for calibration
# samples within 1500 km of the target centroid.  The Scottish Highlands
# are in a low-calibration-density region; the nearest sites are typically
# in Scandinavia, the British Isles, or northern France.
# ---------------------------------------------------------------------------

wf.find_calibration_data(dataset_id=20, max_dist_km=1500, min_samples=3)

# ---------------------------------------------------------------------------
# Step 3. Calibrate — derive P_ref, then compute P_local per target
# ---------------------------------------------------------------------------
# include_global=True (default) always reports the global default alongside
# the locally derived rate so users can judge how much they diverge.

wf.calibrate(include_global=True)

# ---------------------------------------------------------------------------
# Step 4. Report: production rates + corrections checklist
# ---------------------------------------------------------------------------

wf.report()

# ---------------------------------------------------------------------------
# Step 5. Date all samples from the CSV using the local production rate
# ---------------------------------------------------------------------------

wf.date_all(use_local=True)
