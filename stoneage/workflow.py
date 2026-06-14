"""
High-level workflow for cosmogenic nuclide production rate estimation and
optional exposure age calculation.

Design principles
-----------------
- Production rate is the primary output; dating is optional.
- All steps are printed transparently so non-experts can follow the logic.
- Local calibration from nearby ICE-D sites is preferred over the global
  default; both are always reported so the user can judge.
- LSDn is the recommended scaling scheme (Borchers et al. 2016, Lifton et al.
  2014), but St and Lm are reported alongside it.
- "Fitting parameter" is never used; the output says "reference production
  rate at sea level, high latitude (SLHL)".

Quick start
-----------
    from stoneage.workflow import ProductionRateWorkflow

    # From a single point
    wf = ProductionRateWorkflow(lat=46.2, lon=-91.8, elv=183)

    # From a location-only CSV (name, lat, lon, elevation)
    wf = ProductionRateWorkflow.from_csv("sites.csv")

    # Or from a full v3-field CSV (adds nuclide, mineral, concentration, …)
    wf = ProductionRateWorkflow.from_csv("samples_with_measurements.csv")

    wf.find_calibration_data()   # fetch ICE-D, rank by distance
    wf.calibrate()               # compute local production rates
    wf.report()                  # print transparent summary

    # Optional dating (single sample, manual nuclide line)
    wf.date("SITE Be-10 quartz 60000 1800 07KNSTD;")

    # Dating all samples loaded from a full CSV
    wf.date_all()
"""

import math
import urllib.request
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

warnings.filterwarnings("ignore")

from .atmosphere import ERA40atm, antatm
from .scaling import stone2000
from .corrections import thickness, Lsp
from .constants import make_consts, Constants
from .input_parser import parse_v3_input, csv_to_v3
from .calculator import get_ages


# ---------------------------------------------------------------------------
# ICE-D dataset catalogue (Be-10 in quartz; extend as needed)
# ---------------------------------------------------------------------------

ICED_BASE = "https://version2.ice-d.org/production%20rate%20calibration%20data/cal_data_set"

ICED_DATASETS = {
    20: "All Be-10 in quartz (most complete)",
    4:  "CRONUS Primary Be-10 (Borchers et al. 2016)",
    3:  "Balco – NE North America 2009",
    2:  "Stroeven – Scandinavia 2015",
    1:  "Young – Baffin Bay 2013",
    10: "Putnam – Rannoch Moor",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TargetSite:
    name: str
    lat: float
    lon: float
    elv: float          # metres
    thick: float = 2.0  # cm sample thickness
    rho:   float = 2.65 # g/cm³
    shield: float = 1.0 # topographic shielding factor
    erosion: float = 0.0
    yr: int = 2010


@dataclass
class CalibrationSite:
    name: str
    lat: float
    lon: float
    dist_km: float
    v3_block: str       # raw v3 text for this sample
    site_label: str     # true_t site name
    age_bp: float
    d_age: float


@dataclass
class CalibrationResult:
    """Scheme-level reference production rate — one per scaling scheme."""
    scheme: str
    P_ref_SLHL: float   # at/g/yr for St/Lm; dimensionless for LSDn
    dP_ref: float
    source: str         # "global default" or "ICE-D local (N sites, X km)"
    n_sites: int = 0
    max_dist_km: float = 0.0
    chi2_p: Optional[float] = None


@dataclass
class PointResult:
    """Per-sample local production rate derived from a CalibrationResult."""
    target_name: str
    lat: float
    lon: float
    elv: float
    scheme: str
    SF: float           # Stone2000 (or LSDn) scaling factor at this point
    tc: float           # thickness correction
    P_mu: float         # muon production rate
    P_local: float      # total local production rate
    dP_local: float     # uncertainty from P_ref only
    source: str         # same as CalibrationResult.source


# Keep a combined alias for backward compatibility
ProductionRateResult = PointResult


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _fetch_iced_dataset(dataset_id: int) -> str:
    url = f"{ICED_BASE}/{dataset_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "stoneage/3.0"})
    html = urllib.request.urlopen(req, timeout=30).read().decode()
    pre = re.findall(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL)
    if not pre:
        raise RuntimeError(f"No data block found in ICE-D dataset {dataset_id}")
    return pre[0].strip()


def _parse_iced_into_samples(v3_text: str) -> List[CalibrationSite]:
    """Parse a v3 calibration text block into CalibrationSite objects."""
    lines = v3_text.split("\n")
    samples = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        parts = line.split()
        if len(parts) >= 5 and parts[4] in ("std", "ant", "pre"):
            name = parts[0]
            lat  = float(parts[1])
            lon  = float(parts[2])
            block_lines = [lines[i]]
            i += 1
            site_label, age_bp, d_age = "", 0.0, 0.0
            while i < len(lines):
                l = lines[i].strip()
                if not l:
                    i += 1
                    break
                lparts = l.split()
                if len(lparts) >= 5 and lparts[4] in ("std", "ant", "pre"):
                    break
                if "true_t" in lparts:
                    site_label = lparts[2] if len(lparts) > 2 else ""
                    age_bp = float(lparts[3].rstrip(";")) if len(lparts) > 3 else 0.0
                    d_age  = float(lparts[4].rstrip(";")) if len(lparts) > 4 else 0.0
                block_lines.append(lines[i])
                i += 1
            samples.append(CalibrationSite(
                name=name, lat=lat, lon=lon, dist_km=0.0,
                v3_block="\n".join(block_lines),
                site_label=site_label, age_bp=age_bp, d_age=d_age,
            ))
        else:
            i += 1
    return samples


def _ewmean(values, errors):
    w = 1.0 / errors ** 2
    mean  = np.sum(w * values) / np.sum(w)
    stdev = 1.0 / np.sqrt(np.sum(w))
    return mean, stdev


def _chi2_p(chi2, dof):
    import scipy.special
    if dof <= 0:
        return 1.0
    return float(1.0 - scipy.special.gammainc(dof / 2.0, chi2 / 2.0))


# ---------------------------------------------------------------------------
# Main workflow class
# ---------------------------------------------------------------------------

class ProductionRateWorkflow:
    """
    Transparent production rate estimation from ICE-D calibration data.

    For a single point:
        wf = ProductionRateWorkflow(lat=46.2, lon=-91.8, elv=183)

    For multiple sample points from a CSV (columns: name, lat, lon, elevation;
    optional: thickness, density, shielding, erosion, year):
        wf = ProductionRateWorkflow.from_csv("samples.csv")
        # by default computes a production rate per point;
        # pass use_centroid=True for one combined rate at the group centroid.
    """

    def __init__(self, lat: float, lon: float, elv: float,
                 thick: float = 2.0, rho: float = 2.65,
                 shield: float = 1.0, erosion: float = 0.0,
                 yr: int = 2010, name: str = "TARGET"):
        self.targets: List[TargetSite] = [
            TargetSite(name=name, lat=lat, lon=lon, elv=elv,
                       thick=thick, rho=rho, shield=shield,
                       erosion=erosion, yr=yr)
        ]
        self._cal_samples: List[CalibrationSite] = []
        self._selected: List[CalibrationSite] = []
        self._cal_results: List[CalibrationResult] = []   # one per scheme
        self._point_results: List[PointResult] = []       # one per (scheme × target)
        self._consts: Optional[Constants] = None
        self._v3_text: Optional[str] = None

    # convenience accessor for single-point workflows
    @property
    def target(self) -> TargetSite:
        return self.targets[0]

    @classmethod
    def from_csv(cls, path: str, use_centroid: bool = False,
                 **kwargs) -> "ProductionRateWorkflow":
        """
        Create from a CSV file.

        Accepts two formats:

        **Location-only CSV** (columns: name, lat, lon, elevation; optional:
        elv_flag, thickness, density, shielding, erosion, year):
            Use for production-rate estimation before samples are measured.

        **Full v3-field CSV** (adds: nuclide, mineral, concentration, uncertainty,
        standard; He-3/Ne-21 also need standard_value):
            Use when samples are already measured. Enables date_all() for
            computing exposure ages directly from the CSV.

        Multiple rows sharing the same sample name are treated as one sample
        with multiple nuclide measurements (dual-nuclide samples).

        Parameters
        ----------
        use_centroid : bool
            If True, compute one production rate at the geographic centroid
            of all points.  Default False: compute per point.
        """
        import csv as csv_mod
        rows = []
        with open(path) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            raise ValueError(f"No data rows in {path}")

        # Detect full-format CSV (has nuclide columns with data)
        has_nuclide = (
            "nuclide" in rows[0]
            and any(r.get("nuclide", "").strip() for r in rows)
        )

        def _flt(r, key, default):
            v = r.get(key, "")
            return float(v) if str(v).strip() else float(default)

        def _site(r):
            return TargetSite(
                name    = r["name"].strip(),
                lat     = float(r["lat"]),
                lon     = float(r["lon"]),
                elv     = float(r["elevation"]),
                thick   = _flt(r, "thickness", 2.0),
                rho     = _flt(r, "density",   2.65),
                shield  = _flt(r, "shielding", 1.0),
                erosion = _flt(r, "erosion",   0.0),
                yr      = int(_flt(r, "year", 2010)),
            )

        # Deduplicate: one TargetSite per unique sample name
        seen: dict = {}
        unique_rows = []
        for r in rows:
            name = r["name"].strip()
            if name not in seen:
                seen[name] = r
                unique_rows.append(r)

        wf = cls.__new__(cls)

        if use_centroid:
            lats = [float(r["lat"])       for r in unique_rows]
            lons = [float(r["lon"])       for r in unique_rows]
            elvs = [float(r["elevation"]) for r in unique_rows]
            centroid = TargetSite(
                name=Path(path).stem + "_centroid",
                lat=float(np.mean(lats)),
                lon=float(np.mean(lons)),
                elv=float(np.mean(elvs)),
                **{k: v for k, v in kwargs.items()
                   if k in ("thick","rho","shield","erosion","yr")},
            )
            wf.targets = [centroid]
        else:
            wf.targets = [_site(r) for r in unique_rows]

        wf._cal_samples   = []
        wf._selected      = []
        wf._cal_results   = []
        wf._point_results = []
        wf._consts        = None
        wf._csv_path      = path
        wf._v3_text       = csv_to_v3(path) if has_nuclide else None
        return wf

    # ── Site physics ────────────────────────────────────────────────────────

    def _centroid(self) -> TargetSite:
        """Geographic centroid of all target sites (for ICE-D search)."""
        lats = [t.lat for t in self.targets]
        lons = [t.lon for t in self.targets]
        elvs = [t.elv for t in self.targets]
        t0   = self.targets[0]
        return TargetSite(
            name="centroid", lat=float(np.mean(lats)),
            lon=float(np.mean(lons)), elv=float(np.mean(elvs)),
            thick=t0.thick, rho=t0.rho, shield=t0.shield,
            erosion=t0.erosion, yr=t0.yr,
        )

    @staticmethod
    def _site_pressure(t: TargetSite) -> float:
        if t.elv >= 0:
            return float(ERA40atm([t.lat], [t.lon], [t.elv])[0])
        return float(antatm([t.elv])[0])

    @staticmethod
    def _scaling_factor(t: TargetSite, pressure: float) -> float:
        return float(stone2000([t.lat], [pressure], Fsp=1.0)[0])

    @staticmethod
    def _thickness_correction(t: TargetSite) -> float:
        return float(thickness([t.thick], [Lsp()], [t.rho])[0])

    @staticmethod
    def _muon_rate(pressure: float, nuclide_idx: int = 3) -> float:
        """Be-10 muon rate (nuclide_idx=3 for N10quartz in the 8-element vector)."""
        consts = make_consts()
        return float(consts.Pmu0[nuclide_idx]
                     * np.exp((1013.25 - pressure) / consts.L_mu_atm[nuclide_idx]))

    # ── ICE-D data retrieval ─────────────────────────────────────────────────

    def load_calibration_text(self, v3_text: str,
                              max_dist_km: float = 1e9) -> "ProductionRateWorkflow":
        """
        Load calibration data from a v3-format text string instead of fetching
        from ICE-D.  Useful when working offline or with custom datasets.
        """
        all_samples = _parse_iced_into_samples(v3_text)
        t = self.target
        for s in all_samples:
            s.dist_km = _haversine(t.lat, t.lon, s.lat, s.lon)
        all_samples.sort(key=lambda s: s.dist_km)
        self._cal_samples = [s for s in all_samples if s.dist_km <= max_dist_km]
        self._selected    = self._cal_samples
        print(f"Loaded {len(self._cal_samples)} calibration samples from text block.")
        return self

    def find_calibration_data(self, dataset_id: int = 20,
                              max_dist_km: float = 1000.0,
                              min_samples: int = 3) -> "ProductionRateWorkflow":
        """
        Fetch ICE-D calibration data and rank by distance to the target area.

        Distance is measured from the centroid of all target points.
        """
        desc = ICED_DATASETS.get(dataset_id, f"ICE-D dataset {dataset_id}")
        print(f"Fetching {desc} from ICE-D...")
        v3_text = _fetch_iced_dataset(dataset_id)
        all_samples = _parse_iced_into_samples(v3_text)
        print(f"  {len(all_samples)} total calibration samples retrieved.")

        ctr = self._centroid()
        for s in all_samples:
            s.dist_km = _haversine(ctr.lat, ctr.lon, s.lat, s.lon)
        all_samples.sort(key=lambda s: s.dist_km)

        nearby = [s for s in all_samples if s.dist_km <= max_dist_km]
        self._cal_samples = nearby

        ref_label = (f"{ctr.lat:.3f}°N, {ctr.lon:.3f}°W"
                     if len(self.targets) > 1 else
                     f"{self.targets[0].lat:.3f}°N, {self.targets[0].lon:.3f}°W")
        print(f"\nNearest calibration samples (distances from {ref_label}):")
        print(f"  {'Dist (km)':>10}  {'Sample':<14}  {'Lat':>7}  {'Lon':>8}  "
              f"{'Age (yr BP)':>11}  Site")
        print("  " + "-" * 65)
        for s in nearby[:10]:
            print(f"  {s.dist_km:>10.0f}  {s.name:<14}  {s.lat:>7.3f}  "
                  f"{s.lon:>8.3f}  {s.age_bp:>11.0f}  {s.site_label}")
        if len(nearby) > 10:
            print(f"  ... and {len(nearby)-10} more within {max_dist_km:.0f} km")
        if len(nearby) < min_samples:
            print(f"\n  Warning: only {len(nearby)} sample(s) within {max_dist_km} km. "
                  f"Consider increasing max_dist_km or using the global default.")

        self._selected = nearby
        return self

    def select_by_distance(self, max_dist_km: float) -> "ProductionRateWorkflow":
        """Restrict calibration samples to within max_dist_km."""
        self._selected = [s for s in self._cal_samples if s.dist_km <= max_dist_km]
        print(f"Selected {len(self._selected)} samples within {max_dist_km:.0f} km.")
        return self

    def select_by_site(self, site_label: str) -> "ProductionRateWorkflow":
        """Restrict to samples from a specific named calibration site."""
        self._selected = [s for s in self._cal_samples
                          if s.site_label.upper() == site_label.upper()]
        print(f"Selected {len(self._selected)} samples from site '{site_label}'.")
        return self

    # ── Calibration ──────────────────────────────────────────────────────────

    def _point_production(self, t: TargetSite, cal: CalibrationResult) -> PointResult:
        """Compute P_local at a single target given an already-calibrated P_ref."""
        P_atm = self._site_pressure(t)
        SF    = self._scaling_factor(t, P_atm)
        tc    = self._thickness_correction(t)
        P_mu  = self._muon_rate(P_atm)
        P_loc = cal.P_ref_SLHL * SF * tc * t.shield + P_mu
        dP_loc = cal.dP_ref    * SF * tc * t.shield
        return PointResult(
            target_name=t.name, lat=t.lat, lon=t.lon, elv=t.elv,
            scheme=cal.scheme, SF=SF, tc=tc, P_mu=P_mu,
            P_local=P_loc, dP_local=dP_loc, source=cal.source,
        )

    def calibrate(self, include_global: bool = True) -> "ProductionRateWorkflow":
        """
        Compute reference production rates, then apply per target point.

        P_ref (SLHL) is derived from the selected calibration samples and/or
        the global default.  P_local is then computed independently for each
        target point, accounting for its specific elevation, latitude, and
        sample geometry.
        """
        consts = make_consts()
        idx    = consts.nuclides.index("N10quartz")

        self._cal_results   = []
        self._point_results = []
        self._consts        = None

        scheme_keys = [
            ("St",   "refP_St",   "delrefP_St"),
            ("Lm",   "refP_Lm",   "delrefP_Lm"),
            ("LSDn", "refP_LSDn", "delrefP_LSDn"),
        ]

        # ── Global defaults ──────────────────────────────────────────────────
        if include_global:
            for sf_name, ref_key, dref_key in scheme_keys:
                cal = CalibrationResult(
                    scheme     = sf_name,
                    P_ref_SLHL = float(getattr(consts, ref_key)[idx]),
                    dP_ref     = float(getattr(consts, dref_key)[idx]),
                    source     = "global default (Borchers et al. 2016)",
                )
                self._cal_results.append(cal)
                for t in self.targets:
                    self._point_results.append(self._point_production(t, cal))

        # ── Local calibration from selected ICE-D samples ───────────────────
        if self._selected:
            v3_block = "\n".join(s.v3_block for s in self._selected)
            data   = parse_v3_input(v3_block)
            result = get_ages(data, control={"cal": 1, "result_type": "long"})
            n_res  = result["n"]
            max_d  = max(s.dist_km for s in self._selected)
            n_samp = len(self._selected)
            source = (f"ICE-D local ({n_samp} sample{'s' if n_samp>1 else ''}, "
                      f"max {max_d:.0f} km)")

            local_consts = make_consts()

            for sf_name, ref_key, _ in scheme_keys:
                P_vals  = n_res[f"calc_P_{sf_name}"]
                dP_vals = n_res[f"calc_delP_{sf_name}"]
                ok = (P_vals > 0) & (dP_vals > 0)
                if ok.sum() == 0:
                    continue
                P_ref, dP_formal = _ewmean(P_vals[ok], dP_vals[ok])
                scatter = float(np.std(P_vals[ok], ddof=1)) if ok.sum() > 1 else dP_formal
                dP_ref  = max(float(dP_formal), scatter)

                chi2   = float(np.sum(((P_vals[ok] - P_ref) / dP_vals[ok]) ** 2))
                dof    = int(ok.sum()) - 1
                p_chi2 = _chi2_p(chi2, dof) if dof > 0 else None

                cal = CalibrationResult(
                    scheme=sf_name, P_ref_SLHL=float(P_ref), dP_ref=dP_ref,
                    source=source, n_sites=n_samp, max_dist_km=max_d,
                    chi2_p=p_chi2,
                )
                self._cal_results.append(cal)
                for t in self.targets:
                    self._point_results.append(self._point_production(t, cal))

                # store for optional dating
                getattr(local_consts, ref_key)[idx] = float(P_ref)

            self._consts = local_consts

        return self

    # ── Reporting ────────────────────────────────────────────────────────────

    def report(self) -> "ProductionRateWorkflow":
        """Print a transparent summary of the production rate results."""
        multi = len(self.targets) > 1

        print()
        print("=" * 72)
        print("  PRODUCTION RATE REPORT")
        print("=" * 72)
        print("  Recommended scaling scheme: LSDn (Borchers et al. 2016,")
        print("  Lifton et al. 2014). St and Lm shown for comparison.")
        print()

        # ── Calibration summary ──────────────────────────────────────────────
        sources = list(dict.fromkeys(c.source for c in self._cal_results))
        for source in sources:
            cals = [c for c in self._cal_results if c.source == source]
            print(f"  Calibration source: {source}")
            if cals[0].chi2_p is not None:
                flag = "consistent" if cals[0].chi2_p > 0.05 else "scattered"
                print(f"    Chi-squared p-value: {cals[0].chi2_p:.3f}  ({flag})")
            print(f"    {'Scheme':<6}  {'P_ref at SLHL':>16}  Unit")
            print("    " + "-" * 36)
            for c in cals:
                unit = "dimensionless" if c.scheme == "LSDn" else "at/g/yr"
                print(f"    {c.scheme:<6}  "
                      f"{c.P_ref_SLHL:.4f} ± {c.dP_ref:.4f}  {unit}")
            print()

        # ── Per-point local production rates ─────────────────────────────────
        for source in sources:
            pts = [p for p in self._point_results if p.source == source]
            print(f"  Local production rates — {source}")
            if multi:
                hdr = (f"    {'Site':<16}  {'Elv(m)':>6}  "
                       f"{'SF':>7}  {'P_local':>9}  {'±':>7}  Scheme")
                print(hdr)
                print("    " + "-" * 62)
                for sf in ["St", "Lm", "LSDn"]:
                    sf_pts = [p for p in pts if p.scheme == sf]
                    unit = "(d'less)" if sf == "LSDn" else "(at/g/yr)"
                    print(f"    ── {sf} {unit} " + "─" * 30)
                    for p in sf_pts:
                        print(f"    {p.target_name:<16}  {p.elv:>6.0f}  "
                              f"{p.SF:>7.4f}  {p.P_local:>9.4f}  "
                              f"{p.dP_local:>7.4f}")
            else:
                t = self.targets[0]
                P_atm_val = self._site_pressure(t)
                print(f"    Location: {t.lat:.4f}°N, {t.lon:.4f}°W, {t.elv:.0f} m")
                print(f"    ERA-40 pressure: {P_atm_val:.2f} hPa")
                print(f"    {'Scheme':<6}  {'SF':>7}  {'P_local':>10}  {'±':>8}  Unit")
                print("    " + "-" * 46)
                for p in pts:
                    unit = "dim'less" if p.scheme == "LSDn" else "at/g/yr"
                    print(f"    {p.scheme:<6}  {p.SF:>7.4f}  "
                          f"{p.P_local:>10.4f}  {p.dP_local:>8.4f}  {unit}")
            print()

        # ── Global vs local comparison ───────────────────────────────────────
        global_cal = {c.scheme: c for c in self._cal_results if c.n_sites == 0}
        local_cal  = {c.scheme: c for c in self._cal_results if c.n_sites > 0}
        if global_cal and local_cal and not multi:
            t = self.targets[0]
            g_pts = {p.scheme: p for p in self._point_results if p.source == next(iter(global_cal.values())).source}
            l_pts = {p.scheme: p for p in self._point_results if p.source == next(iter(local_cal.values())).source}
            print("  Global vs. local comparison at this site:")
            print(f"    {'Scheme':<6}  {'Global':>10}  {'Local':>10}  {'Diff':>8}  {'Rel %':>7}")
            print("    " + "-" * 48)
            for sf in ["St", "Lm", "LSDn"]:
                if sf in g_pts and sf in l_pts:
                    g = g_pts[sf].P_local
                    l = l_pts[sf].P_local
                    print(f"    {sf:<6}  {g:>10.4f}  {l:>10.4f}  "
                          f"{l-g:>+8.4f}  {100*(l-g)/g:>+7.2f}%")
            print()

        self._print_corrections_checklist()
        return self

    def _print_corrections_checklist(self):
        """Print a summary of which corrections are applied and which are not."""
        t0 = self.targets[0]
        has_erosion  = any(t.erosion > 0 for t in self.targets)
        has_shielding = any(t.shield < 1.0 for t in self.targets)

        print("  ── Corrections checklist ─────────────────────────────────────")
        print("  [OK] Sample thickness correction applied")
        print(f"  [OK] Erosion rate: "
              + ("user-supplied (> 0)" if has_erosion else "0 cm/yr (assumed stable surface)"))
        print(f"  [OK] Topographic shielding: "
              + ("user-supplied scalar" if has_shielding else "1.0 (no shielding assumed)"))
        print()
        print("  [  ] Snow shielding          — not applied; assess for seasonal snow cover")
        print("  [  ] Vegetation / soil cover  — not applied; assess for forested / soil-mantled sites")
        print("  [  ] Surface geometry / dip   — assumes horizontal planar surface")
        print("  [  ] Boulder stability        — assumes no tilting or rolling since deposition")
        print("  [  ] Rock exfoliation         — episodic spalling not distinguished from steady erosion")
        print("  [  ] Paleoelevation / rebound — uses current elevation throughout")
        print("  [  ] Submergence / water cover — not applied")
        print("  [  ] Ice cover history        — not applied; zero production assumed during glaciation")
        print("  [  ] Inheritance / pre-exposure — no check; assess from field context")
        print("  [  ] Nucleogenic He-3 / Ne-21 — no correction applied")
        print("  [  ] He-3 diffusive loss      — not modelled")
        print()
        print("  See README 'Before computing exposure ages' for details on each item.")
        print()

    # ── Optional dating ──────────────────────────────────────────────────────

    def date(self, nuclide_text: str, use_local: bool = True,
             result_type: str = "long") -> dict:
        """
        Compute an exposure age for a sample at the target location.

        Parameters
        ----------
        nuclide_text : str
            One or more nuclide lines in v3 format, e.g.:
            "SITE Be-10 quartz 60000 1800 07KNSTD;"
            The sample name in nuclide_text must match the name used here.
        use_local : bool
            If True and a local calibration exists, use it. Otherwise global.
        result_type : str
            'long' for all three scaling schemes (default), 'short' for St only.
        """
        t = self.target
        sample_line = (f"{t.name} {t.lat} {t.lon} {t.elv} std "
                       f"{t.thick} {t.rho} {t.shield} {t.erosion} {t.yr};")
        # Ensure nuclide lines use the same sample name
        nuc_lines = "\n".join(
            re.sub(r"^\S+", t.name, line) if line.strip() else line
            for line in nuclide_text.strip().split("\n")
        )
        full_text = sample_line + "\n" + nuc_lines

        consts_arg = self._consts if (use_local and self._consts) else None
        data   = parse_v3_input(full_text)
        result = get_ages(data, control={"result_type": result_type},
                          consts=consts_arg)
        n      = result["n"]

        source_label = "local calibration" if (use_local and self._consts) else "global default"
        print(f"\n  Exposure age ({source_label}):")
        print(f"  {'Scheme':<6}  {'Age (yr)':>10}  {'±int':>8}  {'±ext':>8}")
        print("  " + "-" * 38)
        for sf in ["St", "Lm", "LSDn"]:
            if f"t_{sf}" in n:
                t_age = n[f"t_{sf}"][0]
                di    = n[f"delt_int_{sf}"][0]
                de    = n[f"delt_ext_{sf}"][0]
                print(f"  {sf:<6}  {t_age:>10,.0f}  {di:>8,.0f}  {de:>8,.0f}")
        return result

    def date_all(self, use_local: bool = True,
                 result_type: str = "long") -> dict:
        """
        Date all samples loaded from a full v3-field CSV.

        Requires that the workflow was created with from_csv() from a CSV that
        includes nuclide columns (nuclide, mineral, concentration, uncertainty,
        standard).

        When a local calibration is available (after calling calibrate()), both
        global-default and locally calibrated ages are printed side by side so
        the effect of the local production rate is immediately visible.

        Returns the locally calibrated get_ages() result when use_local=True
        and local data exist, otherwise the global-default result.
        """
        if not self._v3_text:
            raise ValueError(
                "No nuclide data available. Load a full v3-field CSV "
                "(with nuclide, mineral, concentration, uncertainty, standard "
                "columns) via from_csv()."
            )

        has_local = use_local and self._consts is not None

        # Always compute global ages
        result_g = get_ages(parse_v3_input(self._v3_text),
                            control={"result_type": result_type})
        n_g = result_g["n"]

        if has_local:
            result_l = get_ages(parse_v3_input(self._v3_text),
                                control={"result_type": result_type},
                                consts=self._consts)
            n_l = result_l["n"]

            # Identify local calibration source label
            local_cals = [c for c in self._cal_results if c.n_sites > 0]
            cal_label  = local_cals[0].source if local_cals else "local calibration"

            print(f"\n  Exposure ages — global default vs. {cal_label}:")
            print(f"  {'Sample':<16}  {'Nuclide':<14}  {'Scheme':<6}  "
                  f"{'Global (yr)':>12}  {'Local (yr)':>11}  {'Shift':>14}")
            print("  " + "-" * 82)

            for i, sname in enumerate(n_g["sample_name"]):
                nuc = n_g["nuclide"][i]
                for sf in ["St", "Lm", "LSDn"]:
                    t_key = f"t_{sf}"
                    if t_key in n_g and i < len(n_g[t_key]):
                        t_g   = n_g[t_key][i]
                        t_l   = n_l[t_key][i]
                        shift = t_l - t_g
                        pct   = 100.0 * shift / t_g if t_g else 0.0
                        print(f"  {sname:<16}  {nuc:<14}  {sf:<6}  "
                              f"{t_g:>12,.0f}  {t_l:>11,.0f}  "
                              f"{shift:>+8,.0f} ({pct:>+.1f}%)")

            return result_l

        else:
            print(f"\n  Exposure ages (global default):")
            print(f"  {'Sample':<16}  {'Nuclide':<14}  "
                  f"{'Scheme':<6}  {'Age (yr)':>10}  {'±int':>8}  {'±ext':>8}")
            print("  " + "-" * 70)

            for i, sname in enumerate(n_g["sample_name"]):
                nuc = n_g["nuclide"][i]
                for sf in ["St", "Lm", "LSDn"]:
                    t_key = f"t_{sf}"
                    if t_key in n_g and i < len(n_g[t_key]):
                        t_age = n_g[t_key][i]
                        di    = n_g[f"delt_int_{sf}"][i]
                        de    = n_g[f"delt_ext_{sf}"][i]
                        print(f"  {sname:<16}  {nuc:<14}  "
                              f"{sf:<6}  {t_age:>10,.0f}  {di:>8,.0f}  {de:>8,.0f}")

            return result_g
