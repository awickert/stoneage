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

    # Or from a CSV with columns  name, lat, lon, elevation
    wf = ProductionRateWorkflow.from_csv("sites.csv")

    wf.find_calibration_data()   # fetch ICE-D, rank by distance
    wf.calibrate()               # compute local production rates
    wf.report()                  # print transparent summary

    # Optional dating
    wf.date("SITE Be-10 quartz 60000 1800 07KNSTD;")
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
from .input_parser import parse_v3_input
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
class ProductionRateResult:
    scheme: str
    P_ref_SLHL: float   # reference production rate (at/g/yr or dimensionless for LSDn)
    dP_ref: float
    SF: float           # site scaling factor
    P_local: float      # production rate at target site
    dP_local: float
    source: str         # "global default" or "ICE-D local (N sites, X km)"
    n_sites: int = 0
    max_dist_km: float = 0.0
    chi2_p: Optional[float] = None


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

    Parameters
    ----------
    lat, lon, elv : float
        Target location (decimal degrees; west negative) and elevation (m).
    thick : float
        Sample thickness (cm). Default 2 cm.
    rho : float
        Sample density (g/cm³). Default 2.65.
    shield : float
        Topographic shielding factor (0–1). Default 1.0.
    name : str
        Optional label for the target site.
    """

    def __init__(self, lat: float, lon: float, elv: float,
                 thick: float = 2.0, rho: float = 2.65,
                 shield: float = 1.0, erosion: float = 0.0,
                 yr: int = 2010, name: str = "TARGET"):
        self.target = TargetSite(name=name, lat=lat, lon=lon, elv=elv,
                                 thick=thick, rho=rho, shield=shield,
                                 erosion=erosion, yr=yr)
        self._cal_samples: List[CalibrationSite] = []
        self._selected: List[CalibrationSite] = []
        self._results: List[ProductionRateResult] = []
        self._consts: Optional[Constants] = None

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "ProductionRateWorkflow":
        """
        Create from a CSV file.

        The CSV must have columns: name, lat, lon, elevation
        Optional columns: thickness, density, shielding, erosion, year

        If multiple rows, the centroid is used as the target location and
        individual rows are stored for later dating.
        """
        import csv
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            raise ValueError(f"No data rows in {path}")

        # Use centroid of all points as the target for calibration
        lats = [float(r["lat"])       for r in rows]
        lons = [float(r["lon"])       for r in rows]
        elvs = [float(r["elevation"]) for r in rows]
        lat_c = float(np.mean(lats))
        lon_c = float(np.mean(lons))
        elv_c = float(np.mean(elvs))

        wf = cls(lat=lat_c, lon=lon_c, elv=elv_c,
                 name=Path(path).stem, **kwargs)
        wf._csv_rows = rows
        wf._csv_path = path
        return wf

    # ── Site physics ────────────────────────────────────────────────────────

    def _site_pressure(self) -> float:
        t = self.target
        if t.elv >= 0:
            return float(ERA40atm([t.lat], [t.lon], [t.elv])[0])
        return float(antatm([t.elv])[0])

    def _scaling_factor(self, pressure: float) -> float:
        return float(stone2000([self.target.lat], [pressure], Fsp=1.0)[0])

    def _thickness_correction(self) -> float:
        return float(thickness([self.target.thick], [Lsp()], [self.target.rho])[0])

    def _muon_rate(self, pressure: float, nuclide_idx: int = 3) -> float:
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
        Fetch ICE-D calibration data and rank by distance to target.

        Parameters
        ----------
        dataset_id : int
            ICE-D dataset to query (default 20 = all Be-10).
        max_dist_km : float
            Maximum distance to include (km). Default 1000 km.
        min_samples : int
            Warn if fewer than this many samples are found within max_dist_km.
        """
        desc = ICED_DATASETS.get(dataset_id, f"ICE-D dataset {dataset_id}")
        print(f"Fetching {desc} from ICE-D...")
        v3_text = _fetch_iced_dataset(dataset_id)
        all_samples = _parse_iced_into_samples(v3_text)
        print(f"  {len(all_samples)} total calibration samples retrieved.")

        t = self.target
        for s in all_samples:
            s.dist_km = _haversine(t.lat, t.lon, s.lat, s.lon)
        all_samples.sort(key=lambda s: s.dist_km)

        nearby = [s for s in all_samples if s.dist_km <= max_dist_km]
        self._cal_samples = nearby

        print(f"\nNearest calibration samples to {t.name} "
              f"({t.lat:.3f}°N, {t.lon:.3f}°W, {t.elv:.0f} m):")
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

    def calibrate(self, include_global: bool = True) -> "ProductionRateWorkflow":
        """
        Compute local and (optionally) global reference production rates.

        Results are stored in self._results and printed via report().
        """
        consts = make_consts()
        idx    = consts.nuclides.index("N10quartz")
        P_atm  = self._site_pressure()
        SF     = self._scaling_factor(P_atm)
        tc     = self._thickness_correction()
        P_mu   = self._muon_rate(P_atm)
        shield = self.target.shield

        self._results = []
        self._consts  = None  # reset

        # ── Global defaults ──────────────────────────────────────────────────
        if include_global:
            for sf_name, ref_key, dref_key in [
                ("St",   "refP_St",   "delrefP_St"),
                ("Lm",   "refP_Lm",   "delrefP_Lm"),
                ("LSDn", "refP_LSDn", "delrefP_LSDn"),
            ]:
                P_ref  = float(getattr(consts, ref_key)[idx])
                dP_ref = float(getattr(consts, dref_key)[idx])
                P_loc  = P_ref * SF * tc * shield + P_mu
                dP_loc = dP_ref * SF * tc * shield
                self._results.append(ProductionRateResult(
                    scheme=sf_name, P_ref_SLHL=P_ref, dP_ref=dP_ref,
                    SF=SF, P_local=P_loc, dP_local=dP_loc,
                    source="global default (Borchers et al. 2016)",
                    n_sites=0, max_dist_km=0.0,
                ))

        # ── Local calibration from selected ICE-D samples ───────────────────
        if self._selected:
            v3_block = "\n".join(s.v3_block for s in self._selected)
            data   = parse_v3_input(v3_block)
            result = get_ages(data, control={"cal": 1, "result_type": "long"})
            n      = result["n"]
            max_d  = max(s.dist_km for s in self._selected)
            n_samp = len(self._selected)
            source = (f"ICE-D local calibration "
                      f"({n_samp} sample{'s' if n_samp>1 else ''}, "
                      f"max {max_d:.0f} km away)")

            for sf_name, ref_key, dref_key in [
                ("St",   "refP_St",   "delrefP_St"),
                ("Lm",   "refP_Lm",   "delrefP_Lm"),
                ("LSDn", "refP_LSDn", "delrefP_LSDn"),
            ]:
                P_vals  = n[f"calc_P_{sf_name}"]
                dP_vals = n[f"calc_delP_{sf_name}"]
                ok = (P_vals > 0) & (dP_vals > 0)
                if ok.sum() == 0:
                    continue
                P_ref, dP_ref = _ewmean(P_vals[ok], dP_vals[ok])

                # chi-squared of calibration
                ewm, _ = _ewmean(P_vals[ok], dP_vals[ok])
                chi2 = float(np.sum(((P_vals[ok] - ewm) / dP_vals[ok]) ** 2))
                dof  = int(ok.sum()) - 1
                p_chi2 = _chi2_p(chi2, dof) if dof > 0 else None

                # scatter uncertainty (larger of formal and scatter)
                scatter = float(np.std(P_vals[ok], ddof=1)) if ok.sum() > 1 else dP_ref
                dP_ref_final = max(dP_ref, scatter)

                P_loc  = P_ref * SF * tc * shield + P_mu
                dP_loc = dP_ref_final * SF * tc * shield

                self._results.append(ProductionRateResult(
                    scheme=sf_name, P_ref_SLHL=float(P_ref),
                    dP_ref=float(dP_ref_final),
                    SF=SF, P_local=float(P_loc), dP_local=float(dP_loc),
                    source=source, n_sites=n_samp, max_dist_km=max_d,
                    chi2_p=p_chi2,
                ))

            # Store locally calibrated consts for optional dating
            local_consts = make_consts()
            for sf_name, ref_key in [("St","refP_St"),("Lm","refP_Lm"),("LSDn","refP_LSDn")]:
                for r in self._results:
                    if r.scheme == sf_name and r.n_sites > 0:
                        getattr(local_consts, ref_key)[idx] = r.P_ref_SLHL
            self._consts = local_consts

        return self

    # ── Reporting ────────────────────────────────────────────────────────────

    def report(self) -> "ProductionRateWorkflow":
        """Print a transparent summary of the production rate results."""
        t   = self.target
        P_atm = self._site_pressure()
        SF    = self._scaling_factor(P_atm)
        tc    = self._thickness_correction()

        print()
        print("=" * 72)
        print("  PRODUCTION RATE REPORT")
        print("=" * 72)
        print(f"  Target site:      {t.name}")
        print(f"  Location:         {t.lat:.4f}°N, {t.lon:.4f}°W")
        print(f"  Elevation:        {t.elv:.0f} m")
        print(f"  ERA-40 pressure:  {P_atm:.2f} hPa")
        print(f"  Stone2000 SF:     {SF:.4f}  (spallation only, Fsp=1)")
        print(f"  Thickness corr:   {tc:.5f}  "
              f"({t.thick} cm, {t.rho} g/cm³)")
        if t.shield < 1.0:
            print(f"  Shielding:        {t.shield:.4f}")
        print()
        print("  Recommended scaling scheme: LSDn (Borchers et al. 2016,")
        print("  Lifton et al. 2014). St and Lm shown for comparison.")
        print()

        # Group by source
        sources = []
        seen = set()
        for r in self._results:
            if r.source not in seen:
                sources.append(r.source)
                seen.add(r.source)

        for source in sources:
            rr = [r for r in self._results if r.source == source]
            print(f"  Source: {source}")
            if rr[0].n_sites > 0 and rr[0].chi2_p is not None:
                print(f"    Chi-squared p-value: {rr[0].chi2_p:.3f}  "
                      f"({'consistent' if rr[0].chi2_p > 0.05 else 'scattered'})")
            print()
            print(f"    {'Scheme':<6}  {'P_ref (SLHL)':>14}  "
                  f"{'P_local':>10}  {'±':>8}  {'Unit'}")
            print("    " + "-" * 56)
            for r in rr:
                unit = "dim'less" if r.scheme == "LSDn" else "at/g/yr"
                slhl_str = f"{r.P_ref_SLHL:.4f} ± {r.dP_ref:.4f}"
                print(f"    {r.scheme:<6}  {slhl_str:>14}  "
                      f"{r.P_local:>10.4f}  {r.dP_local:>8.4f}  {unit}")
            print()

        # Comparison table if both global and local present
        global_r = {r.scheme: r for r in self._results if r.n_sites == 0}
        local_r  = {r.scheme: r for r in self._results if r.n_sites > 0}
        if global_r and local_r:
            print("  Local vs. global comparison (P_local):")
            print(f"    {'Scheme':<6}  {'Global':>10}  {'Local':>10}  {'Diff':>8}  {'Rel %':>7}")
            print("    " + "-" * 46)
            for sf in ["St", "Lm", "LSDn"]:
                if sf in global_r and sf in local_r:
                    g = global_r[sf].P_local
                    l = local_r[sf].P_local
                    print(f"    {sf:<6}  {g:>10.4f}  {l:>10.4f}  "
                          f"{l-g:>+8.4f}  {100*(l-g)/g:>+7.2f}%")
            print()

        return self

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

        consts = self._consts if (use_local and self._consts) else None
        data   = parse_v3_input(full_text)
        result = get_ages(data, control={"result_type": result_type},
                          consts=consts)
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
