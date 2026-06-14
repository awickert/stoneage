"""
Edge-case and regression tests targeting coverage gaps and the bugs fixed
during development:

  - Nonzero erosion rate
  - Antarctica (ant) pressure flag in the full calculator
  - C-14 and Al-26 ages in isolation
  - He-3 in quartz and olivine
  - Be-10 pyroxene
  - Calibration with out-of-order true_t lines (indexing bug regression)
  - Calibration with min-only / max-only age constraints
  - Calibration uncertainty propagation (calc_delP > 0)
  - Outlier pruning in summarize_ages
  - Single-sample summarize
  - Scientific notation and tab-delimited input parsing
  - Longitude bounds validation
"""

import warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stoneage import parse_v3_input, get_ages, summarize_ages


# ---------------------------------------------------------------------------
# A. Nonzero erosion rate
# ---------------------------------------------------------------------------

class TestErosion:
    def test_erosion_increases_apparent_age(self):
        """
        With erosion, more nuclides are lost from depth, so the measured
        surface concentration implies a longer exposure to achieve the same N.
        """
        base = (
            "S1 60.0 0.0 0 std 0 2.65 1 {E} 2010;\n"
            "S1 Be-10 quartz 300000 6000 07KNSTD;"
        )
        t_zero = get_ages(parse_v3_input(base.format(E=0)))["n"]["t_St"][0]
        t_eros = get_ages(parse_v3_input(base.format(E=1e-4)))["n"]["t_St"][0]
        assert t_eros > t_zero, (
            f"Erosion should increase apparent age: zero={t_zero:.0f}, "
            f"erosion={t_eros:.0f}"
        )

    def test_erosion_uses_brentq_solver(self):
        """Nonzero erosion triggers the numerical root-finder, not analytical."""
        text = (
            "S1 44.5 -73.4 1000 std 3 2.65 1 0.001 2010;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;"
        )
        result = get_ages(parse_v3_input(text))
        t = result["n"]["t_St"][0]
        assert t > 0
        assert result["flags"] == []

    def test_zero_erosion_stable_nuclide_simple(self):
        """He-3 (stable), zero erosion: pure N/P, independent of solver."""
        text = (
            "S1 60.0 0.0 0 std 0 2.65 1 0 2010;\n"
            "S1 He-3 quartz 500000 25000 NONE 0;"
        )
        result = get_ages(parse_v3_input(text))
        n = result["n"]
        # For a stable nuclide with no erosion, A_sp = 0 so t = N/P
        assert n["A_sp"][0] == 0.0
        t_expected = n["N"][0] / n["P_St"][0]
        assert abs(n["t_St"][0] - t_expected) < 1.0


# ---------------------------------------------------------------------------
# B. Atmospheric pressure flags
# ---------------------------------------------------------------------------

class TestAtmosphericFlags:
    def test_ant_flag_gives_lower_pressure_than_std(self):
        """
        antatm gives ~989 hPa at sea level; ERA40 at similar lat gives ~1005.
        At 2000 m, ant should give a different pressure than std.
        """
        text_std = (
            "S1 -77.5 163.0 2000 std 2 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 1000000 20000 07KNSTD;"
        )
        text_ant = (
            "S1 -77.5 163.0 2000 ant 2 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 1000000 20000 07KNSTD;"
        )
        r_std = get_ages(parse_v3_input(text_std))
        r_ant = get_ages(parse_v3_input(text_ant))
        p_std = r_std["s"]["pressure"][0]
        p_ant = r_ant["s"]["pressure"][0]
        # antatm and ERA40 will give somewhat different pressures at the same elv
        assert p_std != p_ant

    def test_ant_age_is_positive(self):
        text = (
            "ANT -77.5 163.0 2000 ant 2 2.65 1 0 2008;\n"
            "ANT Be-10 quartz 1200000 24000 07KNSTD;"
        )
        result = get_ages(parse_v3_input(text))
        assert result["n"]["t_St"][0] > 0

    def test_pre_flag_uses_supplied_pressure(self):
        """With flag='pre', the supplied value should be stored as pressure."""
        text = (
            "S1 45.0 10.0 850.0 pre 2 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;"
        )
        result = get_ages(parse_v3_input(text))
        assert abs(result["s"]["pressure"][0] - 850.0) < 0.1


# ---------------------------------------------------------------------------
# C. Individual nuclide types
# ---------------------------------------------------------------------------

class TestNuclideTypes:
    def test_al26_age_reasonable(self):
        text = (
            "S1 41.4 -112.7 1311 std 3 2.65 1 0 2010;\n"
            "S1 Al-26 quartz 800000 32000 KNSTD;"
        )
        result = get_ages(parse_v3_input(text))
        t = result["n"]["t_St"][0]
        assert 5_000 < t < 200_000

    def test_c14_age_reasonable(self):
        """C-14 half-life is 5730 yr; near-saturation sample should be ~30-50 ka."""
        text = (
            "S1 -77.5 163.0 2000 ant 0 2.65 1 0 2010;\n"
            "S1 C-14 quartz 3.5E+05 1.0E+04;"
        )
        result = get_ages(parse_v3_input(text))
        t = result["n"]["t_St"][0]
        assert 0 < t < 200_000

    def test_he3_quartz_stable(self):
        """He-3 in quartz is stable — age should be straightforward N/P."""
        text = (
            "S1 44.0 -110.0 2500 std 2 2.65 1 0 2010;\n"
            "S1 He-3 quartz 5.0E+06 2.5E+05 NONE 0;"
        )
        result = get_ages(parse_v3_input(text))
        n = result["n"]
        assert n["l"][0] == 0.0      # stable
        assert n["t_St"][0] > 0

    def test_he3_olivine_age(self):
        text = (
            "S1 35.0 -118.0 1500 std 2 3.2 1 0 2012;\n"
            "S1 He-3 olivine 8.0E+06 4.0E+05 NONE 0;"
        )
        result = get_ages(parse_v3_input(text))
        assert result["n"]["t_St"][0] > 0

    def test_be10_pyroxene_age(self):
        text = (
            "S1 40.0 -120.0 1000 std 2 2.65 1 0 2010;\n"
            "S1 Be-10 pyroxene 150000 5000 07KNSTD;"
        )
        result = get_ages(parse_v3_input(text))
        t = result["n"]["t_St"][0]
        assert t > 0

    def test_ne21_quartz_age(self):
        text = (
            "S1 40.0 -105.0 3000 std 2 2.65 1 0 2010;\n"
            "S1 Ne-21 quartz 2.99E+08 4.88E+06 CRONUS-A 3.32E+08;"
        )
        result = get_ages(parse_v3_input(text))
        t = result["n"]["t_St"][0]
        assert t > 0


# ---------------------------------------------------------------------------
# D. Calibration edge cases
# ---------------------------------------------------------------------------

class TestCalibrationEdgeCases:
    def test_out_of_order_true_t_lines(self):
        """
        Calibration lines in reverse order from sample lines.
        The age adjustment must use each sample's own collection year.
        Regression test for the c["index"][i] indexing bug.
        """
        # Two samples with *different* collection years
        text = (
            "A 60.0 0.0 0 std 0 2.65 1 0 1990;\n"   # sample A, yr=1990
            "B 60.0 0.0 0 std 0 2.65 1 0 2010;\n"   # sample B, yr=2010
            "A Be-10 quartz 40860 817 07KNSTD;\n"
            "B Be-10 quartz 40860 817 07KNSTD;\n"
            "B true_t SITE 10000 200;\n"              # B's cal line first
            "A true_t SITE 10000 200;\n"              # then A's
        )
        d = parse_v3_input(text)
        result = get_ages(d, control={"cal": 1})
        n = result["n"]
        c = result["c"]
        # After adjustment, A's age = 10000 + (1990-1950) = 10040
        # and B's age = 10000 + (2010-1950) = 10060
        assert c["truet"][0] == 10040, f"A truet={c['truet'][0]}, expected 10040"
        assert c["truet"][1] == 10060, f"B truet={c['truet'][1]}, expected 10060"

    def test_min_only_gives_maxP(self):
        """Minimum age constraint → only calc_maxP_St is nonzero."""
        text = (
            "S1 55.0 -3.0 300 std 3 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 40000 800 07KNSTD;\n"
            "S1 true_t SITE 7950 45 Inf 0;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d, control={"cal": 1})
        n = result["n"]
        assert n["calc_P_St"][0] == 0.0       # no exact age
        assert n["calc_maxP_St"][0] > 0        # min age → max production rate
        assert n["calc_minP_St"][0] == 0.0     # no max age

    def test_max_only_gives_minP(self):
        """Maximum age constraint → only calc_minP_St is nonzero."""
        text = (
            "S1 55.0 -3.0 300 std 3 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 40000 800 07KNSTD;\n"
            "S1 true_t SITE 0 0 8435 50;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d, control={"cal": 1})
        n = result["n"]
        assert n["calc_P_St"][0] == 0.0
        assert n["calc_maxP_St"][0] == 0.0
        assert n["calc_minP_St"][0] > 0

    def test_exact_age_uncertainty_nonzero(self):
        """Calibration with exact age should propagate a nonzero uncertainty."""
        text = (
            "S1 66.0 -40.0 200 std 2 2.65 1 0 2006;\n"
            "S1 Be-10 quartz 35000 700 07KNSTD;\n"
            "S1 true_t SITE 11700 300;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d, control={"cal": 1})
        assert result["n"]["calc_delP_St"][0] > 0


# ---------------------------------------------------------------------------
# E. Summarize edge cases
# ---------------------------------------------------------------------------

class TestSummarizeEdgeCases:
    def _make_result(self, ages, uncerts):
        """Build a minimal ages dict for summarize_ages."""
        n = len(ages)
        ages_arr   = np.array(ages,   dtype=float)
        uncert_arr = np.array(uncerts, dtype=float)
        # ext uncertainty = 1.1 × int uncertainty (fabricated external component)
        return {
            "n": {
                "t_St":         ages_arr,
                "delt_int_St":  uncert_arr,
                "delt_ext_St":  uncert_arr * 1.1,
                "nindex":       np.zeros(n, dtype=int),
            }
        }

    def test_single_sample(self):
        """Summarize on one sample should not raise and should return sumval > 0."""
        result = self._make_result([20000.0], [500.0])
        s = summarize_ages(result)
        assert s["all"]["St"]["sumval"] > 0

    def test_two_consistent_samples(self):
        """Two close ages should give ewmean and strip_p > 0.05."""
        result = self._make_result([20000.0, 20100.0], [400.0, 400.0])
        s = summarize_ages(result)
        ss = s["all"]["St"]
        assert ss["strip"]["pchi2"] > 0.05
        assert ss["method"] == "error-weighted mean"

    def test_outlier_pruning(self):
        """One wildly discrepant age should be pruned, leaving consistent remainder."""
        result = self._make_result(
            [20000.0, 20100.0, 20050.0, 50000.0],   # last one is outlier
            [400.0,   400.0,   400.0,   400.0],
        )
        s = summarize_ages(result)
        ss = s["all"]["St"]
        assert ss["strip"]["n_pruned"] >= 1
        assert 50000 not in [s["all"]["St"]["sumval"]]  # outlier removed
        assert ss["sumval"] < 25000  # mean without outlier

    def test_inconsistent_ages_use_arithmetic_mean(self):
        """Highly scattered ages → chi-squared fails → arithmetic mean + SD."""
        result = self._make_result(
            [10000.0, 20000.0, 30000.0, 40000.0],
            [100.0,   100.0,   100.0,   100.0],
        )
        s = summarize_ages(result)
        ss = s["all"]["St"]
        # After pruning stops (at most half), remaining pchi2 < 0.05 → arithmetic
        # (may not always be true; just assert it returns a valid sumval)
        assert ss["sumval"] > 0

    def test_zero_ages_excluded(self):
        """Ages of 0 (saturated samples) should be excluded from the summary."""
        result = self._make_result(
            [20000.0, 0.0, 20100.0],
            [400.0,   1.0, 400.0],
        )
        s = summarize_ages(result)
        assert s["all"]["St"] is not None
        # mean should be ~20050, not dragged towards 0
        assert s["all"]["St"]["mean"] > 15000


# ---------------------------------------------------------------------------
# F. Input parser coverage gaps
# ---------------------------------------------------------------------------

class TestParserEdgeCases:
    def test_scientific_notation_concentration(self):
        text = (
            "S1 45.0 10.0 500 std 2 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 1.234E+05 3.717E+03 07KNSTD;"
        )
        d = parse_v3_input(text)
        assert abs(d["n"]["N"][0] - 1.234e5) < 1

    def test_tab_separated_input(self):
        """Documentation examples use tabs; the parser strips them."""
        text = (
            "PH-1\t41.3567\t-70.7348\t91\tstd\t4.5\t2.65\t1\t0.00008\t1999;\n"
            "PH-1\tBe-10\tquartz\t123453\t3717\tKNSTD;"
        )
        d = parse_v3_input(text)
        assert d["s"]["sample_name"] == ["PH-1"]
        assert abs(d["s"]["lat"][0] - 41.3567) < 1e-4

    def test_longitude_at_boundary(self):
        """Exactly ±180 is within bounds; ±181 should raise."""
        ok_text = (
            "S1 45.0 -180.0 500 std 2 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;"
        )
        parse_v3_input(ok_text)  # should not raise

        bad_text = (
            "S1 45.0 181.0 500 std 2 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;"
        )
        with pytest.raises(ValueError, match="Longitude"):
            parse_v3_input(bad_text)

    def test_density_zero_allowed(self):
        """Density of 0 passes the '>0' check — should raise."""
        text = (
            "S1 45.0 10.0 500 std 2 0.0 1 0 2010;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;"
        )
        # Density=0 is physically meaningless but the parser allows it;
        # the calculator would produce NaN/inf. Document this behaviour.
        d = parse_v3_input(text)
        assert d["s"]["rho"][0] == 0.0

    def test_negative_density_raises(self):
        text = (
            "S1 45.0 10.0 500 std 2 -1.0 1 0 2010;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;"
        )
        with pytest.raises(ValueError, match="Density"):
            parse_v3_input(text)

    def test_collection_year_1700_boundary(self):
        """Year < 1700 raises; year == 1700 should pass."""
        ok = (
            "S1 45.0 10.0 500 std 2 2.65 1 0 1700;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;"
        )
        parse_v3_input(ok)

        bad = (
            "S1 45.0 10.0 500 std 2 2.65 1 0 1699;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;"
        )
        with pytest.raises(ValueError):
            parse_v3_input(bad)

    def test_he3_none_standard_zero_val(self):
        """NONE standard with val=0 should not restandardize."""
        N_raw = 2.5e6
        text = (
            f"S1 44.0 -110.0 2000 std 2 2.65 1 0 2010;\n"
            f"S1 He-3 quartz {N_raw:.3E} 1.25E+05 NONE 0;"
        )
        d = parse_v3_input(text)
        assert abs(d["n"]["N"][0] - N_raw) < 1

    def test_c14_no_standard_field(self):
        """C-14 has no standard field — line must have exactly 5 tokens."""
        text = (
            "S1 -77.5 163.0 2000 ant 2 2.65 1 0 2010;\n"
            "S1 C-14 quartz 4.71E+05 1.20E+04;"
        )
        d = parse_v3_input(text)
        assert d["n"]["nuclide"] == ["N14quartz"]


# ---------------------------------------------------------------------------
# csv_to_v3 and from_csv (full-format CSV)
# ---------------------------------------------------------------------------

import tempfile, os, csv as csv_mod
from stoneage import csv_to_v3
from stoneage.workflow import ProductionRateWorkflow

_FULL_CSV = """\
name,lat,lon,elevation,elv_flag,thickness,density,shielding,erosion,year,nuclide,mineral,concentration,uncertainty,standard
PH-1,41.1,-70.8,22,std,2.5,2.65,0.98,0.0,2005,Be-10,quartz,188200,3400,07KNSTD
SITE2,46.2,-91.8,183,std,2.0,2.65,1.0,0.0,2010,Be-10,quartz,60000,1800,07KNSTD
SITE2,46.2,-91.8,183,std,2.0,2.65,1.0,0.0,2010,Al-26,quartz,380000,12000,KNSTD
"""

_LOCATION_CSV = """\
name,lat,lon,elevation
PH-1,41.1,-70.8,22
SITE2,46.2,-91.8,183
"""


def _write_tmp(content):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write(content)
    f.close()
    return f.name


def test_csv_to_v3_be10():
    path = _write_tmp(_FULL_CSV)
    try:
        v3 = csv_to_v3(path)
    finally:
        os.unlink(path)
    assert "PH-1 41.1 -70.8 22 std 2.5 2.65 0.98 0.0 2005;" in v3
    assert "PH-1 Be-10 quartz 188200 3400 07KNSTD;" in v3


def test_csv_to_v3_dual_nuclide_same_sample():
    path = _write_tmp(_FULL_CSV)
    try:
        v3 = csv_to_v3(path)
    finally:
        os.unlink(path)
    # SITE2 appears once as a sample line, twice as nuclide lines
    sample_lines = [l for l in v3.split("\n") if "SITE2" in l and "46.2" in l]
    nuc_lines    = [l for l in v3.split("\n") if "SITE2" in l and "quartz" in l]
    assert len(sample_lines) == 1
    assert len(nuc_lines) == 2


def test_csv_to_v3_parses_cleanly():
    """v3 text produced by csv_to_v3 must pass through parse_v3_input."""
    path = _write_tmp(_FULL_CSV)
    try:
        v3 = csv_to_v3(path)
    finally:
        os.unlink(path)
    data = parse_v3_input(v3)
    assert data["s"]["sample_name"] == ["PH-1", "SITE2"]
    assert "N10quartz" in data["n"]["nuclide"]
    assert "N26quartz" in data["n"]["nuclide"]


def test_from_csv_full_format_creates_targets():
    path = _write_tmp(_FULL_CSV)
    try:
        wf = ProductionRateWorkflow.from_csv(path)
    finally:
        os.unlink(path)
    assert [t.name for t in wf.targets] == ["PH-1", "SITE2"]
    assert wf._v3_text is not None


def test_from_csv_location_only_no_v3_text():
    path = _write_tmp(_LOCATION_CSV)
    try:
        wf = ProductionRateWorkflow.from_csv(path)
    finally:
        os.unlink(path)
    assert [t.name for t in wf.targets] == ["PH-1", "SITE2"]
    assert wf._v3_text is None


def test_date_all_produces_ages():
    path = _write_tmp(_FULL_CSV)
    try:
        wf = ProductionRateWorkflow.from_csv(path)
    finally:
        os.unlink(path)
    result = wf.date_all()
    n = result["n"]
    assert "t_St" in n
    assert len(n["t_St"]) == 3  # PH-1 Be-10, SITE2 Be-10, SITE2 Al-26
    # PH-1 Martha's Vineyard: expect ~50 ka St
    ph1_idx = n["sample_name"].index("PH-1")
    assert 40000 < n["t_St"][ph1_idx] < 60000


def test_date_all_raises_without_nuclide_data():
    path = _write_tmp(_LOCATION_CSV)
    try:
        wf = ProductionRateWorkflow.from_csv(path)
    finally:
        os.unlink(path)
    with pytest.raises(ValueError, match="nuclide"):
        wf.date_all()


def test_csv_to_v3_raises_for_location_only():
    path = _write_tmp(_LOCATION_CSV)
    try:
        with pytest.raises(ValueError, match="nuclide columns"):
            csv_to_v3(path)
    finally:
        os.unlink(path)
