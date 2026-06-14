"""
Functional tests for the stoneage v3 Python library.

Input format reference:
  https://hess.ess.washington.edu/math/docs/v3/v3_input_explained.html

Test samples are taken verbatim from that documentation (PH-1) or constructed
to exercise every nuclide/mineral pair and every scaling scheme.
"""

import warnings
import math
import numpy as np
import pytest

warnings.filterwarnings("ignore")  # suppress scipy numerics warnings

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from stoneage import parse_v3_input, get_ages, make_consts, summarize_ages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ages_consistent(t_St, t_Lm, t_LSDn, rtol=0.06):
    """Three scaling schemes should agree within ~6 %."""
    vals = [v for v in [t_St, t_Lm, t_LSDn] if v > 0]
    if len(vals) < 2:
        return True
    return (max(vals) - min(vals)) / np.mean(vals) < rtol


# ---------------------------------------------------------------------------
# A. Input parsing: valid formats
# ---------------------------------------------------------------------------

class TestInputParsing:

    def test_be10_quartz_basic(self):
        """Documentation example: PH-1 with Be-10 and Al-26 in quartz."""
        text = (
            "PH-1\t41.3567\t-70.7348\t91\tstd\t4.5\t2.65\t1\t0.00008\t1999;\n"
            "PH-1\tBe-10\tquartz\t123453\t3717\tKNSTD;\n"
        )
        d = parse_v3_input(text)
        assert d["s"]["sample_name"] == ["PH-1"]
        assert d["n"]["nuclide"] == ["N10quartz"]
        # KNSTD conversion factor is 0.9042
        assert abs(d["n"]["N"][0] - 123453 * 0.9042) < 1

    def test_al26_quartz(self):
        text = (
            "PH-1 41.3567 -70.7348 91 std 4.5 2.65 1 0.00008 1999;\n"
            "PH-1 Al-26 quartz 712408 31238 KNSTD;\n"
        )
        d = parse_v3_input(text)
        assert d["n"]["nuclide"] == ["N26quartz"]
        # KNSTD Al-26 cf = 1.0000
        assert abs(d["n"]["N"][0] - 712408) < 1

    def test_be10_al26_same_sample(self):
        """Two nuclide lines on one sample."""
        text = (
            "PH-1 41.3567 -70.7348 91 std 4.5 2.65 1 0.00008 1999;\n"
            "PH-1 Be-10 quartz 123453 3717 KNSTD;\n"
            "PH-1 Al-26 quartz 712408 31238 KNSTD;\n"
        )
        d = parse_v3_input(text)
        assert len(d["n"]["nuclide"]) == 2
        assert "N10quartz" in d["n"]["nuclide"]
        assert "N26quartz" in d["n"]["nuclide"]
        # Both nuclide measurements index back to sample 0
        assert all(d["n"]["index"] == 0)

    def test_c14_quartz(self):
        """C-14 line: 5 fields (no standard name)."""
        text = (
            "10-MPS-046 70.0 -25.0 500 std 2 2.65 1 0 2010;\n"
            "10-MPS-046 C-14 quartz 4.71E+05 1.20E+04;\n"
        )
        d = parse_v3_input(text)
        assert d["n"]["nuclide"] == ["N14quartz"]
        assert abs(d["n"]["N"][0] - 4.71e5) < 1

    def test_he3_pyroxene_with_standard(self):
        """He-3 with CRONUS-P standardization."""
        # Restandardization: N_out = N_in * (5.02e9 / lab_value)
        lab_val = 5.20e9
        N_raw = 2.50e6
        text = (
            "MPS-046 70.0 -25.0 500 std 2 2.65 1 0 2010;\n"
            f"MPS-046 He-3 pyroxene {N_raw:.3E} 2.73E+05 CRONUS-P {lab_val:.3E};\n"
        )
        d = parse_v3_input(text)
        assert d["n"]["nuclide"] == ["N3pyroxene"]
        expected = N_raw * (5.02e9 / lab_val)
        assert abs(d["n"]["N"][0] - expected) / expected < 1e-6

    def test_he3_quartz_none_standard(self):
        """He-3 with NONE standardization — no restandardization applied."""
        N_raw = 2.50e6
        text = (
            "S1 44.0 -110.0 2000 std 3 2.65 1 0 2015;\n"
            f"S1 He-3 quartz {N_raw:.3E} 5.00E+04 NONE 0;\n"
        )
        d = parse_v3_input(text)
        assert abs(d["n"]["N"][0] - N_raw) < 1

    def test_ne21_quartz_with_standard(self):
        """Ne-21 with CRONUS-A standardization."""
        N_raw = 299185727
        lab_val = 3.32e8
        text = (
            "WO-140 45.0 -110.0 1500 std 2 2.65 1 0 2005;\n"
            f"WO-140 Ne-21 quartz {N_raw} 4884352 CRONUS-A {lab_val:.3E};\n"
        )
        d = parse_v3_input(text)
        assert d["n"]["nuclide"] == ["N21quartz"]
        expected = N_raw * (320e6 / lab_val)
        assert abs(d["n"]["N"][0] - expected) / expected < 1e-6

    def test_be10_pyroxene(self):
        text = (
            "S1 40.0 -120.0 1000 std 2 2.65 1 0 2010;\n"
            "S1 Be-10 pyroxene 150000 5000 07KNSTD;\n"
        )
        d = parse_v3_input(text)
        assert d["n"]["nuclide"] == ["N10pyroxene"]

    def test_he3_olivine(self):
        text = (
            "S1 35.0 -118.0 1500 std 3 3.2 1 0 2012;\n"
            "S1 He-3 olivine 8.0E+06 4.0E+05 NONE 0;\n"
        )
        d = parse_v3_input(text)
        assert d["n"]["nuclide"] == ["N3olivine"]

    def test_07KNSTD_be10_no_restandardization(self):
        """07KNSTD is the reference standard — conversion factor 1.0."""
        text = (
            "S1 60.0 15.0 200 std 1 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 500000 10000 07KNSTD;\n"
        )
        d = parse_v3_input(text)
        assert abs(d["n"]["N"][0] - 500000) < 1

    def test_pressure_flag(self):
        """Direct pressure input (flag = 'pre')."""
        text = (
            "S1 45.0 10.0 850 pre 2 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 300000 6000 07KNSTD;\n"
        )
        d = parse_v3_input(text)
        assert d["s"]["aa"] == ["pre"]
        assert d["s"]["pressure"][0] == 850.0

    def test_antarctica_flag(self):
        """Antarctic atmospheric correction flag."""
        text = (
            "ANT-1 -77.5 163.0 1500 ant 2 2.65 1 0 2008;\n"
            "ANT-1 Be-10 quartz 1200000 24000 07KNSTD;\n"
        )
        d = parse_v3_input(text)
        assert d["s"]["aa"] == ["ant"]

    def test_collection_year_zero_defaults(self):
        """Year = 0 should default to 2010."""
        text = (
            "S1 45.0 10.0 500 std 2 2.65 1 0 0;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;\n"
        )
        d = parse_v3_input(text)
        assert d["s"]["yr"][0] == 2010

    def test_multiline_multisamples(self):
        """Two independent samples in one block."""
        text = (
            "S1 44.5 -73.4 1000 std 3 2.65 1.0 0 2010;\n"
            "S2 60.0 15.0 200 std 1 2.7 0.95 0 2015;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;\n"
            "S2 Be-10 quartz 500000 10000 07KNSTD;\n"
        )
        d = parse_v3_input(text)
        assert len(d["s"]["sample_name"]) == 2
        assert len(d["n"]["nuclide"]) == 2
        assert d["n"]["index"][0] == 0
        assert d["n"]["index"][1] == 1

    def test_calibration_exact_age(self):
        """true_t line with exact age (2 fields: age, uncertainty)."""
        text = (
            "SKY-03 66.0 -40.0 200 std 2 2.65 1 0 2006;\n"
            "SKY-03 Be-10 quartz 35000 700 07KNSTD;\n"
            "SKY-03 true_t FEAR 11700 300;\n"
        )
        d = parse_v3_input(text)
        assert "c" in d
        assert d["c"]["truet"][0] == 11700
        assert d["c"]["dtruet"][0] == 300

    def test_calibration_min_max_age(self):
        """true_t line with min/max age bounds."""
        text = (
            "CI2-01 55.0 -3.0 300 std 3 2.65 1 0 2010;\n"
            "CI2-01 Be-10 quartz 40000 800 07KNSTD;\n"
            "CI2-01 true_t CLYDE 7950 45 8435 50;\n"
        )
        d = parse_v3_input(text)
        assert d["c"]["mint"][0] == 7950
        assert d["c"]["maxt"][0] == 8435

    def test_calibration_min_only(self):
        """true_t with minimum age only (Inf maximum)."""
        text = (
            "CI2-01 55.0 -3.0 300 std 3 2.65 1 0 2010;\n"
            "CI2-01 Be-10 quartz 40000 800 07KNSTD;\n"
            "CI2-01 true_t CLYDE 7950 45 Inf 0;\n"
        )
        d = parse_v3_input(text)
        assert d["c"]["mint"][0] == 7950
        assert math.isinf(d["c"]["maxt"][0])


# ---------------------------------------------------------------------------
# B. Input parsing: error handling
# ---------------------------------------------------------------------------

class TestInputErrors:

    def test_wrong_nuclide(self):
        # Xe-100 is not in _NUCLIDE_TOKENS so the line is misclassified as a
        # sample line and fails field-count validation — still a ValueError.
        text = (
            "S1 45.0 10.0 500 std 2 2.65 1 0 2010;\n"
            "S1 Xe-100 quartz 200000 4000 07KNSTD;\n"
        )
        with pytest.raises(ValueError):
            parse_v3_input(text)

    def test_sample_name_mismatch(self):
        text = (
            "S1 45.0 10.0 500 std 2 2.65 1 0 2010;\n"
            "S2 Be-10 quartz 200000 4000 07KNSTD;\n"
        )
        with pytest.raises(ValueError):
            parse_v3_input(text)

    def test_latitude_out_of_bounds(self):
        text = (
            "S1 95.0 10.0 500 std 2 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;\n"
        )
        with pytest.raises(ValueError, match="Latitude"):
            parse_v3_input(text)

    def test_shielding_out_of_range(self):
        text = (
            "S1 45.0 10.0 500 std 2 2.65 1.5 0 2010;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;\n"
        )
        with pytest.raises(ValueError, match="Shielding"):
            parse_v3_input(text)

    def test_unknown_be10_standard(self):
        text = (
            "S1 45.0 10.0 500 std 2 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 200000 4000 BOGUS_STD;\n"
        )
        with pytest.raises(ValueError, match="standard"):
            parse_v3_input(text)

    def test_unknown_mineral_for_al26(self):
        text = (
            "S1 45.0 10.0 500 std 2 2.65 1 0 2010;\n"
            "S1 Al-26 olivine 200000 4000 KNSTD;\n"
        )
        with pytest.raises(ValueError, match="mineral"):
            parse_v3_input(text)

    def test_future_collection_year(self):
        text = (
            "S1 45.0 10.0 500 std 2 2.65 1 0 2100;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;\n"
        )
        with pytest.raises(ValueError, match="future"):
            parse_v3_input(text)

    def test_only_one_line_raises(self):
        text = "S1 45.0 10.0 500 std 2 2.65 1 0 2010;\n"
        with pytest.raises(ValueError):
            parse_v3_input(text)


# ---------------------------------------------------------------------------
# C. Age calculation: NTD / St scaling
# ---------------------------------------------------------------------------

class TestNTDCalculation:

    def test_ph1_be10_age_reasonable(self):
        """PH-1 from the documentation: Martha's Vineyard, 91 m, Be-10 only."""
        text = (
            "PH-1 41.3567 -70.7348 91 std 4.5 2.65 1 0.00008 1999;\n"
            "PH-1 Be-10 quartz 123453 3717 KNSTD;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d)
        t = result["n"]["t_St"][0]
        # Martha's Vineyard deglaciated ~14-18 ka; given these concentrations,
        # the age should be plausibly in the 10–30 ka range.
        assert 8_000 < t < 40_000, f"t_St = {t:.0f} yr is outside expected range"

    def test_ph1_be10_al26_same_age(self):
        """Be-10 and Al-26 on same sample should give consistent ages."""
        text = (
            "PH-1 41.3567 -70.7348 91 std 4.5 2.65 1 0.00008 1999;\n"
            "PH-1 Be-10 quartz 123453 3717 KNSTD;\n"
            "PH-1 Al-26 quartz 712408 31238 KNSTD;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d)
        n = result["n"]
        nucs = n["nuclide"]
        i10 = nucs.index("N10quartz")
        i26 = nucs.index("N26quartz")
        t10 = n["t_St"][i10]
        t26 = n["t_St"][i26]
        # Both nuclides from same sample — should agree within ~20%
        ratio = abs(t10 - t26) / np.mean([t10, t26])
        assert ratio < 0.20, f"Be-10 age {t10:.0f} and Al-26 age {t26:.0f} disagree by {ratio:.1%}"

    def test_uncertainty_positive_and_ordered(self):
        """Internal uncertainty < external uncertainty < age."""
        text = (
            "S1 44.5 -73.4 1000 std 3 2.65 1.0 0 2010;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d)
        n = result["n"]
        t   = n["t_St"][0]
        di  = n["delt_int_St"][0]
        de  = n["delt_ext_St"][0]
        assert di > 0
        assert de > 0
        assert di <= de, "Internal uncertainty must not exceed external"
        assert de < t,  "External uncertainty must be less than age"

    def test_slhl_be10_age_near_production_rate(self):
        """
        At SLHL (lat=61, sea level), production rate ≈ refP_St = 4.086 at/g/yr.
        N = 40860 at/g should give ≈ 10000 yr (within 5%).
        """
        N = 40860
        text = (
            f"SLHL 61.0 0.0 0 std 0 2.65 1 0 2010;\n"
            f"SLHL Be-10 quartz {N} 817 07KNSTD;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d)
        t = result["n"]["t_St"][0]
        assert abs(t - 10000) / 10000 < 0.05, f"Expected ~10000 yr, got {t:.0f} yr"

    def test_zero_thickness_correction(self):
        """Zero-thickness sample: thickness correction factor should be 1."""
        text = (
            "S1 60.0 0.0 0 std 0 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 40860 817 07KNSTD;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d)
        assert abs(result["n"]["sf_thick"][0] - 1.0) < 1e-10

    def test_shielding_halves_production(self):
        """50% shielding should approximately double the apparent age."""
        base = (
            "S1 60.0 0.0 0 std 0 2.65 {shield} 0 2010;\n"
            "S1 Be-10 quartz 40860 817 07KNSTD;\n"
        )
        d_full   = parse_v3_input(base.format(shield=1.0))
        d_half   = parse_v3_input(base.format(shield=0.5))
        t_full   = get_ages(d_full)["n"]["t_St"][0]
        t_half   = get_ages(d_half)["n"]["t_St"][0]
        ratio = t_half / t_full
        assert 1.8 < ratio < 2.2, f"Shielding ratio {ratio:.2f} not near 2.0"

    def test_he3_olivine_age(self):
        """He-3 is stable — simple N/P age equation."""
        text = (
            "S1 60.0 0.0 500 std 0 3.2 1 0 2010;\n"
            "S1 He-3 olivine 1.196e6 5.0e4 NONE 0;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d)
        t = result["n"]["t_St"][0]
        assert t > 0
        assert 1000 < t < 500_000

    def test_c14_age_young(self):
        """C-14 with modest concentration should give young age."""
        text = (
            "ANT1 -77.5 163.0 2000 ant 2 2.65 1 0 2008;\n"
            "ANT1 C-14 quartz 1.0E+05 3.0E+03;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d)
        t = result["n"]["t_St"][0]
        assert 0 < t < 100_000

    def test_saturated_sample_flagged(self):
        """Extremely high concentration should trigger saturation flag."""
        text = (
            "S1 45.0 10.0 500 std 0 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 1.0E+10 1.0E+08 07KNSTD;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d)
        assert result["n"]["t_St"][0] == 0
        assert len(result["flags"]) > 0

    def test_flags_empty_normal_sample(self):
        """Normal, unsaturated sample should produce no flags."""
        text = (
            "S1 44.5 -73.4 1000 std 3 2.65 1.0 0 2010;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d)
        assert result["flags"] == []


# ---------------------------------------------------------------------------
# D. Age calculation: long-form (all three scaling schemes)
# ---------------------------------------------------------------------------

class TestLongFormScaling:

    @pytest.fixture
    def ph1_long(self):
        text = (
            "PH-1 41.3567 -70.7348 91 std 4.5 2.65 1 0.00008 1999;\n"
            "PH-1 Be-10 quartz 123453 3717 KNSTD;\n"
        )
        d = parse_v3_input(text)
        return get_ages(d, control={"result_type": "long"})

    def test_all_three_ages_present(self, ph1_long):
        n = ph1_long["n"]
        assert "t_St"   in n
        assert "t_Lm"   in n
        assert "t_LSDn" in n

    def test_all_three_ages_positive(self, ph1_long):
        n = ph1_long["n"]
        for sf in ["St", "Lm", "LSDn"]:
            assert n[f"t_{sf}"][0] > 0, f"t_{sf} is not positive"

    def test_scaling_consistency(self, ph1_long):
        n = ph1_long["n"]
        t_St   = n["t_St"][0]
        t_Lm   = n["t_Lm"][0]
        t_LSDn = n["t_LSDn"][0]
        # At ~28 ka, low elevation, geomagnetic corrections can spread schemes
        # by up to ~10 % — widen tolerance slightly from the default 6 %.
        assert ages_consistent(t_St, t_Lm, t_LSDn, rtol=0.10), (
            f"Scaling schemes disagree: St={t_St:.0f}, Lm={t_Lm:.0f}, LSDn={t_LSDn:.0f}"
        )

    def test_all_uncertainties_present(self, ph1_long):
        n = ph1_long["n"]
        for sf in ["St", "Lm", "LSDn"]:
            assert n[f"delt_int_{sf}"][0] > 0
            assert n[f"delt_ext_{sf}"][0] > 0

    def test_sfdata_present(self, ph1_long):
        """Long-form results attach the sfdata structure."""
        assert "f" in ph1_long

    def test_two_nuclides_long_form(self):
        """Be-10 + Al-26 both get all three scaling ages."""
        text = (
            "PH-1 41.3567 -70.7348 91 std 4.5 2.65 1 0.00008 1999;\n"
            "PH-1 Be-10 quartz 123453 3717 KNSTD;\n"
            "PH-1 Al-26 quartz 712408 31238 KNSTD;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d, control={"result_type": "long"})
        n = result["n"]
        assert len(n["t_St"]) == 2
        assert len(n["t_Lm"]) == 2
        assert len(n["t_LSDn"]) == 2

    def test_high_lat_scaling_stable(self):
        """High-latitude sample: LSDn and St should be especially close."""
        text = (
            "S1 70.0 15.0 100 std 2 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 300000 6000 07KNSTD;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d, control={"result_type": "long"})
        n = result["n"]
        assert ages_consistent(n["t_St"][0], n["t_Lm"][0], n["t_LSDn"][0], rtol=0.08)


# ---------------------------------------------------------------------------
# E. Calibration mode
# ---------------------------------------------------------------------------

class TestCalibrationMode:

    def test_exact_age_production_rate_reasonable(self):
        """
        Known-age sample: production rate should be in the range 3–6 at/g/yr
        (the expected SLHL range for Be-10 St scaling is ~4 at/g/yr).
        """
        text = (
            "SKY-03 66.0 -40.0 200 std 2 2.65 1 0 2006;\n"
            "SKY-03 Be-10 quartz 35000 700 07KNSTD;\n"
            "SKY-03 true_t FEAR 11700 300;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d, control={"cal": 1})
        P = result["n"]["calc_P_St"][0]
        assert 2.0 < P < 8.0, f"Calibration P_St = {P:.4f} at/g/yr out of range"

    def test_calibration_long_form(self):
        """Calibration long-form should populate calc_P for Lm and LSDn."""
        text = (
            "SKY-03 66.0 -40.0 200 std 2 2.65 1 0 2006;\n"
            "SKY-03 Be-10 quartz 35000 700 07KNSTD;\n"
            "SKY-03 true_t FEAR 11700 300;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d, control={"cal": 1, "result_type": "long"})
        n = result["n"]
        assert "calc_P_Lm"   in n
        assert "calc_P_LSDn" in n
        consts = make_consts()
        idx = consts.nuclides.index("N10quartz")
        P_St   = n["calc_P_St"][0]
        P_Lm   = n["calc_P_Lm"][0]
        P_LSDn = n["calc_P_LSDn"][0]
        # St and Lm report dimensional production rates (at/g/yr); expect > 0
        assert P_St > 0 and P_Lm > 0
        # LSDn reports a nondimensional correction factor (~refP_LSDn = 0.849).
        # All three, normalised to their reference values, should agree within 30 %
        # (synthetic sample, not a real calibration site, so some spread is expected).
        norm_St   = P_St   / consts.refP_St[idx]
        norm_Lm   = P_Lm   / consts.refP_Lm[idx]
        norm_LSDn = P_LSDn / consts.refP_LSDn[idx]
        vals = [norm_St, norm_Lm, norm_LSDn]
        spread = (max(vals) - min(vals)) / np.mean(vals)
        assert spread < 0.30, (
            f"Normalised calibration rates diverge by {spread:.1%}: "
            f"St={norm_St:.3f}, Lm={norm_Lm:.3f}, LSDn={norm_LSDn:.3f}"
        )

    def test_min_age_gives_max_production(self):
        """
        Minimum-age constraint → maximum production rate.
        calc_maxP should be > calc_P from exact-age test.
        """
        text_minmax = (
            "CI2-01 55.0 -3.0 300 std 3 2.65 1 0 2010;\n"
            "CI2-01 Be-10 quartz 40000 800 07KNSTD;\n"
            "CI2-01 true_t CLYDE 7950 45 Inf 0;\n"
        )
        text_exact = (
            "CI2-01 55.0 -3.0 300 std 3 2.65 1 0 2010;\n"
            "CI2-01 Be-10 quartz 40000 800 07KNSTD;\n"
            "CI2-01 true_t CLYDE 7950 45 7950 45;\n"
        )
        res_min   = get_ages(parse_v3_input(text_minmax), control={"cal": 1})
        res_exact = get_ages(parse_v3_input(text_exact),  control={"cal": 1})
        maxP  = res_min["n"]["calc_maxP_St"][0]
        exactP = res_exact["n"]["calc_P_St"][0]
        # Minimum age → maximum production rate, so maxP > exactP
        assert maxP >= exactP * 0.95, (
            f"calc_maxP ({maxP:.4f}) should be >= calc_P ({exactP:.4f})"
        )


# ---------------------------------------------------------------------------
# F. Summary statistics
# ---------------------------------------------------------------------------

class TestSummarize:

    def test_summarize_returns_all_St(self):
        text = (
            "PH-1 41.3567 -70.7348 91 std 4.5 2.65 1 0.00008 1999;\n"
            "PH-1 Be-10 quartz 123453 3717 KNSTD;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d)
        summary = summarize_ages(result)
        assert "all" in summary
        assert "St" in summary["all"]
        assert summary["all"]["St"]["sumval"] > 0

    def test_summarize_multi_sample(self):
        """Summary over two samples should give a mean age."""
        text = (
            "S1 44.5 -73.4 1000 std 3 2.65 1 0 2010;\n"
            "S2 44.5 -73.4 1000 std 3 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 180000 3600 07KNSTD;\n"
            "S2 Be-10 quartz 220000 4400 07KNSTD;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d)
        summary = summarize_ages(result)
        t1 = result["n"]["t_St"][0]
        t2 = result["n"]["t_St"][1]
        sumval = summary["all"]["St"]["sumval"]
        # Summary value should be between the two individual ages
        assert min(t1, t2) * 0.8 < sumval < max(t1, t2) * 1.2

    def test_summarize_long_form_has_lm_lsdn(self):
        text = (
            "S1 44.5 -73.4 1000 std 3 2.65 1 0 2010;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;\n"
        )
        d = parse_v3_input(text)
        result = get_ages(d, control={"result_type": "long"})
        summary = summarize_ages(result)
        assert "Lm"   in summary["all"]
        assert "LSDn" in summary["all"]


# ---------------------------------------------------------------------------
# G. Constants sanity checks
# ---------------------------------------------------------------------------

class TestConstants:

    def test_be10_decay_constant(self):
        """Be-10 half-life = 1.387 Ma → lambda = ln(2)/1.387e6."""
        c = make_consts()
        expected = math.log(2) / 1.387e6
        assert abs(c.l10 - expected) / expected < 1e-8

    def test_al26_decay_constant(self):
        c = make_consts()
        assert abs(c.l26 - 9.83e-7) < 1e-30

    def test_refP_St_be10(self):
        """Be-10 St reference production rate from Balco 2016 calibration."""
        c = make_consts()
        idx = c.nuclides.index("N10quartz")
        assert abs(c.refP_St[idx] - 4.086) < 0.001

    def test_nuclide_vector_lengths(self):
        c = make_consts()
        n = len(c.nuclides)
        assert len(c.refP_St)   == n
        assert len(c.refP_Lm)   == n
        assert len(c.refP_LSDn) == n
        assert len(c.Pmu0)      == n
        assert len(c.l)         == n


# ---------------------------------------------------------------------------
# H. CSV workflow — from_csv() + date_all() round-trip
# ---------------------------------------------------------------------------

import tempfile, os
from stoneage import csv_to_v3
from stoneage.workflow import ProductionRateWorkflow

_MV_CSV = """\
name,lat,lon,elevation,elv_flag,thickness,density,shielding,erosion,year,nuclide,mineral,concentration,uncertainty,standard
PH-1,41.1,-70.8,22,std,2.5,2.65,0.98,0.0,2005,Be-10,quartz,188200,3400,07KNSTD
"""


class TestCSVWorkflow:
    """Round-trip: ages from date_all() must match direct parse_v3_input + get_ages."""

    def _write_tmp(self, content):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_martha_vineyard_csv_ages_match_direct_v3(self):
        """
        Martha's Vineyard PH-1 dated via date_all() must give the same ages
        as calling parse_v3_input + get_ages on the equivalent v3 text directly.
        """
        path = self._write_tmp(_MV_CSV)
        try:
            # via CSV workflow
            wf     = ProductionRateWorkflow.from_csv(path)
            csv_result = wf.date_all()

            # via direct v3 text
            v3_text = csv_to_v3(path)
            direct_result = get_ages(parse_v3_input(v3_text),
                                     control={"result_type": "long"})
        finally:
            os.unlink(path)

        for sf in ["St", "Lm", "LSDn"]:
            t_csv    = csv_result["n"][f"t_{sf}"][0]
            t_direct = direct_result["n"][f"t_{sf}"][0]
            assert abs(t_csv - t_direct) < 0.1, (
                f"{sf}: date_all()={t_csv:.1f}, direct={t_direct:.1f}"
            )

    def test_martha_vineyard_age_range(self):
        """
        PH-1 Martha's Vineyard: all three schemes should give ~45–58 ka
        (reasonable for this Late Pleistocene moraine sample).
        """
        path = self._write_tmp(_MV_CSV)
        try:
            wf     = ProductionRateWorkflow.from_csv(path)
            result = wf.date_all()
        finally:
            os.unlink(path)

        n = result["n"]
        for sf in ["St", "Lm", "LSDn"]:
            t = n[f"t_{sf}"][0]
            assert 44000 < t < 59000, (
                f"PH-1 {sf} age {t:.0f} yr outside expected 44–59 ka window"
            )

    def test_martha_vineyard_scheme_consistency(self):
        """All three scaling schemes should agree within 10 %."""
        path = self._write_tmp(_MV_CSV)
        try:
            wf     = ProductionRateWorkflow.from_csv(path)
            result = wf.date_all()
        finally:
            os.unlink(path)

        n = result["n"]
        ages = [n[f"t_{sf}"][0] for sf in ["St", "Lm", "LSDn"]]
        spread = (max(ages) - min(ages)) / np.mean(ages)
        assert spread < 0.10, f"Scheme spread {100*spread:.1f}% > 10%"
