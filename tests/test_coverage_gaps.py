"""
Tests targeting the remaining coverage gaps after the initial test suite.
Each test is annotated with the specific lines it exercises.
"""

import warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stoneage import parse_v3_input, get_ages


# ---------------------------------------------------------------------------
# calculator.py lines 308-309
# Stable nuclide (A_sp == 0) in long-form time integration.
# The A_sp=0 branch computes int_sp = pp * dt  (no decay factor).
# ---------------------------------------------------------------------------

def test_stable_nuclide_long_form_uses_linear_integration():
    """
    He-3 in pyroxene is stable (l=0) and with zero erosion A_sp = l + E*rho/L = 0.
    In long-form mode the cumulation loop takes the A_sp==0 branch.
    """
    text = (
        "HE3 40.0 -105.0 3000 std 2 3.0 1 0 2015;\n"
        "HE3 He-3 pyroxene 2.50E+06 1.25E+05 CRONUS-P 5.20E+09;"
    )
    d = parse_v3_input(text)
    result = get_ages(d, control={"result_type": "long"})
    n = result["n"]
    # Stable nuclide: l=0, A_sp=0
    assert n["l"][0] == 0.0
    assert n["A_sp"][0] == 0.0
    # All three TD ages should be positive
    for sf in ["St", "Lm", "LSDn"]:
        assert n[f"t_{sf}"][0] > 0, f"t_{sf} is not positive"


# ---------------------------------------------------------------------------
# calculator.py lines 332-333
# Saturation flag in Lm or LSDn long-form.
# Triggered when measured N exceeds the maximum cumulated N in the TD scheme.
# ---------------------------------------------------------------------------

def test_saturation_flagged_in_td_scaling():
    """
    A concentration far above saturation should trigger the saturation flag
    for the time-dependent schemes (Lm, LSDn).
    """
    text = (
        "SAT 60.0 0.0 1000 std 0 2.65 1 0 2010;\n"
        "SAT Be-10 quartz 1.0E+10 1.0E+08 07KNSTD;"
    )
    d = parse_v3_input(text)
    result = get_ages(d, control={"result_type": "long"})
    n = result["n"]
    # At least one TD scheme should report t=0 (saturated)
    assert n["t_Lm"][0] == 0 or n["t_LSDn"][0] == 0
    assert any("saturated" in f.lower() for f in result["flags"])


# ---------------------------------------------------------------------------
# calculator.py lines 367-368
# Calibration long-form: mint > 0 branch (minimum age → max production rate).
# ---------------------------------------------------------------------------

def test_calibration_long_form_mint_branch():
    """
    Minimum-age constraint in calibration long-form mode exercises the
    interp1 call for mint in the Lm/LSDn calibration loop.
    """
    text = (
        "S1 55.0 -3.0 300 std 3 2.65 1 0 2015;\n"
        "S1 Be-10 quartz 40000 800 07KNSTD;\n"
        "S1 true_t SITE 7950 45 Inf 0;\n"
    )
    d = parse_v3_input(text)
    result = get_ages(d, control={"cal": 1, "result_type": "long"})
    n = result["n"]
    # mint > 0 → calc_maxP should be populated for Lm and LSDn
    assert n["calc_maxP_Lm"][0] > 0
    assert n["calc_maxP_LSDn"][0] > 0
    assert n["calc_P_Lm"][0] == 0.0    # no exact age
    assert n["calc_minP_Lm"][0] == 0.0  # no max age


# ---------------------------------------------------------------------------
# calculator.py lines 371-372
# Calibration long-form: maxt < inf branch (maximum age → min production rate).
# ---------------------------------------------------------------------------

def test_calibration_long_form_maxt_branch():
    """
    Maximum-age constraint in calibration long-form mode exercises the
    maxt < inf branch in the Lm/LSDn calibration loop.
    """
    text = (
        "S1 55.0 -3.0 300 std 3 2.65 1 0 2015;\n"
        "S1 Be-10 quartz 40000 800 07KNSTD;\n"
        "S1 true_t SITE 0 0 8435 50;\n"
    )
    d = parse_v3_input(text)
    result = get_ages(d, control={"cal": 1, "result_type": "long"})
    n = result["n"]
    # maxt < inf → calc_minP should be populated for Lm and LSDn
    assert n["calc_minP_Lm"][0] > 0
    assert n["calc_minP_LSDn"][0] > 0
    assert n["calc_P_Lm"][0] == 0.0    # no exact age
    assert n["calc_maxP_Lm"][0] == 0.0  # no min age


# ---------------------------------------------------------------------------
# calculator.py lines 447-449
# Stable nuclide (A_sp == 0) branch in the St calibration uncertainty block.
# Triggered when a stable nuclide (He-3, Ne-21) is used in calibration mode.
# ---------------------------------------------------------------------------

def test_stable_nuclide_calibration_uncertainty():
    """
    He-3 calibration: A_sp == 0, so uncertainty propagation takes the
    A_sp==0 branch (SFeff = Nsp * t / P, not involving the decay exponential).
    """
    text = (
        "HE3 40.0 -105.0 3000 std 2 3.0 1 0 2015;\n"
        "HE3 He-3 pyroxene 2.50E+06 1.25E+05 CRONUS-P 5.20E+09;\n"
        "HE3 true_t SITE 10000 500;\n"
    )
    d = parse_v3_input(text)
    result = get_ages(d, control={"cal": 1})
    n = result["n"]
    assert n["A_sp"][0] == 0.0           # stable nuclide, no erosion
    assert n["calc_P_St"][0] > 0         # production rate computed
    assert n["calc_delP_St"][0] > 0      # uncertainty propagated through A_sp=0 branch


# ---------------------------------------------------------------------------
# cutoff_rigidity.py lines 115-116
# yr > tzero (tzero = 2010): sample collected after the paleomagnetic
# reference year. Sets temptmin[0] = 0 and assigns SPhi directly.
# ---------------------------------------------------------------------------

def test_get_DipRc_yr_after_tzero():
    """
    Collection year 2015 > tzero=2010 exercises the yr > tzero branch in
    get_DipRc, which sets the first time-step lower bound to zero.
    """
    text = (
        "S1 44.5 -73.4 1000 std 3 2.65 1 0 2015;\n"
        "S1 Be-10 quartz 200000 4000 07KNSTD;"
    )
    d = parse_v3_input(text)
    result = get_ages(d, control={"result_type": "long"})
    n = result["n"]
    # All three TD ages should be valid and positive
    for sf in ["St", "Lm", "LSDn"]:
        assert n[f"t_{sf}"][0] > 0
    # The first tmin in the Rc series should be 0 (yr > tzero sets it)
    assert result["f"]["tmin"][0][0] == 0.0


# ---------------------------------------------------------------------------
# cutoff_rigidity.py lines 147-151
# req_t > t2[-1] (~20 Ma): exposure duration exceeds the entire long record.
# A single tail step is appended to cover the remainder.
# ---------------------------------------------------------------------------

def test_get_DipRc_beyond_long_record():
    """
    A He-3 sample with enormous concentration gives an age >20 Ma, which
    exceeds the paleomagnetic long record (~20 Ma). The code appends a
    single tail step with the mean MM02 value.
    """
    # He-3 at high altitude (61°N, 3000 m) has P_St ~1340 at/g/yr.
    # req_t = 1.5 * t_St must exceed the long-record limit (~20.2 Ma),
    # so t_St must be > 13.5 Ma, requiring N > 1.5 * 13.5e6 * 1340 ≈ 2.7e10.
    text = (
        "OLD 61.0 0.0 3000 std 0 2.65 1 0 2010;\n"
        "OLD He-3 quartz 3.0E+10 1.5E+09 NONE 0;"
    )
    d = parse_v3_input(text)
    result = get_ages(d, control={"result_type": "long"})
    n = result["n"]
    assert n["t_St"][0] > 1.35e7   # > 13.5 Ma (so req_t > 20.2 Ma)
    # The last tmax in the Rc series should equal req_t (the huge requested age)
    last_tmax = result["f"]["tmax"][0][-1]
    assert last_tmax > 2.02e7      # > 20.2 Ma, confirming the tail branch was hit
