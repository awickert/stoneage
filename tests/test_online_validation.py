"""
Validation tests: compare Python library output against the live MATLAB
calculator at stoneage.ice-d.org.

These tests require an internet connection and are marked ``network`` so
they can be skipped in offline environments::

    pytest -m "not network"          # skip these
    pytest -m network                 # run only these

The online calculator returns XML with integer-rounded ages.  Each test
asserts that the Python result rounds to the same integer as the server,
i.e. |t_py - t_web| < 1 year for ages and within 1 year for uncertainties.
"""

import subprocess
import xml.etree.ElementTree as ET
import warnings
import pytest
import numpy as np

warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stoneage import parse_v3_input, get_ages

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ONLINE_URL = "https://stoneage.ice-d.org/cgi-bin/matweb"


def _post(text_block: str) -> ET.Element:
    """POST to the online calculator; return the parsed XML root element."""
    r = subprocess.run(
        [
            "curl", "-s", "--max-time", "30",
            "-X", "POST", ONLINE_URL,
            "-d", "mlmfile=age_input_v3",
            "-d", "reportType=XML",
            "-d", "resultType=long",
            "--data-urlencode", f"text_block={text_block}",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        pytest.skip("Online calculator unreachable")
    try:
        return ET.fromstring(r.stdout.strip())
    except ET.ParseError:
        pytest.skip("Online calculator returned non-XML (server may be down)")


def _web_age(root: ET.Element, nuclide_tag: str, sf: str) -> float:
    """Extract an age from the XML response; e.g. nuclide_tag='t10quartz', sf='St'."""
    el = root.find(f".//{nuclide_tag}_{sf}")
    assert el is not None, f"Tag {nuclide_tag}_{sf} not found in XML response"
    return float(el.text)


def _web_uncert(root: ET.Element, prefix: str, kind: str, sf: str) -> float:
    """Extract an uncertainty; kind is 'int' or 'ext'."""
    el = root.find(f".//del{prefix}_{kind}_{sf}")
    assert el is not None, f"Tag del{prefix}_{kind}_{sf} not found in XML"
    return float(el.text)


def _run_py(text: str):
    """Parse + calculate with all three scaling schemes."""
    d = parse_v3_input(text)
    return get_ages(d, control={"result_type": "long"})


def _assert_age(t_py: float, t_web: float, label: str):
    """Ages must agree to the nearest year (XML output is integer-rounded)."""
    assert abs(t_py - t_web) < 1.0, (
        f"{label}: Python={t_py:.1f}, online={t_web:.1f}, diff={t_py-t_web:+.1f} yr"
    )


def _assert_uncert(du_py: float, du_web: float, label: str):
    assert abs(du_py - du_web) < 1.0, (
        f"{label}: Python={du_py:.1f}, online={du_web:.1f}, diff={du_py-du_web:+.1f} yr"
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestOnlineValidation:
    """Ages and uncertainties must match the live MATLAB calculator to 1 yr."""

    def test_ph1_be10_all_schemes(self):
        """
        PH-1: Be-10 in quartz, 91 m, New England, ~28 ka.
        Documentation example — exercises the standard low-elevation pathway.
        """
        text = (
            "PH-1 41.3567 -70.7348 91 std 4.5 2.65 1 0.00008 1999;\n"
            "PH-1 Be-10 quartz 123453 3717 KNSTD;"
        )
        root   = _post(text)
        result = _run_py(text)
        n      = result["n"]

        for sf in ["St", "Lm", "LSDn"]:
            t_web = _web_age(root, "t10quartz", sf)
            t_py  = n[f"t_{sf}"][0]
            _assert_age(t_py, t_web, f"PH-1 Be-10 {sf}")

            for kind in ["int", "ext"]:
                du_web = _web_uncert(root, "t10quartz", kind, sf)
                du_py  = n[f"delt_{kind}_{sf}"][0]
                _assert_uncert(du_py, du_web, f"PH-1 Be-10 delt_{kind}_{sf}")

    def test_slhl_be10(self):
        """
        Sea level / high latitude Be-10, ~10 ka.
        Tests the reference point where production rates are defined.
        """
        text = (
            "SLHL 61.0 0.0 0 std 0 2.65 1 0 2010;\n"
            "SLHL Be-10 quartz 40860 817 07KNSTD;"
        )
        root   = _post(text)
        result = _run_py(text)
        n      = result["n"]

        for sf in ["St", "Lm", "LSDn"]:
            _assert_age(n[f"t_{sf}"][0], _web_age(root, "t10quartz", sf),
                        f"SLHL Be-10 {sf}")

    def test_antarctic_be10_high_elevation(self):
        """
        Antarctic Be-10, 2000 m, old sample (~42 ka).
        High altitude: validates the Fsp=1 fix — double-counting muons in
        stone2000 caused a 1.4% error here before the fix.
        """
        text = (
            "ANT1 -77.5 163.0 2000 ant 2 2.65 1 0 2008;\n"
            "ANT1 Be-10 quartz 1200000 24000 07KNSTD;"
        )
        root   = _post(text)
        result = _run_py(text)
        n      = result["n"]

        for sf in ["St", "Lm", "LSDn"]:
            _assert_age(n[f"t_{sf}"][0], _web_age(root, "t10quartz", sf),
                        f"Antarctic Be-10 {sf}")

        for kind in ["int", "ext"]:
            _assert_uncert(
                n["delt_int_St"][0] if kind == "int" else n["delt_ext_St"][0],
                _web_uncert(root, "t10quartz", kind, "St"),
                f"Antarctic Be-10 delt_{kind}_St",
            )

    def test_be10_al26_same_sample(self):
        """
        Be-10 + Al-26 on the same sample — exercises multi-nuclide indexing.
        Both ages must match the online calculator independently.
        """
        text = (
            "PH-1 41.3567 -70.7348 91 std 4.5 2.65 1 0.00008 1999;\n"
            "PH-1 Be-10 quartz 123453 3717 KNSTD;\n"
            "PH-1 Al-26 quartz 712408 31238 KNSTD;"
        )
        root   = _post(text)
        result = _run_py(text)
        n      = result["n"]

        nucs = n["nuclide"]
        for sf in ["St", "Lm", "LSDn"]:
            i10 = nucs.index("N10quartz")
            i26 = nucs.index("N26quartz")
            _assert_age(n[f"t_{sf}"][i10], _web_age(root, "t10quartz", sf),
                        f"Be-10 {sf}")
            _assert_age(n[f"t_{sf}"][i26], _web_age(root, "t26quartz", sf),
                        f"Al-26 {sf}")

    def test_he3_pyroxene_cronus_p_standard(self):
        """
        He-3 in pyroxene with CRONUS-P restandardization.
        Stable nuclide: uses the simple N/P age equation.
        """
        text = (
            "HE3-1 40.0 -105.0 3000 std 2 3.0 1 0 2010;\n"
            "HE3-1 He-3 pyroxene 2.50E+06 1.25E+05 CRONUS-P 5.20E+09;"
        )
        root   = _post(text)
        result = _run_py(text)
        n      = result["n"]

        for sf in ["St", "Lm", "LSDn"]:
            _assert_age(n[f"t_{sf}"][0], _web_age(root, "t3pyroxene", sf),
                        f"He-3 px {sf}")

    def test_ne21_quartz_cronus_a_standard(self):
        """
        Ne-21 in quartz with CRONUS-A restandardization.
        Stable nuclide, large concentration.
        """
        text = (
            "NE21-1 40.0 -105.0 3000 std 2 2.65 1 0 2010;\n"
            "NE21-1 Ne-21 quartz 2.99E+08 4.88E+06 CRONUS-A 3.32E+08;"
        )
        root   = _post(text)
        result = _run_py(text)
        n      = result["n"]

        for sf in ["St", "Lm", "LSDn"]:
            _assert_age(n[f"t_{sf}"][0], _web_age(root, "t21quartz", sf),
                        f"Ne-21 {sf}")

    def test_pressure_flag_pre(self):
        """
        Direct pressure input (flag='pre') instead of elevation.
        Same sample as PH-1 but with pressure supplied explicitly.
        """
        text = (
            "PH-1P 41.3567 -70.7348 1005.1 pre 4.5 2.65 1 0.00008 1999;\n"
            "PH-1P Be-10 quartz 123453 3717 KNSTD;"
        )
        root   = _post(text)
        result = _run_py(text)
        n      = result["n"]

        for sf in ["St", "Lm", "LSDn"]:
            _assert_age(n[f"t_{sf}"][0], _web_age(root, "t10quartz", sf),
                        f"pre-flag Be-10 {sf}")

    def test_two_independent_samples(self):
        """
        Two samples in one block — nuclide index mapping must be correct.
        """
        text = (
            "S1 44.5 -73.4 1000 std 3 2.65 1.0 0 2010;\n"
            "S2 61.0 15.0 200 std 1 2.7 0.95 0 2015;\n"
            "S1 Be-10 quartz 200000 4000 07KNSTD;\n"
            "S2 Be-10 quartz 500000 10000 07KNSTD;"
        )
        root   = _post(text)
        result = _run_py(text)
        n      = result["n"]

        for sf in ["St", "Lm", "LSDn"]:
            for b, sample_tag in enumerate(["t10quartz", "t10quartz"]):
                t_web = float(
                    root.findall(f".//{sample_tag}_{sf}")[b].text
                )
                _assert_age(n[f"t_{sf}"][b], t_web, f"S{b+1} {sf}")
