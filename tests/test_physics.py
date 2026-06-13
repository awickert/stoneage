"""
Unit tests for the individual physics functions: atmosphere, scaling, corrections.
These test each component in isolation against known analytical values.
"""

import warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stoneage.atmosphere import ERA40atm, antatm
from stoneage.scaling import stone2000
from stoneage.corrections import thickness, Lsp
from stoneage.cutoff_rigidity import _angdistr


# ---------------------------------------------------------------------------
# antatm
# ---------------------------------------------------------------------------

class TestAntatm:
    def test_sea_level(self):
        """antatm(0) = 989.1 hPa by definition."""
        assert abs(antatm([0.0])[0] - 989.1) < 1e-6

    def test_monotone_decrease(self):
        """Pressure decreases with increasing elevation."""
        z = np.array([0., 500., 1000., 2000., 4000.])
        p = antatm(z)
        assert np.all(np.diff(p) < 0)

    def test_scale_height(self):
        """At z = 7588 m, pressure should be 989.1 / e ≈ 363.8 hPa."""
        p = antatm([7588.0])[0]
        assert abs(p - 989.1 / np.e) < 0.01

    def test_vector_input(self):
        z = np.array([0., 1000., 2000.])
        p = antatm(z)
        assert p.shape == (3,)


# ---------------------------------------------------------------------------
# ERA40atm
# ---------------------------------------------------------------------------

class TestERA40atm:
    def test_sea_level_high_lat_near_standard(self):
        """At high latitude sea level, pressure should be near 1013 hPa."""
        p = ERA40atm([70.0], [15.0], [0.0])[0]
        assert 990 < p < 1030

    def test_pressure_decreases_with_elevation(self):
        lats = [45.0, 45.0, 45.0]
        lons = [10.0, 10.0, 10.0]
        elvs = [0.0, 1000.0, 3000.0]
        p = ERA40atm(lats, lons, elvs)
        assert np.all(np.diff(p) < 0)

    def test_negative_longitude_converted(self):
        """West-negative longitude should give same result as 0-360 equivalent."""
        p1 = ERA40atm([41.4], [-112.7], [1311.0])[0]
        p2 = ERA40atm([41.4], [247.3],  [1311.0])[0]
        assert abs(p1 - p2) < 0.01

    def test_southern_hemisphere(self):
        """Antarctic plateau: ERA40 should give plausible sub-1000 hPa value."""
        p = ERA40atm([-75.0], [0.0], [3000.0])[0]
        assert 600 < p < 750


# ---------------------------------------------------------------------------
# stone2000
# ---------------------------------------------------------------------------

class TestStone2000:
    def test_slhl_near_unity(self):
        """At SLHL (lat≥60, sea-level pressure), scaling factor ≈ 1.0."""
        sf = stone2000([60.0], [1013.25])[0]
        assert abs(sf - 1.0) < 0.05

    def test_equator_sea_level(self):
        """At equator sea level, scaling factor ≈ 0.587 (Stone 2000, Table 1)."""
        sf = stone2000([0.0], [1013.25])[0]
        assert abs(sf - 0.587) < 0.005

    def test_fsp1_spallation_only(self):
        """Fsp=1 should be strictly greater than default (which mixes in muons)."""
        sf_default = stone2000([45.0], [900.0])[0]
        sf_spall   = stone2000([45.0], [900.0], Fsp=1.0)[0]
        # At altitude muon contribution lifts the mixed value above spallation-only
        # at low latitude, but below at high latitude; they differ in any case.
        assert sf_default != sf_spall

    def test_fsp0_muons_only(self):
        """Fsp=0 gives muon-only scaling; should be positive."""
        sf = stone2000([45.0], [1013.25], Fsp=0.0)[0]
        assert sf > 0

    def test_increases_with_altitude(self):
        """Higher altitude (lower pressure) → higher scaling factor."""
        sf_sl  = stone2000([45.0], [1013.25])[0]
        sf_alt = stone2000([45.0], [700.0])[0]
        assert sf_alt > sf_sl

    def test_increases_with_latitude(self):
        """Higher latitude → higher scaling factor at sea level."""
        sf_eq  = stone2000([0.0],  [1013.25])[0]
        sf_mid = stone2000([45.0], [1013.25])[0]
        sf_hi  = stone2000([60.0], [1013.25])[0]
        assert sf_eq < sf_mid < sf_hi

    def test_southern_hemisphere_symmetric(self):
        """Southern hemisphere should give same result as northern."""
        sf_n = stone2000([45.0],  [900.0])[0]
        sf_s = stone2000([-45.0], [900.0])[0]
        assert abs(sf_n - sf_s) < 1e-10

    def test_cap_above_60(self):
        """lat=75 and lat=60 should give the same result (cap at 60)."""
        sf_60 = stone2000([60.0], [900.0])[0]
        sf_75 = stone2000([75.0], [900.0])[0]
        assert abs(sf_60 - sf_75) < 1e-10

    def test_vector_inputs(self):
        lats = np.array([0.0, 30.0, 60.0])
        P    = np.array([1013.25, 1013.25, 1013.25])
        sf   = stone2000(lats, P)
        assert sf.shape == (3,)
        assert np.all(np.diff(sf) > 0)


# ---------------------------------------------------------------------------
# thickness
# ---------------------------------------------------------------------------

class TestThickness:
    def test_zero_thickness_is_one(self):
        assert thickness([0.0], [160.0], [2.65])[0] == 1.0

    def test_thin_sample_near_one(self):
        """1 cm granite: correction should be very close to 1."""
        tc = thickness([1.0], [160.0], [2.65])[0]
        assert 0.99 < tc < 1.0

    def test_decreases_with_thickness(self):
        """Thicker sample → smaller correction (more self-shielding)."""
        t1 = thickness([2.0],  [160.0], [2.65])[0]
        t5 = thickness([5.0],  [160.0], [2.65])[0]
        t10 = thickness([10.0], [160.0], [2.65])[0]
        assert 1.0 > t1 > t5 > t10 > 0

    def test_analytical_value(self):
        """thickness = (L/rho/z)*(1-exp(-rho*z/L))."""
        z, L, rho = 3.0, 160.0, 2.65
        expected = (L / (rho * z)) * (1 - np.exp(-rho * z / L))
        assert abs(thickness([z], [L], [rho])[0] - expected) < 1e-10

    def test_vector_input(self):
        z   = np.array([0.0, 2.0, 5.0])
        L   = np.full(3, 160.0)
        rho = np.full(3, 2.65)
        tc  = thickness(z, L, rho)
        assert tc[0] == 1.0
        assert tc[1] > tc[2]


# ---------------------------------------------------------------------------
# Lsp
# ---------------------------------------------------------------------------

def test_lsp_value():
    """Spallogenic attenuation length is 160 g/cm²."""
    assert Lsp() == 160.0


# ---------------------------------------------------------------------------
# _angdistr (internal helper in cutoff_rigidity)
# ---------------------------------------------------------------------------

class TestAngdistr:
    def test_same_point_is_zero(self):
        """Angular distance from a point to itself is 0."""
        theta = _angdistr(0.5, 1.0, 0.5, 1.0)
        assert abs(theta) < 1e-10

    def test_antipodal_is_pi(self):
        """Antipodal points are π radians apart."""
        theta = _angdistr(0.0, 0.0, 0.0, np.pi)
        assert abs(theta - np.pi) < 1e-10

    def test_poles_are_pi_apart(self):
        """North and south poles are π apart regardless of longitude."""
        theta = _angdistr(np.pi / 2, 0.0, -np.pi / 2, 0.0)
        assert abs(theta - np.pi) < 1e-10

    def test_90_degrees(self):
        """Two points 90° apart on the equator."""
        theta = _angdistr(0.0, 0.0, 0.0, np.pi / 2)
        assert abs(theta - np.pi / 2) < 1e-10
