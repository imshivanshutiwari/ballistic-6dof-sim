"""
Unit tests for Interior and Terminal Ballistics modules.
"""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.interior_ballistics import InteriorBallisticsSolver, GUN_155MM_M109, GUN_127MM_M2
from src.terminal_ballistics import TerminalBallisticsCalculator


class TestInteriorBallistics:
    """Test the LeDuc interior ballistics solver."""

    def test_muzzle_velocity_positive(self):
        solver = InteriorBallisticsSolver(GUN_155MM_M109)
        result = solver.solve()
        assert result.muzzle_velocity_ms > 0

    def test_muzzle_velocity_reasonable_155mm(self):
        """155mm M109 should produce ~560 m/s for Zone 7 charge."""
        solver = InteriorBallisticsSolver(GUN_155MM_M109)
        result = solver.solve()
        assert 500 < result.muzzle_velocity_ms < 1000

    def test_spin_rate_positive(self):
        solver = InteriorBallisticsSolver(GUN_155MM_M109)
        result = solver.solve()
        assert result.muzzle_spin_rate_rad_s > 0

    def test_peak_pressure_exceeds_muzzle(self):
        solver = InteriorBallisticsSolver(GUN_155MM_M109)
        result = solver.solve()
        assert result.peak_pressure_MPa > result.muzzle_pressure_MPa

    def test_velocity_monotonically_increases(self):
        solver = InteriorBallisticsSolver(GUN_155MM_M109)
        result = solver.solve()
        assert np.all(np.diff(result.velocity_ms) >= 0)

    def test_12_7mm_muzzle_velocity(self):
        """12.7mm should produce 800-1000 m/s."""
        solver = InteriorBallisticsSolver(GUN_127MM_M2)
        result = solver.solve()
        assert 700 < result.muzzle_velocity_ms < 1100

    def test_profile_length_matches(self):
        solver = InteriorBallisticsSolver(GUN_155MM_M109)
        result = solver.solve(n_points=100)
        assert len(result.travel_m) == 100
        assert len(result.velocity_ms) == 100
        assert len(result.pressure_MPa) == 100


class TestTerminalBallistics:
    """Test the terminal ballistics calculator."""

    def test_kinetic_energy_positive(self):
        calc = TerminalBallisticsCalculator(mass_kg=43.5, diameter_m=0.155)
        v = np.array([200.0, 0.0, -300.0])
        result = calc.compute(v, 1.0)
        assert result.kinetic_energy_MJ > 0

    def test_angle_of_fall_45deg(self):
        """When vx = |vz|, angle of fall should be 45°."""
        calc = TerminalBallisticsCalculator(mass_kg=43.5, diameter_m=0.155)
        v = np.array([300.0, 0.0, -300.0])
        result = calc.compute(v, 1.0)
        assert result.angle_of_fall_deg == pytest.approx(45.0, abs=0.1)

    def test_angle_of_fall_vertical(self):
        """Pure vertical impact → 90°."""
        calc = TerminalBallisticsCalculator(mass_kg=43.5, diameter_m=0.155)
        v = np.array([0.0, 0.0, -500.0])
        result = calc.compute(v, 1.0)
        assert result.angle_of_fall_deg == pytest.approx(90.0, abs=0.1)

    def test_armor_penetration_positive(self):
        calc = TerminalBallisticsCalculator(mass_kg=43.5, diameter_m=0.155)
        v = np.array([200.0, 0.0, -300.0])
        result = calc.compute(v, 1.0)
        assert result.armor_penetration_mm_RHA > 0

    def test_crater_diameter_for_HE(self):
        calc = TerminalBallisticsCalculator(mass_kg=43.5, diameter_m=0.155, is_he=True)
        v = np.array([200.0, 0.0, -300.0])
        result = calc.compute(v, 1.0)
        assert result.crater_diameter_m > 0

    def test_no_crater_for_non_HE(self):
        calc = TerminalBallisticsCalculator(mass_kg=43.5, diameter_m=0.155, is_he=False)
        v = np.array([200.0, 0.0, -300.0])
        result = calc.compute(v, 1.0)
        assert result.crater_diameter_m == 0.0

    def test_lethal_radius_for_HE(self):
        calc = TerminalBallisticsCalculator(mass_kg=43.5, diameter_m=0.155, is_he=True)
        v = np.array([200.0, 0.0, -300.0])
        result = calc.compute(v, 1.0)
        assert result.lethal_radius_m > 0


class TestGyroscopicStability:
    """Test the Sg factor computation."""

    def test_stable_shell(self):
        from src.aerodynamics import AeroModel
        aero = AeroModel(
            Cd=0.3, Cl=0.05, Cm=0.003,
            reference_area_m2=0.01887,
            reference_diameter_m=0.155,
            Cma=-3.0,
        )
        Sg = aero.gyroscopic_stability_factor(
            spin_rate_rad_s=1800, speed_ms=800,
            density_kg_m3=1.225, Ix=0.065, Iy=0.30,
        )
        assert Sg > 1.0  # should be stable

    def test_low_spin_unstable(self):
        from src.aerodynamics import AeroModel
        aero = AeroModel(
            Cd=0.3, Cl=0.05, Cm=0.003,
            reference_area_m2=0.01887,
            reference_diameter_m=0.155,
            Cma=-3.0,
        )
        Sg = aero.gyroscopic_stability_factor(
            spin_rate_rad_s=10, speed_ms=800,
            density_kg_m3=1.225, Ix=0.065, Iy=0.30,
        )
        assert Sg < 1.0  # should be unstable


class TestAltitudeGravity:
    """Test altitude-dependent gravity."""

    def test_sea_level(self):
        from src.equations_of_motion import gravity_at_altitude, GRAVITY_MS2
        g = gravity_at_altitude(0.0)
        assert g == pytest.approx(GRAVITY_MS2, rel=1e-10)

    def test_decreases_with_altitude(self):
        from src.equations_of_motion import gravity_at_altitude
        g_0 = gravity_at_altitude(0.0)
        g_10k = gravity_at_altitude(10000.0)
        g_20k = gravity_at_altitude(20000.0)
        assert g_0 > g_10k > g_20k

    def test_10km_gravity(self):
        """At 10 km, g should be about 9.776 m/s²."""
        from src.equations_of_motion import gravity_at_altitude
        g = gravity_at_altitude(10000.0)
        assert g == pytest.approx(9.776, abs=0.01)
