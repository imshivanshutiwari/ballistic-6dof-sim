"""
Unit tests for the Aerodynamic Force and Moment model.

Tests cover:
    - Drag force direction and magnitude
    - Mach-dependent Cd interpolation
    - Lift force perpendicularity
    - Magnus force cross-product direction
    - Zero-velocity edge cases
    - Moment computation (pitch-damping, roll-damping, Magnus moment)
"""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.aerodynamics import (
    AeroModel,
    MACH_TABLE_155MM,
    CD_TABLE_155MM,
)


@pytest.fixture
def aero_const():
    """AeroModel with constant Cd (no Mach table)."""
    return AeroModel(
        Cd=0.3, Cl=0.05, Cm=0.003,
        reference_area_m2=0.01887,
        reference_diameter_m=0.155,
    )


@pytest.fixture
def aero_mach():
    """AeroModel with Mach-dependent Cd table."""
    return AeroModel(
        Cd=0.3, Cl=0.05, Cm=0.003,
        reference_area_m2=0.01887,
        reference_diameter_m=0.155,
        mach_table=MACH_TABLE_155MM,
        cd_table=CD_TABLE_155MM,
    )


class TestMachDependentCd:
    """Test Mach-dependent drag coefficient interpolation."""

    def test_subsonic_cd(self, aero_mach):
        cd = aero_mach.get_Cd(0.4)
        assert cd == pytest.approx(0.15, abs=0.01)

    def test_transonic_cd_peak(self, aero_mach):
        cd = aero_mach.get_Cd(1.05)
        assert cd == pytest.approx(0.44, abs=0.01)

    def test_supersonic_cd(self, aero_mach):
        cd = aero_mach.get_Cd(2.0)
        assert cd == pytest.approx(0.30, abs=0.01)

    def test_constant_cd_fallback(self, aero_const):
        cd = aero_const.get_Cd(1.5)
        assert cd == 0.3  # constant regardless of Mach

    def test_cd_monotonic_beyond_transonic(self, aero_mach):
        """Cd should decrease from transonic peak toward hypersonic."""
        cd_1_1 = aero_mach.get_Cd(1.1)
        cd_2_0 = aero_mach.get_Cd(2.0)
        cd_3_0 = aero_mach.get_Cd(3.0)
        assert cd_1_1 > cd_2_0 > cd_3_0


class TestDragForce:
    """Test drag force computation."""

    def test_opposes_velocity(self, aero_const):
        v = np.array([100.0, 0.0, 0.0])
        F = aero_const.drag_force(v, 1.225)
        assert F[0] < 0  # opposes x-velocity
        assert abs(F[1]) < 1e-10
        assert abs(F[2]) < 1e-10

    def test_magnitude_scales_with_speed_squared(self, aero_const):
        rho = 1.225
        F1 = np.linalg.norm(aero_const.drag_force(np.array([100, 0, 0]), rho))
        F2 = np.linalg.norm(aero_const.drag_force(np.array([200, 0, 0]), rho))
        # F ∝ v², so F2/F1 ≈ 4
        assert F2 / F1 == pytest.approx(4.0, rel=0.01)

    def test_zero_velocity_returns_zero(self, aero_const):
        F = aero_const.drag_force(np.zeros(3), 1.225)
        np.testing.assert_array_equal(F, np.zeros(3))

    def test_mach_dependent_drag(self, aero_mach):
        """Transonic drag should be higher than subsonic."""
        v_sub = np.array([200.0, 0.0, 0.0])   # ~Mach 0.6
        v_trans = np.array([340.0, 0.0, 0.0])  # ~Mach 1.0
        rho = 1.225
        F_sub = np.linalg.norm(aero_mach.drag_force(v_sub, rho, mach=0.6))
        F_trans = np.linalg.norm(aero_mach.drag_force(v_trans, rho, mach=1.0))
        # Transonic has higher Cd AND higher speed → much more drag
        assert F_trans > F_sub


class TestLiftForce:
    """Test lift force computation."""

    def test_perpendicular_to_velocity(self, aero_const):
        v = np.array([100.0, 0.0, 50.0])
        F_lift = aero_const.lift_force(v, 1.225)
        # Lift should be perpendicular to velocity
        dot = np.dot(F_lift, v)
        assert abs(dot) < 1e-6

    def test_zero_when_vertical(self, aero_const):
        """Lift is zero when velocity is purely vertical (parallel to z)."""
        v = np.array([0.0, 0.0, 100.0])
        F_lift = aero_const.lift_force(v, 1.225)
        np.testing.assert_allclose(F_lift, np.zeros(3), atol=1e-10)


class TestMagnusForce:
    """Test Magnus force computation."""

    def test_cross_product_direction(self, aero_const):
        v = np.array([100.0, 0.0, 0.0])
        omega = np.array([1000.0, 0.0, 0.0])  # spin about x
        F = aero_const.magnus_force(v, omega, 1.225)
        # omega × v_hat = [1000,0,0] × [1,0,0] = [0,0,0]
        # Parallel spin and velocity → zero Magnus
        np.testing.assert_allclose(F, np.zeros(3), atol=1e-10)

    def test_nonzero_for_misaligned_spin(self, aero_const):
        v = np.array([100.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 1000.0])  # spin about z
        F = aero_const.magnus_force(v, omega, 1.225)
        assert np.linalg.norm(F) > 0


class TestMoments:
    """Test aerodynamic moment computation."""

    def test_pitch_damping_opposes_transverse_rate(self, aero_const):
        omega = np.array([0.0, 10.0, 0.0])  # pitch rate
        M = aero_const.pitch_damping_moment(omega, 300.0, 1.225)
        # Should oppose: Cmq < 0 → moment in same direction as omega when Cmq*omega<0
        assert M[1] != 0
        # The sign: Cmq is negative, omega_y > 0, so moment_y < 0 (opposing)
        assert M[1] * omega[1] < 0  # opposing sign

    def test_roll_damping_opposes_spin(self, aero_const):
        omega = np.array([1000.0, 0.0, 0.0])  # spin about x
        M = aero_const.roll_damping_moment(omega, 300.0, 1.225)
        assert M[0] != 0
        assert M[0] * omega[0] < 0  # opposing sign

    def test_total_moment_at_zero_speed(self, aero_const):
        omega = np.array([1000.0, 5.0, 5.0])
        M = aero_const.compute_total_moment(omega, 0.1, 0.0, 1.225)
        np.testing.assert_allclose(M, np.zeros(3), atol=1e-10)
