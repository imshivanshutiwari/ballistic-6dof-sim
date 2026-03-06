"""Tests for Level 300 features: fire solution, base bleed, moving target, MRSI, advanced terminal."""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fire_solution import solve_fire_solution, vacuum_range
from src.base_bleed import BaseBleedUnit, BASEBLEED_155MM
from src.moving_target import create_target, compute_lead, TARGET_PROFILES
from src.mrsi import solve_mrsi
from src.terminal_ballistics import TerminalBallisticsCalculator


class TestFireSolution:
    def test_vacuum_range_45deg(self):
        """At 45 deg, vacuum range = v^2/g."""
        r = vacuum_range(800.0, 45.0)
        assert r == pytest.approx(800**2 / 9.81, rel=0.01)

    def test_bisection_converges(self):
        """Fire solution should converge on a simple parabolic trajectory."""
        def fake_trajectory(elev_deg):
            r = vacuum_range(500.0, elev_deg)
            tof = 2 * 500.0 * math.sin(math.radians(elev_deg)) / 9.81
            return r, tof, 300.0

        result = solve_fire_solution(
            muzzle_velocity=500.0,
            target_range_m=20000.0,
            run_trajectory_fn=fake_trajectory,
            tolerance_m=50.0,
        )
        assert result["converged"] is True
        assert abs(result["error_m"]) < 50.0


class TestBaseBleed:
    def test_no_reduction_before_ignition(self):
        bb = BaseBleedUnit(BASEBLEED_155MM)
        state = bb.get_state(0.1)  # before ignition delay (0.5s)
        assert state.is_active is False
        assert state.drag_reduction_factor == 0.0

    def test_peak_reduction_during_burn(self):
        bb = BaseBleedUnit(BASEBLEED_155MM)
        state = bb.get_state(10.0)  # well into sustained burn
        assert state.is_active is True
        assert state.drag_reduction_factor == pytest.approx(0.30, abs=0.01)

    def test_no_reduction_after_burnout(self):
        bb = BaseBleedUnit(BASEBLEED_155MM)
        state = bb.get_state(30.0)  # after total duration
        assert state.is_active is False
        assert state.drag_reduction_factor == 0.0

    def test_drag_multiplier_during_peak(self):
        bb = BaseBleedUnit(BASEBLEED_155MM)
        mult = bb.drag_multiplier(10.0)
        assert mult == pytest.approx(0.70, abs=0.01)

    def test_ramp_up(self):
        bb = BaseBleedUnit(BASEBLEED_155MM)
        # At 50% ramp: t = ignition_delay + ramp_up/2 = 0.5 + 1.0 = 1.5
        state = bb.get_state(1.5)
        assert 0.1 < state.drag_reduction_factor < 0.25


class TestMovingTarget:
    def test_create_target_with_profile(self):
        tgt = create_target(range_m=5000, speed_ms=15, profile_name="tank")
        assert tgt.description == "Main Battle Tank (MBT)"
        assert np.linalg.norm(tgt.velocity_ms) == pytest.approx(15.0, abs=0.1)

    def test_lead_converges_perpendicular(self):
        """A crossing tank should require lateral lead."""
        tgt = create_target(range_m=5000, speed_ms=15, heading_deg=90.0)

        def est_tof(r):
            return 10.0  # ~10s flight

        lead = compute_lead(tgt, est_tof)
        assert lead.converged is True
        assert lead.lead_distance_m > 100  # tank moves ~150m in 10s

    def test_all_profiles_exist(self):
        for name in ["tank", "apc", "helicopter", "truck"]:
            assert name in TARGET_PROFILES


class TestMRSI:
    def test_schedule_basic(self):
        """MRSI with a simple vacuum model should produce 3 rounds."""
        def fake_traj(elev_deg):
            r = vacuum_range(800.0, elev_deg)
            tof = 2 * 800 * math.sin(math.radians(elev_deg)) / 9.81
            return r, tof, 400.0

        schedule = solve_mrsi(
            target_range_m=40000.0,
            muzzle_velocity=800.0,
            run_trajectory_fn=fake_traj,
            num_rounds=3,
            range_tolerance_m=5000.0,
        )
        assert schedule.num_rounds >= 2
        if schedule.num_rounds >= 2:
            # First round should have longest TOF (fired first)
            assert schedule.rounds[0].time_of_flight_s >= schedule.rounds[-1].time_of_flight_s
            # Fire delays should increase
            assert schedule.rounds[0].fire_delay_s == 0.0


class TestAdvancedTerminal:
    def test_concrete_penetration_positive(self):
        calc = TerminalBallisticsCalculator(
            mass_kg=43.5, diameter_m=0.155, is_he=True,
        )
        vel = np.array([200.0, 0.0, -300.0])
        result = calc.compute(vel, 1.0)
        assert result.concrete_penetration_m > 0.0

    def test_spall_greater_than_penetration(self):
        """Spall occurs at thicknesses greater than direct penetration."""
        calc = TerminalBallisticsCalculator(
            mass_kg=43.5, diameter_m=0.155, is_he=True,
        )
        vel = np.array([200.0, 0.0, -300.0])
        result = calc.compute(vel, 1.0)
        assert result.spall_thickness_mm > result.armor_penetration_mm_RHA

    def test_behind_armor_velocity_positive(self):
        calc = TerminalBallisticsCalculator(
            mass_kg=43.5, diameter_m=0.155, is_he=True,
        )
        vel = np.array([200.0, 0.0, -300.0])
        result = calc.compute(vel, 1.0)
        assert result.behind_armor_velocity_ms > 0.0

    def test_debris_cone_angle_valid(self):
        calc = TerminalBallisticsCalculator(
            mass_kg=43.5, diameter_m=0.155, is_he=True,
        )
        vel = np.array([200.0, 0.0, -300.0])
        result = calc.compute(vel, 1.0)
        assert 20.0 <= result.behind_armor_cone_deg <= 45.0
