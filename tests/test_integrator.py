"""
Unit tests for the Trajectory Integrator.

Tests cover:
    - Basic trajectory runs without crash
    - Ground-impact event terminates integration
    - Vacuum trajectory matches analytical range v²/g
    - Energy conservation in vacuum
    - Result dataclass contains expected fields
    - Derived quantities (speed, Mach) are physically reasonable
"""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.atmosphere import ISAAtmosphere
from src.aerodynamics import AeroModel
from src.integrator import TrajectoryIntegrator, TrajectoryResult


GRAVITY_MS2 = 9.80665


@pytest.fixture
def atm():
    return ISAAtmosphere()


@pytest.fixture
def zero_aero():
    return AeroModel(
        Cd=0.0, Cl=0.0, Cm=0.0,
        reference_area_m2=0.01,
        reference_diameter_m=0.1,
        Cmq=0.0, Clp=0.0, Cnpa=0.0, Cma=0.0,
    )


@pytest.fixture
def integrator():
    return TrajectoryIntegrator()


def _make_state(v0, elev_deg):
    """Create initial state for a given speed and elevation."""
    elev = math.radians(elev_deg)
    return np.array([
        0, 0, 0,
        v0 * math.cos(elev), 0, v0 * math.sin(elev),
        1, 0, 0, 0,
        0, 0, 0,
    ], dtype=float)


class TestBasicIntegration:
    """Test that the integrator runs and returns valid results."""

    def test_returns_trajectory_result(self, integrator, zero_aero, atm):
        state0 = _make_state(100, 45)
        result = integrator.integrate(
            state0, (0, 100), 0.1, 1.0,
            np.diag([0.001]*3), zero_aero, atm, np.zeros(3),
        )
        assert isinstance(result, TrajectoryResult)

    def test_time_array_is_increasing(self, integrator, zero_aero, atm):
        state0 = _make_state(100, 45)
        result = integrator.integrate(
            state0, (0, 100), 0.1, 1.0,
            np.diag([0.001]*3), zero_aero, atm, np.zeros(3),
        )
        assert np.all(np.diff(result.time_s) > 0)

    def test_state_shape(self, integrator, zero_aero, atm):
        state0 = _make_state(100, 45)
        result = integrator.integrate(
            state0, (0, 100), 0.1, 1.0,
            np.diag([0.001]*3), zero_aero, atm, np.zeros(3),
        )
        assert result.state.shape[1] == 13
        assert result.state.shape[0] == len(result.time_s)


class TestGroundImpact:
    """Test that integration terminates at ground impact."""

    def test_final_altitude_near_zero(self, integrator, zero_aero, atm):
        state0 = _make_state(100, 45)
        result = integrator.integrate(
            state0, (0, 100), 0.05, 1.0,
            np.diag([0.001]*3), zero_aero, atm, np.zeros(3),
        )
        # Final z should be near zero (ground)
        assert result.state[-1, 2] <= 2.0  # within 2 m of ground (RK45 event tolerance)


class TestVacuumTrajectory:
    """Validate vacuum range against analytical v²/g."""

    def test_range_matches_analytical(self, integrator, zero_aero, atm):
        v0 = 300.0
        state0 = _make_state(v0, 45)
        result = integrator.integrate(
            state0, (0, 200), 0.01, 1.0,
            np.diag([0.001]*3), zero_aero, atm, np.zeros(3),
        )
        sim_range = result.state[-1, 0]
        analytical_range = v0**2 / GRAVITY_MS2
        error_pct = abs(sim_range - analytical_range) / analytical_range * 100
        assert error_pct < 1.0, f"Vacuum range error {error_pct:.2f}% exceeds 1%"


class TestEnergyConservation:
    """In vacuum, total mechanical energy must be conserved."""

    def test_energy_deviation_below_threshold(self, integrator, zero_aero, atm):
        v0 = 300.0
        mass = 1.0
        state0 = _make_state(v0, 45)
        result = integrator.integrate(
            state0, (0, 200), 0.01, mass,
            np.diag([0.001]*3), zero_aero, atm, np.zeros(3),
        )
        z = result.state[:, 2]
        speed = result.speed_ms
        KE = 0.5 * mass * speed**2
        PE = mass * GRAVITY_MS2 * z
        E_total = KE + PE
        E0 = E_total[0]
        max_dev_pct = np.max(np.abs(E_total - E0)) / abs(E0) * 100
        assert max_dev_pct < 0.1, f"Energy deviation {max_dev_pct:.4f}% exceeds 0.1%"


class TestDerivedQuantities:
    """Test that derived quantities are physically reasonable."""

    def test_speed_is_positive(self, integrator, zero_aero, atm):
        state0 = _make_state(200, 45)
        result = integrator.integrate(
            state0, (0, 100), 0.1, 1.0,
            np.diag([0.001]*3), zero_aero, atm, np.zeros(3),
        )
        assert np.all(result.speed_ms >= 0)

    def test_mach_number_reasonable(self, integrator, zero_aero, atm):
        state0 = _make_state(200, 45)
        result = integrator.integrate(
            state0, (0, 100), 0.1, 1.0,
            np.diag([0.001]*3), zero_aero, atm, np.zeros(3),
        )
        # At 200 m/s, Mach should be around 0.5-0.6
        assert result.mach_number[0] == pytest.approx(200 / 340.29, abs=0.05)
