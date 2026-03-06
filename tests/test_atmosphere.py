"""
Unit tests for the ISA Atmosphere model.

Tests cover:
    - Sea-level standard values (temperature, pressure, density, speed of sound)
    - Tropopause boundary conditions
    - Stratosphere isothermal region
    - Edge cases (altitude = 0, altitude = 20 000 m)
    - Invalid altitude handling
"""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.atmosphere import (
    ISAAtmosphere,
    SEA_LEVEL_TEMPERATURE_K,
    SEA_LEVEL_PRESSURE_PA,
    SEA_LEVEL_DENSITY_KG_M3,
    TROPOPAUSE_TEMPERATURE_K,
    TROPOPAUSE_ALTITUDE_M,
    STRATOSPHERE_CEILING_M,
)


@pytest.fixture
def atm():
    """Provide an ISAAtmosphere instance."""
    return ISAAtmosphere()


class TestSeaLevel:
    """Validate sea-level (h = 0) atmospheric properties."""

    def test_temperature_exact(self, atm):
        props = atm.get_properties(0.0)
        assert props["temperature_K"] == 288.15

    def test_pressure_exact(self, atm):
        props = atm.get_properties(0.0)
        assert props["pressure_Pa"] == pytest.approx(101325.0, rel=1e-6)

    def test_density_within_tolerance(self, atm):
        props = atm.get_properties(0.0)
        assert props["density_kg_m3"] == pytest.approx(1.225, rel=1e-3)

    def test_speed_of_sound(self, atm):
        props = atm.get_properties(0.0)
        # Expected: sqrt(1.4 * 287.05 * 288.15) ≈ 340.29 m/s
        assert props["speed_of_sound_m_s"] == pytest.approx(340.29, abs=0.1)


class TestTroposphere:
    """Validate troposphere (0 – 11 km)."""

    def test_temperature_at_5000m(self, atm):
        props = atm.get_properties(5000.0)
        expected_T = 288.15 - 0.0065 * 5000  # = 255.65 K
        assert props["temperature_K"] == pytest.approx(expected_T, abs=0.01)

    def test_density_decreases_with_altitude(self, atm):
        rho_0 = atm.get_properties(0)["density_kg_m3"]
        rho_5 = atm.get_properties(5000)["density_kg_m3"]
        rho_10 = atm.get_properties(10000)["density_kg_m3"]
        assert rho_0 > rho_5 > rho_10

    def test_tropopause_boundary(self, atm):
        props = atm.get_properties(TROPOPAUSE_ALTITUDE_M)
        assert props["temperature_K"] == pytest.approx(
            TROPOPAUSE_TEMPERATURE_K, abs=0.01
        )


class TestStratosphere:
    """Validate lower stratosphere (11 – 20 km, isothermal)."""

    def test_isothermal_temperature(self, atm):
        for h in [12000, 15000, 18000, 20000]:
            props = atm.get_properties(h)
            assert props["temperature_K"] == pytest.approx(
                TROPOPAUSE_TEMPERATURE_K, abs=0.01
            )

    def test_pressure_decay(self, atm):
        p_11 = atm.get_properties(11000)["pressure_Pa"]
        p_15 = atm.get_properties(15000)["pressure_Pa"]
        p_20 = atm.get_properties(20000)["pressure_Pa"]
        assert p_11 > p_15 > p_20


class TestEdgeCases:
    """Test boundary and invalid inputs."""

    def test_altitude_zero(self, atm):
        # Should not raise
        atm.get_properties(0.0)

    def test_altitude_ceiling(self, atm):
        # Should not raise
        atm.get_properties(STRATOSPHERE_CEILING_M)

    def test_negative_altitude_raises(self, atm):
        with pytest.raises(ValueError, match="outside the valid ISA range"):
            atm.get_properties(-1.0)

    def test_above_ceiling_raises(self, atm):
        with pytest.raises(ValueError, match="outside the valid ISA range"):
            atm.get_properties(20001.0)
