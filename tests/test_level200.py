"""Tests for Level 200 features: WGS-84, weather, propulsion, propellant temp."""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.wgs84_gravity import wgs84_gravity, somigliana_gravity
from src.weather import WeatherModel, WEATHER_PROFILES
from src.propulsion import RocketMotor, MOTOR_122MM_GRAD
from src.interior_ballistics import InteriorBallisticsSolver, GUN_155MM_M109


class TestWGS84Gravity:
    def test_equator_sea_level(self):
        g = wgs84_gravity(0.0, 0.0)
        assert g == pytest.approx(9.780, abs=0.01)

    def test_pole_sea_level(self):
        g = wgs84_gravity(90.0, 0.0)
        assert g == pytest.approx(9.832, abs=0.01)

    def test_pole_greater_than_equator(self):
        assert wgs84_gravity(90.0, 0.0) > wgs84_gravity(0.0, 0.0)

    def test_decreases_with_altitude(self):
        g0 = wgs84_gravity(45.0, 0.0)
        g10k = wgs84_gravity(45.0, 10000.0)
        assert g0 > g10k

    def test_pune_latitude(self):
        """Gravity at Pune (18.5°N) should be ~9.788 m/s²."""
        g = wgs84_gravity(18.5, 0.0)
        assert 9.78 < g < 9.80


class TestWeatherModel:
    def test_standard_matches_isa(self):
        w = WeatherModel(profile_name="standard")
        props = w.get_properties(0.0)
        assert props["temperature_K"] == pytest.approx(288.15, abs=0.1)

    def test_hot_day_warmer(self):
        std = WeatherModel(profile_name="standard")
        hot = WeatherModel(profile_name="hot")
        T_std = std.get_properties(0.0)["temperature_K"]
        T_hot = hot.get_properties(0.0)["temperature_K"]
        assert T_hot > T_std

    def test_cold_day_colder(self):
        std = WeatherModel(profile_name="standard")
        cold = WeatherModel(profile_name="cold")
        T_std = std.get_properties(0.0)["temperature_K"]
        T_cold = cold.get_properties(0.0)["temperature_K"]
        assert T_cold < T_std

    def test_hot_day_lower_density(self):
        """Hot air is less dense → less drag → longer range."""
        std = WeatherModel(profile_name="standard")
        hot = WeatherModel(profile_name="hot")
        rho_std = std.get_properties(0.0)["density_kg_m3"]
        rho_hot = hot.get_properties(0.0)["density_kg_m3"]
        assert rho_hot < rho_std

    def test_humidity_reduces_density(self):
        """High humidity → lighter air (water vapor lighter than N2/O2)."""
        dry = WeatherModel(delta_T_K=20.0, relative_humidity=0.0)
        humid = WeatherModel(delta_T_K=20.0, relative_humidity=0.95)
        rho_dry = dry.get_properties(0.0)["density_kg_m3"]
        rho_humid = humid.get_properties(0.0)["density_kg_m3"]
        assert rho_humid < rho_dry

    def test_all_profiles_exist(self):
        for name in ["standard", "hot", "cold", "tropical", "mild"]:
            assert name in WEATHER_PROFILES


class TestRocketMotor:
    def test_thrust_at_peak(self):
        motor = RocketMotor(MOTOR_122MM_GRAD)
        state = motor.get_state(0.10)
        assert state.thrust_N == pytest.approx(25000, abs=100)

    def test_no_thrust_after_burnout(self):
        motor = RocketMotor(MOTOR_122MM_GRAD)
        state = motor.get_state(2.0)
        assert state.thrust_N == 0.0
        assert state.is_burning is False

    def test_mass_decreases_during_burn(self):
        motor = RocketMotor(MOTOR_122MM_GRAD)
        m_start = motor.get_state(0.05).mass_kg
        m_end = motor.get_state(1.5).mass_kg
        assert m_end < m_start

    def test_dry_mass_after_burnout(self):
        motor = RocketMotor(MOTOR_122MM_GRAD)
        state = motor.get_state(5.0)
        assert state.mass_kg == pytest.approx(11.3, abs=0.1)

    def test_total_impulse_positive(self):
        motor = RocketMotor(MOTOR_122MM_GRAD)
        assert motor.total_impulse_Ns > 10000


class TestPropellantTemperature:
    def test_hot_propellant_higher_velocity(self):
        solver = InteriorBallisticsSolver(GUN_155MM_M109)
        cold = solver.solve(propellant_temp_C=0.0)
        hot = solver.solve(propellant_temp_C=50.0)
        assert hot.muzzle_velocity_ms > cold.muzzle_velocity_ms

    def test_standard_temp_unchanged(self):
        solver = InteriorBallisticsSolver(GUN_155MM_M109)
        r1 = solver.solve(propellant_temp_C=21.0)
        r2 = solver.solve()  # default = 21
        assert r1.muzzle_velocity_ms == pytest.approx(r2.muzzle_velocity_ms, rel=1e-10)
