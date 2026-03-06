"""
Advanced Weather & Non-Standard Atmosphere Model.

Extends the ISA model with real-world corrections:

    1. **Non-Standard Day profiles** (Hot Day, Cold Day, Tropical, MIL-STD-210)
    2. **Temperature offset** (ΔT from ISA)
    3. **Humidity correction** (virtual temperature for density)
    4. **Custom surface conditions** (arbitrary T₀, P₀)

The key insight: a +15 °C temperature deviation from ISA reduces air density
by ~5 %, which extends artillery range by 200–400 m on a 20 km shot.

References:
    - MIL-STD-210C: Climatic Information to Determine Design and Test
      Requirements for Military Systems and Equipment. 1987.
    - MIL-HDBK-310: Global Climatic Data for Developing Military Products. 1997.
"""

import math
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard atmosphere constants (same as ISA)
# ---------------------------------------------------------------------------
_T0_ISA = 288.15          # ISA sea-level temperature [K]
_P0_ISA = 101_325.0       # ISA sea-level pressure [Pa]
_LAPSE_RATE = -0.0065     # troposphere lapse rate [K/m]
_TROPOPAUSE_ALT = 11_000  # tropopause altitude [m]
_R_AIR = 287.058          # specific gas constant for dry air [J/(kg·K)]
_GAMMA_AIR = 1.4          # ratio of specific heats
_G0 = 9.80665             # standard gravity [m/s²]


# ---------------------------------------------------------------------------
# Pre-defined weather profiles  (ΔT from ISA at sea level)
# ---------------------------------------------------------------------------
WEATHER_PROFILES = {
    "standard": {
        "name": "ISA Standard Day",
        "delta_T_K": 0.0,
        "relative_humidity": 0.0,
        "description": "Standard 15 °C / 1013.25 hPa day",
    },
    "hot": {
        "name": "MIL-STD-210 Hot Day (+25 K)",
        "delta_T_K": +25.0,
        "relative_humidity": 0.30,
        "description": "Desert conditions, ~40 °C at sea level",
    },
    "cold": {
        "name": "MIL-STD-210 Cold Day (−35 K)",
        "delta_T_K": -35.0,
        "relative_humidity": 0.60,
        "description": "Arctic / high-altitude winter, ~−20 °C at sea level",
    },
    "tropical": {
        "name": "Tropical Day (+20 K, high humidity)",
        "delta_T_K": +20.0,
        "relative_humidity": 0.85,
        "description": "Hot and humid equatorial conditions",
    },
    "mild": {
        "name": "Temperate Day (+5 K)",
        "delta_T_K": +5.0,
        "relative_humidity": 0.50,
        "description": "Mild European summer",
    },
}


def saturation_vapour_pressure(T_K: float) -> float:
    """Compute saturation vapour pressure using the Buck equation.

    Parameters
    ----------
    T_K : float
        Temperature [K].

    Returns
    -------
    float
        Saturation vapour pressure [Pa].
    """
    T_C = T_K - 273.15
    if T_C < -40:
        return 0.0  # negligible below −40 °C
    return 611.21 * math.exp((18.678 - T_C / 234.5) * (T_C / (257.14 + T_C)))


class WeatherModel:
    """Non-standard atmosphere with temperature offset and humidity.

    Parameters
    ----------
    delta_T_K : float
        Temperature offset from ISA at all altitudes [K].
        +25 = hot day, −35 = cold day.
    relative_humidity : float
        Relative humidity [0.0 – 1.0].
    surface_pressure_Pa : float, optional
        Override surface pressure [Pa].  Default: 101325 Pa.
    profile_name : str, optional
        Name of a pre-defined weather profile.  Overrides delta_T_K and
        relative_humidity if provided.
    """

    def __init__(
        self,
        delta_T_K: float = 0.0,
        relative_humidity: float = 0.0,
        surface_pressure_Pa: float = _P0_ISA,
        profile_name: Optional[str] = None,
    ):
        if profile_name and profile_name in WEATHER_PROFILES:
            prof = WEATHER_PROFILES[profile_name]
            self.delta_T = prof["delta_T_K"]
            self.RH = prof["relative_humidity"]
            self.name = prof["name"]
            logger.info("Weather: %s", self.name)
        else:
            self.delta_T = delta_T_K
            self.RH = max(0.0, min(1.0, relative_humidity))
            self.name = f"Custom (ΔT={delta_T_K:+.1f} K, RH={relative_humidity:.0%})"

        self.P0 = surface_pressure_Pa

    def get_properties(self, altitude_m: float) -> dict:
        """Compute atmosphere properties at a given altitude.

        Parameters
        ----------
        altitude_m : float
            Geometric altitude above sea level [m].  Clamped to [0, 20000].

        Returns
        -------
        dict
            Keys: temperature_K, pressure_Pa, density_kg_m3,
            speed_of_sound_ms, virtual_temperature_K
        """
        alt = max(0.0, min(altitude_m, 20_000.0))

        # --- Temperature (ISA + offset) ---
        if alt <= _TROPOPAUSE_ALT:
            T_isa = _T0_ISA + _LAPSE_RATE * alt
        else:
            T_isa = 216.65  # isothermal stratosphere

        T = T_isa + self.delta_T
        T = max(T, 100.0)  # physical floor

        # --- Pressure (hypsometric with offset) ---
        if alt <= _TROPOPAUSE_ALT:
            T0_mod = _T0_ISA + self.delta_T
            exponent = _G0 / (-_LAPSE_RATE * _R_AIR)
            P = self.P0 * (T / T0_mod) ** exponent
        else:
            # Pressure at tropopause
            T_trop = (_T0_ISA + _LAPSE_RATE * _TROPOPAUSE_ALT) + self.delta_T
            T0_mod = _T0_ISA + self.delta_T
            exponent = _G0 / (-_LAPSE_RATE * _R_AIR)
            P_trop = self.P0 * (T_trop / T0_mod) ** exponent
            # Isothermal decay above tropopause
            T_strat = 216.65 + self.delta_T
            T_strat = max(T_strat, 100.0)
            P = P_trop * math.exp(-_G0 * (alt - _TROPOPAUSE_ALT) / (_R_AIR * T_strat))

        # --- Humidity correction (virtual temperature) ---
        e_sat = saturation_vapour_pressure(T)
        e = self.RH * e_sat
        # Virtual temperature: accounts for water vapour being lighter than
        # dry air → reduces effective density
        T_v = T / (1.0 - (e / P) * (1.0 - 0.622)) if P > e else T

        # --- Density (ideal gas with virtual temperature) ---
        rho = P / (_R_AIR * T_v)

        # --- Speed of sound ---
        a = math.sqrt(_GAMMA_AIR * _R_AIR * T)

        return {
            "temperature_K": T,
            "pressure_Pa": P,
            "density_kg_m3": rho,
            "speed_of_sound_ms": a,
            "virtual_temperature_K": T_v,
        }
