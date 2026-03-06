"""
International Standard Atmosphere (ISA) Model.

Implements the ISA model for altitudes from 0 to 20,000 metres, covering
the troposphere (0–11 km) and lower stratosphere (11–20 km).  Provides
temperature, pressure, air density, and speed of sound at any given
geometric altitude.

Reference:
    U.S. Standard Atmosphere, 1976.  NOAA / NASA / USAF.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# ISA Constants
# ---------------------------------------------------------------------------
SEA_LEVEL_TEMPERATURE_K = 288.15          # T0  [K]
SEA_LEVEL_PRESSURE_PA = 101325.0          # P0  [Pa]
SEA_LEVEL_DENSITY_KG_M3 = 1.225          # rho0 [kg/m^3]
TROPOSPHERE_LAPSE_RATE_K_PER_M = -0.0065  # dT/dh in troposphere [K/m]
TROPOPAUSE_ALTITUDE_M = 11000.0           # boundary altitude [m]
TROPOPAUSE_TEMPERATURE_K = 216.65         # isothermal temperature [K]
TROPOPAUSE_PRESSURE_PA = 22632.0          # pressure at tropopause [Pa]
STRATOSPHERE_CEILING_M = 20000.0          # max modelled altitude [m]
SPECIFIC_GAS_CONSTANT_J_PER_KG_K = 287.05  # R for dry air [J/(kg·K)]
GRAVITY_MS2 = 9.80665                     # standard gravity [m/s^2]
RATIO_OF_SPECIFIC_HEATS = 1.4             # gamma for dry air


class ISAAtmosphere:
    """International Standard Atmosphere (ISA) model (0–20 km).

    Provides atmospheric properties as a function of geometric altitude.

    Usage
    -----
    >>> atm = ISAAtmosphere()
    >>> props = atm.get_properties(5000.0)
    >>> print(props['density_kg_m3'])
    """

    # Pre-computed exponent for the troposphere barometric formula
    _TROPOSPHERE_PRESSURE_EXPONENT = (
        GRAVITY_MS2
        / (-TROPOSPHERE_LAPSE_RATE_K_PER_M * SPECIFIC_GAS_CONSTANT_J_PER_KG_K)
    )  # ≈ 5.2561

    # Pre-computed constant for the stratosphere exponential decay
    _STRATOSPHERE_DECAY_RATE = (
        GRAVITY_MS2
        / (SPECIFIC_GAS_CONSTANT_J_PER_KG_K * TROPOPAUSE_TEMPERATURE_K)
    )  # ≈ 1.577e-4  [1/m]

    def get_properties(self, altitude_m: float) -> dict:
        """Return atmospheric properties at a given geometric altitude.

        Parameters
        ----------
        altitude_m : float
            Geometric altitude above mean sea level [m].
            Must be in the range [0, 20 000].

        Returns
        -------
        dict
            Keys:
            - ``temperature_K``       : static temperature [K]
            - ``pressure_Pa``         : static pressure [Pa]
            - ``density_kg_m3``       : air density [kg/m³]
            - ``speed_of_sound_m_s``  : speed of sound [m/s]

        Raises
        ------
        ValueError
            If *altitude_m* is outside the range [0, 20 000] m.
        """
        if altitude_m < 0.0 or altitude_m > STRATOSPHERE_CEILING_M:
            raise ValueError(
                f"Altitude {altitude_m:.1f} m is outside the valid ISA range "
                f"[0, {STRATOSPHERE_CEILING_M:.0f}] m."
            )

        if altitude_m <= TROPOPAUSE_ALTITUDE_M:
            # --- TROPOSPHERE (0 – 11 km) ---
            temperature_K = (
                SEA_LEVEL_TEMPERATURE_K
                + TROPOSPHERE_LAPSE_RATE_K_PER_M * altitude_m
            )
            pressure_Pa = SEA_LEVEL_PRESSURE_PA * (
                temperature_K / SEA_LEVEL_TEMPERATURE_K
            ) ** self._TROPOSPHERE_PRESSURE_EXPONENT
        else:
            # --- LOWER STRATOSPHERE (11 – 20 km, isothermal) ---
            temperature_K = TROPOPAUSE_TEMPERATURE_K
            pressure_Pa = TROPOPAUSE_PRESSURE_PA * math.exp(
                -self._STRATOSPHERE_DECAY_RATE
                * (altitude_m - TROPOPAUSE_ALTITUDE_M)
            )

        density_kg_m3 = pressure_Pa / (
            SPECIFIC_GAS_CONSTANT_J_PER_KG_K * temperature_K
        )
        speed_of_sound_m_s = math.sqrt(
            RATIO_OF_SPECIFIC_HEATS
            * SPECIFIC_GAS_CONSTANT_J_PER_KG_K
            * temperature_K
        )

        return {
            "temperature_K": temperature_K,
            "pressure_Pa": pressure_Pa,
            "density_kg_m3": density_kg_m3,
            "speed_of_sound_m_s": speed_of_sound_m_s,
        }
