"""
Rocket Propulsion Module — Time-Dependent Thrust & Mass Depletion.

Models a solid-fuel rocket motor with:

    - **Thrust curve** T(t) — time-dependent thrust profile [N]
    - **Mass depletion** m(t) — decreasing mass as fuel burns
    - **Specific impulse** Isp — propulsive efficiency [s]
    - **Burn time** — duration of powered flight [s]
    - **Centre-of-gravity shift** — optional Xcg(t) during burn

The thrust profile is defined by breakpoint pairs (time, thrust) and
interpolated linearly.  After burnout, thrust = 0 and mass = dry mass.

References:
    - Sutton, G. P. & Biblarz, O.  *Rocket Propulsion Elements*, 9th ed.
      Wiley, 2017.
    - Zarchan, P.  *Tactical and Strategic Missile Guidance*, 6th ed.
      AIAA, 2012.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PropulsionState:
    """Instantaneous propulsion state at time t.

    Attributes
    ----------
    thrust_N : float
        Current thrust [N].  Zero after burnout.
    mass_kg : float
        Current total mass (structural + remaining fuel) [kg].
    is_burning : bool
        True if the motor is still burning.
    fuel_remaining_kg : float
        Remaining fuel mass [kg].
    """
    thrust_N: float
    mass_kg: float
    is_burning: bool
    fuel_remaining_kg: float


# ---------------------------------------------------------------------------
# Pre-defined rocket motor profiles
# ---------------------------------------------------------------------------

MOTOR_122MM_GRAD = {
    "name": "9M22U Motor  (BM-21 Grad 122 mm)",
    "total_mass_kg": 20.0,
    "fuel_mass_kg": 8.7,
    "dry_mass_kg": 11.3,
    "Isp_s": 230.0,
    "burn_time_s": 1.7,
    "thrust_profile": np.array([
        [0.00,     0], [0.05, 15000], [0.10, 25000], [0.30, 23000],
        [1.00, 22000], [1.50, 18000], [1.65, 10000], [1.70,     0],
    ], dtype=float),
}

# --- Generic Ballistic Missile Motors (Estimates based on range and mass) ---

MOTOR_V2 = {
    "name": "V-2 Liquid Rocket Engine",
    "total_mass_kg": 12500.0, "fuel_mass_kg": 8610.0, "dry_mass_kg": 3890.0,
    "Isp_s": 210.0, "burn_time_s": 65.0,
    "thrust_profile": np.array([
        [0.0, 0], [2.0, 250000], [60.0, 250000], [65.0, 0]
    ], dtype=float),
}

MOTOR_SCUD_B = {
    "name": "Scud-B Isayev Engine",
    "total_mass_kg": 5900.0, "fuel_mass_kg": 3700.0, "dry_mass_kg": 2200.0,
    "Isp_s": 230.0, "burn_time_s": 62.0,
    "thrust_profile": np.array([
        [0.0, 0], [1.0, 130000], [60.0, 130000], [62.0, 0]
    ], dtype=float),
}

MOTOR_M26_MLRS = {
    "name": "M26 Solid Rocket Motor",
    "total_mass_kg": 306.0, "fuel_mass_kg": 150.0, "dry_mass_kg": 156.0,
    "Isp_s": 240.0, "burn_time_s": 4.0,
    "thrust_profile": np.array([
        [0.0, 0], [0.2, 80000], [3.5, 75000], [4.0, 0]
    ], dtype=float),
}

MOTOR_TOMAHAWK_BOOSTER = {
    "name": "Tomahawk Solid Booster (Launch phase only)",
    "total_mass_kg": 1600.0, "fuel_mass_kg": 285.0, "dry_mass_kg": 1315.0,
    "Isp_s": 250.0, "burn_time_s": 12.0,
    "thrust_profile": np.array([
        [0.0, 0], [0.5, 55000], [11.0, 50000], [12.0, 0]
    ], dtype=float),
}

# Representative 3-stage solid motor profile for heavy ICBMs/IRBMs (Agni/Minuteman)
# Modeled as a continuous burn sequence for simplicity in 6-DOF
MOTOR_HEAVY_ICBM_AGNI_V = {
    "name": "3-Stage Solid ICBM Motor (Agni-V Estimate)",
    "total_mass_kg": 50000.0, "fuel_mass_kg": 40000.0, "dry_mass_kg": 10000.0,
    "Isp_s": 280.0, "burn_time_s": 180.0,
    "thrust_profile": np.array([
        [0.0, 0], [2.0, 1500000], [60.0, 1500000],    # Stage 1
        [61.0, 800000], [120.0, 800000],              # Stage 2
        [121.0, 300000], [178.0, 300000], [180.0, 0]  # Stage 3
    ], dtype=float),
}

MOTOR_MEDIUM_IRBM_AGNI_II = {
    "name": "2-Stage Solid IRBM Motor (Agni-II Estimate)",
    "total_mass_kg": 17000.0, "fuel_mass_kg": 13000.0, "dry_mass_kg": 4000.0,
    "Isp_s": 260.0, "burn_time_s": 100.0,
    "thrust_profile": np.array([
        [0.0, 0], [1.0, 600000], [50.0, 600000],      # Stage 1
        [51.0, 250000], [98.0, 200000], [100.0, 0]    # Stage 2
    ], dtype=float),
}


ALL_MOTOR_CONFIGS = {
    "122mm": MOTOR_122MM_GRAD,
    "v2": MOTOR_V2,
    "scudb": MOTOR_SCUD_B,
    "m26": MOTOR_M26_MLRS,
    "tomahawk": MOTOR_TOMAHAWK_BOOSTER,
    # Assign the generic heavy motor to the large ICBMs
    "agni5": MOTOR_HEAVY_ICBM_AGNI_V,
    "agni3": MOTOR_HEAVY_ICBM_AGNI_V, 
    "minuteman3": MOTOR_HEAVY_ICBM_AGNI_V,
    # Assign the medium motor to the IRBMs/SRBMs
    "agni1": MOTOR_MEDIUM_IRBM_AGNI_II,
    "agni2": MOTOR_MEDIUM_IRBM_AGNI_II,
    "agni4": MOTOR_MEDIUM_IRBM_AGNI_II,
    "agnip": MOTOR_MEDIUM_IRBM_AGNI_II,
}


# ---------------------------------------------------------------------------
# Propulsion model
# ---------------------------------------------------------------------------

class RocketMotor:
    """Time-dependent solid-rocket motor model.

    Parameters
    ----------
    motor_config : dict
        Motor configuration dictionary with thrust profile, masses, and Isp.
    """

    def __init__(self, motor_config: dict):
        self.config = motor_config
        self.name = motor_config["name"]
        self.total_mass = motor_config["total_mass_kg"]
        self.fuel_mass = motor_config["fuel_mass_kg"]
        self.dry_mass = motor_config["dry_mass_kg"]
        self.Isp = motor_config["Isp_s"]
        self.burn_time = motor_config["burn_time_s"]

        # Thrust profile breakpoints
        profile = motor_config["thrust_profile"]
        self._t_bp = profile[:, 0]
        self._T_bp = profile[:, 1]

        # Total impulse (integral of thrust curve)
        self.total_impulse_Ns = float(np.trapezoid(self._T_bp, self._t_bp))

        # Mass flow rate profile (derived from thrust / (Isp × g₀))
        g0 = 9.80665
        self._mdot_bp = self._T_bp / (self.Isp * g0)

        # Cumulative fuel consumed at each breakpoint
        self._fuel_consumed = np.zeros_like(self._t_bp)
        for i in range(1, len(self._t_bp)):
            dt = self._t_bp[i] - self._t_bp[i - 1]
            avg_mdot = 0.5 * (self._mdot_bp[i - 1] + self._mdot_bp[i])
            self._fuel_consumed[i] = self._fuel_consumed[i - 1] + avg_mdot * dt

        logger.info(
            "Rocket motor: %s  |  Fuel=%.1f kg  Isp=%.0f s  "
            "Burn=%.2f s  Impulse=%.0f N·s",
            self.name, self.fuel_mass, self.Isp,
            self.burn_time, self.total_impulse_Ns,
        )

    def get_state(self, t: float) -> PropulsionState:
        """Get the propulsion state at time t.

        Parameters
        ----------
        t : float
            Time since launch [s].

        Returns
        -------
        PropulsionState
        """
        if t < 0 or t >= self.burn_time:
            return PropulsionState(
                thrust_N=0.0,
                mass_kg=self.dry_mass,
                is_burning=False,
                fuel_remaining_kg=0.0,
            )

        # Interpolate thrust
        thrust = float(np.interp(t, self._t_bp, self._T_bp))

        # Interpolate fuel consumed
        fuel_used = float(np.interp(t, self._t_bp, self._fuel_consumed))
        fuel_used = min(fuel_used, self.fuel_mass)
        fuel_remaining = self.fuel_mass - fuel_used

        current_mass = self.dry_mass + fuel_remaining

        return PropulsionState(
            thrust_N=thrust,
            mass_kg=current_mass,
            is_burning=True,
            fuel_remaining_kg=fuel_remaining,
        )
