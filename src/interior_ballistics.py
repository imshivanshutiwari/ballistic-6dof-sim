"""
Interior Ballistics Module — LeDuc Pressure Model.

Simulates the projectile's acceleration inside the gun barrel, from
ignition to muzzle exit.  Provides:

    - In-barrel velocity profile  v(x)
    - Chamber pressure curve      P(x)
    - Muzzle velocity             V_muzzle
    - Muzzle spin rate            ω_muzzle  (from rifling twist)

The LeDuc model is a classical closed-form interior ballistics method
widely used for preliminary design and sanity-checking.  It relates
the projectile travel distance *x* inside the barrel to velocity via:

    v(x) = V_lim * x / (l_0 + x)

where V_lim is the limiting (asymptotic) velocity and l_0 is the
LeDuc pressure-distance parameter.

From this, the peak (breech) pressure and the muzzle pressure can be
derived analytically.

References:
    Carlucci, D. E. & Jacobson, S. S.  *Ballistics: Theory and Design
    of Guns and Ammunition*, 2nd ed.  CRC Press, 2013.

    Corner, J.  *Theory of the Interior Ballistics of Guns*.  Wiley, 1950.
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
class InteriorBallisticsResult:
    """Container for interior ballistics computation results.

    Attributes
    ----------
    muzzle_velocity_ms : float
        Velocity of the projectile at the muzzle [m/s].
    muzzle_spin_rate_rad_s : float
        Axial spin rate at muzzle exit [rad/s].
    peak_pressure_MPa : float
        Peak (breech) chamber pressure [MPa].
    muzzle_pressure_MPa : float
        Gas pressure at the moment of muzzle exit [MPa].
    travel_m : np.ndarray
        Projectile travel distance inside the barrel [m].
    velocity_ms : np.ndarray
        Velocity profile along the barrel [m/s].
    pressure_MPa : np.ndarray
        Pressure profile along the barrel [MPa].
    """
    muzzle_velocity_ms: float
    muzzle_spin_rate_rad_s: float
    peak_pressure_MPa: float
    muzzle_pressure_MPa: float
    travel_m: np.ndarray
    velocity_ms: np.ndarray
    pressure_MPa: np.ndarray


# ---------------------------------------------------------------------------
# Gun / propellant configuration presets
# ---------------------------------------------------------------------------

GUN_155MM_M109 = {
    "name": "155 mm / 39 cal  (M109 Howitzer)",
    "barrel_length_m": 6.045,       # 39 calibres × 0.155 m
    "bore_area_m2": 0.01887,        # π/4 × 0.155²
    "chamber_volume_m3": 0.01872,   # ~18.72 litres
    "projectile_mass_kg": 43.5,
    "propellant_mass_kg": 10.4,     # Zone 7 full charge
    "propellant_force_J_kg": 1.00e6,  # ~ 1 MJ/kg for single-base propellant
    "covolume_m3_kg": 1.0e-3,       # co-volume of propellant gas
    "gamma": 1.25,                  # ratio of specific heats for propellant gas
    "twist_calibres": 20.0,         # 1 turn in 20 calibres (right-hand)
    "bore_diameter_m": 0.155,
}

GUN_127MM_M2 = {
    "name": "12.7 mm / .50 BMG  (M2 Heavy Machine Gun)",
    "barrel_length_m": 1.143,       # 45 inches
    "bore_area_m2": 1.267e-4,       # π/4 × 0.0127²
    "chamber_volume_m3": 1.54e-5,   # ~15.4 cm³
    "projectile_mass_kg": 0.046,
    "propellant_mass_kg": 0.0156,   # ~15.6 g
    "propellant_force_J_kg": 1.05e6,
    "covolume_m3_kg": 1.0e-3,
    "gamma": 1.25,
    "twist_calibres": 32.0,        # 1 turn in 32 calibres (left-hand)
    "bore_diameter_m": 0.0127,
}

# Used to launch missiles from rest (0 m/s)
ZERO_LAUNCH_PAD = {
    "name": "Launch Pad / Silo (Zero velocity)",
    "barrel_length_m": 0.01,        # negligible travel
    "bore_area_m2": 1.0,            # arbitrary
    "chamber_volume_m3": 1.0,       # arbitrary
    "projectile_mass_kg": 1000.0,   # arbitrary (overridden by missile)
    "propellant_mass_kg": 0.001,    # negligible propellant
    "propellant_force_J_kg": 1.0,   # negligible force -> ~0 m/s exit
    "covolume_m3_kg": 1.0e-3,
    "gamma": 1.25,
    "twist_calibres": 1e6,          # practically no spin
    "bore_diameter_m": 1.0,         # arbitrary
}

ALL_GUN_CONFIGS = {
    "155mm": GUN_155MM_M109,
    "12.7mm": GUN_127MM_M2,
    "vertical_launch": ZERO_LAUNCH_PAD,
}


# ---------------------------------------------------------------------------
# Interior ballistics solver
# ---------------------------------------------------------------------------

class InteriorBallisticsSolver:
    """LeDuc-method interior ballistics solver.

    Parameters
    ----------
    gun_config : dict
        Gun and propellant configuration dictionary.
    """

    def __init__(self, gun_config: dict):
        self.config = gun_config
        self.name = gun_config["name"]

    def solve(
        self,
        n_points: int = 200,
        propellant_temp_C: float = 21.0,
    ) -> InteriorBallisticsResult:
        """Compute the interior ballistics solution.

        Parameters
        ----------
        n_points : int
            Number of points along the barrel for the output profile.
        propellant_temp_C : float
            Propellant temperature [°C].  Standard is 21 °C.
            Higher temperatures increase burn rate and muzzle velocity.
            Lower temperatures decrease them.

        Returns
        -------
        InteriorBallisticsResult
        """
        cfg = self.config
        m_proj = cfg["projectile_mass_kg"]
        m_prop = cfg["propellant_mass_kg"]
        F_prop_base = cfg["propellant_force_J_kg"]
        V_ch = cfg["chamber_volume_m3"]
        A_bore = cfg["bore_area_m2"]
        L_barrel = cfg["barrel_length_m"]
        gamma = cfg["gamma"]
        eta = cfg["covolume_m3_kg"]
        d = cfg["bore_diameter_m"]
        n_twist = cfg["twist_calibres"]

        # --- Propellant temperature sensitivity ---
        # Typical single-base propellant: ~0.08% increase in force per °C
        # above standard (21 °C).  MIL-STD: coefficient ≈ 0.0008 /K
        TEMP_COEFF = 0.0008  # per kelvin
        delta_T = propellant_temp_C - 21.0
        temp_factor = 1.0 + TEMP_COEFF * delta_T
        F_prop = F_prop_base * temp_factor

        logger.info("Interior ballistics: %s", self.name)
        if abs(delta_T) > 0.1:
            logger.info(
                "  Propellant temp: %.0f °C  (dT = %+.0f K -> "
                "force factor = %.4f)", propellant_temp_C, delta_T, temp_factor,
            )

        # --- LeDuc parameters ---
        # Effective mass (projectile + 1/2 propellant gas)
        m_eff = m_proj + 0.5 * m_prop

        # Total energy available from propellant
        E_total_J = F_prop * m_prop

        # Limiting velocity (all energy → kinetic)
        V_lim = math.sqrt(2.0 * E_total_J / m_eff)

        # LeDuc pressure-distance parameter
        # l_0 = V_ch / A_bore  (simplified)
        l_0 = V_ch / A_bore

        logger.debug("  V_lim = %.1f m/s,  l_0 = %.4f m", V_lim, l_0)

        # --- Velocity profile: v(x) = V_lim * x / (l_0 + x) ---
        x = np.linspace(0.0, L_barrel, n_points)
        v = V_lim * x / (l_0 + x)

        # --- Pressure profile ---
        # From energy balance: P(x) = m_eff * v * dv/dx / A_bore
        # dv/dx = V_lim * l_0 / (l_0 + x)^2
        dv_dx = V_lim * l_0 / (l_0 + x) ** 2
        P_Pa = m_eff * v * dv_dx / A_bore
        P_MPa = P_Pa / 1e6

        # Peak pressure occurs at x_peak = l_0 (analytical)
        x_peak = l_0
        v_peak = V_lim * x_peak / (l_0 + x_peak)
        dv_peak = V_lim * l_0 / (l_0 + x_peak) ** 2
        P_peak_Pa = m_eff * v_peak * dv_peak / A_bore
        P_peak_MPa = P_peak_Pa / 1e6

        # Muzzle values
        V_muzzle = V_lim * L_barrel / (l_0 + L_barrel)
        dv_muzzle = V_lim * l_0 / (l_0 + L_barrel) ** 2
        P_muzzle_Pa = m_eff * V_muzzle * dv_muzzle / A_bore
        P_muzzle_MPa = P_muzzle_Pa / 1e6

        # --- Spin rate from rifling ---
        # Twist rate: 1 revolution per n_twist calibres
        twist_pitch_m = n_twist * d
        omega_muzzle = 2.0 * math.pi * V_muzzle / twist_pitch_m

        logger.info(
            "  Muzzle velocity: %.1f m/s,  Spin: %.0f rad/s,  "
            "Peak pressure: %.1f MPa,  Muzzle pressure: %.1f MPa",
            V_muzzle, omega_muzzle, P_peak_MPa, P_muzzle_MPa,
        )

        return InteriorBallisticsResult(
            muzzle_velocity_ms=V_muzzle,
            muzzle_spin_rate_rad_s=omega_muzzle,
            peak_pressure_MPa=P_peak_MPa,
            muzzle_pressure_MPa=P_muzzle_MPa,
            travel_m=x,
            velocity_ms=v,
            pressure_MPa=P_MPa,
        )
