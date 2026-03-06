"""
Pre-built Projectile Configurations — Full Model.

Each configuration contains:
    - Physical properties (mass, geometry, inertia tensor)
    - Aerodynamic force coefficients (Cd, Cl, Cm)
    - Aerodynamic moment coefficients (Cmq, Clp, Cnpa)
    - Mach-Cd lookup table for Mach-dependent drag

Configurations:
    1. **155 mm Artillery Shell** (M107-class)
    2. **12.7 mm Heavy Machine Gun Round** (.50 BMG-class)
    3. **Generic 122 mm Rocket** (Grad-class)

References:
    McCoy, R. L.  *Modern Exterior Ballistics*, 2nd ed.  Schiffer, 2012.
    STANAG 4355 — Ballistic Trajectory Models.
"""

import math
import numpy as np

from .aerodynamics import (
    MACH_TABLE_155MM, CD_TABLE_155MM,
    MACH_TABLE_127MM, CD_TABLE_127MM,
    MACH_TABLE_122MM, CD_TABLE_122MM,
)


def _cylinder_inertia(mass_kg: float, radius_m: float, length_m: float) -> np.ndarray:
    """Approximate inertia tensor for a solid cylinder (spin axis = x).

    Parameters
    ----------
    mass_kg : float
        Mass [kg].
    radius_m : float
        Radius [m].
    length_m : float
        Length [m].

    Returns
    -------
    np.ndarray, shape (3, 3)
        Diagonal inertia tensor [kg·m²].
    """
    Ixx = 0.5 * mass_kg * radius_m ** 2
    Iyy = (1.0 / 12.0) * mass_kg * (3 * radius_m ** 2 + length_m ** 2)
    Izz = Iyy
    return np.diag([Ixx, Iyy, Izz])


# ---------------------------------------------------------------------------
# 155 mm Artillery Shell  (M107-class)
# ---------------------------------------------------------------------------
_SHELL_155_MASS_KG = 43.5
_SHELL_155_DIAMETER_M = 0.155
_SHELL_155_LENGTH_M = 0.605
_SHELL_155_RADIUS_M = _SHELL_155_DIAMETER_M / 2.0
_SHELL_155_REF_AREA_M2 = math.pi * _SHELL_155_RADIUS_M ** 2

SHELL_155MM = {
    "name": "155 mm Artillery Shell (M107-class)",
    "mass_kg": _SHELL_155_MASS_KG,
    "diameter_m": _SHELL_155_DIAMETER_M,
    "length_m": _SHELL_155_LENGTH_M,
    "Cd": 0.30,
    "Cl": 0.05,
    "Cm": 0.003,
    "Cmq": -8.0,       # pitch damping
    "Clp": -0.010,      # roll damping
    "Cnpa": -0.50,      # Magnus moment
    "Cma": -3.0,        # overturning moment (statically stable)
    "reference_area_m2": _SHELL_155_REF_AREA_M2,
    "inertia_tensor": _cylinder_inertia(
        _SHELL_155_MASS_KG, _SHELL_155_RADIUS_M, _SHELL_155_LENGTH_M
    ),
    "mach_table": MACH_TABLE_155MM,
    "cd_table": CD_TABLE_155MM,
}

# ---------------------------------------------------------------------------
# 12.7 mm HMG Round  (.50 BMG-class)
# ---------------------------------------------------------------------------
_HMG_127_MASS_KG = 0.046
_HMG_127_DIAMETER_M = 0.0127
_HMG_127_LENGTH_M = 0.058
_HMG_127_RADIUS_M = _HMG_127_DIAMETER_M / 2.0
_HMG_127_REF_AREA_M2 = math.pi * _HMG_127_RADIUS_M ** 2

ROUND_12_7MM = {
    "name": "12.7 mm HMG Round (.50 BMG-class)",
    "mass_kg": _HMG_127_MASS_KG,
    "diameter_m": _HMG_127_DIAMETER_M,
    "length_m": _HMG_127_LENGTH_M,
    "Cd": 0.28,
    "Cl": 0.04,
    "Cm": 0.002,
    "Cmq": -6.0,
    "Clp": -0.008,
    "Cnpa": -0.30,
    "Cma": -2.5,
    "reference_area_m2": _HMG_127_REF_AREA_M2,
    "inertia_tensor": _cylinder_inertia(
        _HMG_127_MASS_KG, _HMG_127_RADIUS_M, _HMG_127_LENGTH_M
    ),
    "mach_table": MACH_TABLE_127MM,
    "cd_table": CD_TABLE_127MM,
}

# ---------------------------------------------------------------------------
# Generic 122 mm Rocket  (Grad-class)
# ---------------------------------------------------------------------------
_ROCKET_122_MASS_KG = 20.0
_ROCKET_122_DIAMETER_M = 0.122
_ROCKET_122_LENGTH_M = 1.0
_ROCKET_122_RADIUS_M = _ROCKET_122_DIAMETER_M / 2.0
_ROCKET_122_REF_AREA_M2 = math.pi * _ROCKET_122_RADIUS_M ** 2

ROCKET_122MM = {
    "name": "Generic 122 mm Rocket (Grad-class)",
    "mass_kg": _ROCKET_122_MASS_KG,
    "diameter_m": _ROCKET_122_DIAMETER_M,
    "length_m": _ROCKET_122_LENGTH_M,
    "Cd": 0.25,
    "Cl": 0.04,
    "Cm": 0.002,
    "Cmq": -5.0,
    "Clp": -0.006,
    "Cnpa": -0.25,
    "Cma": -2.0,
    "reference_area_m2": _ROCKET_122_REF_AREA_M2,
    "inertia_tensor": _cylinder_inertia(
        _ROCKET_122_MASS_KG, _ROCKET_122_RADIUS_M, _ROCKET_122_LENGTH_M
    ),
    "mach_table": MACH_TABLE_122MM,
    "cd_table": CD_TABLE_122MM,
}

# ===========================================================================
#  BALLISTIC MISSILES & ROCKETS — Publicly Available Specifications
#  Aerodynamic coefficients are textbook estimates for slender ogive bodies.
#  Sources: Wikipedia, CSIS Missile Threat Project, Jane's, open DRDO data.
# ===========================================================================

# --- Generic Mach-Cd table for large missiles (ogive + cylinder + fin) ---
_MISSILE_MACH_TABLE = np.array([0.0, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0])
_MISSILE_CD_TABLE   = np.array([0.20, 0.22, 0.28, 0.38, 0.50, 0.48, 0.44, 0.38, 0.32, 0.28, 0.24, 0.22])

# --- Generic Mach-Cd table for cruise missiles (subsonic, streamlined) ---
_CRUISE_MACH_TABLE = np.array([0.0, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
_CRUISE_CD_TABLE   = np.array([0.15, 0.16, 0.18, 0.20, 0.25, 0.32, 0.45, 0.60])


def _missile_config(name, mass_kg, diameter_m, length_m, Cd=0.30, 
                     mach_table=None, cd_table=None):
    """Helper to build a missile config dict with estimated aero coefficients."""
    radius_m = diameter_m / 2.0
    ref_area = math.pi * radius_m ** 2
    return {
        "name": name,
        "mass_kg": mass_kg,
        "diameter_m": diameter_m,
        "length_m": length_m,
        "Cd": Cd,
        "Cl": 0.03,          # small lift for finned body
        "Cm": 0.002,
        "Cmq": -6.0,         # moderate pitch damping
        "Clp": -0.005,       # roll damping
        "Cnpa": -0.20,       # Magnus moment
        "Cma": -2.5,         # overturning (statically stable)
        "reference_area_m2": ref_area,
        "inertia_tensor": _cylinder_inertia(mass_kg, radius_m, length_m),
        "mach_table": mach_table if mach_table is not None else _MISSILE_MACH_TABLE,
        "cd_table": cd_table if cd_table is not None else _MISSILE_CD_TABLE,
    }


# ---------------------------------------------------------------------------
# [IN] INDIA — Agni Series (DRDO / SFC)
# ---------------------------------------------------------------------------

AGNI_I = _missile_config(
    "[IN] Agni-I SRBM (700-1200 km)", 
    mass_kg=12000, diameter_m=1.0, length_m=15.0, Cd=0.28)

AGNI_II = _missile_config(
    "[IN] Agni-II MRBM (2000-3500 km)", 
    mass_kg=17000, diameter_m=1.0, length_m=21.0, Cd=0.26)

AGNI_III = _missile_config(
    "[IN] Agni-III IRBM (3500-5000 km)", 
    mass_kg=48000, diameter_m=2.0, length_m=17.0, Cd=0.28)

AGNI_IV = _missile_config(
    "[IN] Agni-IV IRBM (3500-4000 km)", 
    mass_kg=17000, diameter_m=1.0, length_m=20.0, Cd=0.26)

AGNI_V = _missile_config(
    "[IN] Agni-V ICBM (5000-8000 km)", 
    mass_kg=50000, diameter_m=2.0, length_m=17.5, Cd=0.27)

AGNI_P = _missile_config(
    "[IN] Agni-Prime (Canister-launched, 1000-2000 km)", 
    mass_kg=11000, diameter_m=1.15, length_m=10.5, Cd=0.25)

# ---------------------------------------------------------------------------
# [DE]/[US] — V-2 (A-4) Rocket  (World's first ballistic missile)
# ---------------------------------------------------------------------------

V2_ROCKET = _missile_config(
    "[DE] V-2 (A-4) Rocket (320 km)", 
    mass_kg=12500, diameter_m=1.65, length_m=14.0, Cd=0.30)

# ---------------------------------------------------------------------------
# [RU] — Scud-B (R-17 Elbrus)
# ---------------------------------------------------------------------------

SCUD_B = _missile_config(
    "[RU] Scud-B (R-17 Elbrus, 300 km)", 
    mass_kg=5900, diameter_m=0.88, length_m=11.25, Cd=0.32)

# ---------------------------------------------------------------------------
# [US] — Tomahawk Cruise Missile (BGM-109)
# ---------------------------------------------------------------------------

TOMAHAWK = _missile_config(
    "[US] Tomahawk BGM-109 Cruise Missile (2500 km)", 
    mass_kg=1315, diameter_m=0.518, length_m=5.56, Cd=0.18,
    mach_table=_CRUISE_MACH_TABLE, cd_table=_CRUISE_CD_TABLE)

# ---------------------------------------------------------------------------
# [US] — M26 MLRS Rocket (M270 system)
# ---------------------------------------------------------------------------

M26_MLRS = _missile_config(
    "[US] M26 MLRS Rocket (32 km)", 
    mass_kg=306, diameter_m=0.227, length_m=3.94, Cd=0.24)

# ---------------------------------------------------------------------------
# [US] — Minuteman III ICBM (LGM-30G)
# ---------------------------------------------------------------------------

MINUTEMAN_III = _missile_config(
    "[US] Minuteman III ICBM (13,000 km)", 
    mass_kg=36000, diameter_m=1.67, length_m=18.3, Cd=0.26)


# ---------------------------------------------------------------------------
# Convenience lookup — ALL projectile and missile configurations
# ---------------------------------------------------------------------------
ALL_CONFIGS = {
    # Original projectiles
    "155mm": SHELL_155MM,
    "12.7mm": ROUND_12_7MM,
    "122mm": ROCKET_122MM,
    # Indian Ballistic Missiles
    "agni1": AGNI_I,
    "agni2": AGNI_II,
    "agni3": AGNI_III,
    "agni4": AGNI_IV,
    "agni5": AGNI_V,
    "agnip": AGNI_P,
    # Historic / International
    "v2": V2_ROCKET,
    "scudb": SCUD_B,
    "tomahawk": TOMAHAWK,
    "m26": M26_MLRS,
    "minuteman3": MINUTEMAN_III,
}

