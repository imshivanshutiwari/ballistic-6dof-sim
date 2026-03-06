"""
Terminal Ballistics Module.

Computes the destructive effects of the projectile at the moment of
ground impact.  Provides:

    - **Impact kinetic energy** [MJ]
    - **Angle of fall** [°] — steepness of the impact trajectory
    - **Armor penetration estimate** [mm RHA] — using the de Marre /
      Krupp empirical formula
    - **Crater diameter estimate** [m] — for HE shells in soft soil
    - **Lethal radius estimate** [m] — fragmentation radius for HE

The Krupp penetration formula:

    e = K · (m^0.5 · v^{0.7}) / (d^0.75)

where:
    e = penetration depth [mm RHA equivalent]
    K = empirical constant (~0.5–0.7 for AP projectiles)
    m = projectile mass [kg]
    v = impact velocity [m/s]
    d = calibre [mm]

References:
    Moss, G. M., Leeming, D. W. & Farrar, C. L.  *Military Ballistics*,
    Revised ed.  Brassey's, 1995.

    Carlucci & Jacobson.  *Ballistics*, 2nd ed.  CRC Press, 2013.
"""

import math
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TerminalBallisticsResult:
    """Container for terminal ballistics results.

    Attributes
    ----------
    impact_velocity_ms : float
        Speed at ground impact [m/s].
    impact_mach : float
        Mach number at impact.
    angle_of_fall_deg : float
        Angle between the velocity vector and the horizontal at impact.
    kinetic_energy_MJ : float
        Translational kinetic energy at impact [MJ].
    armor_penetration_mm_RHA : float
        Estimated penetration depth in Rolled Homogeneous Armor [mm].
    crater_diameter_m : float
        Estimated crater diameter in soft soil [m] (HE shells only).
    lethal_radius_m : float
        Estimated fragmentation lethal radius [m] (HE shells only).
    concrete_penetration_m : float
        Estimated penetration into reinforced concrete [m].
    spall_thickness_mm : float
        Maximum armor thickness that produces spall on the far side [mm].
    behind_armor_cone_deg : float
        Debris cone half-angle behind armor after perforation [deg].
    behind_armor_velocity_ms : float
        Estimated residual velocity of debris behind armor [m/s].
    """
    impact_velocity_ms: float
    impact_mach: float
    angle_of_fall_deg: float
    kinetic_energy_MJ: float
    armor_penetration_mm_RHA: float
    crater_diameter_m: float
    lethal_radius_m: float
    concrete_penetration_m: float = 0.0
    spall_thickness_mm: float = 0.0
    behind_armor_cone_deg: float = 0.0
    behind_armor_velocity_ms: float = 0.0


class TerminalBallisticsCalculator:
    """Compute terminal effects from trajectory impact data.

    Parameters
    ----------
    mass_kg : float
        Projectile mass [kg].
    diameter_m : float
        Projectile calibre [m].
    is_he : bool
        True if the projectile is a High-Explosive (HE) type.
    he_fill_kg : float, optional
        Mass of HE fill [kg].  Defaults to 15% of projectile mass.
    krupp_constant : float, optional
        Empirical constant K for the Krupp formula.
    """

    def __init__(
        self,
        mass_kg: float,
        diameter_m: float,
        is_he: bool = True,
        he_fill_kg: float = None,
        krupp_constant: float = 0.50,
    ):
        self.mass_kg = mass_kg
        self.diameter_m = diameter_m
        self.diameter_mm = diameter_m * 1000.0
        self.is_he = is_he
        self.he_fill_kg = he_fill_kg if he_fill_kg else 0.15 * mass_kg
        self.K = krupp_constant

    def compute(
        self,
        impact_velocity_vector_ms: np.ndarray,
        impact_mach: float,
    ) -> TerminalBallisticsResult:
        """Compute terminal ballistics from the impact state.

        Parameters
        ----------
        impact_velocity_vector_ms : np.ndarray, shape (3,)
            Velocity vector at impact [m/s] (inertial frame, z-up).
        impact_mach : float
            Mach number at impact.

        Returns
        -------
        TerminalBallisticsResult
        """
        vx, vy, vz = impact_velocity_vector_ms
        v_horiz = math.sqrt(vx**2 + vy**2)
        v_total = math.sqrt(vx**2 + vy**2 + vz**2)

        # --- Angle of fall ---
        if v_horiz > 1e-6:
            angle_of_fall_deg = math.degrees(math.atan2(abs(vz), v_horiz))
        else:
            angle_of_fall_deg = 90.0

        # --- Kinetic energy ---
        KE_J = 0.5 * self.mass_kg * v_total**2
        KE_MJ = KE_J / 1e6

        # --- Krupp armor penetration ---
        penetration_mm = (
            self.K
            * (self.mass_kg ** 0.5)
            * (v_total ** 0.7)
            / (self.diameter_mm ** 0.75)
        )

        # --- Crater diameter (HE in soft soil) ---
        if self.is_he and self.he_fill_kg > 0:
            E_he_MJ = self.he_fill_kg * 4.184
            crater_diameter_m = 0.8 * (E_he_MJ ** (1.0 / 3.0))
        else:
            crater_diameter_m = 0.0

        # --- Lethal radius (fragmentation) ---
        if self.is_he and self.he_fill_kg > 0:
            lethal_radius_m = 12.0 * (self.he_fill_kg ** (1.0 / 3.0))
        else:
            lethal_radius_m = 0.0

        # --- Concrete penetration (modified ConWep/NDRC formula) ---
        # D = K_c * N * (m^0.5 * v^0.8) / (d^0.2 * fc^0.5)
        # Simplified: penetration_concrete_m ~ 0.00036 * m^0.4 * v^0.9 / d^0.8
        # where fc = 35 MPa (standard reinforced concrete)
        fc_MPa = 35.0  # compressive strength of reinforced concrete
        concrete_pen_m = (
            0.00036
            * (self.mass_kg ** 0.4)
            * (v_total ** 0.9)
            / ((self.diameter_m ** 0.8) * (fc_MPa ** 0.5))
        )

        # --- Spallation thickness ---
        # Spall occurs when the stress wave from impact reflects off the
        # far side of the armor plate, causing fragmentation on the
        # protected side even without full penetration.
        # Spall threshold ~ 0.65 * penetration depth (empirical for RHA)
        spall_thickness_mm = penetration_mm / 0.65

        # --- Behind-armor effects ---
        # If the projectile perforates, estimate residual velocity
        # using the Thompson energy balance: v_r = v * sqrt(1 - (t/e)^2)
        # where t = armor thickness, e = max penetration
        # Assume target armor = 50% of penetration capability for demo
        assumed_armor_mm = penetration_mm * 0.5
        if penetration_mm > assumed_armor_mm and penetration_mm > 0:
            ratio = assumed_armor_mm / penetration_mm
            behind_armor_vel = v_total * math.sqrt(max(0, 1.0 - ratio**2))

            # Debris cone half-angle: typically 20-45 degrees
            # Wider cone at lower velocity ratios
            cone_deg = 20.0 + 25.0 * ratio
        else:
            behind_armor_vel = 0.0
            cone_deg = 0.0

        logger.info(
            "Terminal ballistics: v=%.1f m/s, AoF=%.1f deg, KE=%.3f MJ, "
            "RHA=%.1f mm, concrete=%.2f m, spall=%.0f mm",
            v_total, angle_of_fall_deg, KE_MJ,
            penetration_mm, concrete_pen_m, spall_thickness_mm,
        )

        return TerminalBallisticsResult(
            impact_velocity_ms=v_total,
            impact_mach=impact_mach,
            angle_of_fall_deg=angle_of_fall_deg,
            kinetic_energy_MJ=KE_MJ,
            armor_penetration_mm_RHA=penetration_mm,
            crater_diameter_m=crater_diameter_m,
            lethal_radius_m=lethal_radius_m,
            concrete_penetration_m=concrete_pen_m,
            spall_thickness_mm=spall_thickness_mm,
            behind_armor_cone_deg=cone_deg,
            behind_armor_velocity_ms=behind_armor_vel,
        )

