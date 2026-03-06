"""
Aerodynamic Force and Moment Model for a Spinning Projectile.

Computes drag, lift, and Magnus forces, as well as pitch-damping,
roll-damping, and Magnus moments acting on a projectile.

Key features:
    - **Mach-dependent drag coefficient** via a piecewise lookup table
      (subsonic → transonic → supersonic), interpolated with numpy.
    - **Aerodynamic moments**: pitch-damping (Cmq), roll-damping (Clp),
      and Magnus moment (Cnpa).
    - All forces returned in the inertial frame as numpy (3,) arrays [N].
    - All moments returned in the body frame as numpy (3,) arrays [N·m].

References:
    McCoy, R. L.  *Modern Exterior Ballistics*, 2nd ed.  Schiffer, 2012.
    STANAG 4355 — Modified Point Mass and Five-DOF Trajectory Models.
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Default Mach-Cd tables for common projectile classes
# ---------------------------------------------------------------------------

# 155 mm artillery shell (M107-class) — typical Cd vs Mach from McCoy
MACH_TABLE_155MM = np.array([
    0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2,
    1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
])
CD_TABLE_155MM = np.array([
    0.15, 0.15, 0.15, 0.18, 0.22, 0.30, 0.42, 0.44, 0.43, 0.40,
    0.36, 0.33, 0.31, 0.30, 0.28, 0.27, 0.26, 0.25,
])

# 12.7 mm (.50 BMG) — typical boat-tail bullet
MACH_TABLE_127MM = np.array([
    0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2,
    1.4, 1.6, 1.8, 2.0, 2.5, 3.0,
])
CD_TABLE_127MM = np.array([
    0.12, 0.12, 0.13, 0.15, 0.20, 0.28, 0.40, 0.42, 0.41, 0.38,
    0.34, 0.31, 0.29, 0.28, 0.26, 0.25,
])

# Generic 122 mm rocket
MACH_TABLE_122MM = np.array([
    0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2,
    1.4, 1.6, 1.8, 2.0, 2.5, 3.0,
])
CD_TABLE_122MM = np.array([
    0.13, 0.13, 0.14, 0.16, 0.20, 0.26, 0.38, 0.40, 0.39, 0.36,
    0.32, 0.30, 0.28, 0.25, 0.24, 0.23,
])


class AeroModel:
    """Aerodynamic force and moment calculator for a spinning projectile.

    Parameters
    ----------
    Cd : float
        Zero-yaw drag coefficient at reference Mach (used as fallback) [–].
    Cl : float
        Lift-curve slope coefficient [–].
    Cm : float
        Magnus force coefficient [–].
    reference_area_m2 : float
        Aerodynamic reference area *S* [m²].
    reference_diameter_m : float
        Reference diameter *d* (calibre) [m].
    Cmq : float, optional
        Pitch-damping moment coefficient [–].  Default −8.0 (stabilising).
    Clp : float, optional
        Roll-damping moment coefficient [–].  Default −0.01.
    Cnpa : float, optional
        Magnus moment coefficient [–].  Default −0.5.
    mach_table : np.ndarray, optional
        Array of Mach numbers for Cd lookup.
    cd_table : np.ndarray, optional
        Corresponding Cd values.  Must be same length as *mach_table*.
    reference_length_m : float, optional
        Reference length for moment coefficients [m].  Defaults to
        *reference_diameter_m*.
    """

    def __init__(
        self,
        Cd: float,
        Cl: float,
        Cm: float,
        reference_area_m2: float,
        reference_diameter_m: float,
        Cmq: float = -8.0,
        Clp: float = -0.01,
        Cnpa: float = -0.5,
        Cma: float = -3.0,
        mach_table: Optional[np.ndarray] = None,
        cd_table: Optional[np.ndarray] = None,
        reference_length_m: Optional[float] = None,
    ):
        self.Cd_ref = Cd
        self.Cl = Cl
        self.Cm = Cm
        self.reference_area_m2 = reference_area_m2
        self.reference_diameter_m = reference_diameter_m
        self.reference_length_m = (
            reference_length_m if reference_length_m is not None
            else reference_diameter_m
        )

        # Moment coefficients
        self.Cmq = Cmq      # pitch-damping (negative = stabilising)
        self.Clp = Clp       # roll-damping   (negative = decelerating)
        self.Cnpa = Cnpa     # Magnus moment  (negative = stabilising)
        self.Cma = Cma       # overturning moment (negative = statically stable)

        # Mach-Cd lookup table
        if mach_table is not None and cd_table is not None:
            assert len(mach_table) == len(cd_table), (
                "mach_table and cd_table must have the same length"
            )
            self._mach_table = np.asarray(mach_table, dtype=float)
            self._cd_table = np.asarray(cd_table, dtype=float)
            self._use_mach_cd = True
        else:
            self._use_mach_cd = False

    # ------------------------------------------------------------------
    # Mach-dependent Cd
    # ------------------------------------------------------------------

    def get_Cd(self, mach: float) -> float:
        """Return the drag coefficient at a given Mach number.

        Uses linear interpolation on the Mach-Cd table if available,
        otherwise returns the constant reference Cd.

        Parameters
        ----------
        mach : float
            Mach number [–].

        Returns
        -------
        float
            Drag coefficient Cd [–].
        """
        if not self._use_mach_cd:
            return self.Cd_ref
        return float(np.interp(mach, self._mach_table, self._cd_table))

    # ------------------------------------------------------------------
    # Force methods (all return inertial-frame vectors in Newtons)
    # ------------------------------------------------------------------

    def drag_force(
        self,
        velocity_vector_ms: np.ndarray,
        density_kg_m3: float,
        mach: float = 0.0,
    ) -> np.ndarray:
        """Compute aerodynamic drag force opposing the velocity vector.

        Uses Mach-dependent Cd when a lookup table is configured.

        Parameters
        ----------
        velocity_vector_ms : np.ndarray, shape (3,)
            Velocity relative to air [m/s].
        density_kg_m3 : float
            Local air density [kg/m³].
        mach : float
            Current Mach number [–].

        Returns
        -------
        np.ndarray, shape (3,)
            Drag force vector [N].
        """
        speed_ms = np.linalg.norm(velocity_vector_ms)
        if speed_ms < 1e-12:
            return np.zeros(3)

        velocity_unit = velocity_vector_ms / speed_ms
        Cd = self.get_Cd(mach)
        dynamic_pressure_Pa = 0.5 * density_kg_m3 * speed_ms ** 2

        return -dynamic_pressure_Pa * Cd * self.reference_area_m2 * velocity_unit

    def lift_force(
        self,
        velocity_vector_ms: np.ndarray,
        density_kg_m3: float,
    ) -> np.ndarray:
        """Compute aerodynamic lift force perpendicular to the velocity.

        Parameters
        ----------
        velocity_vector_ms : np.ndarray, shape (3,)
            Velocity relative to air [m/s].
        density_kg_m3 : float
            Local air density [kg/m³].

        Returns
        -------
        np.ndarray, shape (3,)
            Lift force vector [N].
        """
        speed_ms = np.linalg.norm(velocity_vector_ms)
        if speed_ms < 1e-12:
            return np.zeros(3)

        velocity_unit = velocity_vector_ms / speed_ms
        z_axis = np.array([0.0, 0.0, 1.0])
        perp = z_axis - np.dot(z_axis, velocity_unit) * velocity_unit
        perp_norm = np.linalg.norm(perp)
        if perp_norm < 1e-12:
            return np.zeros(3)
        lift_direction = perp / perp_norm

        dynamic_pressure_Pa = 0.5 * density_kg_m3 * speed_ms ** 2

        return (
            dynamic_pressure_Pa * self.Cl * self.reference_area_m2 * lift_direction
        )

    def magnus_force(
        self,
        velocity_vector_ms: np.ndarray,
        spin_rate_vector_rad_s: np.ndarray,
        density_kg_m3: float,
    ) -> np.ndarray:
        """Compute the Magnus (spin-induced lateral) force.

        Parameters
        ----------
        velocity_vector_ms : np.ndarray, shape (3,)
            Velocity relative to air [m/s].
        spin_rate_vector_rad_s : np.ndarray, shape (3,)
            Angular-velocity vector [rad/s].
        density_kg_m3 : float
            Local air density [kg/m³].

        Returns
        -------
        np.ndarray, shape (3,)
            Magnus force vector [N].
        """
        speed_ms = np.linalg.norm(velocity_vector_ms)
        if speed_ms < 1e-12:
            return np.zeros(3)

        velocity_unit = velocity_vector_ms / speed_ms
        magnus_direction = np.cross(spin_rate_vector_rad_s, velocity_unit)

        return (
            0.5 * density_kg_m3 * speed_ms
            * self.Cm * self.reference_area_m2 * self.reference_diameter_m
            * magnus_direction
        )

    # ------------------------------------------------------------------
    # Moment methods (all return body-frame vectors in N·m)
    # ------------------------------------------------------------------

    def pitch_damping_moment(
        self,
        omega_body: np.ndarray,
        speed_ms: float,
        density_kg_m3: float,
    ) -> np.ndarray:
        """Compute pitch-damping moment (Cmq) in the body frame.

        Opposes transverse angular velocity (wy, wz) to damp nutation.

        Parameters
        ----------
        omega_body : np.ndarray, shape (3,)
            Angular velocity in body frame [rad/s].
        speed_ms : float
            Airspeed [m/s].
        density_kg_m3 : float
            Local air density [kg/m³].

        Returns
        -------
        np.ndarray, shape (3,)
            Pitch-damping moment vector in body frame [N·m].
        """
        if speed_ms < 1e-12:
            return np.zeros(3)

        q_dyn = 0.5 * density_kg_m3 * speed_ms ** 2
        S = self.reference_area_m2
        d = self.reference_length_m

        # Pitch damping acts on transverse angular velocities only
        factor = q_dyn * S * d * (d / (2.0 * speed_ms)) * self.Cmq
        return np.array([0.0, factor * omega_body[1], factor * omega_body[2]])

    def roll_damping_moment(
        self,
        omega_body: np.ndarray,
        speed_ms: float,
        density_kg_m3: float,
    ) -> np.ndarray:
        """Compute roll-damping moment (Clp) in the body frame.

        Opposes axial spin (wx) due to viscous friction.

        Parameters
        ----------
        omega_body : np.ndarray, shape (3,)
            Angular velocity in body frame [rad/s].
        speed_ms : float
            Airspeed [m/s].
        density_kg_m3 : float
            Local air density [kg/m³].

        Returns
        -------
        np.ndarray, shape (3,)
            Roll-damping moment vector in body frame [N·m].
        """
        if speed_ms < 1e-12:
            return np.zeros(3)

        q_dyn = 0.5 * density_kg_m3 * speed_ms ** 2
        S = self.reference_area_m2
        d = self.reference_length_m

        factor = q_dyn * S * d * (d / (2.0 * speed_ms)) * self.Clp
        return np.array([factor * omega_body[0], 0.0, 0.0])

    def magnus_moment(
        self,
        omega_body: np.ndarray,
        alpha_rad: float,
        speed_ms: float,
        density_kg_m3: float,
    ) -> np.ndarray:
        """Compute the Magnus moment in the body frame.

        The Magnus moment is proportional to the product of spin rate
        and angle of attack, and acts in the yaw plane.

        Parameters
        ----------
        omega_body : np.ndarray, shape (3,)
            Angular velocity in body frame [rad/s].
        alpha_rad : float
            Angle of attack [rad].
        speed_ms : float
            Airspeed [m/s].
        density_kg_m3 : float
            Local air density [kg/m³].

        Returns
        -------
        np.ndarray, shape (3,)
            Magnus moment vector in body frame [N·m].
        """
        if speed_ms < 1e-12:
            return np.zeros(3)

        q_dyn = 0.5 * density_kg_m3 * speed_ms ** 2
        S = self.reference_area_m2
        d = self.reference_length_m
        spin_x = omega_body[0]  # axial spin

        # Magnus moment acts perpendicular to the spin axis and AoA plane
        pd_over_2V = (spin_x * d) / (2.0 * speed_ms)
        M_magnus = q_dyn * S * d * self.Cnpa * alpha_rad * pd_over_2V

        # Apply in the yaw (z-body) direction as a first-order model
        return np.array([0.0, 0.0, M_magnus])

    def overturning_moment(
        self,
        alpha_rad: float,
        speed_ms: float,
        density_kg_m3: float,
    ) -> np.ndarray:
        """Compute the overturning (static restoring) moment.

        The overturning moment is proportional to the angle of attack
        and acts to pitch the projectile nose toward (Cma < 0, stable)
        or away from (Cma > 0, unstable) the velocity vector.

        Parameters
        ----------
        alpha_rad : float
            Angle of attack [rad].
        speed_ms : float
            Airspeed [m/s].
        density_kg_m3 : float
            Local air density [kg/m³].

        Returns
        -------
        np.ndarray, shape (3,)
            Overturning moment vector in body frame [N·m].
        """
        if speed_ms < 1e-12:
            return np.zeros(3)

        q_dyn = 0.5 * density_kg_m3 * speed_ms ** 2
        S = self.reference_area_m2
        d = self.reference_length_m

        M_ot = q_dyn * S * d * self.Cma * alpha_rad
        # Overturning moment acts in the pitch (y-body) plane
        return np.array([0.0, M_ot, 0.0])

    def gyroscopic_stability_factor(
        self,
        spin_rate_rad_s: float,
        speed_ms: float,
        density_kg_m3: float,
        Ix: float,
        Iy: float,
    ) -> float:
        """Compute the gyroscopic stability factor Sg.

        A projectile is gyroscopically stable when Sg > 1.0.
        Values below 1.0 indicate the shell will tumble.

        Parameters
        ----------
        spin_rate_rad_s : float
            Axial spin rate [rad/s].
        speed_ms : float
            Airspeed [m/s].
        density_kg_m3 : float
            Local air density [kg/m³].
        Ix : float
            Axial moment of inertia [kg·m²].
        Iy : float
            Transverse moment of inertia [kg·m²].

        Returns
        -------
        float
            Gyroscopic stability factor Sg [–].
            Sg > 1 means stable.  Sg < 1 means tumbling.
        """
        if speed_ms < 1e-12 or abs(self.Cma) < 1e-12:
            return float('inf')  # no concern

        q_dyn = 0.5 * density_kg_m3 * speed_ms ** 2
        S = self.reference_area_m2
        d = self.reference_length_m

        M_alpha = q_dyn * S * d * abs(self.Cma)  # magnitude
        Sg = (Ix * spin_rate_rad_s) ** 2 / (4.0 * Iy * M_alpha)
        return Sg

    def compute_total_moment(
        self,
        omega_body: np.ndarray,
        alpha_rad: float,
        speed_ms: float,
        density_kg_m3: float,
    ) -> np.ndarray:
        """Compute total aerodynamic moment in body frame.

        Sums overturning, pitch-damping, roll-damping, and Magnus moments.

        Parameters
        ----------
        omega_body : np.ndarray, shape (3,)
            Angular velocity in body frame [rad/s].
        alpha_rad : float
            Angle of attack [rad].
        speed_ms : float
            Airspeed [m/s].
        density_kg_m3 : float
            Local air density [kg/m³].

        Returns
        -------
        np.ndarray, shape (3,)
            Total aerodynamic moment vector in body frame [N·m].
        """
        M_overturn = self.overturning_moment(alpha_rad, speed_ms, density_kg_m3)
        M_pitch = self.pitch_damping_moment(omega_body, speed_ms, density_kg_m3)
        M_roll = self.roll_damping_moment(omega_body, speed_ms, density_kg_m3)
        M_magnus = self.magnus_moment(omega_body, alpha_rad, speed_ms, density_kg_m3)
        return M_overturn + M_pitch + M_roll + M_magnus
