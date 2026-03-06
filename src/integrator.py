"""
Trajectory Integrator — Full Model.

Wraps ``scipy.integrate.solve_ivp`` (RK45) around the 6-DOF equations of
motion.  Supports Mach-dependent drag, aerodynamic moments, altitude-
dependent wind profiles, and Coriolis correction.

Provides derived quantities at each saved time-step: speed, dynamic
pressure, angle of attack, and Mach number.
"""

import logging
import math
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Optional, Callable

from .equations_of_motion import eom_6dof

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryResult:
    """Container for trajectory integration results.

    Attributes
    ----------
    time_s : np.ndarray, shape (N,)
        Time stamps [s].
    state : np.ndarray, shape (N, 13)
        Full state history.
    speed_ms : np.ndarray, shape (N,)
        Scalar speed at each time-step [m/s].
    dynamic_pressure_Pa : np.ndarray, shape (N,)
        Dynamic pressure *q* at each time-step [Pa].
    angle_of_attack_deg : np.ndarray, shape (N,)
        Angle of attack at each time-step [°].
    mach_number : np.ndarray, shape (N,)
        Mach number at each time-step [–].
    """
    time_s: np.ndarray
    state: np.ndarray
    speed_ms: np.ndarray
    dynamic_pressure_Pa: np.ndarray
    angle_of_attack_deg: np.ndarray
    mach_number: np.ndarray


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------

class TrajectoryIntegrator:
    """RK45-based trajectory integrator with ground-impact termination.

    Supports optional Coriolis correction and altitude-dependent wind
    profiles via keyword arguments to :meth:`integrate`.
    """

    def integrate(
        self,
        initial_state: np.ndarray,
        t_span: tuple,
        dt: float,
        mass_kg: float,
        inertia_tensor: np.ndarray,
        aero_model,
        atmosphere,
        wind_vector_ms: np.ndarray,
        wind_profile: Optional[Callable[[float], np.ndarray]] = None,
        enable_coriolis: bool = False,
        latitude_deg: float = 0.0,
        use_wgs84: bool = False,
        rocket_motor=None,
        base_bleed_unit=None,
    ) -> TrajectoryResult:
        """Integrate the 6-DOF equations of motion.

        Parameters
        ----------
        initial_state : np.ndarray, shape (13,)
            Initial state vector.
        t_span : tuple of (float, float)
            Start and end time [s].
        dt : float
            Maximum integration time-step [s].
        mass_kg : float
            Projectile mass [kg].
        inertia_tensor : np.ndarray, shape (3, 3)
            Diagonal inertia tensor [kg·m²].
        aero_model : AeroModel
            Aerodynamic model instance (with Mach-Cd table and moments).
        atmosphere : ISAAtmosphere or WeatherModel
            Atmosphere model instance.
        wind_vector_ms : np.ndarray, shape (3,)
            Constant wind vector [m/s].  Ignored if *wind_profile* given.
        wind_profile : callable, optional
            ``f(altitude_m) -> wind_vector[3]`` for altitude-dependent wind.
        enable_coriolis : bool
            Enable Coriolis correction (default False).
        latitude_deg : float
            Launch-site latitude [°] (used if Coriolis is enabled).
        use_wgs84 : bool
            If True, use WGS-84 oblate-earth gravity.
        rocket_motor : RocketMotor or None
            If provided, applies time-dependent thrust and mass depletion.

        Returns
        -------
        TrajectoryResult
        """
        logger.info(
            "Starting integration: t_span=%s, dt=%.4f, mass=%.2f kg, "
            "coriolis=%s, lat=%.1f°, wgs84=%s, rocket=%s",
            t_span, dt, mass_kg, enable_coriolis, latitude_deg,
            use_wgs84, rocket_motor is not None,
        )

        # --- Ground-impact event ---
        def ground_impact_event(t, state, *args):
            return state[2]  # z-coordinate
        ground_impact_event.terminal = True
        ground_impact_event.direction = -1

        # --- ODE wrapper ---
        def rhs(t, state):
            return eom_6dof(
                t, state, mass_kg, inertia_tensor,
                aero_model, atmosphere, wind_vector_ms,
                wind_profile=wind_profile,
                enable_coriolis=enable_coriolis,
                latitude_deg=latitude_deg,
                use_wgs84=use_wgs84,
                rocket_motor=rocket_motor,
                base_bleed_unit=base_bleed_unit,
            )

        # --- Integrate ---
        t_eval = np.arange(t_span[0], t_span[1], dt)
        sol = solve_ivp(
            rhs, t_span, initial_state,
            method="RK45",
            t_eval=t_eval,
            events=ground_impact_event,
            max_step=dt,
            rtol=1e-3,
            atol=1e-6,
        )

        time_s = sol.t
        state_history = sol.y.T  # shape (N, 13)
        n_steps = len(time_s)

        logger.info(
            "Integration complete: %d steps, final_t=%.2f s",
            n_steps, time_s[-1] if n_steps > 0 else 0.0,
        )

        # --- Derived quantities ---
        speed_ms = np.zeros(n_steps)
        dynamic_pressure_Pa = np.zeros(n_steps)
        angle_of_attack_deg = np.zeros(n_steps)
        mach_number = np.zeros(n_steps)

        for i in range(n_steps):
            vel = state_history[i, 3:6]
            quat = state_history[i, 6:10]
            q_norm = np.linalg.norm(quat)
            if q_norm > 1e-12:
                quat = quat / q_norm
            alt = float(np.clip(state_history[i, 2], 0.0, 20000.0))

            spd = np.linalg.norm(vel)
            speed_ms[i] = spd

            atm = atmosphere.get_properties(alt)
            rho = atm["density_kg_m3"]
            a = atm.get("speed_of_sound_m_s", atm.get("speed_of_sound_ms", 340.0))

            dynamic_pressure_Pa[i] = 0.5 * rho * spd ** 2
            mach_number[i] = spd / a if a > 0 else 0.0

            # Angle of attack
            q0, q1, q2, q3 = quat
            body_x_inertial = np.array([
                1 - 2*(q2**2 + q3**2),
                2*(q1*q2 + q0*q3),
                2*(q1*q3 - q0*q2),
            ])
            if spd > 1e-12:
                cos_alpha = np.clip(np.dot(vel / spd, body_x_inertial), -1.0, 1.0)
                angle_of_attack_deg[i] = math.degrees(math.acos(cos_alpha))
            else:
                angle_of_attack_deg[i] = 0.0

        return TrajectoryResult(
            time_s=time_s,
            state=state_history,
            speed_ms=speed_ms,
            dynamic_pressure_Pa=dynamic_pressure_Pa,
            angle_of_attack_deg=angle_of_attack_deg,
            mach_number=mach_number,
        )
