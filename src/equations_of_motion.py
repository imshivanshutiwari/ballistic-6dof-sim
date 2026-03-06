"""
Six-Degree-of-Freedom Equations of Motion — Full Model.

Defines the state-derivative function for a rigid projectile subject to
gravity, aerodynamic drag (Mach-dependent), lift, Magnus force,
aerodynamic moments (pitch-damping, roll-damping, Magnus moment),
altitude-dependent wind profiles, and optional Coriolis correction.

State vector layout (13 elements):
    [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]

    - (x, y, z)          : position in the inertial frame [m]
    - (vx, vy, vz)       : velocity in the inertial frame [m/s]
    - (q0, q1, q2, q3)   : attitude quaternion (scalar-first) [–]
    - (wx, wy, wz)       : angular velocity in the body frame [rad/s]

References:
    Etkin, B.  *Dynamics of Flight*, 3rd ed.  Wiley, 1996.
    McCoy, R. L.  *Modern Exterior Ballistics*, 2nd ed.  Schiffer, 2012.
    U.S. Standard Atmosphere, 1976.
"""

import math
import numpy as np
from typing import Optional, Callable

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GRAVITY_MS2 = 9.80665         # standard gravitational acceleration [m/s^2]
EARTH_ROTATION_RAD_S = 7.2921159e-5  # Earth's angular velocity [rad/s]
EARTH_RADIUS_M = 6_371_000.0  # mean Earth radius [m]


def gravity_at_altitude(altitude_m: float) -> float:
    """Return gravitational acceleration at a given altitude.

    Uses Newton's inverse-square law:  g(h) = g₀ (Rₑ / (Rₑ + h))²

    Parameters
    ----------
    altitude_m : float
        Altitude above sea level [m].

    Returns
    -------
    float
        Gravitational acceleration [m/s²].
    """
    return GRAVITY_MS2 * (EARTH_RADIUS_M / (EARTH_RADIUS_M + altitude_m)) ** 2


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def quaternion_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Multiply two quaternions *q* and *r* (scalar-first convention).

    Parameters
    ----------
    q, r : np.ndarray, shape (4,)
        Quaternions ``[q0, q1, q2, q3]``.

    Returns
    -------
    np.ndarray, shape (4,)
        Product ``q ⊗ r``.
    """
    q0, q1, q2, q3 = q
    r0, r1, r2, r3 = r
    return np.array([
        q0*r0 - q1*r1 - q2*r2 - q3*r3,
        q0*r1 + q1*r0 + q2*r3 - q3*r2,
        q0*r2 - q1*r3 + q2*r0 + q3*r1,
        q0*r3 + q1*r2 - q2*r1 + q3*r0,
    ])


def quaternion_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector *v* from body frame to inertial frame using *q*.

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Unit quaternion (body → inertial).
    v : np.ndarray, shape (3,)
        Vector in body-frame coordinates.

    Returns
    -------
    np.ndarray, shape (3,)
        Vector in inertial-frame coordinates.
    """
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2*(q2**2 + q3**2),  2*(q1*q2 - q0*q3),    2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),      1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),      2*(q2*q3 + q0*q1),     1 - 2*(q1**2 + q2**2)],
    ])
    return R @ v


def quaternion_rotate_vector_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector *v* from inertial frame to body frame using *q*.

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Unit quaternion (body → inertial).
    v : np.ndarray, shape (3,)
        Vector in inertial-frame coordinates.

    Returns
    -------
    np.ndarray, shape (3,)
        Vector in body-frame coordinates.
    """
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    return quaternion_rotate_vector(q_conj, v)


# ---------------------------------------------------------------------------
# Coriolis acceleration
# ---------------------------------------------------------------------------

def coriolis_acceleration(
    velocity_inertial_ms: np.ndarray,
    latitude_rad: float,
) -> np.ndarray:
    """Compute Coriolis acceleration for a moving projectile.

    Parameters
    ----------
    velocity_inertial_ms : np.ndarray, shape (3,)
        Velocity in the inertial frame [m/s] (x=East, y=North, z=Up).
    latitude_rad : float
        Geodetic latitude of the launch site [rad].

    Returns
    -------
    np.ndarray, shape (3,)
        Coriolis acceleration [m/s²] (inertial frame).

    Notes
    -----
    Earth's angular velocity vector projected to local ENU:
        Ω = ω_e * [0, cos(lat), sin(lat)]
    Coriolis acceleration = -2 Ω × v
    """
    omega_earth = EARTH_ROTATION_RAD_S * np.array([
        0.0,
        math.cos(latitude_rad),
        math.sin(latitude_rad),
    ])
    return -2.0 * np.cross(omega_earth, velocity_inertial_ms)


# ---------------------------------------------------------------------------
# Wind profile
# ---------------------------------------------------------------------------

def default_wind_profile(altitude_m: float) -> np.ndarray:
    """Return a zero wind vector (no wind).

    Parameters
    ----------
    altitude_m : float
        Altitude [m].

    Returns
    -------
    np.ndarray, shape (3,)
        Wind velocity [m/s].
    """
    return np.zeros(3)


def create_wind_profile(
    altitude_breakpoints_m: np.ndarray,
    wind_east_ms: np.ndarray,
    wind_north_ms: np.ndarray,
) -> Callable[[float], np.ndarray]:
    """Create an altitude-dependent wind profile via linear interpolation.

    Parameters
    ----------
    altitude_breakpoints_m : np.ndarray
        Altitude breakpoints [m].  Must be monotonically increasing.
    wind_east_ms : np.ndarray
        East-component of wind at each breakpoint [m/s].
    wind_north_ms : np.ndarray
        North-component of wind at each breakpoint [m/s].

    Returns
    -------
    Callable[[float], np.ndarray]
        Function that takes altitude [m] and returns wind vector [m/s]
        in the format [east, north, 0] (no vertical wind component).
    """
    alt = np.asarray(altitude_breakpoints_m, dtype=float)
    we = np.asarray(wind_east_ms, dtype=float)
    wn = np.asarray(wind_north_ms, dtype=float)

    def wind_profile(altitude_m: float) -> np.ndarray:
        wx = float(np.interp(altitude_m, alt, we))
        wy = float(np.interp(altitude_m, alt, wn))
        return np.array([wx, wy, 0.0])

    return wind_profile


# ---------------------------------------------------------------------------
# Main EOM function
# ---------------------------------------------------------------------------

def eom_6dof(
    t: float,
    state: np.ndarray,
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
) -> np.ndarray:
    """Compute the time-derivative of the 13-element state vector.

    Parameters
    ----------
    t : float
        Current time [s].
    state : np.ndarray, shape (13,)
        ``[x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]``.
    mass_kg : float
        Total projectile mass [kg].
    inertia_tensor : np.ndarray
        Diagonal moment-of-inertia tensor in the body frame [kg·m²].
    aero_model : AeroModel
        Instance providing force and moment methods.
    atmosphere : ISAAtmosphere or WeatherModel
        Instance providing ``get_properties(altitude_m)``.
    wind_vector_ms : np.ndarray, shape (3,)
        Constant wind velocity in the inertial frame [m/s].
    wind_profile : callable, optional
        Function ``f(altitude_m) -> np.ndarray(3,)`` for altitude-dependent wind.
    enable_coriolis : bool
        If True, include Coriolis acceleration.
    latitude_deg : float
        Launch-site latitude [°].
    use_wgs84 : bool
        If True, use WGS-84 oblate-earth gravity.
    rocket_motor : RocketMotor or None
        If provided, computes thrust and time-varying mass.

    Returns
    -------
    np.ndarray, shape (13,)
        State derivative ``d(state)/dt``.
    """
    # --- Unpack state ---
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]
    omega_body = state[10:13]

    # Normalise quaternion
    quat_norm = np.linalg.norm(quat)
    if quat_norm > 1e-12:
        quat = quat / quat_norm

    # --- Atmospheric properties ---
    altitude_m = float(np.clip(pos[2], 0.0, 20000.0))
    atm = atmosphere.get_properties(altitude_m)
    rho = atm["density_kg_m3"]
    speed_of_sound = atm.get("speed_of_sound_m_s", atm.get("speed_of_sound_ms", 340.0))

    # --- Wind ---
    if wind_profile is not None:
        wind = wind_profile(altitude_m)
    else:
        wind = wind_vector_ms

    # --- Velocity relative to air ---
    vel_rel = vel - wind
    speed_rel = np.linalg.norm(vel_rel)
    mach = speed_rel / speed_of_sound if speed_of_sound > 0 else 0.0

    # --- Compute angle of attack ---
    body_x_inertial = quaternion_rotate_vector(quat, np.array([1.0, 0.0, 0.0]))
    if speed_rel > 1e-12:
        cos_alpha = np.clip(np.dot(vel_rel / speed_rel, body_x_inertial), -1.0, 1.0)
        alpha_rad = math.acos(cos_alpha)
    else:
        alpha_rad = 0.0

    # --- Angular velocity in inertial frame (for Magnus force) ---
    omega_inertial = quaternion_rotate_vector(quat, omega_body)

    # --- Aerodynamic forces (inertial frame, Newtons) ---
    F_drag_N = aero_model.drag_force(vel_rel, rho, mach)

    # --- Base bleed drag reduction ---
    if base_bleed_unit is not None:
        drag_mult = base_bleed_unit.drag_multiplier(t)
        F_drag_N = F_drag_N * drag_mult

    F_lift_N = aero_model.lift_force(vel_rel, rho)
    F_magnus_N = aero_model.magnus_force(vel_rel, omega_inertial, rho)

    # --- Gravity (WGS-84 oblate earth or inverse-square) ---
    if use_wgs84:
        from src.wgs84_gravity import wgs84_gravity
        g_local = wgs84_gravity(latitude_deg, altitude_m)
    else:
        g_local = gravity_at_altitude(altitude_m)

    # --- Rocket thrust and mass ---
    current_mass = mass_kg
    F_thrust_N = np.zeros(3)
    if rocket_motor is not None:
        prop_state = rocket_motor.get_state(t)
        current_mass = prop_state.mass_kg
        if prop_state.is_burning:
            thrust_body = np.array([prop_state.thrust_N, 0.0, 0.0])
            F_thrust_N = quaternion_rotate_vector(quat, thrust_body)

    F_gravity_N = np.array([0.0, 0.0, -current_mass * g_local])

    # --- Total force ---
    F_total_N = F_gravity_N + F_drag_N + F_lift_N + F_magnus_N + F_thrust_N

    # --- Translational acceleration ---
    accel_ms2 = F_total_N / current_mass

    # --- Coriolis correction ---
    if enable_coriolis:
        lat_rad = math.radians(latitude_deg)
        accel_ms2 += coriolis_acceleration(vel, lat_rad)

    # --- Quaternion kinematic equation ---
    omega_quat = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]])
    q_dot = 0.5 * quaternion_multiply(quat, omega_quat)

    # --- Rotational dynamics (body frame, with aero moments) ---
    if inertia_tensor.ndim == 2:
        I_vec = np.array([inertia_tensor[0, 0],
                          inertia_tensor[1, 1],
                          inertia_tensor[2, 2]])
    else:
        I_vec = inertia_tensor.copy()

    I_omega = I_vec * omega_body

    # Aerodynamic moments (body frame)
    M_aero_Nm = aero_model.compute_total_moment(
        omega_body, alpha_rad, speed_rel, rho,
    )

    alpha_body = (M_aero_Nm - np.cross(omega_body, I_omega)) / I_vec

    # --- Assemble state derivative ---
    dstate = np.zeros(13)
    dstate[0:3] = vel
    dstate[3:6] = accel_ms2
    dstate[6:10] = q_dot
    dstate[10:13] = alpha_body

    return dstate
