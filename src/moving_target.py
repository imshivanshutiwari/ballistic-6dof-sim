"""
Moving Target Interception Module.

Computes the ballistic lead required to hit a moving target (e.g., a tank
driving in a straight line, or a helicopter flying at constant speed).

The problem:
    - Target is at position P0 moving at velocity V_target
    - Projectile takes T_flight seconds to reach that range
    - During those T_flight seconds, the target moves V_target * T_flight meters
    - We must aim WHERE THE TARGET WILL BE, not where it is now

Solution method:
    Iterative convergence:
    1. Estimate time-of-flight to current target position
    2. Predict where target will be after that time
    3. Re-estimate time-of-flight to the new predicted position
    4. Repeat until convergence (typically 3-5 iterations)

References:
    - FM 6-40: Tactics, Techniques, and Procedures for Field Artillery
      Manual Cannonry, Chapter 15 (Moving Target Engagement)
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class TargetState:
    """Describes the state of a moving target."""
    position_m: np.ndarray       # [x, y, z] current position
    velocity_ms: np.ndarray      # [vx, vy, vz] velocity vector
    description: str = "Unknown"


@dataclass
class InterceptionSolution:
    """Result of the lead computation."""
    aim_point_m: np.ndarray       # [x, y, z] where to aim
    lead_distance_m: float        # how far ahead of the target to aim
    lead_angle_deg: float         # angular lead
    predicted_tof_s: float        # time of flight to interception point
    target_at_impact_m: np.ndarray  # predicted target position at impact
    iterations: int
    converged: bool


# ---------------------------------------------------------------------------
# Pre-defined target profiles
# ---------------------------------------------------------------------------

TARGET_PROFILES = {
    "tank": {
        "description": "Main Battle Tank (MBT)",
        "speed_ms": 15.0,  # ~54 km/h
        "heading_deg": 90.0,  # moving perpendicular to line of fire
        "altitude_m": 0.0,
    },
    "apc": {
        "description": "Armored Personnel Carrier",
        "speed_ms": 22.0,  # ~80 km/h
        "heading_deg": 90.0,
        "altitude_m": 0.0,
    },
    "helicopter": {
        "description": "Attack Helicopter",
        "speed_ms": 70.0,  # ~250 km/h
        "heading_deg": 45.0,
        "altitude_m": 500.0,
    },
    "truck": {
        "description": "Supply Truck Convoy",
        "speed_ms": 25.0,  # ~90 km/h
        "heading_deg": 90.0,
        "altitude_m": 0.0,
    },
}


def create_target(
    range_m: float,
    speed_ms: float,
    heading_deg: float = 90.0,
    altitude_m: float = 0.0,
    cross_range_m: float = 0.0,
    profile_name: Optional[str] = None,
) -> TargetState:
    """Create a moving target state.

    Parameters
    ----------
    range_m : float
        Down-range distance to target [m].
    speed_ms : float
        Target speed [m/s].
    heading_deg : float
        Target heading relative to line of fire [deg].
        0 = moving directly away, 90 = moving perpendicular (crossing),
        180 = moving directly toward the gun.
    altitude_m : float
        Target altitude [m].
    cross_range_m : float
        Lateral offset of target [m].
    profile_name : str, optional
        Use a pre-defined target profile.

    Returns
    -------
    TargetState
    """
    if profile_name and profile_name in TARGET_PROFILES:
        prof = TARGET_PROFILES[profile_name]
        speed_ms = prof["speed_ms"]
        heading_deg = prof["heading_deg"]
        altitude_m = prof["altitude_m"]
        desc = prof["description"]
    else:
        desc = f"Custom target ({speed_ms:.0f} m/s)"

    heading_rad = math.radians(heading_deg)
    velocity = np.array([
        speed_ms * math.cos(heading_rad),  # along line of fire
        speed_ms * math.sin(heading_rad),  # perpendicular
        0.0,
    ])

    position = np.array([range_m, cross_range_m, altitude_m])

    logger.info(
        "Target: %s at %.0f m, moving %.0f m/s heading %.0f deg",
        desc, range_m, speed_ms, heading_deg,
    )

    return TargetState(
        position_m=position,
        velocity_ms=velocity,
        description=desc,
    )


def compute_lead(
    target: TargetState,
    estimate_tof_fn,
    max_iterations: int = 10,
    tolerance_m: float = 5.0,
) -> InterceptionSolution:
    """Compute the ballistic lead to intercept a moving target.

    Parameters
    ----------
    target : TargetState
        Current target state.
    estimate_tof_fn : callable
        Function f(range_m) -> tof_s that estimates time-of-flight
        to a given range.
    max_iterations : int
        Maximum convergence iterations.
    tolerance_m : float
        Convergence tolerance on aim point shift [m].

    Returns
    -------
    InterceptionSolution
    """
    aim_point = target.position_m.copy()

    for i in range(max_iterations):
        # Estimate TOF to current aim point
        aim_range = np.linalg.norm(aim_point[:2])  # horizontal range
        tof = estimate_tof_fn(aim_range)

        # Predict where target will be after TOF
        old_aim = aim_point.copy()
        aim_point = target.position_m + target.velocity_ms * tof

        shift = np.linalg.norm(aim_point - old_aim)
        logger.debug(
            "  Lead iter %d: aim=(%.0f, %.0f), tof=%.2f s, shift=%.1f m",
            i + 1, aim_point[0], aim_point[1], tof, shift,
        )

        if shift < tolerance_m:
            lead_dist = np.linalg.norm(aim_point - target.position_m)
            lead_angle = math.degrees(math.atan2(
                lead_dist,
                np.linalg.norm(target.position_m[:2]),
            ))

            logger.info(
                "  Lead converged: aim=(%.0f, %.0f, %.0f) m, "
                "lead=%.0f m, tof=%.2f s",
                aim_point[0], aim_point[1], aim_point[2],
                lead_dist, tof,
            )

            return InterceptionSolution(
                aim_point_m=aim_point,
                lead_distance_m=lead_dist,
                lead_angle_deg=lead_angle,
                predicted_tof_s=tof,
                target_at_impact_m=aim_point,
                iterations=i + 1,
                converged=True,
            )

    lead_dist = np.linalg.norm(aim_point - target.position_m)
    return InterceptionSolution(
        aim_point_m=aim_point,
        lead_distance_m=lead_dist,
        lead_angle_deg=0.0,
        predicted_tof_s=tof,
        target_at_impact_m=aim_point,
        iterations=max_iterations,
        converged=False,
    )
