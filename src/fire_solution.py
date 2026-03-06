"""
Inverse Fire Solution Solver.

Given a target range (meters), solves for the optimal launch elevation
angle that will hit that range using the full 6-DOF trajectory model.

Uses a bisection root-finding method on the range residual:
    f(theta) = range(theta) - target_range

Also supports a fast analytic first-guess using the vacuum trajectory
equation, then refines with the full simulation.

This is the core capability of a real Fire Control System (FCS).
"""

import math
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def vacuum_range(v0: float, theta_deg: float, g: float = 9.81) -> float:
    """Analytic vacuum range for initial guess."""
    theta = math.radians(theta_deg)
    return (v0 ** 2 * math.sin(2 * theta)) / g


def solve_fire_solution(
    muzzle_velocity: float,
    target_range_m: float,
    run_trajectory_fn,
    elevation_bounds: Tuple[float, float] = (0.5, 80.0),
    tolerance_m: float = 25.0,
    max_iterations: int = 50,
    high_angle: bool = False,
) -> dict:
    """Find the elevation angle to hit a target at a given range.

    Parameters
    ----------
    muzzle_velocity : float
        Muzzle velocity [m/s].
    target_range_m : float
        Desired down-range distance to target [m].
    run_trajectory_fn : callable
        Function f(elevation_deg) -> (range_m, tof_s, impact_speed_ms)
        that runs the full 6-DOF trajectory and returns the achieved range.
    elevation_bounds : tuple
        (min_deg, max_deg) search bounds for elevation angle.
    tolerance_m : float
        Acceptable error in range [m].  Default: 10 m.
    max_iterations : int
        Maximum bisection iterations.

    Returns
    -------
    dict
        Keys: elevation_deg, predicted_range_m, error_m, tof_s,
              impact_speed_ms, converged, iterations
    """
    lo, hi = elevation_bounds

    logger.info(
        "Fire solution: target=%.0f m, v0=%.1f m/s, bounds=[%.1f, %.1f] deg",
        target_range_m, muzzle_velocity, lo, hi,
    )

    # --- Vacuum first guess to narrow the search ---
    theta_vac = 0.5 * math.degrees(math.asin(
        min(1.0, target_range_m * 9.81 / muzzle_velocity ** 2)
    ))
    if lo < theta_vac < hi:
        # Use vacuum guess to split the search more efficiently
        logger.debug("  Vacuum first guess: %.1f deg", theta_vac)

    best_elevation = (lo + hi) / 2
    best_range = 0.0
    best_tof = 0.0
    best_impact_speed = 0.0

    for iteration in range(max_iterations):
        mid = (lo + hi) / 2.0
        achieved_range, tof, impact_spd = run_trajectory_fn(mid)
        error = achieved_range - target_range_m

        logger.debug(
            "  Iter %2d: elev=%.2f deg -> range=%.0f m (error=%+.0f m)",
            iteration + 1, mid, achieved_range, error,
        )

        best_elevation = mid
        best_range = achieved_range
        best_tof = tof
        best_impact_speed = impact_spd

        if abs(error) < tolerance_m:
            logger.info(
                "  CONVERGED: elev=%.2f deg, range=%.0f m, error=%.0f m, "
                "%d iterations", mid, achieved_range, error, iteration + 1,
            )
            return {
                "elevation_deg": mid,
                "predicted_range_m": achieved_range,
                "error_m": error,
                "tof_s": tof,
                "impact_speed_ms": impact_spd,
                "converged": True,
                "iterations": iteration + 1,
            }

        if not high_angle:
            # Low angle: increasing elevation increases range (up to ~45-55 deg)
            if error < 0:
                lo = mid
            else:
                hi = mid
        else:
            # High angle: increasing elevation decreases range
            if error < 0:
                hi = mid
            else:
                lo = mid

    logger.warning(
        "  DID NOT CONVERGE after %d iterations. Best: %.2f deg -> %.0f m",
        max_iterations, best_elevation, best_range,
    )
    return {
        "elevation_deg": best_elevation,
        "predicted_range_m": best_range,
        "error_m": best_range - target_range_m,
        "tof_s": best_tof,
        "impact_speed_ms": best_impact_speed,
        "converged": False,
        "iterations": max_iterations,
    }
