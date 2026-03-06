"""
Multi-Round Simultaneous Impact (MRSI) Solver.

MRSI is a devastating modern artillery tactic where a single gun fires
multiple rounds at different elevation angles (and thus different
trajectories and flight times) so that ALL rounds arrive on the target
at the exact same instant.

A skilled crew with an M109 Howitzer can fire 3-4 rounds that all impact
within a 2-second window, giving the enemy zero time to react.

How it works:
    1. The gun fires the FIRST round at high elevation (long TOF, e.g. 70s)
    2. Immediately fires SECOND round at medium elevation (shorter TOF, e.g. 50s)
    3. Fires THIRD round at low elevation (shortest TOF, e.g. 35s)
    4. All three rounds land on the target within 1-2 seconds

For 3+ rounds, the gun uses different propellant charge zones (e.g.,
Zone 7 full charge, Zone 5 reduced charge) to produce different muzzle
velocities. Each velocity + elevation combination gives a unique
trajectory to the same target.

This module computes the firing schedule: what elevation angles and
charge zones to use, and how many seconds to wait between shots.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

from src.fire_solution import solve_fire_solution

logger = logging.getLogger(__name__)


@dataclass
class MRSIRound:
    """A single round in the MRSI firing schedule."""
    round_number: int
    elevation_deg: float
    time_of_flight_s: float
    fire_delay_s: float        # seconds after first round is fired
    predicted_range_m: float
    trajectory_type: str       # "HIGH", "MEDIUM", "LOW"
    charge_zone: str           # e.g. "Zone 7 (Full)", "Zone 5 (Reduced)"


@dataclass
class MRSISchedule:
    """Complete MRSI firing schedule."""
    target_range_m: float
    num_rounds: int
    rounds: List[MRSIRound]
    total_window_s: float      # time from first shot to last shot
    impact_spread_s: float     # predicted spread at impact
    converged: bool


# Charge zone definitions (velocity multipliers)
# Based on M109 Howitzer charge zones
CHARGE_ZONES = [
    {"name": "Zone 7 (Full)",    "scale": 1.00},
    {"name": "Zone 5 (Reduced)", "scale": 0.85},
    {"name": "Zone 3 (White Bag)", "scale": 0.72},
]


def solve_mrsi(
    target_range_m: float,
    muzzle_velocity: float,
    run_trajectory_fn,
    num_rounds: int = 3,
    elevation_candidates: Optional[List[float]] = None,
    range_tolerance_m: float = 50.0,
) -> MRSISchedule:
    """Compute an MRSI firing schedule using multiple charge zones.

    Parameters
    ----------
    target_range_m : float
        Distance to target [m].
    muzzle_velocity : float
        Full-charge muzzle velocity [m/s].
    run_trajectory_fn : callable
        Function f(elevation_deg, velocity_scale=1.0) -> (range_m, tof_s, impact_speed_ms)
        that runs the full trajectory simulation. Must accept a velocity_scale parameter.
    num_rounds : int
        Number of simultaneous rounds (2-5 typical). Default: 3.
    elevation_candidates : list of float, optional
        Unused (kept for backward compatibility).
    range_tolerance_m : float
        Maximum acceptable range error per round [m].

    Returns
    -------
    MRSISchedule
    """
    logger.info(
        "MRSI solver: target=%.0f m, %d rounds, v0=%.0f m/s",
        target_range_m, num_rounds, muzzle_velocity,
    )

    valid_solutions = []

    # For each charge zone, find LOW and HIGH angle solutions
    for zone in CHARGE_ZONES:
        scale = zone["scale"]
        v0_scaled = muzzle_velocity * scale
        zone_name = zone["name"]

        # Create a wrapper that fixes the velocity scale
        def make_fn(s):
            return lambda elev: run_trajectory_fn(elev, velocity_scale=s)

        traj_fn = make_fn(scale)

        # --- Find LOW angle solution (0.5 to 44.9 deg) ---
        sol_low = solve_fire_solution(
            v0_scaled, target_range_m, traj_fn,
            elevation_bounds=(0.5, 44.9), tolerance_m=range_tolerance_m,
            max_iterations=20,
        )
        if sol_low["converged"]:
            sol_low["_zone"] = zone_name
            sol_low["_type"] = "LOW"
            valid_solutions.append(sol_low)
            logger.debug(
                "  %s LOW: %.1f deg, TOF=%.1f s, range=%.0f m",
                zone_name, sol_low["elevation_deg"], sol_low["tof_s"],
                sol_low["predicted_range_m"],
            )

        # --- Find HIGH angle solution (45.1 to 85 deg) ---
        sol_hi = solve_fire_solution(
            v0_scaled, target_range_m, traj_fn,
            elevation_bounds=(45.1, 85.0), tolerance_m=range_tolerance_m,
            max_iterations=20, high_angle=True,
        )
        if sol_hi["converged"]:
            sol_hi["_zone"] = zone_name
            sol_hi["_type"] = "HIGH"
            valid_solutions.append(sol_hi)
            logger.debug(
                "  %s HIGH: %.1f deg, TOF=%.1f s, range=%.0f m",
                zone_name, sol_hi["elevation_deg"], sol_hi["tof_s"],
                sol_hi["predicted_range_m"],
            )

        # Stop searching if we have enough solutions
        if len(valid_solutions) >= num_rounds:
            break

    if len(valid_solutions) < num_rounds:
        logger.warning(
            "  Only %d valid solutions found (need %d).",
            len(valid_solutions), num_rounds,
        )
        num_rounds = min(num_rounds, len(valid_solutions))

    if num_rounds <= 1:
        return MRSISchedule(
            target_range_m=target_range_m,
            num_rounds=0,
            rounds=[],
            total_window_s=0.0,
            impact_spread_s=0.0,
            converged=False,
        )

    # Sort by TOF (longest first = first to fire)
    valid_solutions.sort(key=lambda s: s["tof_s"], reverse=True)
    selected = valid_solutions[:num_rounds]

    # Deduplicate: remove solutions with TOF too close (< 5s difference)
    deduped = [selected[0]]
    for sol in selected[1:]:
        if all(abs(sol["tof_s"] - d["tof_s"]) > 3.0 for d in deduped):
            deduped.append(sol)
    selected = deduped[:num_rounds]

    # --- Build the schedule ---
    longest_tof = selected[0]["tof_s"]
    mrsi_rounds = []

    for i, sol in enumerate(selected):
        elev_deg = float(sol["elevation_deg"])
        tof = float(sol["tof_s"])
        delay = longest_tof - tof
        rng = float(sol["predicted_range_m"])
        ttype = sol.get("_type", "MEDIUM")
        zone_name = sol.get("_zone", "Full")

        mrsi_rounds.append(
            MRSIRound(
                round_number=i + 1,
                elevation_deg=elev_deg,
                time_of_flight_s=tof,
                fire_delay_s=max(0.0, delay),
                predicted_range_m=rng,
                trajectory_type=ttype,
                charge_zone=zone_name,
            )
        )

    # Impact spread
    impact_times = [r.fire_delay_s + r.time_of_flight_s for r in mrsi_rounds]
    impact_spread = max(impact_times) - min(impact_times)
    total_window = max(impact_times)

    return MRSISchedule(
        target_range_m=target_range_m,
        num_rounds=len(mrsi_rounds),
        rounds=mrsi_rounds,
        total_window_s=total_window,
        impact_spread_s=impact_spread,
        converged=True,
    )
