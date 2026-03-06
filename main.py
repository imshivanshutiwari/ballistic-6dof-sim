"""
6-DOF Ballistic Trajectory Simulator — Full 3-Stage Pipeline.

    Stage 1: Interior Ballistics  (in-barrel → muzzle velocity & spin)
    Stage 2: Exterior Ballistics  (flight → trajectory)
    Stage 3: Terminal Ballistics  (impact → damage assessment)

Features:
    - Full CLI via argparse
    - Mach-dependent drag with interpolation tables
    - Aerodynamic moments (overturning, pitch-damping, roll-damping, Magnus)
    - Altitude-dependent gravity (inverse-square law)
    - Optional Coriolis correction
    - Animated 3-D trajectory (FuncAnimation)
    - Automatic result export to CSV
    - Gyroscopic stability factor (Sg) monitoring
    - Python logging with configurable verbosity

Usage:
    python main.py
    python main.py --velocity 684 --elevation 30 --projectile 155mm
    python main.py --interior --projectile 155mm
    python main.py --coriolis --latitude 26.9
    python main.py --help
"""

import argparse
import logging
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, ".")

from src.atmosphere import ISAAtmosphere
from src.aerodynamics import AeroModel
from src.integrator import TrajectoryIntegrator
from src.projectile_config import ALL_CONFIGS, SHELL_155MM
from src.interior_ballistics import InteriorBallisticsSolver, ALL_GUN_CONFIGS
from src.terminal_ballistics import TerminalBallisticsCalculator
from src.weather import WeatherModel, WEATHER_PROFILES
from src.wgs84_gravity import wgs84_gravity
from src.propulsion import RocketMotor, ALL_MOTOR_CONFIGS
from src.base_bleed import BaseBleedUnit, ALL_BASEBLEED_CONFIGS
from src.fire_solution import solve_fire_solution
from src.moving_target import create_target, compute_lead, TARGET_PROFILES
from src.mrsi import solve_mrsi


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="6-DOF Ballistic Trajectory Simulator  (Interior -> Exterior -> Terminal)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --interior\n"
            "  python main.py --velocity 684 --elevation 30\n"
            "  python main.py --projectile 12.7mm --velocity 900\n"
            "  python main.py --coriolis --latitude 26.9\n"
        ),
    )
    parser.add_argument(
        "--projectile", choices=list(ALL_CONFIGS.keys()),
        default="155mm", help="Projectile configuration (default: 155mm)",
    )
    parser.add_argument(
        "--velocity", type=float, default=None,
        help="Muzzle velocity [m/s].  If --interior is set, this is IGNORED "
             "and computed from the gun model.  Default: 827 m/s for 155mm.",
    )
    parser.add_argument(
        "--elevation", type=float, default=45.0,
        help="Launch elevation angle [°] (default: 45.0)",
    )
    parser.add_argument(
        "--spin", type=float, default=None,
        help="Axial spin rate [rad/s].  If --interior is set, this is "
             "computed from the rifling twist.  Default: 1800 rad/s.",
    )
    parser.add_argument(
        "--interior", action="store_true",
        help="Run Stage 1 (interior ballistics) to compute muzzle velocity "
             "and spin from the gun model instead of using --velocity/--spin.",
    )
    parser.add_argument(
        "--wind-x", type=float, default=0.0,
        help="Headwind / tailwind [m/s] (default: 0.0)",
    )
    parser.add_argument(
        "--wind-y", type=float, default=0.0,
        help="Crosswind [m/s] (default: 0.0)",
    )
    parser.add_argument(
        "--coriolis", action="store_true",
        help="Enable Coriolis correction",
    )
    parser.add_argument(
        "--latitude", type=float, default=26.9,
        help="Launch-site latitude [°] (default: 26.9  — Pune, India)",
    )
    parser.add_argument(
        "--dt", type=float, default=0.05,
        help="Integration time-step [s] (default: 0.05)",
    )
    parser.add_argument(
        "--max-time", type=float, default=300.0,
        help="Maximum simulation time [s] (default: 300.0)",
    )
    parser.add_argument(
        "--no-animation", action="store_true",
        help="Show static 3-D plot instead of animation",
    )
    parser.add_argument(
        "--output", type=str, default="trajectory_results.csv",
        help="Output CSV filename (default: trajectory_results.csv)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    # --- Level 200 features ---
    parser.add_argument(
        "--wgs84", action="store_true",
        help="Use WGS-84 oblate-earth gravity (latitude+altitude dependent)",
    )
    parser.add_argument(
        "--weather", type=str, default="standard",
        choices=list(WEATHER_PROFILES.keys()),
        help="Weather profile: standard, hot, cold, tropical, mild (default: standard)",
    )
    parser.add_argument(
        "--propellant-temp", type=float, default=21.0,
        help="Propellant temperature [°C] (default: 21.0).  Affects muzzle velocity via interior ballistics.",
    )
    parser.add_argument(
        "--thrust", action="store_true",
        help="Enable rocket propulsion (only for 122mm rocket).",
    )
    # --- Level 300 features ---
    parser.add_argument(
        "--base-bleed", action="store_true",
        help="Enable base bleed drag reduction (30%% less drag for 25s).",
    )
    parser.add_argument(
        "--target-range", type=float, default=None,
        help="Target range [m]. Solves for the optimal elevation angle.",
    )
    parser.add_argument(
        "--target-speed", type=float, default=None,
        help="Moving target speed [m/s]. Computes ballistic lead.",
    )
    parser.add_argument(
        "--target-heading", type=float, default=90.0,
        help="Moving target heading [deg]. 90=crossing, 0=away (default: 90).",
    )
    parser.add_argument(
        "--mrsi", type=int, default=None,
        help="MRSI mode: number of simultaneous rounds (2-5).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def build_aero_model(config: dict) -> AeroModel:
    """Instantiate an AeroModel from a configuration dictionary."""
    return AeroModel(
        Cd=config["Cd"],
        Cl=config["Cl"],
        Cm=config["Cm"],
        reference_area_m2=config["reference_area_m2"],
        reference_diameter_m=config["diameter_m"],
        Cmq=config.get("Cmq", -8.0),
        Clp=config.get("Clp", -0.01),
        Cnpa=config.get("Cnpa", -0.5),
        Cma=config.get("Cma", -3.0),
        mach_table=config.get("mach_table"),
        cd_table=config.get("cd_table"),
        reference_length_m=config.get("length_m", config["diameter_m"]),
    )


def build_initial_state(
    muzzle_velocity_ms: float,
    elevation_deg: float,
    spin_rate_rad_s: float,
) -> np.ndarray:
    """Create the 13-element initial state vector."""
    elev_rad = math.radians(elevation_deg)
    vx = muzzle_velocity_ms * math.cos(elev_rad)
    vy = 0.0
    vz = muzzle_velocity_ms * math.sin(elev_rad)

    half = elev_rad / 2.0
    q0 = math.cos(half)
    q1 = 0.0
    q2 = -math.sin(half)
    q3 = 0.0

    return np.array([
        0.0, 0.0, 0.0,
        vx, vy, vz,
        q0, q1, q2, q3,
        spin_rate_rad_s, 0.0, 0.0,
    ])


def export_results(result, output_path: str) -> None:
    """Export trajectory results to a CSV file."""
    df = pd.DataFrame({
        "time_s": result.time_s,
        "x_m": result.state[:, 0],
        "y_m": result.state[:, 1],
        "z_m": result.state[:, 2],
        "vx_ms": result.state[:, 3],
        "vy_ms": result.state[:, 4],
        "vz_ms": result.state[:, 5],
        "q0": result.state[:, 6],
        "q1": result.state[:, 7],
        "q2": result.state[:, 8],
        "q3": result.state[:, 9],
        "wx_rads": result.state[:, 10],
        "wy_rads": result.state[:, 11],
        "wz_rads": result.state[:, 12],
        "speed_ms": result.speed_ms,
        "dynamic_pressure_Pa": result.dynamic_pressure_Pa,
        "angle_of_attack_deg": result.angle_of_attack_deg,
        "mach_number": result.mach_number,
    })
    df.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_animated_3d(x, y, z, title: str):
    """Create an animated 3-D trajectory plot."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    margin = 0.05
    ax.set_xlim(np.min(x) - margin * np.ptp(x), np.max(x) + margin * np.ptp(x))
    ax.set_ylim(np.min(y) - margin * max(np.ptp(y), 1.0),
                np.max(y) + margin * max(np.ptp(y), 1.0))
    ax.set_zlim(0, np.max(z) * 1.1)
    ax.set_xlabel("Down-range  x  [m]")
    ax.set_ylabel("Cross-range  y  [m]")
    ax.set_zlabel("Altitude  z  [m]")
    ax.set_title(title)

    line, = ax.plot([], [], [], linewidth=1.5, color="#1f77b4")
    point, = ax.plot([], [], [], "ro", markersize=6)

    n_total = len(x)
    n_frames = min(200, n_total)
    indices = np.linspace(0, n_total - 1, n_frames, dtype=int)

    def update(frame_idx):
        i = indices[frame_idx]
        line.set_data(x[:i+1], y[:i+1])
        line.set_3d_properties(z[:i+1])
        point.set_data([x[i]], [y[i]])
        point.set_3d_properties([z[i]])
        return line, point

    anim = FuncAnimation(fig, update, frames=n_frames, interval=30, blit=False)
    return fig, anim


def plot_static_3d(x, y, z, title: str):
    """Create a static 3-D trajectory plot."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, linewidth=1.5, color="#1f77b4")
    ax.scatter([x[0]], [y[0]], [z[0]], color="green", s=80, label="Launch", zorder=5)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=80, label="Impact", zorder=5)
    ax.set_xlabel("Down-range  x  [m]")
    ax.set_ylabel("Cross-range  y  [m]")
    ax.set_zlabel("Altitude  z  [m]")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_panels(result):
    """Create the two 2-panel figures (speed/alt and AoA/Mach)."""
    t = result.time_s
    x = result.state[:, 0]
    z = result.state[:, 2]

    fig2, (ax_speed, ax_alt) = plt.subplots(2, 1, figsize=(12, 8))
    ax_speed.plot(t, result.speed_ms, linewidth=1.2, color="#ff7f0e")
    ax_speed.set_xlabel("Time [s]")
    ax_speed.set_ylabel("Speed [m/s]")
    ax_speed.set_title("Speed vs. Time")
    ax_speed.grid(True, alpha=0.3)

    ax_alt.plot(x / 1000, z / 1000, linewidth=1.2, color="#2ca02c")
    ax_alt.set_xlabel("Down-range [km]")
    ax_alt.set_ylabel("Altitude [km]")
    ax_alt.set_title("Altitude vs. Range")
    ax_alt.grid(True, alpha=0.3)
    fig2.tight_layout()

    fig3, (ax_aoa, ax_mach) = plt.subplots(2, 1, figsize=(12, 8))
    ax_aoa.plot(t, result.angle_of_attack_deg, linewidth=1.2, color="#d62728")
    ax_aoa.set_xlabel("Time [s]")
    ax_aoa.set_ylabel("Angle of Attack [°]")
    ax_aoa.set_title("Angle of Attack vs. Time")
    ax_aoa.grid(True, alpha=0.3)

    ax_mach.plot(t, result.mach_number, linewidth=1.2, color="#9467bd")
    ax_mach.set_xlabel("Time [s]")
    ax_mach.set_ylabel("Mach Number [–]")
    ax_mach.set_title("Mach Number vs. Time")
    ax_mach.grid(True, alpha=0.3)
    fig3.tight_layout()

    return fig2, fig3


def plot_interior(ib_result):
    """Plot interior ballistics: velocity and pressure vs barrel travel."""
    fig, (ax_v, ax_p) = plt.subplots(2, 1, figsize=(12, 8))

    ax_v.plot(ib_result.travel_m * 100, ib_result.velocity_ms,
              linewidth=1.5, color="#1f77b4")
    ax_v.set_xlabel("Barrel Travel [cm]")
    ax_v.set_ylabel("Projectile Velocity [m/s]")
    ax_v.set_title("Interior Ballistics — Velocity Profile")
    ax_v.grid(True, alpha=0.3)

    ax_p.plot(ib_result.travel_m * 100, ib_result.pressure_MPa,
              linewidth=1.5, color="#d62728")
    ax_p.set_xlabel("Barrel Travel [cm]")
    ax_p.set_ylabel("Chamber Pressure [MPa]")
    ax_p.set_title("Interior Ballistics — Pressure Curve")
    ax_p.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full 3-stage ballistic simulation."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    import sys

# Configure root logger to output to stdout instead of stderr (fixes PowerShell red text)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger("ballistic-sim")

    config = ALL_CONFIGS[args.projectile]

    print("=" * 70)
    print("  6-DOF BALLISTIC TRAJECTORY SIMULATOR")
    print(f"  {config['name']}")
    print("  Interior -> Exterior -> Terminal  Pipeline")
    print("=" * 70)

    # ===================================================================
    #  STAGE 1 — INTERIOR BALLISTICS
    # ===================================================================

    if args.interior and args.projectile in ALL_GUN_CONFIGS:
        gun_config = ALL_GUN_CONFIGS[args.projectile]
        solver = InteriorBallisticsSolver(gun_config)
        ib_result = solver.solve(propellant_temp_C=args.propellant_temp)

        muzzle_velocity = ib_result.muzzle_velocity_ms
        spin_rate = ib_result.muzzle_spin_rate_rad_s

        print(f"\n{'-' * 70}")
        print(f"  STAGE 1 - INTERIOR BALLISTICS  ({gun_config['name']})")
        print(f"{'-' * 70}")
        print(f"  Barrel length        : {gun_config['barrel_length_m']:.3f} m")
        print(f"  Propellant mass      : {gun_config['propellant_mass_kg']:.3f} kg")
        print(f"  Propellant temp      : {args.propellant_temp:.0f} °C")
        print(f"  Peak chamber pressure: {ib_result.peak_pressure_MPa:.1f} MPa")
        print(f"  Muzzle pressure      : {ib_result.muzzle_pressure_MPa:.1f} MPa")
        print(f"  Muzzle velocity      : {muzzle_velocity:.1f} m/s")
        print(f"  Muzzle spin rate     : {spin_rate:.0f} rad/s")
        print(f"  Rifling twist        : 1 in {gun_config['twist_calibres']:.0f} calibres")
    else:
        ib_result = None
        muzzle_velocity = args.velocity if args.velocity else 827.0
        spin_rate = args.spin if args.spin else 1800.0

        if args.interior and args.projectile not in ALL_GUN_CONFIGS:
            print(f"\n  [!] No gun config for '{args.projectile}'. "
                  f"Using manual v={muzzle_velocity} m/s, w={spin_rate} rad/s.")

    # ===================================================================
    #  STAGE 2 - EXTERIOR BALLISTICS
    # ===================================================================

    print(f"\n{'-' * 70}")
    print(f"  STAGE 2 - EXTERIOR BALLISTICS  (6-DOF Flight)")
    print(f"{'-' * 70}")

    # --- Atmosphere: ISA or advanced weather model ---
    if args.weather != "standard":
        atmosphere = WeatherModel(profile_name=args.weather)
        print(f"  Weather model        : {atmosphere.name}")
    else:
        atmosphere = ISAAtmosphere()
        print(f"  Weather model        : ISA Standard Atmosphere")

    aero = build_aero_model(config)
    integrator = TrajectoryIntegrator()
    initial_state = build_initial_state(muzzle_velocity, args.elevation, spin_rate)
    wind = np.array([args.wind_x, args.wind_y, 0.0])

    # --- Base bleed unit ---
    base_bleed = None
    if args.base_bleed and args.projectile in ALL_BASEBLEED_CONFIGS:
        base_bleed = BaseBleedUnit(ALL_BASEBLEED_CONFIGS[args.projectile])
        print(f"  Base bleed           : {base_bleed.name}")
        print(f"    Burn duration      : {base_bleed.burn_duration:.0f} s")
        print(f"    Peak drag cut      : {base_bleed.peak_reduction*100:.0f}%")
    elif args.base_bleed:
        print(f"  [!] No base bleed config for '{args.projectile}'.")

    # --- Rocket motor ---
    rocket_motor = None
    if args.thrust and args.projectile in ALL_MOTOR_CONFIGS:
        rocket_motor = RocketMotor(ALL_MOTOR_CONFIGS[args.projectile])
        print(f"  Rocket motor         : {rocket_motor.name}")
        print(f"    Fuel mass          : {rocket_motor.fuel_mass:.1f} kg")
        print(f"    Burn time          : {rocket_motor.burn_time:.2f} s")
        print(f"    Total impulse      : {rocket_motor.total_impulse_Ns:.0f} N.s")
    elif args.thrust:
        print(f"  [!] No motor config for '{args.projectile}'. Thrust disabled.")

    print(f"  Muzzle velocity      : {muzzle_velocity:.1f} m/s")
    print(f"  Elevation angle      : {args.elevation:.1f} deg")
    print(f"  Axial spin rate      : {spin_rate:.0f} rad/s")
    print(f"  Drag model           : Mach-dependent Cd (interpolated)")
    print(f"  Aero moments         : Cma={config.get('Cma','--')}, "
          f"Cmq={config.get('Cmq','--')}, Clp={config.get('Clp','--')}, "
          f"Cnpa={config.get('Cnpa','--')}")
    print(f"  Gravity model        : {'WGS-84 Oblate Earth' if args.wgs84 else 'Altitude-dependent (inverse-square)'}")
    print(f"  Wind                 : {wind} m/s")
    print(f"  Coriolis             : {'ON' if args.coriolis else 'OFF'}"
          + (f" (lat={args.latitude} deg)" if args.coriolis else ""))

    # Gyroscopic stability check at launch
    atm_sl = atmosphere.get_properties(0.0)
    Sg = aero.gyroscopic_stability_factor(
        spin_rate, muzzle_velocity, atm_sl["density_kg_m3"],
        Ix=config["inertia_tensor"][0, 0],
        Iy=config["inertia_tensor"][1, 1],
    )
    if Sg < 1.0:
        print(f"  [!] GYROSCOPIC STABILITY: Sg = {Sg:.2f}  (UNSTABLE -- will tumble!)")
    else:
        print(f"  Gyroscopic stability : Sg = {Sg:.2f}  (stable)")

    print(f"\n  Integrating (dt={args.dt}s, max_t={args.max_time}s) ...")

    result = integrator.integrate(
        initial_state=initial_state,
        t_span=(0.0, args.max_time),
        dt=args.dt,
        mass_kg=config["mass_kg"],
        inertia_tensor=config["inertia_tensor"],
        aero_model=aero,
        atmosphere=atmosphere,
        wind_vector_ms=wind,
        enable_coriolis=args.coriolis,
        latitude_deg=args.latitude,
        use_wgs84=args.wgs84,
        rocket_motor=rocket_motor,
        base_bleed_unit=base_bleed,
    )

    t = result.time_s
    x = result.state[:, 0]
    y = result.state[:, 1]
    z = result.state[:, 2]

    print(f"\n  Maximum range        : {x[-1]:,.0f} m  ({x[-1] / 1000:.2f} km)")
    print(f"  Maximum altitude     : {np.max(z):,.0f} m  ({np.max(z) / 1000:.2f} km)")
    print(f"  Time of flight       : {t[-1]:.2f} s")
    print(f"  Impact speed         : {result.speed_ms[-1]:.1f} m/s")
    print(f"  Impact Mach          : {result.mach_number[-1]:.2f}")
    if abs(y[-1]) > 0.01:
        print(f"  Cross-range drift    : {y[-1]:+.1f} m")

    # ===================================================================
    #  STAGE 3 - TERMINAL BALLISTICS (expanded)
    # ===================================================================

    print(f"\n{'-' * 70}")
    print(f"  STAGE 3 - TERMINAL BALLISTICS  (Impact Effects)")
    print(f"{'-' * 70}")

    terminal_calc = TerminalBallisticsCalculator(
        mass_kg=config["mass_kg"],
        diameter_m=config["diameter_m"],
        is_he=True,
    )

    impact_velocity_vector = result.state[-1, 3:6]
    impact_mach = result.mach_number[-1]

    terminal = terminal_calc.compute(impact_velocity_vector, impact_mach)

    print(f"  Impact velocity      : {terminal.impact_velocity_ms:.1f} m/s")
    print(f"  Impact Mach          : {terminal.impact_mach:.2f}")
    print(f"  Angle of fall        : {terminal.angle_of_fall_deg:.1f} deg")
    print(f"  Kinetic energy       : {terminal.kinetic_energy_MJ:.3f} MJ")
    print(f"  --- Steel Armor ---")
    print(f"  Armor penetration    : {terminal.armor_penetration_mm_RHA:.0f} mm RHA eq.")
    print(f"  Spall threshold      : {terminal.spall_thickness_mm:.0f} mm")
    print(f"  Behind-armor velocity: {terminal.behind_armor_velocity_ms:.0f} m/s")
    print(f"  Debris cone angle    : {terminal.behind_armor_cone_deg:.0f} deg")
    print(f"  --- Concrete ---")
    print(f"  Concrete penetration : {terminal.concrete_penetration_m:.2f} m")
    print(f"  --- Soft Target ---")
    print(f"  Crater diameter      : {terminal.crater_diameter_m:.1f} m  (soft soil)")
    print(f"  Lethal radius        : {terminal.lethal_radius_m:.0f} m  (fragmentation)")

    # ===================================================================
    #  STAGE 4 - FIRE SOLUTION (if --target-range given)
    # ===================================================================

    if args.target_range is not None:
        print(f"\n{'-' * 70}")
        print(f"  STAGE 4 - FIRE SOLUTION  (Inverse Problem)")
        print(f"{'-' * 70}")
        print(f"  Target range         : {args.target_range:,.0f} m")

        def run_traj_for_fire_sol(elev_deg):
            state0 = build_initial_state(muzzle_velocity, elev_deg, spin_rate)
            res = integrator.integrate(
                initial_state=state0,
                t_span=(0.0, args.max_time),
                dt=args.dt,
                mass_kg=config["mass_kg"],
                inertia_tensor=config["inertia_tensor"],
                aero_model=aero,
                atmosphere=atmosphere,
                wind_vector_ms=wind,
                enable_coriolis=args.coriolis,
                latitude_deg=args.latitude,
                use_wgs84=args.wgs84,
                rocket_motor=rocket_motor,
                base_bleed_unit=base_bleed,
            )
            rng = res.state[-1, 0]
            tof = res.time_s[-1]
            spd = res.speed_ms[-1]
            return rng, tof, spd

        fs = solve_fire_solution(
            muzzle_velocity, args.target_range, run_traj_for_fire_sol,
        )
        print(f"  Solution elevation   : {fs['elevation_deg']:.2f} deg")
        print(f"  Predicted range      : {fs['predicted_range_m']:,.0f} m")
        print(f"  Error                : {fs['error_m']:+.0f} m")
        print(f"  Time of flight       : {fs['tof_s']:.1f} s")
        print(f"  Impact speed         : {fs['impact_speed_ms']:.0f} m/s")
        print(f"  Converged            : {'YES' if fs['converged'] else 'NO'} "
              f"({fs['iterations']} iterations)")

    # ===================================================================
    #  STAGE 5 - MOVING TARGET (if --target-speed given)
    # ===================================================================

    if args.target_speed is not None:
        print(f"\n{'-' * 70}")
        print(f"  STAGE 5 - MOVING TARGET INTERCEPTION")
        print(f"{'-' * 70}")

        target_range = args.target_range if args.target_range else x[-1]
        tgt = create_target(
            range_m=target_range,
            speed_ms=args.target_speed,
            heading_deg=args.target_heading,
        )
        print(f"  Target               : {tgt.description}")
        print(f"  Target range         : {target_range:,.0f} m")
        print(f"  Target speed         : {args.target_speed:.0f} m/s")
        print(f"  Target heading       : {args.target_heading:.0f} deg")

        def estimate_tof(range_m):
            # Quick estimate: use flight time proportional to range
            return t[-1] * (range_m / max(x[-1], 1.0))

        lead = compute_lead(tgt, estimate_tof)
        print(f"  Lead distance        : {lead.lead_distance_m:.0f} m")
        print(f"  Lead angle           : {lead.lead_angle_deg:.1f} deg")
        print(f"  Predicted TOF        : {lead.predicted_tof_s:.1f} s")
        print(f"  Aim point            : ({lead.aim_point_m[0]:,.0f}, "
              f"{lead.aim_point_m[1]:,.0f}, {lead.aim_point_m[2]:.0f}) m")
        print(f"  Converged            : {'YES' if lead.converged else 'NO'} "
              f"({lead.iterations} iterations)")

    # ===================================================================
    #  STAGE 6 - MRSI (if --mrsi given)
    # ===================================================================

    if args.mrsi is not None:
        print(f"\n{'-' * 70}")
        print(f"  STAGE 6 - MULTI-ROUND SIMULTANEOUS IMPACT (MRSI)")
        print(f"{'-' * 70}")

        mrsi_range = args.target_range if args.target_range else x[-1]
        print(f"  Target range         : {mrsi_range:,.0f} m")
        print(f"  Rounds               : {args.mrsi}")

        def run_traj_for_mrsi(elev_deg):
            state0 = build_initial_state(muzzle_velocity, elev_deg, spin_rate)
            res = integrator.integrate(
                initial_state=state0,
                t_span=(0.0, args.max_time),
                dt=args.dt,
                mass_kg=config["mass_kg"],
                inertia_tensor=config["inertia_tensor"],
                aero_model=aero,
                atmosphere=atmosphere,
                wind_vector_ms=wind,
                enable_coriolis=args.coriolis,
                latitude_deg=args.latitude,
                use_wgs84=args.wgs84,
                rocket_motor=rocket_motor,
                base_bleed_unit=base_bleed,
            )
            return res.state[-1, 0], res.time_s[-1], res.speed_ms[-1]

        schedule = solve_mrsi(
            mrsi_range, muzzle_velocity, run_traj_for_mrsi,
            num_rounds=args.mrsi,
            range_tolerance_m=200.0,
            elevation_candidates=list(np.arange(15.0, 75.0, 2.0)),
        )

        if schedule.converged:
            print(f"\n  FIRING SCHEDULE:")
            print(f"  {'Round':>5}  {'Elevation':>10}  {'TOF':>8}  {'Fire At':>10}  {'Arc':>8}")
            print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
            for r in schedule.rounds:
                print(f"  {r.round_number:>5}  {r.elevation_deg:>9.1f}  "
                      f"{r.time_of_flight_s:>7.1f}s  "
                      f"T+{r.fire_delay_s:>7.1f}s  {r.trajectory_type:>8}")
            print(f"\n  Firing window        : {schedule.total_window_s:.1f} s")
            print(f"  Impact spread        : {schedule.impact_spread_s:.2f} s")
        else:
            print(f"  [!] MRSI solver did not converge.")

    # ===================================================================
    #  EXPORT & PLOTS
    # ===================================================================

    print(f"\n{'-' * 70}")
    print(f"  OUTPUT")
    print(f"{'-' * 70}")

    export_results(result, args.output)
    print(f"  Trajectory CSV       : {args.output}")

    # Interior ballistics plot
    if ib_result is not None:
        plot_interior(ib_result)

    # Exterior trajectory
    title = f"{config['name']} - 3-D Trajectory"
    if args.no_animation:
        plot_static_3d(x, y, z, title)
    else:
        fig1, anim = plot_animated_3d(x, y, z, title)

    plot_panels(result)

    plt.show()
    print("\nDone.  Close plot windows to exit.")


if __name__ == "__main__":
    main()

