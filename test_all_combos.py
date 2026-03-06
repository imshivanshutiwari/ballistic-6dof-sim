"""
Exhaustive test: runs the simulation with every projectile, gun, and fire mode combo.
Prints PASS/FAIL + the exception for each.
"""
import sys, os, math, traceback
import numpy as np

from src.atmosphere import ISAAtmosphere
from src.aerodynamics import AeroModel
from src.integrator import TrajectoryIntegrator
from src.projectile_config import ALL_CONFIGS
from src.interior_ballistics import InteriorBallisticsSolver, ALL_GUN_CONFIGS
from src.terminal_ballistics import TerminalBallisticsCalculator
from src.weather import WeatherModel, WEATHER_PROFILES
from src.propulsion import RocketMotor, ALL_MOTOR_CONFIGS
from src.base_bleed import BaseBleedUnit, ALL_BASEBLEED_CONFIGS
from src.fire_solution import solve_fire_solution
from src.moving_target import create_target, compute_lead, TARGET_PROFILES
from src.mrsi import solve_mrsi

def run_sim(proj_name, gun_name, elevation, temp=21.0, spin=1800.0,
            weather="standard", wx=0.0, wy=0.0, coriolis=False, lat=26.9,
            wgs84=True, thrust=False, base_bleed=False, dt=0.5, max_t=600.0):
    cfg = ALL_CONFIGS[proj_name]
    if gun_name in ALL_GUN_CONFIGS:
        gcfg = ALL_GUN_CONFIGS[gun_name]
    else:
        gcfg = ALL_GUN_CONFIGS[list(ALL_GUN_CONFIGS.keys())[0]]

    ib_solver = InteriorBallisticsSolver(gcfg)
    ib_res = ib_solver.solve(propellant_temp_C=float(temp))
    v0 = ib_res.muzzle_velocity_ms

    if weather and weather != "standard" and weather in WEATHER_PROFILES:
        atm = WeatherModel(profile_name=weather)
    else:
        atm = ISAAtmosphere()

    aero = AeroModel(
        Cd=cfg["Cd"], Cl=cfg["Cl"], Cm=cfg["Cm"],
        reference_area_m2=cfg["reference_area_m2"],
        reference_diameter_m=cfg["diameter_m"],
        Cmq=cfg.get("Cmq", -8.0), Clp=cfg.get("Clp", -0.01),
        Cnpa=cfg.get("Cnpa", -0.5), Cma=cfg.get("Cma", -3.0),
        mach_table=cfg.get("mach_table"), cd_table=cfg.get("cd_table"),
        reference_length_m=cfg.get("length_m", cfg["diameter_m"]),
    )

    motor = None
    if thrust and proj_name in ALL_MOTOR_CONFIGS:
        motor = RocketMotor(ALL_MOTOR_CONFIGS[proj_name])
    bleed = None
    if base_bleed and proj_name in ALL_BASEBLEED_CONFIGS:
        bleed = BaseBleedUnit(ALL_BASEBLEED_CONFIGS[proj_name])

    v_horiz = v0 * math.cos(math.radians(float(elevation)))
    v_vert = v0 * math.sin(math.radians(float(elevation)))
    state0 = np.array([0.0, 0.0, 0.0, v_horiz, 0.0, v_vert,
                       1.0, 0.0, 0.0, 0.0, float(spin), 0.0, 0.0])

    integrator = TrajectoryIntegrator()
    res = integrator.integrate(
        initial_state=state0, t_span=(0.0, float(max_t)), dt=float(dt),
        mass_kg=cfg["mass_kg"], inertia_tensor=cfg["inertia_tensor"],
        aero_model=aero, atmosphere=atm,
        wind_vector_ms=np.array([float(wx), float(wy), 0.0]),
        enable_coriolis=bool(coriolis), latitude_deg=float(lat),
        use_wgs84=bool(wgs84), rocket_motor=motor, base_bleed_unit=bleed
    )

    term_calc = TerminalBallisticsCalculator(cfg["mass_kg"], cfg["diameter_m"], is_he=True)
    term = term_calc.compute(res.state[-1, 3:6], res.mach_number[-1])
    return ib_res, res, term


pass_count = 0
fail_count = 0
failures = []

# ============================================================
# TEST 1: Every projectile + every gun in Direct Fire mode
# ============================================================
print("=" * 80)
print("TEST 1: Direct Fire - Every Projectile x Every Gun")
print("=" * 80)

for proj_key in ALL_CONFIGS:
    for gun_key in ALL_GUN_CONFIGS:
        label = f"  {proj_key:20s} + {gun_key:15s}"
        try:
            ib, res, term = run_sim(proj_key, gun_key, 45.0)
            rng = res.state[-1, 0]
            alt = np.max(res.state[:, 2])
            print(f"{label} -> Range={rng/1000:.1f} km, Alt={alt/1000:.1f} km [PASS]")
            pass_count += 1
        except Exception as e:
            short = str(e).split('\n')[0][:80]
            print(f"{label} -> {short} [FAIL]")
            failures.append((proj_key, gun_key, "Direct", traceback.format_exc()))
            fail_count += 1

# ============================================================
# TEST 2: Inverse Fire with first projectile + each gun
# ============================================================
print("\n" + "=" * 80)
print("TEST 2: Inverse Fire - first projectile + each gun")
print("=" * 80)

first_proj = list(ALL_CONFIGS.keys())[0]
for gun_key in ALL_GUN_CONFIGS:
    label = f"  {first_proj:20s} + {gun_key:15s}"
    try:
        def run_traj(elev, p=first_proj, g=gun_key):
            _, r, t = run_sim(p, g, elev)
            return r.state[-1, 0], r.time_s[-1], t.impact_velocity_ms

        gcfg = ALL_GUN_CONFIGS[gun_key]
        v0 = InteriorBallisticsSolver(gcfg).solve(21.0).muzzle_velocity_ms
        sol = solve_fire_solution(v0, 10000.0, run_traj, max_iterations=10)
        status = "CONV" if sol["converged"] else "NOCONV"
        print(f"{label} -> elev={sol['elevation_deg']:.1f}, rng={sol['predicted_range_m']:.0f}m ({status}) [PASS]")
        pass_count += 1
    except Exception as e:
        short = str(e).split('\n')[0][:80]
        print(f"{label} -> {short} [FAIL]")
        failures.append((first_proj, gun_key, "Inverse", traceback.format_exc()))
        fail_count += 1

# ============================================================
# TEST 3: Weather profiles
# ============================================================
print("\n" + "=" * 80)
print("TEST 3: Weather Profiles")
print("=" * 80)

for wp in ["standard", "hot", "cold", "tropical", "mild"]:
    label = f"  weather={wp:12s}"
    try:
        ib, res, term = run_sim(first_proj, list(ALL_GUN_CONFIGS.keys())[0], 45.0, weather=wp)
        print(f"{label} -> Range={res.state[-1,0]/1000:.1f} km [PASS]")
        pass_count += 1
    except Exception as e:
        short = str(e).split('\n')[0][:80]
        print(f"{label} -> {short} [FAIL]")
        failures.append((first_proj, "default", f"Weather-{wp}", traceback.format_exc()))
        fail_count += 1

# ============================================================
# TEST 4: Thrust + Base Bleed
# ============================================================
print("\n" + "=" * 80)
print("TEST 4: Thrust + Base Bleed combos")
print("=" * 80)

gun0 = list(ALL_GUN_CONFIGS.keys())[0]
for proj_key in ALL_CONFIGS:
    for t_flag, bb_flag in [(False, False), (True, False), (False, True), (True, True)]:
        label = f"  {proj_key:20s} thrust={str(t_flag):5s} bb={str(bb_flag):5s}"
        try:
            ib, res, term = run_sim(proj_key, gun0, 45.0, thrust=t_flag, base_bleed=bb_flag)
            print(f"{label} -> Range={res.state[-1,0]/1000:.1f} km [PASS]")
            pass_count += 1
        except Exception as e:
            short = str(e).split('\n')[0][:80]
            print(f"{label} -> {short} [FAIL]")
            failures.append((proj_key, gun0, f"T={t_flag},BB={bb_flag}", traceback.format_exc()))
            fail_count += 1

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print(f"TOTAL: {pass_count} PASSED, {fail_count} FAILED")
print("=" * 80)

if failures:
    print("\n\nDETAILED FAILURE TRACEBACKS:")
    for proj, gun, mode, tb in failures:
        print(f"\n{'='*60}")
        print(f"FAILED: {proj} + {gun} [{mode}]")
        print(f"{'='*60}")
        # Only print last 15 lines of traceback to keep it short
        lines = tb.strip().split('\n')
        for line in lines[-15:]:
            print(line)
