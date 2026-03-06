"""Quick test: all 4 fire modes."""
import sys, math
import numpy as np
from src.projectile_config import ALL_CONFIGS
from src.interior_ballistics import InteriorBallisticsSolver, ALL_GUN_CONFIGS
from src.terminal_ballistics import TerminalBallisticsCalculator
from src.atmosphere import ISAAtmosphere
from src.aerodynamics import AeroModel
from src.integrator import TrajectoryIntegrator
from src.fire_solution import solve_fire_solution
from src.moving_target import create_target, compute_lead, TARGET_PROFILES
from src.mrsi import solve_mrsi

proj = list(ALL_CONFIGS.keys())[0]
gun = list(ALL_GUN_CONFIGS.keys())[0]
cfg = ALL_CONFIGS[proj]
gcfg = ALL_GUN_CONFIGS[gun]
print(f"Using projectile: {proj}, gun: {gun}")

def run_sim(elev, velocity_scale=1.0):
    ib = InteriorBallisticsSolver(gcfg)
    ib_res = ib.solve(propellant_temp_C=21.0)
    v0 = ib_res.muzzle_velocity_ms * velocity_scale
    atm = ISAAtmosphere()
    aero = AeroModel(Cd=cfg['Cd'], Cl=cfg['Cl'], Cm=cfg['Cm'],
        reference_area_m2=cfg['reference_area_m2'], reference_diameter_m=cfg['diameter_m'],
        Cmq=cfg.get('Cmq',-8.0), Clp=cfg.get('Clp',-0.01), Cnpa=cfg.get('Cnpa',-0.5),
        Cma=cfg.get('Cma',-3.0), mach_table=cfg.get('mach_table'), cd_table=cfg.get('cd_table'),
        reference_length_m=cfg.get('length_m', cfg['diameter_m']))
    vh = v0 * math.cos(math.radians(elev))
    vv = v0 * math.sin(math.radians(elev))
    s0 = np.array([0,0,0,vh,0,vv,1,0,0,0,1800.0,0,0])
    res = TrajectoryIntegrator().integrate(s0, (0,300), 0.5, cfg['mass_kg'], cfg['inertia_tensor'], aero, atm, np.zeros(3))
    tc = TerminalBallisticsCalculator(cfg['mass_kg'], cfg['diameter_m'], is_he=True)
    term = tc.compute(res.state[-1,3:6], res.mach_number[-1])
    return res.state[-1,0], res.time_s[-1], term.impact_velocity_ms

print()
# Test 1: Direct Fire
print("=== TEST 1: Direct Fire ===")
try:
    rng, tof, vf = run_sim(45.0)
    print(f"  PASS: range={rng/1000:.1f} km, tof={tof:.1f} s")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 2: Inverse Fire
print("=== TEST 2: Inverse Fire ===")
try:
    v0_est = InteriorBallisticsSolver(gcfg).solve(propellant_temp_C=21.0).muzzle_velocity_ms
    sol = solve_fire_solution(v0_est, 10000.0, run_sim)
    conv = "CONVERGED" if sol["converged"] else "NOT CONVERGED"
    print(f"  PASS: elev={sol['elevation_deg']:.1f} deg, range={sol['predicted_range_m']:.0f} m ({conv})")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Moving Target
print("=== TEST 3: Moving Target ===")
try:
    v0_est = InteriorBallisticsSolver(gcfg).solve(propellant_temp_C=21.0).muzzle_velocity_ms
    def est_tof(rng):
        s = solve_fire_solution(v0_est, rng, run_sim)
        return s['tof_s'] if s['converged'] else 0.0
    tgt = create_target(5000.0, 15.0, 90.0)
    lead = compute_lead(tgt, est_tof)
    conv = "CONVERGED" if lead.converged else "NOT CONVERGED"
    print(f"  PASS: lead={lead.lead_distance_m:.0f} m, tof={lead.predicted_tof_s:.1f} s ({conv})")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()

# Test 4: MRSI
print("=== TEST 4: MRSI ===")
try:
    v0_est = InteriorBallisticsSolver(gcfg).solve(propellant_temp_C=21.0).muzzle_velocity_ms
    sched = solve_mrsi(10000.0, v0_est, run_sim, num_rounds=3)
    conv = "CONVERGED" if sched.converged else "NOT CONVERGED"
    print(f"  PASS: rounds={sched.num_rounds}, spread={sched.impact_spread_s:.2f} s ({conv})")
    for r in sched.rounds:
        print(f"    R{r.round_number}: elev={r.elevation_deg:.1f} deg, TOF={r.time_of_flight_s:.1f} s, delay={r.fire_delay_s:.1f} s [{r.charge_zone}] ({r.trajectory_type})")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()

print()
print("DONE - All 4 fire modes tested!")
