"""Streamlit Web UI for 6-DOF Ballistics Simulator."""

import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

# Override logging so it doesn't mess up Streamlit stdout
import logging
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

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

# --- Advanced Fire Control Modes ---
from src.fire_solution import solve_fire_solution
from src.moving_target import create_target, compute_lead, TARGET_PROFILES
from src.mrsi import solve_mrsi

st.set_page_config(page_title="6-DOF Ballistics Simulator", layout="wide")

# ==============================================================================
# UI SIDEBAR SETUP
# ==============================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=100)
st.sidebar.title("Configuration")

with st.sidebar.expander("Projectile Selection", expanded=True):
    # Map the human readable string to the short key
    proj_map = {cfg["name"]: key for key, cfg in ALL_CONFIGS.items()}
    proj_name_str = st.selectbox("Projectile Type", list(proj_map.keys()), index=0)
    proj_name = proj_map[proj_name_str]  # Internal key
    config = ALL_CONFIGS[proj_name]
    mass_str = f"{config['mass_kg']/1000:.1f} t" if config['mass_kg'] >= 1000 else f"{config['mass_kg']:.1f} kg"
    dia_str = f"{config['diameter_m']:.2f} m" if config['diameter_m'] >= 0.5 else f"{config['diameter_m']*1000:.0f} mm"
    st.write(f"Mass: {mass_str} | Dia: {dia_str} | L: {config.get('length_m', 0):.1f} m")

with st.sidebar.expander("Gun & Launch", expanded=True):
    gun_map = {cfg["name"]: key for key, cfg in ALL_GUN_CONFIGS.items()}
    gun_name_str = st.selectbox("Gun Profile", list(gun_map.keys()), index=0)
    gun_name = gun_map[gun_name_str] # Internal key
    propellant_temp = st.slider("Propellant Temp (C)", -20.0, 50.0, value=21.0, step=1.0)
    spin_rate = st.number_input("Axial Spin (rad/s)", value=1800.0, min_value=0.0)

with st.sidebar.expander("Environment & Weather", expanded=True):
    weather = st.selectbox("Weather Profile", ["standard"] + list(WEATHER_PROFILES.keys()), index=0)
    wind_x = st.number_input("Wind X (+Tail/-Head) [m/s]", value=0.0)
    wind_y = st.number_input("Wind Y (+Right/-Left) [m/s]", value=0.0)
    coriolis = st.checkbox("Enable Coriolis Effect", value=False)
    latitude = st.number_input("Firing Latitude (deg)", value=26.9, min_value=-90.0, max_value=90.0) if coriolis else 26.9
    wgs84 = st.checkbox("Use WGS-84 Gravity", value=True)

with st.sidebar.expander("Advanced Physics (Level 200/300)", expanded=True):
    thrust = st.checkbox("Enable Rocket Thrust (RAP)")
    base_bleed = st.checkbox("Enable Base Bleed (Drag Reduction)")

st.sidebar.markdown("---")
st.sidebar.subheader("Fire Control System")

fire_mode = st.sidebar.radio(
    "Engagement Mode",
    ["1. Direct Fire (Manual)", "2. Inverse Fire (Auto-Aim)", "3. Moving Target Lead", "4. MRSI (Multi-Round)"]
)

# Initialize variables to avoid unbound errors
elevation = 45.0
target_range = 10000.0
mrsi_rounds = 3
target_profile_name = "tank"

if fire_mode == "1. Direct Fire (Manual)":
    elevation = st.sidebar.slider("Elevation Angle (deg)", 5.0, 85.0, value=45.0, step=1.0)
elif fire_mode == "2. Inverse Fire (Auto-Aim)":
    target_range = st.sidebar.number_input("Target Range (m)", 1000.0, 2000000.0, 15000.0, step=1000.0)
elif fire_mode == "3. Moving Target Lead":
    target_profile_name = st.sidebar.selectbox("Target Type", list(TARGET_PROFILES.keys()))
    target_range = st.sidebar.number_input("Initial Target Range (m)", 1000.0, 50000.0, 5000.0, step=500.0)
elif fire_mode == "4. MRSI (Multi-Round)":
    target_range = st.sidebar.number_input("MRSI Target Range (m)", 1000.0, 50000.0, 15000.0, step=1000.0)
    mrsi_rounds = st.sidebar.slider("Rounds to Fire", 2, 5, 3)

st.sidebar.markdown("---")
dt = st.sidebar.number_input("Integration Step Size dt (s)", value=0.5, min_value=0.01)
max_t = st.sidebar.number_input("Max Time (s)", value=600.0, min_value=10.0, max_value=2000.0)

# ==============================================================================
# MAIN ENGINE WRAPPERS
# ==============================================================================
@st.cache_data(show_spinner=False)
def run_simulation(proj_name, gun_name, elevation, temp, spin, weather, wx, wy, coriolis, lat, wgs84, thrust, base_bleed, dt, max_t, velocity_scale=1.0):
    cfg = ALL_CONFIGS[proj_name]
    
    # Safe gun config lookup: fall back to first available gun if key missing
    if gun_name in ALL_GUN_CONFIGS:
        gcfg = ALL_GUN_CONFIGS[gun_name]
    else:
        fallback_key = list(ALL_GUN_CONFIGS.keys())[0]
        gcfg = ALL_GUN_CONFIGS[fallback_key]

    # STAGE 1: Interior
    ib_solver = InteriorBallisticsSolver(gcfg)
    ib_res = ib_solver.solve(propellant_temp_C=float(temp))
    v0 = ib_res.muzzle_velocity_ms * velocity_scale
    
    # Environment Setup
    if weather and weather != "standard" and weather in WEATHER_PROFILES:
        atm = WeatherModel(profile_name=weather)
    else:
        atm = ISAAtmosphere()
    
    aero = AeroModel(
        Cd=cfg["Cd"],
        Cl=cfg["Cl"],
        Cm=cfg["Cm"],
        reference_area_m2=cfg["reference_area_m2"],
        reference_diameter_m=cfg["diameter_m"],
        Cmq=cfg.get("Cmq", -8.0),
        Clp=cfg.get("Clp", -0.01),
        Cnpa=cfg.get("Cnpa", -0.5),
        Cma=cfg.get("Cma", -3.0),
        mach_table=cfg.get("mach_table"),
        cd_table=cfg.get("cd_table"),
        reference_length_m=cfg.get("length_m", cfg["diameter_m"]),
    )
    
    # Motor & Bleed
    motor = None
    if thrust and proj_name in ALL_MOTOR_CONFIGS:
        motor = RocketMotor(ALL_MOTOR_CONFIGS[proj_name])
    
    bleed = None
    if base_bleed and proj_name in ALL_BASEBLEED_CONFIGS:
        bleed = BaseBleedUnit(ALL_BASEBLEED_CONFIGS[proj_name])

    # STAGE 2: Exterior
    v_horiz = v0 * math.cos(math.radians(float(elevation)))
    v_vert = v0 * math.sin(math.radians(float(elevation)))
    state0 = np.array([0.0, 0.0, 0.0, v_horiz, 0.0, v_vert, 1.0, 0.0, 0.0, 0.0, float(spin), 0.0, 0.0])
    
    integrator = TrajectoryIntegrator()
    res = integrator.integrate(
        initial_state=state0, t_span=(0.0, float(max_t)), dt=float(dt),
        mass_kg=cfg["mass_kg"], inertia_tensor=cfg["inertia_tensor"],
        aero_model=aero, atmosphere=atm, wind_vector_ms=np.array([float(wx), float(wy), 0.0]),
        enable_coriolis=bool(coriolis), latitude_deg=float(lat), use_wgs84=bool(wgs84),
        rocket_motor=motor, base_bleed_unit=bleed
    )
    
    # STAGE 3: Terminal
    term_calc = TerminalBallisticsCalculator(cfg["mass_kg"], cfg["diameter_m"], is_he=True)
    term = term_calc.compute(res.state[-1, 3:6], res.mach_number[-1])
    
    return ib_res, res, term

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================
st.title("6-DOF Advanced Ballistic Trajectory Simulator")
st.markdown("Fully coupled interior, exterior, and terminal ballistics using numerical integration.")

run_btn = st.sidebar.button("Run Simulation", use_container_width=True, type="primary")

if run_btn:
    st.session_state.simulation_run = True

if st.session_state.get("simulation_run", False):
    mrsi_schedule = None  # Always initialize

    try:
        with st.spinner(f"Computing {fire_mode}..."):
            # --- Helper to wrap the core simulation for solvers ---
            def run_trajectory(test_elevation, velocity_scale=1.0):
                _ib, _res, _term = run_simulation(
                    proj_name, gun_name, test_elevation, propellant_temp, spin_rate,
                    weather, wind_x, wind_y, coriolis, latitude, wgs84, thrust, base_bleed, dt, max_t,
                    velocity_scale=velocity_scale
                )
                return _res.state[-1, 0], _res.time_s[-1], _term.impact_velocity_ms
            
            # 1. Direct Fire (Manual)
            if fire_mode == "1. Direct Fire (Manual)":
                ib_res, res, term = run_simulation(
                    proj_name, gun_name, elevation, propellant_temp, spin_rate,
                    weather, wind_x, wind_y, coriolis, latitude, wgs84, thrust, base_bleed, dt, max_t
                )

            # 2. Inverse Fire (Auto-Aim)
            elif fire_mode == "2. Inverse Fire (Auto-Aim)":
                ib_config = ALL_GUN_CONFIGS.get(gun_name, list(ALL_GUN_CONFIGS.values())[0])
                v0_est = InteriorBallisticsSolver(ib_config).solve(propellant_temp_C=float(propellant_temp)).muzzle_velocity_ms
                
                sol = solve_fire_solution(v0_est, target_range, run_trajectory)
                if not sol["converged"]:
                    st.warning(f"Solver could not hit {target_range}m. Best effort: {sol['predicted_range_m']:.0f}m at {sol['elevation_deg']:.1f} deg")
                
                elevation = sol["elevation_deg"]
                ib_res, res, term = run_simulation(
                    proj_name, gun_name, elevation, propellant_temp, spin_rate,
                    weather, wind_x, wind_y, coriolis, latitude, wgs84, thrust, base_bleed, dt, max_t
                )

            # 3. Moving Target Lead
            elif fire_mode == "3. Moving Target Lead":
                ib_config = ALL_GUN_CONFIGS.get(gun_name, list(ALL_GUN_CONFIGS.values())[0])
                v0_est = InteriorBallisticsSolver(ib_config).solve(propellant_temp_C=float(propellant_temp)).muzzle_velocity_ms
                
                def estimate_tof(rng):
                    s = solve_fire_solution(v0_est, rng, run_trajectory)
                    return s["tof_s"] if s["converged"] else 0.0

                tgt = create_target(target_range, TARGET_PROFILES[target_profile_name]["speed_ms"], 
                                   TARGET_PROFILES[target_profile_name]["heading_deg"])
                
                lead_sol = compute_lead(tgt, estimate_tof)
                if not lead_sol.converged:
                    st.warning("Lead solver failed to converge.")
                
                final_aim_range = float(np.linalg.norm(lead_sol.aim_point_m[:2]))
                fire_sol = solve_fire_solution(v0_est, final_aim_range, run_trajectory)
                elevation = fire_sol["elevation_deg"]
                
                ib_res, res, term = run_simulation(
                    proj_name, gun_name, elevation, propellant_temp, spin_rate,
                    weather, wind_x, wind_y, coriolis, latitude, wgs84, thrust, base_bleed, dt, max_t
                )
                
            # 4. MRSI (Multi-Round)
            elif fire_mode == "4. MRSI (Multi-Round)":
                ib_config = ALL_GUN_CONFIGS.get(gun_name, list(ALL_GUN_CONFIGS.values())[0])
                v0_est = InteriorBallisticsSolver(ib_config).solve(propellant_temp_C=float(propellant_temp)).muzzle_velocity_ms
                
                mrsi_schedule = solve_mrsi(target_range, v0_est, run_trajectory, num_rounds=mrsi_rounds)
                
                if not mrsi_schedule.converged or mrsi_schedule.num_rounds == 0:
                    st.error(f"Cannot compute MRSI schedule for {target_range}m. The gun/projectile lacks the necessary elevation spread.")
                    st.session_state.simulation_run = False
                    st.stop()
                
                from src.mrsi import CHARGE_ZONES
                r1 = mrsi_schedule.rounds[0]
                elevation = r1.elevation_deg
                vscale = next((z["scale"] for z in CHARGE_ZONES if z["name"] == r1.charge_zone), 1.0)
                ib_res, res, term = run_simulation(
                    proj_name, gun_name, elevation, propellant_temp, spin_rate,
                    weather, wind_x, wind_y, coriolis, latitude, wgs84, thrust, base_bleed, dt, max_t,
                    velocity_scale=vscale
                )
            
            else:
                # Fallback
                ib_res, res, term = run_simulation(
                    proj_name, gun_name, elevation, propellant_temp, spin_rate,
                    weather, wind_x, wind_y, coriolis, latitude, wgs84, thrust, base_bleed, dt, max_t
                )

    except Exception as e:
        st.error(f"Simulation failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.info("Try changing the parameters. Some combinations (e.g. certain gun/projectile pairings or extreme values) may not be supported by the numerical solver.")
        st.session_state.simulation_run = False
        st.stop()

    st.success("Simulation Complete!")

    # Resolve display names for the config
    disp_proj_name = config.get("name", proj_name)
    if gun_name in ALL_GUN_CONFIGS:
        disp_gun_name = ALL_GUN_CONFIGS[gun_name].get("name", gun_name)
    else:
        disp_gun_name = gun_name

    # Top KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Muzzle Velocity", f"{ib_res.muzzle_velocity_ms:.1f} m/s")
    col2.metric("Max Range", f"{res.state[-1, 0]/1000:.2f} km")
    col3.metric("Max Altitude", f"{np.max(res.state[:, 2])/1000:.2f} km")
    col4.metric("Time of Flight", f"{res.time_s[-1]:.1f} s")
    col5.metric("Drift (Cross-range)", f"{res.state[-1, 1]:+.1f} m")

    tabs = st.tabs(["3D Trajectory", "Flight Data", "Terminal Effects"])

    # Extract core state vectors for all tabs
    x_pos, y_pos, z_pos = res.state[:, 0], res.state[:, 1], res.state[:, 2]
    vx_vel, vy_vel, vz_vel = res.state[:, 3], res.state[:, 4], res.state[:, 5]
    q0_q, q1_q, q2_q, q3_q = res.state[:, 6], res.state[:, 7], res.state[:, 8], res.state[:, 9]
    wx_w, wy_w, wz_w = res.state[:, 10], res.state[:, 11], res.state[:, 12]

    # Sub-sample the frames globally
    n_frames_global = min(150, len(x_pos))
    indices = np.linspace(0, len(x_pos)-1, n_frames_global, dtype=int)

    # --- Helper to create 3D animations with AUTO-ORBITING camera ---
    def make_3d_anim_multi(trajectories, title, ax_labels, colors=None):
        """
        trajectories: list of dicts {"x": array, "y": array, "z": array, "name": str}
        """
        if not colors:
            colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Convert numpy arrays to lists for safe Plotly serialization
        for traj in trajectories:
            traj["x"] = list(traj["x"]) if hasattr(traj["x"], 'tolist') else list(traj["x"])
            traj["y"] = list(traj["y"]) if hasattr(traj["y"], 'tolist') else list(traj["y"])
            traj["z"] = list(traj["z"]) if hasattr(traj["z"], 'tolist') else list(traj["z"])
            
        fig = go.Figure()
        
        # 1. Add static path lines
        for i, traj in enumerate(trajectories):
            fig.add_trace(go.Scatter3d(
                x=traj["x"], y=traj["y"], z=traj["z"], mode='lines',
                line=dict(color='rgba(0, 0, 0, 0.4)', width=3), name=f"{traj['name']} Path",
                showlegend=False
            ))
            
        # 2. Add the dynamic markers
        for i, traj in enumerate(trajectories):
            fig.add_trace(go.Scatter3d(
                x=[traj["x"][0]], y=[traj["y"][0]], z=[traj["z"][0]], mode='markers',
                marker=dict(color=colors[i % len(colors)], size=6, line=dict(color='black', width=1)),
                name=traj['name']
            ))
            
        # Max frames
        max_len = max(len(t["x"]) for t in trajectories)
        n_frames = min(150, max_len)
        
        # Bounding box for camera orbit (use unique var names to avoid shadowing)
        flat_x = [val for t in trajectories for val in t["x"]]
        flat_y = [val for t in trajectories for val in t["y"]]
        flat_z = [val for t in trajectories for val in t["z"]]
        cx = (min(flat_x) + max(flat_x)) / 2
        cy = (min(flat_y) + max(flat_y)) / 2
        cz = (min(flat_z) + max(flat_z)) / 2
        span = max(max(flat_x)-min(flat_x), max(flat_y)-min(flat_y), max(flat_z)-min(flat_z))
        if span < 1e-6:
            span = 1.0  # Prevent division by zero
        R = span * 1.8
        
        # Build frames
        frames = []
        for frame_num in range(n_frames):
            frame_data = []
            for i, traj in enumerate(trajectories):
                idx = int(frame_num * (len(traj["x"]) - 1) / max(1, (n_frames - 1)))
                frame_data.append(go.Scatter3d(x=[traj["x"][idx]], y=[traj["y"][idx]], z=[traj["z"][idx]], mode='markers'))
            
            theta = 2 * math.pi * frame_num / n_frames
            eye_x = cx + R * math.cos(theta)
            eye_y = cy + R * math.sin(theta)
            
            frames.append(go.Frame(
                data=frame_data,
                traces=list(range(len(trajectories), len(trajectories)*2)),
                layout=go.Layout(scene=dict(
                    camera=dict(
                        eye=dict(x=(eye_x-cx)/span, y=(eye_y-cy)/span, z=0.6),
                        center=dict(x=0, y=0, z=0)
                    )
                )),
                name=str(frame_num)
            ))
            
        fig.frames = list(frames)
        
        fig.update_layout(
            title=f"{title} (Play for orbiting camera)",
            scene=dict(xaxis_title=ax_labels[0], yaxis_title=ax_labels[1], zaxis_title=ax_labels[2]),
            margin=dict(l=0, r=0, b=0, t=40), height=700,
            updatemenus=[dict(
                type="buttons", showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                ]
            )]
        )
        return fig

    # =========================================================================
    # TAB 0: 3D TRAJECTORY
    # =========================================================================
    with tabs[0]:
        st.subheader("3D Flight Path (Auto-Orbiting)")
        
        traj_list = []
        if fire_mode == "4. MRSI (Multi-Round)" and mrsi_schedule is not None:
            for r in mrsi_schedule.rounds:
                try:
                    from src.mrsi import CHARGE_ZONES
                    vscale = next((z["scale"] for z in CHARGE_ZONES if z["name"] == r.charge_zone), 1.0)
                    _, r_res, _ = run_simulation(
                        proj_name, gun_name, r.elevation_deg, propellant_temp, spin_rate,
                        weather, wind_x, wind_y, coriolis, latitude, wgs84, thrust, base_bleed, dt, max_t,
                        velocity_scale=vscale
                    )
                    traj_list.append({
                        "x": r_res.state[:, 0],
                        "y": r_res.state[:, 1],
                        "z": r_res.state[:, 2],
                        "name": f"Round {r.round_number} ({r.trajectory_type})"
                    })
                except Exception:
                    pass
        
        if not traj_list:
            traj_list.append({
                "x": x_pos, "y": y_pos, "z": z_pos, "name": "Projectile"
            })
            
        fig3d = make_3d_anim_multi(
            traj_list,
            title=f"Trajectory: {disp_proj_name} ({disp_gun_name})", 
            ax_labels=['Downrange (m)', 'Crossrange (m)', 'Altitude (m)']
        )
        st.plotly_chart(fig3d, use_container_width=True)

        # --- 5x 3D Phase Space Animations ---
        st.subheader("Interactive 3D Physics Phase Spaces")
        st.info("ℹ️ **Why do the graphs look the same?** In Direct Fire, you pick the angle. But in **Inverse Fire**, **Moving Target**, and **MRSI**, the solvers automatically aim the gun to hit your target. Since they are all successfully hitting the same Target Range, the physical trajectory arcs out to that range look virtually identical. MRSI displays all 3 rounds in this 3D view!")
        st.write("Explore the projectile's physical evolution through 5 distinct mathematical dimensions.")
        
        view_selection = st.radio(
            "Select 3D Animation View:",
            ["1. Translational Trajectory [X, Y, Z]", 
             "2. Velocity Hodograph [Vx, Vy, Vz]", 
             "3. Gyroscopic Angular Velocity Space [Wx, Wy, Wz]", 
             "4. Projectile Nose Direction (Attitude Unit Sphere)", 
             "5. Quaternion State Evolution [q1, q2, q3]"],
            horizontal=True
        )

        if "1." in view_selection:
            st.info("This is the actual physical path the projectile takes through the air.")
            fig1 = make_3d_anim_multi([{"x": x_pos, "y": y_pos, "z": z_pos, "name": "Path"}], "3D Physical Flight Path", ["Downrange X (m)", "Crossrange Y (m)", "Altitude Z (m)"], colors=['red'])
            st.plotly_chart(fig1, use_container_width=True)
            
        elif "2." in view_selection:
            st.info("This shows how fast the projectile is going in 3D. As gravity pulls it down and air slows it down, its speed arrows change shape.")
            fig2v = make_3d_anim_multi([{"x": vx_vel, "y": vy_vel, "z": vz_vel, "name": "Velocity"}], "3D Velocity Vector Phase Space", ["Vx (m/s)", "Vy (m/s)", "Vz (m/s)"], colors=['orange'])
            st.plotly_chart(fig2v, use_container_width=True)
            
        elif "3." in view_selection:
            st.info("This tracks how the projectile is spinning. It wobbles (nutation) and its nose makes tiny circles (precession) to stay stable.")
            fig3w = make_3d_anim_multi([{"x": wx_w, "y": wy_w, "z": wz_w, "name": "Omega"}], "3D Angular Velocity (Gyroscopic Precession/Nutation)", ["Roll Wx (rad/s)", "Yaw Wy (rad/s)", "Pitch Wz (rad/s)"], colors=['green'])
            st.plotly_chart(fig3w, use_container_width=True)
            
        elif "4." in view_selection:
            st.info("Imagine a laser pointer stuck to the nose of the projectile. This sphere shows exactly where that laser is pointing in the sky as the projectile arcs and wobbles.")
            nx = q0_q**2 + q1_q**2 - q2_q**2 - q3_q**2
            ny = 2 * (q1_q*q2_q + q0_q*q3_q)
            nz = 2 * (q1_q*q3_q - q0_q*q2_q)
            fig4a = make_3d_anim_multi([{"x": nx, "y": ny, "z": nz, "name": "Attitude"}], "Nose Direction (Attitude mapped to Unit Sphere)", ["Inertial X", "Inertial Y", "Inertial Z"], colors=['purple'])
            st.plotly_chart(fig4a, use_container_width=True)
            
        elif "5." in view_selection:
            st.info("This is a super-math version of 3D rotation used by video games and rockets to completely prevent 'gimbal lock' when doing crazy flips in the air.")
            fig5q = make_3d_anim_multi([{"x": q1_q, "y": q2_q, "z": q3_q, "name": "Quaternion"}], "Quaternion Attitude Vector Space", ["q1 (Vector i)", "q2 (Vector j)", "q3 (Vector k)"], colors=['blue'])
            st.plotly_chart(fig5q, use_container_width=True)

    # =========================================================================
    # TAB 1: ADVANCED TELEMETRY (10+ PLOTS)
    # =========================================================================
    with tabs[1]:
        st.subheader("Aerodynamic & Kinematic Telemetry (Animated)")
        st.info("Imagine packing a tiny computer inside the projectile. These 10 graphs are the data from that computer: showing exactly how fast it spins, how much the wind pushes it sideways, and how strong the air resistance is at every single second of its flight.")
        fig2t = make_subplots(
            rows=5, cols=2, 
            subplot_titles=(
                "Altitude vs Time (m)", "Speed vs Time (m/s)", 
                "Mach Number vs Range", "Dynamic Pressure vs Time (kPa)",
                "Angle of Attack vs Time (deg)", "Cross-Range Wind Drift (m)",
                "Axial Spin Rate vs Time (rad/s)", "Pitch Rate vs Time (rad/s)",
                "Yaw Rate vs Time (rad/s)", "Velocity Vector Magnitude vs Range"
            ),
            vertical_spacing=0.08
        )
        
        base_traces = [
            (res.time_s, z_pos, "Altitude", '#00CC96', 1, 1),
            (res.time_s, res.speed_ms, "Speed", '#FF9900', 1, 2),
            (x_pos, res.mach_number, "Mach", '#EF553B', 2, 1),
            (res.time_s, res.dynamic_pressure_Pa/1000, "q (kPa)", '#AB63FA', 2, 2),
            (res.time_s, res.angle_of_attack_deg, "AoA (deg)", '#FFA15A', 3, 1),
            (res.time_s, y_pos, "Drift (m)", '#19D3F3', 3, 2),
            (res.time_s, res.state[:, 10], "Spin (rad/s)", '#FF6692', 4, 1),
            (res.time_s, res.state[:, 12], "Pitch Rate", '#B6E880', 4, 2),
            (res.time_s, res.state[:, 11], "Yaw Rate", '#FF97FF', 5, 1),
            (x_pos, res.speed_ms, "V vs X (m/s)", '#FECB52', 5, 2)
        ]
        
        for xd, yd, name, color, row, col in base_traces:
            fig2t.add_trace(go.Scatter(x=xd, y=yd, name=name, line=dict(color=color)), row=row, col=col)
            
        marker_trace_indices = []
        for xd, yd, name, color, row, col in base_traces:
            fig2t.add_trace(
                go.Scatter(x=[xd[0]], y=[yd[0]], mode='markers', marker=dict(color='white', size=10, line=dict(color='black', width=2)), showlegend=False),
                row=row, col=col
            )
            marker_trace_indices.append(len(fig2t.data) - 1)
            
        frames2 = []
        for i in indices:
            frame_data = []
            for base_idx in range(len(base_traces)):
                xd, yd = base_traces[base_idx][0], base_traces[base_idx][1]
                frame_data.append(go.Scatter(x=[xd[i]], y=[yd[i]]))
            frames2.append(go.Frame(data=frame_data, traces=marker_trace_indices, name=str(i)))
            
        fig2t.frames = frames2
        
        fig2t.update_layout(
            height=1400, showlegend=False,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(
                    label="Play All 10 Metrics",
                    method="animate",
                    args=[[str(i) for i in indices], {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
                )]
            )]
        )
        st.plotly_chart(fig2t, use_container_width=True)

    # =========================================================================
    # TAB 2: TERMINAL EFFECTS
    # =========================================================================
    with tabs[2]:
        st.subheader("Impact Kinematics")
        t1, t2, t3 = st.columns(3)
        t1.metric("Impact Speed", f"{term.impact_velocity_ms:.1f} m/s")
        t2.metric("Impact Mach", f"{term.impact_mach:.2f}")
        t3.metric("Angle of Fall", f"{term.angle_of_fall_deg:.1f} deg")

        st.subheader("Destructive Potential")
        c1, c2, c3 = st.columns(3)
        c1.metric("Kinetic Energy", f"{term.kinetic_energy_MJ:.2f} MJ")
        c2.metric("Armor Pen. (RHA)", f"{term.armor_penetration_mm_RHA:.0f} mm")
        c3.metric("Spall Threshold", f"{term.spall_thickness_mm:.0f} mm")

        c4, c5, c6 = st.columns(3)
        c4.metric("Concrete Penetration", f"{term.concrete_penetration_m:.2f} m")
        c5.metric("Crater Diameter", f"{term.crater_diameter_m:.1f} m")
        c6.metric("Lethal Fragmentation", f"{term.lethal_radius_m:.1f} m")
        
    st.markdown("---")
    st.caption("Powered by Advanced Agentic Ballistics Engine")
else:
    st.info("Configure the parameters in the sidebar and click **Run Simulation** to begin.")
