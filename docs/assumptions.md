# Assumptions and Limitations

## 6-DOF Ballistic Trajectory Simulator — v1.0.0

This document lists all modelling assumptions and known limitations.

---

## Assumptions

1. **Rigid body** — The projectile is a rigid body with constant mass and
   inertia tensor.  No mass variation (propellant burn) is modelled.

2. **Flat, non-rotating Earth** (default) — The coordinate system is a flat-
   Earth inertial frame (z-up).  **Coriolis correction is available** as an
   option (`--coriolis`) but assumes flat earth geometry with a latitude-
   dependent correction term.  Centrifugal acceleration is neglected.

3. **International Standard Atmosphere (ISA)** — Air properties from the U.S.
   Standard Atmosphere 1976 (0–20 km).  Real weather, humidity, and non-
   standard day profiles are not included.

4. **Mach-dependent drag coefficient** — $C_D$ is interpolated from a
   piecewise lookup table (McCoy-derived) as a function of Mach number.
   This captures the subsonic-transonic-supersonic drag rise correctly.
   However, the tables are representative and not calibrated to a specific
   lot of ammunition.

5. **Constant lift and Magnus force coefficients** — $C_L$ and $C_M$ are
   treated as constants, independent of Mach and angle of attack.

6. **Aerodynamic moments** — Pitch-damping ($C_{mq}$), roll-damping
   ($C_{lp}$), and Magnus moment ($C_{npa}$) are included but modelled with
   constant coefficients.  A full model would use Mach-dependent moment
   derivatives.

7. **Simplified lift direction** — Lift is computed perpendicular to velocity
   in the vertical plane.  A full model would resolve normal and side forces
   from body-axis angles.

8. **No aerodynamic overturning moment** — The static stability derivative
   ($C_{m\alpha}$) is not included.  The pitch-damping moment damps
   transverse rates but there is no restoring moment.

9. **Point-mass launch (no barrel dynamics)** — Simulation starts at the
   muzzle; in-barrel ballistics are not modelled.

10. **No terminal effects** — Impact is detected when $z \leq 0$.  No
    penetration, ricochet, or fuze model.

---

## Known Limitations

| Limitation | Effect |
|---|---|
| No $C_{m\alpha}$ (overturning moment) | Static stability margin not enforced |
| Constant $C_L$, $C_M$ | Inaccurate at high AoA or near transonic |
| Cd table is generic | Not lot-specific; range predictions ±5–10% |
| No wind shear profile in CLI | CLI accepts constant wind; altitude-dependent wind available via API |
| Flat Earth | Significant errors for trajectories > 100 km |
| No Coriolis by default | Must opt in with `--coriolis` flag |
| ISA only (0–20 km) | Cannot simulate very-high-altitude trajectories |
| No propulsion | Cannot simulate rocket-boost or sustainer phases |
| No spin-up model | Spin rate is set at launch, not computed from rifling |
| No gyroscopic stability number | $S_g$ not computed or checked |

---

## What Is Modelled vs. Production Fire Control

| Feature | This Simulator | Production FCS |
|---|---|---|
| 6-DOF dynamics | ✅ | ✅ |
| Quaternion attitude | ✅ | ✅ |
| Mach-dependent $C_D$ | ✅ | ✅ (lot-specific) |
| Aero moments (Cmq, Clp) | ✅ (constant) | ✅ (Mach-dependent) |
| Magnus force & moment | ✅ | ✅ |
| Coriolis | ✅ (optional) | ✅ (always on) |
| Wind profile | ✅ (API) | ✅ (met data) |
| Overturning moment | ❌ | ✅ |
| WGS-84 oblate earth | ❌ | ✅ |
| Real met data | ❌ | ✅ |
| Propulsion model | ❌ | ✅ (for rockets) |
| Lot-specific coefficients | ❌ | ✅ |

---

## References

- Etkin, B. *Dynamics of Flight: Stability and Control*, 3rd ed. Wiley, 1996.
- McCoy, R. L. *Modern Exterior Ballistics*, 2nd ed. Schiffer, 2012.
- U.S. Standard Atmosphere, 1976. NOAA / NASA / USAF.
- STANAG 4355 — Modified Point Mass and Five-DOF Trajectory Models.
