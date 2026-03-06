# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-03-06

### Added
- **6-DOF equations of motion** with quaternion attitude representation
- **ISA atmosphere model** (0–20 km) with troposphere and stratosphere layers
- **Mach-dependent drag coefficient** with piecewise interpolation tables for
  155 mm, 12.7 mm, and 122 mm projectiles
- **Aerodynamic moments**: pitch-damping (Cmq), roll-damping (Clp), and Magnus
  moment (Cnpa)
- **Magnus force** (spin-induced lateral force)
- **Coriolis correction** (optional, latitude-configurable)
- **Altitude-dependent wind profile** support via interpolation
- **CLI interface** (`argparse`) with full configuration options
- **Animated 3-D trajectory** visualization using `FuncAnimation`
- **Result export** to CSV with full state + derived quantities
- **Three pre-built projectile configurations**: 155 mm M107 shell,
  12.7 mm .50 BMG round, 122 mm Grad-class rocket
- **Validation notebook** (`01_validation.ipynb`) with vacuum trajectory,
  energy conservation, and ISA sea-level tests
- **Range table notebook** (`02_range_table.ipynb`) generating artillery
  firing tables
- **Wind sensitivity notebook** (`03_wind_sensitivity.ipynb`) with
  200-run Monte-Carlo dispersion analysis and CEP computation
- **Automated unit tests** (`tests/`) with pytest covering atmosphere,
  aerodynamics, EOM, and integrator modules
- **CI/CD pipeline** (GitHub Actions) with matrix testing across 3 OS × 4
  Python versions
- `pyproject.toml` for modern packaging
- `CONTRIBUTING.md` and `CHANGELOG.md`
- Professional `README.md` with LaTeX equations, architecture diagram,
  and full documentation
