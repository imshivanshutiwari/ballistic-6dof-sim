# Contributing to Ballistic 6-DOF Simulator

Thank you for considering contributing!  This document provides guidelines
to make the process smooth and consistent.

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/ballistic-6dof-sim.git
cd ballistic-6dof-sim
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
# source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
pip install pytest pytest-cov
```

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

All tests must pass before submitting a pull request.

## Code Style

- **Docstrings**: NumPy-style on every public function and class.
- **Constants**: Named with units (e.g. `GRAVITY_MS2`, not `g`).
- **Type hints**: Use Python type annotations on all function signatures.
- **No magic numbers**: Extract all constants.

## Submitting Changes

1. Fork the repo and create a feature branch from `main`.
2. Write/update tests for your changes.
3. Ensure `pytest` passes and coverage does not decrease.
4. Update documentation (README, docstrings, assumptions).
5. Open a pull request with a clear description.

## Reporting Issues

Open a GitHub issue with:
- A clear title and description
- Steps to reproduce (if applicable)
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under
the MIT License.
