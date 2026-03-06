"""
WGS-84 Oblate Earth Gravity Model.

Computes gravitational acceleration that varies with both **latitude** and
**altitude** using the WGS-84 reference ellipsoid and Somigliana's formula.

On a perfect sphere, gravity is the same at the equator and the poles.
On the real Earth (which bulges at the equator), gravity at the poles
(~9.832 m/s²) is stronger than at the equator (~9.780 m/s²).

The model implements:
    1. **Somigliana normal gravity** on the ellipsoid surface  γ₀(φ)
    2. **Free-air correction** for altitude above the ellipsoid  γ(φ, h)

References:
    - WGS 84 Implementation Manual, Eurocontrol / IfEN, 1998.
    - Torge, W. *Geodesy*, 3rd ed. de Gruyter, 2001.
    - Moritz, H. *Geodetic Reference System 1980*. Bull. Géodésique, 1980.
"""

import math

# ---------------------------------------------------------------------------
# WGS-84 Ellipsoid Constants
# ---------------------------------------------------------------------------
_A = 6_378_137.0              # semi-major axis [m] (equatorial radius)
_B = 6_356_752.3142           # semi-minor axis [m] (polar radius)
_E2 = 1.0 - (_B / _A) ** 2   # first eccentricity squared  ≈ 0.00669

_GM = 3.986004418e14          # geocentric gravitational constant [m³/s²]
_OMEGA = 7.292115e-5          # Earth rotation rate [rad/s]

# Normal gravity on the ellipsoid (Somigliana constants)
_GAMMA_A = 9.7803253359       # gravity at equator [m/s²]
_GAMMA_B = 9.8321849378       # gravity at pole [m/s²]
_K_SOM = ((_B * _GAMMA_B) / (_A * _GAMMA_A)) - 1.0   # Somigliana constant


def somigliana_gravity(latitude_deg: float) -> float:
    """Compute normal gravity on the WGS-84 ellipsoid surface.

    Uses Somigliana's closed-form formula:

        γ₀(φ) = γₐ (1 + k·sin²φ) / √(1 − e²·sin²φ)

    Parameters
    ----------
    latitude_deg : float
        Geodetic latitude [°].

    Returns
    -------
    float
        Normal gravity on the ellipsoid surface [m/s²].
    """
    phi = math.radians(latitude_deg)
    sin2 = math.sin(phi) ** 2
    num = 1.0 + _K_SOM * sin2
    den = math.sqrt(1.0 - _E2 * sin2)
    return _GAMMA_A * num / den


def wgs84_gravity(latitude_deg: float, altitude_m: float) -> float:
    """Compute gravity at a given latitude and altitude above WGS-84.

    Applies the second-order free-air reduction to the Somigliana surface
    gravity:

        γ(φ, h) = γ₀(φ) · [1 − (2/a)(1 + f + m − 2f·sin²φ)h + (3/a²)h²]

    where:
        f = flattening  ≈ 1/298.257
        m = ω²a²b / GM  ≈ 0.00345

    For most ballistic applications the first-order term dominates.

    Parameters
    ----------
    latitude_deg : float
        Geodetic latitude [°].
    altitude_m : float
        Altitude above the ellipsoid [m].

    Returns
    -------
    float
        Gravitational acceleration [m/s²].
    """
    gamma_0 = somigliana_gravity(latitude_deg)

    # First-order free-air gradient  ≈ −3.086 × 10⁻⁶ m/s² per metre
    # (accurate to < 0.01% up to 20 km)
    f = (_A - _B) / _A                     # flattening ≈ 1/298.257
    m = _OMEGA ** 2 * _A ** 2 * _B / _GM   # ≈ 0.00345

    phi = math.radians(latitude_deg)
    sin2 = math.sin(phi) ** 2

    # Free-air correction (2nd-order)
    c1 = (2.0 / _A) * (1.0 + f + m - 2.0 * f * sin2)
    c2 = 3.0 / _A ** 2

    gamma = gamma_0 * (1.0 - c1 * altitude_m + c2 * altitude_m ** 2)
    return max(gamma, 0.0)   # clamp to non-negative
