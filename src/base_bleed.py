"""
Base Bleed & Rocket-Assisted Projectile (RAP) Drag Reduction Model.

Base bleed is a technique where a small pyrotechnic gas generator in the
base of an artillery shell emits hot gas into the low-pressure wake behind
the projectile.  This fills the vacuum, raises base pressure, and
dramatically reduces base drag (which accounts for ~50% of total drag on
a typical artillery shell).

Effect:
    - Drag reduction factor of 15-30% during the bleed phase
    - Typical bleed duration: 20-30 seconds for 155mm shells
    - Range extension: +25-40%

The model applies a time-dependent drag multiplier to the aerodynamic model.

References:
    - Klingenberg, G. & Heimerl, J.M. "Gun Muzzle Blast and Flash",
      AIAA Progress in Astronautics and Aeronautics, Vol 139, 1992.
    - Danberg, J.E. & Nietubicz, C.J. "Predicted flight performance of
      base-bleed projectiles", J. Spacecraft and Rockets, 29(3), 1992.
"""

import math
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BaseBleedState:
    """Current state of the base bleed unit at a given time."""
    is_active: bool
    drag_reduction_factor: float   # 0.0 = no reduction, 0.30 = 30% drag cut
    elapsed_s: float
    remaining_s: float


# ---------------------------------------------------------------------------
# Pre-defined base bleed configurations
# ---------------------------------------------------------------------------

BASEBLEED_155MM = {
    "name": "155mm M864 DPICM Base Bleed",
    "burn_duration_s": 25.0,
    "peak_drag_reduction": 0.30,    # 30% drag reduction at peak
    "ramp_up_s": 2.0,               # time to reach peak bleed
    "ramp_down_s": 3.0,             # tail-off time near end of burn
    "ignition_delay_s": 0.5,        # delay after muzzle exit
}

BASEBLEED_122MM = {
    "name": "122mm Extended Range Base Bleed",
    "burn_duration_s": 18.0,
    "peak_drag_reduction": 0.25,
    "ramp_up_s": 1.5,
    "ramp_down_s": 2.5,
    "ignition_delay_s": 0.3,
}

ALL_BASEBLEED_CONFIGS = {
    "155mm": BASEBLEED_155MM,
    "122mm": BASEBLEED_122MM,
}


class BaseBleedUnit:
    """Time-dependent base bleed drag reduction model.

    The drag reduction follows a trapezoidal time profile:
        - Ignition delay (no reduction)
        - Ramp-up to peak reduction
        - Sustained peak reduction
        - Ramp-down to zero

    Parameters
    ----------
    config : dict
        Base bleed configuration dictionary.
    """

    def __init__(self, config: dict):
        self.name = config["name"]
        self.burn_duration = config["burn_duration_s"]
        self.peak_reduction = config["peak_drag_reduction"]
        self.ramp_up = config["ramp_up_s"]
        self.ramp_down = config["ramp_down_s"]
        self.ignition_delay = config["ignition_delay_s"]
        self.total_duration = self.ignition_delay + self.burn_duration

        logger.info("Base bleed: %s (%.0fs burn, %.0f%% peak reduction)",
                     self.name, self.burn_duration, self.peak_reduction * 100)

    def get_state(self, t: float) -> BaseBleedState:
        """Get the base bleed state at time t [s] after launch.

        Parameters
        ----------
        t : float
            Time since muzzle exit [s].

        Returns
        -------
        BaseBleedState
        """
        # Before ignition
        if t < self.ignition_delay:
            return BaseBleedState(
                is_active=False,
                drag_reduction_factor=0.0,
                elapsed_s=0.0,
                remaining_s=self.total_duration - t,
            )

        # After burn-out
        if t > self.total_duration:
            return BaseBleedState(
                is_active=False,
                drag_reduction_factor=0.0,
                elapsed_s=t - self.ignition_delay,
                remaining_s=0.0,
            )

        # Active phase
        t_burn = t - self.ignition_delay  # time since ignition
        remaining = self.total_duration - t

        # Trapezoidal profile
        if t_burn < self.ramp_up:
            # Ramp up
            factor = self.peak_reduction * (t_burn / self.ramp_up)
        elif t_burn > (self.burn_duration - self.ramp_down):
            # Ramp down
            t_from_end = self.burn_duration - t_burn
            factor = self.peak_reduction * (t_from_end / self.ramp_down)
        else:
            # Sustained peak
            factor = self.peak_reduction

        factor = max(0.0, min(factor, self.peak_reduction))

        return BaseBleedState(
            is_active=True,
            drag_reduction_factor=factor,
            elapsed_s=t_burn,
            remaining_s=remaining,
        )

    def drag_multiplier(self, t: float) -> float:
        """Return the drag multiplier at time t.

        Returns
        -------
        float
            1.0 = full drag, 0.70 = 30% reduction.
        """
        state = self.get_state(t)
        return 1.0 - state.drag_reduction_factor
