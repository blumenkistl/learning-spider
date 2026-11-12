"""What the brain receives each moment: its own joint angles, forward speed and yaw rate through
a population code (tuned inputs, quiet away from their value), and one raw flag per foot that
says whether it touches the ground. No constant drive input, no target, no answer.

Under position servos (``Spider(motors="target")``) each joint also has a held setpoint, state
of the body as the angle is; ``Senses(setpoints=True)`` reads the sixteen setpoints too, through
the same code, so that the brain sees whether a joint is on its way or at rest. It is an option
named in every reading made under it (README, "The setpoint senses").

``Senses(compass=True)`` adds a compass: the cosine and sine of the body's heading in the
world, through the same code. Without it the brain senses its yaw rate and nothing of where it
is pointed, which is all the plain law asks of it; the course law (README, "The course") pays
progress along a fixed world heading, and the compass is the sense that heading can be read
from. It is an option named in every reading made under it (README, "The compass").
"""

from __future__ import annotations

import numpy as np
from cadence import PopulationCode

from .body import HIP_RANGE, KNEE_RANGE, LEGS, Spider

SPEED_RANGE = (-0.3, 0.3)       # forward speed, m/s
YAW_RANGE = (-2.0, 2.0)         # yaw rate, rad/s
PER = 8                         # tuned inputs per reading (a candidate gene; the chamber used 16)

READING_NAMES = (
    [f"hip{i}" for i in range(LEGS)] + [f"knee{i}" for i in range(LEGS)] + ["v_fwd", "omega"]
)
RANGES = [HIP_RANGE] * LEGS + [KNEE_RANGE] * LEGS + [SPEED_RANGE, YAW_RANGE]
SETPOINT_NAMES = [f"hip{i}_set" for i in range(LEGS)] + [f"knee{i}_set" for i in range(LEGS)]
SETPOINT_RANGES = [HIP_RANGE] * LEGS + [KNEE_RANGE] * LEGS
COMPASS_NAMES = ["heading_cos", "heading_sin"]     # the heading as a point on the circle: no seam at ±pi
COMPASS_RANGES = [(-1.0, 1.0), (-1.0, 1.0)]


class Senses:
    """A fixed population code over the readings plus the planted flags; with ``setpoints``
    the servos' setpoints are readings too (only a body with target motors has them), with
    ``compass`` the heading's cosine and sine."""

    def __init__(self, per: int = PER, setpoints: bool = False, compass: bool = False):
        self.setpoints = bool(setpoints)
        self.compass = bool(compass)
        self.names = list(READING_NAMES) + (SETPOINT_NAMES if self.setpoints else []) + (COMPASS_NAMES if self.compass else [])
        self.ranges = list(RANGES) + (SETPOINT_RANGES if self.setpoints else []) + (COMPASS_RANGES if self.compass else [])
        self.code = PopulationCode(self.ranges, per=per)
        self.per = per
        self.inputs = self.code.inputs + LEGS

    def readings(self, spider: Spider) -> np.ndarray:
        own = [spider.joints[:, 0], spider.joints[:, 1], [spider.vx, spider.omega]]
        if self.setpoints:
            if spider.motors != "target":
                raise ValueError("the setpoint senses need a body with target motors")
            own += [spider.targets[:, 0], spider.targets[:, 1]]
        if self.compass:
            own += [[np.cos(spider.heading), np.sin(spider.heading)]]
        return np.concatenate(own)

    def observe(self, spider: Spider) -> np.ndarray:
        coded = self.code(self.readings(spider))                  # (1, per * 18; 34 with setpoints, 2 more with the compass)
        flags = spider.planted().astype(np.float64)[None, :]      # (1, 8)
        return np.concatenate([coded, flags], axis=1)

    def input_names(self) -> list[str]:
        names = []
        for name, (lo, hi) in zip(self.names, self.ranges):
            for value in np.linspace(lo, hi, self.per):
                names.append(f"{name}@{value:+.2f}")
        names += [f"planted{i}" for i in range(LEGS)]
        return names

    def options(self) -> dict:
        """What a sidecar or a birth names of the senses, beside ``per``."""
        return {"per": self.per, "setpoints": self.setpoints, "compass": self.compass}


DEFAULT = Senses()
INPUTS = DEFAULT.inputs


def observe(spider: Spider, senses: Senses = DEFAULT) -> np.ndarray:
    """One row of ``senses.inputs`` values in ``[0, 1]``."""
    return senses.observe(spider)


def input_names(senses: Senses = DEFAULT) -> list[str]:
    return senses.input_names()
