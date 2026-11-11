"""What the brain receives each moment: its own joint angles, forward speed and yaw rate through
a population code (tuned inputs, quiet away from their value), and one raw flag per foot that
says whether it touches the ground. No constant drive input, no target, no answer.
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


class Senses:
    """A fixed population code over the readings plus the planted flags."""

    def __init__(self, per: int = PER):
        self.code = PopulationCode(RANGES, per=per)
        self.per = per
        self.inputs = self.code.inputs + LEGS

    def readings(self, spider: Spider) -> np.ndarray:
        return np.concatenate([spider.joints[:, 0], spider.joints[:, 1], [spider.vx, spider.omega]])

    def observe(self, spider: Spider) -> np.ndarray:
        coded = self.code(self.readings(spider))                  # (1, per * 18)
        flags = spider.planted().astype(np.float64)[None, :]      # (1, 8)
        return np.concatenate([coded, flags], axis=1)

    def input_names(self) -> list[str]:
        names = []
        for name, (lo, hi) in zip(READING_NAMES, RANGES):
            for value in np.linspace(lo, hi, self.per):
                names.append(f"{name}@{value:+.2f}")
        names += [f"planted{i}" for i in range(LEGS)]
        return names


DEFAULT = Senses()
INPUTS = DEFAULT.inputs


def observe(spider: Spider, senses: Senses = DEFAULT) -> np.ndarray:
    """One row of ``senses.inputs`` values in ``[0, 1]``."""
    return senses.observe(spider)


def input_names(senses: Senses = DEFAULT) -> list[str]:
    return senses.input_names()
