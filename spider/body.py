"""The spider's body: a rigid body with a head, eight legs of two joints each, and a contact
model that turns the motion of planted feet into motion of the body.

Frame: z up, x forward, y left. Angles in radians, lengths in metres, one moment is ``DT``
seconds. The sixteen joints are commanded only as ``decrease / hold / increase`` of a target
angle at a fixed rate; there is no controller between those commands and the joints.

A foot is planted when it is on the ground, which the knee reaches at and below its zero (a
pressing leg touches the ground where its tibia crosses it). Propulsion is the negated rigid
two-dimensional fit (translation and yaw) of every foot that was planted both at the previous
moment and at this one, needing at least two such feet. A foot that is lifted pushes nothing;
a planted foot that does not move anchors the body; a closed joint cycle with the same planted
set integrates to zero displacement. Moving forward therefore needs lifting, swinging and
re-planting legs in a pattern: a gait, which is the brain's to find.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DT = 0.05                               # seconds per moment (20 Hz)
LEGS = 8
JOINTS = 2 * LEGS

BODY = (0.15, 0.10, 0.06)               # ellipsoid semi-axes (x, y, z)
HEAD_RADIUS = 0.06
HEAD_X = 0.19                           # head sphere centre, body frame
HEIGHT = 0.10                           # body centre above ground when standing
SAG = 0.03                              # visual drop when fewer than STANCE_MIN feet are planted

HIP_X = (0.12, 0.04, -0.04, -0.12)      # legs 0-3 left front->back, 4-7 right front->back
HIP_Y = 0.10
FEMUR = 0.10                            # hip -> knee
FEMUR_RISE = 0.35                       # the femur's fixed elevation above horizontal
TIBIA = 0.14                            # knee -> foot
TIBIA_REST = math.acos((HEIGHT + FEMUR * math.sin(FEMUR_RISE)) / TIBIA)   # foot on the ground at knee 0

HIP_RANGE = (-0.6, 0.6)                 # swing: 0 straight out, + forward
KNEE_RANGE = (-0.5, 1.0)                # lift: 0 foot on the ground, below 0 the leg presses into it
HIP_RATE = 1.5                          # rad/s per commanded step
KNEE_RATE = 2.5

CONTACT_Z = 0.005                       # a foot this close to the ground is planted
STANCE_MIN = 2                          # fewer planted feet than this push nothing (two fix translation and yaw)
GRIP = 10.0                             # 1/s: how fast the body takes the speed its feet impose
SLIDE = 6.0                             # 1/s: velocity decay with too few feet planted
ARENA_RADIUS = 5.0                      # beyond this the spider is recentred (not a reset)

SIDE = np.array([1.0] * 4 + [-1.0] * 4)             # +1 left, -1 right
HIPS = np.array([(HIP_X[i % 4], SIDE[i] * HIP_Y, 0.0) for i in range(LEGS)])
RATES = np.array([HIP_RATE, KNEE_RATE])
LOW = np.array([HIP_RANGE[0], KNEE_RANGE[0]])
HIGH = np.array([HIP_RANGE[1], KNEE_RANGE[1]])


def kinematics(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Knee and foot positions in the body frame, ``(8, 3)`` each, for ``joints`` ``(8, 2)``
    of hip swing and knee lift. The body frame's origin is the body centre at ``HEIGHT``;
    a foot on the ground has ``z == -HEIGHT``."""
    theta = joints[:, 0]
    beta = TIBIA_REST + joints[:, 1]
    d = np.stack([np.sin(theta), SIDE * np.cos(theta), np.zeros(LEGS)], axis=1)
    up = np.array([0.0, 0.0, 1.0])
    knee = HIPS + FEMUR * (math.cos(FEMUR_RISE) * d + math.sin(FEMUR_RISE) * up)
    foot = knee + TIBIA * (np.sin(beta)[:, None] * d - np.cos(beta)[:, None] * up)
    # A leg pressed below the ground plane touches it where the tibia crosses it: the body's
    # height is fixed, so the leg flexes against the ground instead of lifting the body.
    below = foot[:, 2] < -HEIGHT
    if np.any(below):
        t = (-HEIGHT - knee[below, 2]) / (foot[below, 2] - knee[below, 2])
        foot = foot.copy()
        foot[below] = knee[below] + t[:, None] * (foot[below] - knee[below])
    return knee, foot


def rotation(heading: float) -> np.ndarray:
    c, s = math.cos(heading), math.sin(heading)
    return np.array([[c, -s], [s, c]])


@dataclass
class Spider:
    """The body's state. ``joints`` is ``(8, 2)``: per leg, hip swing then knee lift."""

    joints: np.ndarray = field(default_factory=lambda: np.zeros((LEGS, 2)))
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    vx: float = 0.0                     # body-frame forward speed, m/s
    vy: float = 0.0                     # body-frame left speed, m/s
    omega: float = 0.0                  # yaw rate, rad/s
    feet_before: np.ndarray | None = None
    planted_before: np.ndarray | None = None
    moments: int = 0
    recentres: int = 0

    def __post_init__(self) -> None:
        if self.feet_before is None or self.planted_before is None:
            _, foot = kinematics(self.joints)
            self.feet_before = foot
            self.planted_before = HEIGHT + foot[:, 2] <= CONTACT_Z

    # --- readings -------------------------------------------------------------------------

    def feet(self) -> tuple[np.ndarray, np.ndarray]:
        return kinematics(self.joints)

    def planted(self) -> np.ndarray:
        _, foot = kinematics(self.joints)
        return HEIGHT + foot[:, 2] <= CONTACT_Z

    @property
    def stance_count(self) -> int:
        return int(self.planted().sum())

    @property
    def z(self) -> float:
        """The body centre's drawn height: it sags when too few feet hold it up."""
        return HEIGHT - (SAG if self.stance_count < STANCE_MIN else 0.0)

    # --- one moment -----------------------------------------------------------------------

    def step(self, commands) -> dict:
        """Apply sixteen commands in ``{0: decrease, 1: hold, 2: increase}`` (joint order
        hip0, knee0, hip1, knee1, ...), move the body by what its planted feet did, and
        return ``progress`` (metres along the heading held at the start of the moment),
        ``turn`` (radians the heading changed this moment), ``stance`` (feet planted at both
        moments) and ``recentred``."""
        cmd = np.asarray(commands, dtype=float).reshape(LEGS, 2) - 1.0
        if not np.all(np.isin(cmd, (-1.0, 0.0, 1.0))):
            raise ValueError("commands must be 0, 1 or 2 per joint")
        self.joints = np.clip(self.joints + cmd * RATES * DT, LOW, HIGH)

        _, foot = kinematics(self.joints)
        planted = HEIGHT + foot[:, 2] <= CONTACT_Z
        stance = planted & self.planted_before
        count = int(stance.sum())
        if count >= STANCE_MIN:
            now = foot[stance, :2]
            before = self.feet_before[stance, :2]
            delta = now - before
            translation = -delta.mean(axis=0)
            r = before - before.mean(axis=0)
            dc = delta - delta.mean(axis=0)
            cross = float(np.sum(r[:, 0] * dc[:, 1] - r[:, 1] * dc[:, 0]))
            dot = float(np.sum(r * r)) + float(np.sum(r * dc))
            yaw = -math.atan2(cross, dot) if count > 1 else 0.0
            v_push = translation / DT
            w_push = yaw / DT
            gain = min(1.0, GRIP * DT)
            self.vx += (v_push[0] - self.vx) * gain
            self.vy += (v_push[1] - self.vy) * gain
            self.omega += (w_push - self.omega) * gain
        else:
            decay = max(0.0, 1.0 - SLIDE * DT)
            self.vx *= decay
            self.vy *= decay
            self.omega *= decay

        heading = self.heading
        progress = self.vx * DT
        turn = self.omega * DT
        world = rotation(heading) @ np.array([self.vx, self.vy]) * DT
        self.x += float(world[0])
        self.y += float(world[1])
        self.heading = math.atan2(math.sin(heading + self.omega * DT), math.cos(heading + self.omega * DT))

        self.feet_before = foot
        self.planted_before = planted
        self.moments += 1

        recentred = False
        if math.hypot(self.x, self.y) > ARENA_RADIUS:
            self.recentre()
            recentred = True
        return {"progress": float(progress), "turn": float(turn), "stance": count, "recentred": recentred}

    def recentre(self) -> None:
        """Put the body back at the origin. Joints, heading, velocity and the contact history
        are kept: this is a change of place, not a new life."""
        self.x = 0.0
        self.y = 0.0
        self.recentres += 1

    # --- continuation ---------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "joints": self.joints.tolist(),
            "x": self.x, "y": self.y, "heading": self.heading,
            "vx": self.vx, "vy": self.vy, "omega": self.omega,
            "feet_before": self.feet_before.tolist(),
            "planted_before": [bool(p) for p in self.planted_before],
            "moments": self.moments, "recentres": self.recentres,
        }

    @classmethod
    def restore(cls, state: dict) -> "Spider":
        return cls(
            joints=np.array(state["joints"], dtype=float).reshape(LEGS, 2),
            x=float(state["x"]), y=float(state["y"]), heading=float(state["heading"]),
            vx=float(state["vx"]), vy=float(state["vy"]), omega=float(state["omega"]),
            feet_before=np.array(state["feet_before"], dtype=float).reshape(LEGS, 3),
            planted_before=np.array(state["planted_before"], dtype=bool),
            moments=int(state["moments"]), recentres=int(state.get("recentres", 0)),
        )

    def pose(self) -> dict:
        """What a drawing needs: body position and heading, the body-frame knee and foot
        points of every leg, the planted flags and the sixteen joint angles."""
        knee, foot = kinematics(self.joints)
        planted = HEIGHT + foot[:, 2] <= CONTACT_Z
        return {
            "x": self.x, "y": self.y, "z": self.z, "heading": self.heading,
            "legs": np.concatenate([knee, foot], axis=1).round(4).tolist(),
            "planted": [bool(p) for p in planted],
            "joints": self.joints.reshape(-1).round(4).tolist(),
        }
