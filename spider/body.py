"""The spider's body: a rigid body with a head, eight legs of two joints each, and a contact
model that turns the motion of planted feet into motion of the body.

Frame: z up, x forward, y left. Angles in radians, lengths in metres, one moment is ``DT``
seconds. The sixteen joints are commanded only as ``decrease / hold / increase``; there is no
controller between those commands and the joints. The body has two muscle models, chosen at its
birth and kept for its life (``muscles``): under ``rate``, the founder until 2026-11-12, a command moves the
joint by its rate for one moment and ``hold`` keeps it where it is; under ``target`` each joint
is a position servo with a setpoint, a command sets the setpoint to that end of the joint's
range, ``hold`` keeps the setpoint, and the joint moves toward it at the same rate. The servo
is the joint's own actuator and knows nothing of gaits, but it is a different model: one
command then drives a whole swing where the founder needs the command sixteen times (README,
"The position commands").

A foot is planted when it is on the ground, which the knee reaches at and below its zero (a
pressing leg touches the ground where its tibia crosses it). Propulsion is the negated rigid
two-dimensional fit (translation and yaw) of every foot that was planted both at the previous
moment and at this one, when those feet support the body: at least one on each side and three
in all (``supported``). Standing on one side's feet alone, the body is down and nothing pushes
it. Neighbouring legs of a side cannot cross: a hip that would turn its leg into the one in
front or behind stops against it (``separate``). A foot that is lifted pushes nothing;
a planted foot that does not move anchors the body; a closed joint cycle with the same planted
set integrates to zero displacement. Moving forward therefore needs lifting, swinging and
re-planting legs in a pattern: a gait, which is the brain's to find.

The world can carry a current (``drift``, metres a second, 0 by default): every moment the
body loses that much ground backward along its own heading, planted or not, on top of what
its feet did. It is the world's, a problem and not a hint: the reward is still the body's
motion, which now includes the ground lost, so standing still and swaying go backward and only
a walk faster than the current nets ground. The nursery's and the server's ``--drift`` set it,
the page switches it live, and every reading made under it names it (README, "The drift").
The muscles run at a rate (``muscle_rate``, a factor on both joint rates, 1 by default): how far
a joint runs toward its command in one moment. It is the body's, not a controller: the brain
still chooses every moment, and the body still moves its joint; a body at another rate is
another body, so the rate is set, shown, saved and counted with the law like the drift
(``--muscle-rate``, the page's law panel) and named in every reading made under it.

The ground is one of three arenas (``arena``, ``flat`` by default). On ``flat`` it is the plane
z = 0, as it always was. On ``hills`` it rises and falls by a few centimetres, a fixed sum of
long waves (``HILLS``) over the world's x and y; ``steep`` is the same waves half as high again
(``STEEP``), about the most the body's model holds without tilting: the body stands at its height above the
ground under its lowest foot, a foot is planted where it reaches the ground beneath it, so a
foot on higher ground is planted with its knee partly lifted and must lift further to swing
free, and one over lower ground is not. The propulsion is the same fit of the planted feet's
motion in the plane; the hills tilt nothing and pull nothing downhill. It is the world's, a
problem and not a hint, and it is set, shown, saved and counted with the law like the drift
(``--arena``, the page's world panel) and named in every reading made under it.

On the terrains the world can pull downhill (``slope_pull``, metres a second at a slope of 1,
0 by default): every moment the body creeps down the ground's steepest slope under its centre
by that factor times the slope, planted or not, on top of what its feet did, the way the drift
takes its ground. The body has no mass and its muscles no energy, so the pull is a creep and
not a force: a walk uphill nets less, a walk downhill more, a body standing on a slope slides
toward the valley and stops where the ground is level. A factor on uphill motion alone would
pay a sway along the slope; a creep pays a sway nothing. It is the world's, a problem and not
a hint, set, shown, saved and counted with the law like the drift (``--slope-pull``, the
page's slope button) and named in every reading made under it; on flat ground it pulls nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DT = 0.05                               # seconds per moment (20 Hz)
LEGS = 8
JOINTS = 2 * LEGS
PER_SIDE = LEGS // 2

BODY = (0.15, 0.10, 0.06)               # ellipsoid semi-axes (x, y, z)
HEAD_RADIUS = 0.06
HEAD_X = 0.19                           # head sphere centre, body frame
HEIGHT = 0.10                           # body centre above ground when standing
SAG = 0.03                              # visual drop when the planted feet do not support the body

HIP_X = (0.12, 0.04, -0.04, -0.12)
#                                       # left legs front->back, then right legs front->back
HIP_Y = 0.10
HIP_SPLAY = (0.3, 0.1, -0.1, -0.3)
#                                       # each leg's resting yaw: front legs point forward, rear legs back,
#                                       # as a tarantula stands (all legs straight out until that body)
FEMUR = 0.16                            # hip -> knee (0.10 until the tarantula body; the legs were stubby)
FEMUR_RISE = 0.35                       # the femur's fixed elevation above horizontal
TIBIA = 0.20                            # knee -> foot (0.14 before)
TIBIA_REST = math.acos((HEIGHT + FEMUR * math.sin(FEMUR_RISE)) / TIBIA)   # foot on the ground at knee 0

HIP_RANGE = (-0.6, 0.6)                 # swing about the leg's resting yaw, + forward
KNEE_RANGE = (-0.5, 1.0)                # lift: 0 foot on the ground, below 0 the leg presses on it (foot held)
LEG_CONVERGE = 0.2                      # rad: how far past parallel two neighbouring legs of a side may turn
                                        # toward each other; at 0.2 their lines meet 0.40 m out, beyond the
                                        # 0.35 m a foot reaches, so legs stop against each other, never cross
                                        # (2026-11-12; before it they turned through each other)
HIP_RATE = 1.5                          # rad/s per commanded step
KNEE_RATE = 2.5

CONTACT_Z = 0.005                       # a foot this close to the ground is planted
SUPPORT_SIDE = 1                        # planted feet a side needs for the body to stand ...
SUPPORT_MIN = 3                         # ... and in all: one side's feet alone cannot hold it up (2026-11-12; two
                                        # feet anywhere until then, "The support rule" in the README)
GRIP = 10.0                             # 1/s: how fast the body takes the speed its feet impose
SLIDE = 6.0                             # 1/s: velocity decay while the body is down
ARENA_RADIUS = 5.0                      # beyond this the spider is recentred (not a reset)
# The hills: (wavelength m, direction rad, amplitude m, phase rad) of each long wave over the
# world's x, y; their sum is the ground's height. Fixed, the same world for every life: up to
# about 6 cm either way and slopes of a few in a hundred, at most about one in five where the
# waves agree, which the legs' lift (some 18 cm) can always clear (2026-11-14).
HILLS = ((2.4, 0.35, 0.022, 0.3), (1.7, 2.0, 0.015, 1.1), (3.1, 3.6, 0.016, 2.0), (1.3, 1.13, 0.009, 0.7))
# The steep hills: the same waves at 1.5 times the amplitude, about 9 cm either way. Under the
# eight feet the ground then differs by some 6 cm in the median and up to 13 cm, against 4 and 9
# on the hills; twice the amplitude would leave a foot over the highest ground unable to lift
# clear of it, and a body that does not tilt half sunk in the slope (2026-11-14).
STEEP = tuple((w, d, round(1.5 * a, 4), ph) for w, d, a, ph in HILLS)
TERRAINS = {"hills": HILLS, "steep": STEEP}
ARENAS = ("flat",) + tuple(TERRAINS)              # the ground: the plane z = 0 (the founder), or one of the terrains
_WAVES = {name: np.array([(2 * math.pi / w * math.cos(d), 2 * math.pi / w * math.sin(d), a, ph) for w, d, a, ph in waves])
          for name, waves in TERRAINS.items()}


def ground_height(arena: str, x, y) -> np.ndarray:
    """The ground's height in metres at world points ``x``, ``y`` (arrays of one shape): 0 on
    ``flat``, the sum of the arena's waves on ``hills`` and ``steep``."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if arena == "flat":
        return np.zeros(np.broadcast(x, y).shape)
    h = np.zeros(np.broadcast(x, y).shape)
    for kx, ky, a, ph in _WAVES[arena]:
        h = h + a * np.cos(kx * x + ky * y + ph)
    return h


def ground_slope(arena: str, x: float, y: float) -> tuple[float, float]:
    """The ground's gradient at a world point, (dh/dx, dh/dy): zero on ``flat``."""
    if arena == "flat":
        return 0.0, 0.0
    gx = gy = 0.0
    for kx, ky, a, ph in _WAVES[arena]:
        s = -a * math.sin(kx * x + ky * y + ph)
        gx += s * kx
        gy += s * ky
    return gx, gy

SIDE = np.array([1.0] * PER_SIDE + [-1.0] * PER_SIDE)   # +1 left, -1 right


def separate(joints: np.ndarray) -> np.ndarray:
    """The hips with neighbouring legs of a side kept from crossing: wherever the leg behind
    has turned more than ``LEG_CONVERGE`` further forward than the leg in front, both are
    pushed back to that stop, halfway each, as two legs pressing on each other would be. The
    split keeps both hips in range: two legs that converge past the stop have a mean swing
    inside ±0.4, and the split puts them 0.2 either side of it. A few passes settle a side."""
    joints = joints.copy()
    for _ in range(8):
        moved = False
        for side in range(0, LEGS, PER_SIDE):
            for i in range(side, side + PER_SIDE - 1):
                front = SPLAY[i] + joints[i, 0]
                behind = SPLAY[i + 1] + joints[i + 1, 0]
                over = behind - front - LEG_CONVERGE
                if over > 1e-12:
                    joints[i, 0] += over / 2
                    joints[i + 1, 0] -= over / 2
                    moved = True
        if not moved:
            break
    return joints


def supported(planted) -> bool:
    """Whether these planted feet hold the body up: at least ``SUPPORT_SIDE`` on each side
    and ``SUPPORT_MIN`` in all. A tripod needs both sides; a body on one side's feet tips."""
    planted = np.asarray(planted, dtype=bool)
    left, right = int(planted[:PER_SIDE].sum()), int(planted[PER_SIDE:].sum())
    return min(left, right) >= SUPPORT_SIDE and left + right >= SUPPORT_MIN
HIPS = np.array([(HIP_X[i % PER_SIDE], SIDE[i] * HIP_Y, 0.0) for i in range(LEGS)])
SPLAY = np.array([HIP_SPLAY[i % PER_SIDE] for i in range(LEGS)])
RATES = np.array([HIP_RATE, KNEE_RATE])
MUSCLES = ("rate", "target")             # the two muscle models; rate was the founder
DEFAULT_MUSCLES = "target"               # the model a newborn body gets when nothing names one (rate until
                                        # 2026-11-12; the README's "The position commands" measured the change)
LOW = np.array([HIP_RANGE[0], KNEE_RANGE[0]])
HIGH = np.array([HIP_RANGE[1], KNEE_RANGE[1]])


def kinematics(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Knee and foot positions in the body frame, ``(8, 3)`` each, for ``joints`` ``(8, 2)``
    of hip swing and knee lift. The body frame's origin is the body centre at ``HEIGHT``;
    a foot on the ground has ``z == -HEIGHT``."""
    theta = SPLAY + joints[:, 0]
    # A knee below zero presses the planted foot against the ground: the body's height is
    # fixed, so the press lifts nothing, and the foot stays where a resting knee puts it. Until
    # 2026-11-12 the pressed leg touched the ground where the tibia crossed the plane, a point
    # that slid in under the knee as the press deepened; the learner rocked on that slide for
    # a forward-only income without moving (README, "The rocking").
    beta = TIBIA_REST + np.maximum(joints[:, 1], 0.0)
    d = np.stack([np.sin(theta), SIDE * np.cos(theta), np.zeros(LEGS)], axis=1)
    up = np.array([0.0, 0.0, 1.0])
    knee = HIPS + FEMUR * (math.cos(FEMUR_RISE) * d + math.sin(FEMUR_RISE) * up)
    foot = knee + TIBIA * (np.sin(beta)[:, None] * d - np.cos(beta)[:, None] * up)
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
    muscles: str = DEFAULT_MUSCLES        # the muscle model, see the module docstring
    drift: float = 0.0                  # the world's current, m/s backward along the heading, see the docstring
    muscle_rate: float = 1.0             # a factor on both joint rates: how far a joint runs in one moment
    targets: np.ndarray | None = None   # the servos' setpoints under target muscles; unused under rate
    arena: str = "flat"                 # the ground, see the module docstring
    slope_pull: float = 0.0             # m/s downhill at a slope of 1, see the module docstring

    def __post_init__(self) -> None:
        if self.muscles not in MUSCLES:
            raise ValueError(f"muscles must be one of {MUSCLES}, not {self.muscles!r}")
        if not math.isfinite(self.drift) or self.drift < 0:
            raise ValueError(f"the drift is a speed backward, 0 or more, not {self.drift}")
        if not math.isfinite(self.muscle_rate) or self.muscle_rate <= 0:
            raise ValueError(f"the muscle rate is a positive factor on the joint rates, not {self.muscle_rate}")
        if self.arena not in ARENAS:
            raise ValueError(f"the arena is one of {ARENAS}, not {self.arena!r}")
        if not math.isfinite(self.slope_pull) or self.slope_pull < 0:
            raise ValueError(f"the slope pull is a speed downhill at a slope of 1, 0 or more, not {self.slope_pull}")
        if self.targets is None:
            self.targets = self.joints.copy()
        if self.feet_before is None or self.planted_before is None:
            _, foot = kinematics(self.joints)
            self.feet_before = foot
            self.planted_before = self.contact(foot)[0]

    # --- readings -------------------------------------------------------------------------

    def feet(self) -> tuple[np.ndarray, np.ndarray]:
        return kinematics(self.joints)

    def contact(self, foot: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """For body-frame feet ``(8, 3)`` at the body's place: which are planted, the height of
        the ground the body stands on (under its lowest foot), and the ground's height under each
        foot. On flat ground that is the founder's test, ``HEIGHT + z <= CONTACT_Z``, to the bit."""
        if self.arena == "flat":
            return HEIGHT + foot[:, 2] <= CONTACT_Z, 0.0, np.zeros(LEGS)
        c, s = math.cos(self.heading), math.sin(self.heading)
        under = ground_height(self.arena, self.x + c * foot[:, 0] - s * foot[:, 1],
                              self.y + s * foot[:, 0] + c * foot[:, 1])
        base = float(under.min())
        return base + HEIGHT + foot[:, 2] - under <= CONTACT_Z, base, under

    def planted(self) -> np.ndarray:
        _, foot = kinematics(self.joints)
        return self.contact(foot)[0]

    @property
    def stance_count(self) -> int:
        return int(self.planted().sum())

    @property
    def z(self) -> float:
        """The body centre's drawn height: above the ground under its lowest foot, and it sags
        when the planted feet do not hold it up."""
        _, foot = kinematics(self.joints)
        planted, base, _ = self.contact(foot)
        return base + HEIGHT - (SAG if not supported(planted) else 0.0)

    # --- one moment -----------------------------------------------------------------------

    def step(self, commands) -> dict:
        """Apply sixteen commands in ``{0: decrease, 1: hold, 2: increase}`` (joint order
        hip0, knee0, hip1, knee1, ...), move the body by what its planted feet did, and
        return ``progress`` (metres along the heading held at the start of the moment, the
        ground lost to the drift taken off),
        ``moved`` (the same motion as a world displacement, x and y, which a recentre does not
        touch), ``turn`` (radians the heading changed this moment), ``stance`` (feet planted
        at both moments) and ``recentred``."""
        cmd = np.asarray(commands, dtype=float).reshape(LEGS, 2) - 1.0
        if not np.all(np.isin(cmd, (-1.0, 0.0, 1.0))):
            raise ValueError("commands must be 0, 1 or 2 per joint")
        if self.muscles == "target":
            # a command sets the servo's setpoint to that end of the range, hold keeps it;
            # the joint moves toward the setpoint by at most its rate
            self.targets = np.where(cmd < 0, LOW, np.where(cmd > 0, HIGH, self.targets))
            reach = RATES * self.muscle_rate * DT                # how far a joint runs in this moment
            move = np.clip(self.targets - self.joints, -reach, reach)
            self.joints = separate(np.clip(self.joints + move, LOW, HIGH))
        else:
            self.joints = separate(np.clip(self.joints + cmd * RATES * self.muscle_rate * DT, LOW, HIGH))

        _, foot = kinematics(self.joints)
        planted = self.contact(foot)[0]
        stance = planted & self.planted_before
        count = int(stance.sum())
        if supported(stance):
            now = foot[stance, :2]
            before = self.feet_before[stance, :2]
            delta = now - before
            translation = -delta.mean(axis=0)
            r = before - before.mean(axis=0)
            dc = delta - delta.mean(axis=0)
            cross = float(np.sum(r[:, 0] * dc[:, 1] - r[:, 1] * dc[:, 0]))
            dot = float(np.sum(r * r)) + float(np.sum(r * dc))
            yaw = -math.atan2(cross, dot)
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
        progress = (self.vx - self.drift) * DT     # the current takes its ground whatever the feet did
        turn = self.omega * DT
        world = rotation(heading) @ np.array([self.vx - self.drift, self.vy]) * DT
        creep = self.creep()
        if creep is not None:                      # the slope takes or gives its ground whatever the feet did
            world = world + creep * DT
            progress += float(creep[0] * math.cos(heading) + creep[1] * math.sin(heading)) * DT
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
        return {"progress": float(progress), "moved": (float(world[0]), float(world[1])), "turn": float(turn),
                "stance": count, "recentred": recentred}

    def ground_velocity(self) -> tuple[float, float]:
        """The body's velocity over the ground in the world's frame, x and y in m/s: the
        body-frame velocity turned by the heading, the drift taken off. What a step adds to
        ``moved`` is this times DT; a recentre does not touch it."""
        world = rotation(self.heading) @ np.array([self.vx - self.drift, self.vy])
        creep = self.creep()
        if creep is not None:
            world = world + creep
        return float(world[0]), float(world[1])

    def creep(self) -> np.ndarray | None:
        """The slope's pull as a world velocity, x and y in m/s, down the ground's gradient at the
        body's centre; None where nothing pulls (no pull, or flat ground)."""
        if not self.slope_pull or self.arena == "flat":
            return None
        gx, gy = ground_slope(self.arena, self.x, self.y)
        return -self.slope_pull * np.array([gx, gy])

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
            "muscles": self.muscles, "targets": self.targets.tolist(),
            "drift": self.drift, "muscle_rate": self.muscle_rate, "arena": self.arena,
            "slope_pull": self.slope_pull,
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
            # a world saved before the model existed is a rate body; one saved before 2026-11-14 names it "motors"
            muscles=str(state.get("muscles", state.get("motors", "rate"))),
            targets=None if state.get("targets") is None else np.array(state["targets"], dtype=float).reshape(LEGS, 2),
            drift=float(state.get("drift", 0.0)),   # a world saved before the current existed had none
            # one saved before the rate existed ran at 1; before 2026-11-14 it was the "motor_rate"
            muscle_rate=float(state.get("muscle_rate", state.get("motor_rate", 1.0))),
            arena=str(state.get("arena", "flat")),   # a world saved before the hills was flat
            slope_pull=float(state.get("slope_pull", 0.0)),   # and one saved before the pull had none
        )

    def pose(self) -> dict:
        """What a drawing needs: body position and heading, the body-frame knee and foot
        points of every leg, the planted flags and the sixteen joint angles; on the hills the
        ground's height under each foot too (``ground``), and a planted foot drawn on the ground
        it stands on rather than in it."""
        knee, foot = kinematics(self.joints)
        planted, base, under = self.contact(foot)
        z = base + HEIGHT - (SAG if not supported(planted) else 0.0)
        if self.arena != "flat":
            foot = foot.copy()
            foot[:, 2] = np.maximum(foot[:, 2], under - base - HEIGHT)
        return {
            "x": self.x, "y": self.y, "z": z, "heading": self.heading,
            "ground": [round(float(h), 4) for h in under],
            "legs": np.concatenate([knee, foot], axis=1).round(4).tolist(),
            "planted": [bool(p) for p in planted],
            "joints": self.joints.reshape(-1).round(4).tolist(),
        }
