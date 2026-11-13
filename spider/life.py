"""One continuing life: the body in its world, the policy that commands it, and the custody of
every outcome.

A tick observes the body, hands the previous command's reward and this moment's observation to
the policy, applies the returned commands to the body, and keeps the reward the body earned for
the next tick. The reward is the world's: progress along the body's own heading (or, under the
course law, along one fixed heading of the world), less two hints about how to go (feet short
of four, radians turned).
Each outcome reaches the brain exactly once; a refusal keeps it owed, a policy switch keeps it
owed, a save carries it, a load delivers it.

``Life`` runs its own daemon thread. One thread owns the policy and the body; readers take the
``latest`` snapshot, which is replaced wholesale.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .body import DEFAULT_MOTORS, DT, HEIGHT, HIPS, JOINTS, LEGS, MOTORS, Spider
from .brain import (DEFAULT_LAYOUT, LAYOUTS, FrozenPolicy, RandomPolicy, SpiderBrain, backend_of, fitted, gene_words,
                    genes_from_text, make_policy)
from .senses import DEFAULT as DEFAULT_SENSES, Senses

PROGRESS_PAY = 40.0     # reward per metre forward: 0.4 a moment at the full stroke speed of 0.2 m/s;
                        # one planted stroke moment all but reaches the cap, so four pushing feet pay
                        # as one does; --pay-per-metre / PAY_PER_METRE set it (README, "The pay per metre")
PROGRESS_CAP = 0.5      # the Robot Arena's cap, under the dopamine cap of 1.0
BACKWARD_PAY = 0.0      # the share of the pay per metre charged for a metre backward: 0 leaves backward
                        # free (the founder, README "The reward"), 1 charges it as forward pays;
                        # --backward-pay / BACKWARD_PAY set it (README, "The backward pay")
DRIFT = 0.0             # the world's current, metres a second the body loses backward along its heading
                        # every moment, planted or not: the world's, not a term of the pay; 0 is still
                        # ground (the founder); --drift / DRIFT set it (README, "The drift")
STANCE_FLOOR = LEGS // 2   # feet the world wants on the ground through every moment: half of them
STANCE_PAY = 0.0        # taken per foot short of the floor: a hint, shaping, not an outcome; off by default,
                        # 0.1 is the measured dose (README, "The stance term"); --stance-pay / STANCE_PAY set it
PROGRESS_WINDOW = 1     # moments the progress term is averaged over: 1 pays each moment's own progress;
                        # a window pays the mean over the last N, which a sway that nets nothing cannot earn
                        # (README, "The rocking"); --window / PROGRESS_WINDOW set it
HIGH_WATER = False      # pay only new ground: progress beyond the farthest point the body has reached along
                        # its heading (its high-water mark). A sway earns on its forward half under the plain
                        # law and nothing under this one; the mark is the life's, saved and loaded with it
                        # (README, "The high-water mark"); --high-water / HIGH_WATER switch it on
OLD_GROUND = 0.0        # under the mark, the share of a moment's progress on old ground that is paid: 0 is the
                        # mark as above, 1 the plain law; between, the mark leaks (README, "The leaking mark");
                        # --old-ground / OLD_GROUND set it
YAW_PAY = 0.0           # taken per radian turned in the moment: the hint to go straight, off (README: ignored at
                        # 0.1 and 1 a radian, obeyed at 5 by standing still); --yaw-pay / YAW_PAY switch it on
COURSE = None           # the course law: pay progress along this fixed heading of the world (radians, 0 is the
                        # heading at birth) instead of along the body's own, so that a circle nets nothing and a
                        # line pays; None is off, the plain law. Not a hint: the same outcome, measured along one
                        # heading instead of a turning one. The brain reads the heading through the compass
                        # (Senses(compass=True)). --course / COURSE set it (README, "The course")
STRICT = False          # the strict course: under a course, the progress term is the lesser of the moment's
                        # motion along the course and along the body's own heading, so that only nose first
                        # on the course pays in full: backward along the course is backward along the nose
                        # (charged as --backward-pay says), nose first off the course pays little. Not a hint:
                        # the same outcome measured twice, the lesser paid. --strict / STRICT set it; it needs
                        # a course (README, "The strict course")
BLOCK = 100             # moments per statistics block
BLOCKS_KEPT = 60
FORMAT = "learning-spider/1"
try:                    # the library the running brain is composed with; written into the sidecar, and a
    from importlib.metadata import version as _version    # load of a life saved under another is noted
    CADENCE_VERSION = _version("cadence-net")
except Exception:       # pragma: no cover - an install that records no metadata
    CADENCE_VERSION = "unknown"


def pay(progress: float, stance: int = STANCE_FLOOR, stance_pay: float = STANCE_PAY,
        turn: float = 0.0, yaw_pay: float = YAW_PAY, per_metre: float = PROGRESS_PAY,
        backward_pay: float = BACKWARD_PAY) -> float:
    """Forward progress pays; backward and standing still pay zero (``backward_pay`` charges
    backward progress that share of the forward pay, under the same cap); every foot short of
    ``STANCE_FLOOR`` planted through the moment takes ``stance_pay``, and every radian turned
    in the moment takes ``yaw_pay``. The progress term is the outcome. The stance and yaw terms
    are hints the world gives about posture and course (reward shaping): the stance term was
    chosen after the learner under progress alone walked on ever fewer feet, the yaw term
    because progress along the body's own heading pays a circle as well as a line (README,
    "Measure it"). The first law paid backward progress negatively, and over 30,000 moments the
    learning brain lifted every foot to avoid it; forward-only pay was chosen by that reading."""
    gain = per_metre * (backward_pay if progress < 0 else 1.0)
    progress_term = float(np.clip(gain * progress, -PROGRESS_CAP, PROGRESS_CAP))
    stance_term = stance_pay * max(0, STANCE_FLOOR - int(stance))
    yaw_term = yaw_pay * abs(float(turn))
    return float(np.clip(progress_term - stance_term - yaw_term, -PROGRESS_CAP, PROGRESS_CAP))


def _replace(path: Path, write) -> None:
    """Write through a temporary file in the same folder, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=path.suffix)
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


LAW_KEYS = ("stance_pay", "yaw_pay", "window", "high_water", "old_ground", "pay_per_metre", "backward_pay", "course",
            "drift", "strict")


def checked_course(course: Any) -> float | None:
    """The course law's heading: None (or false) is off, true is the birth heading (0), a number
    is that heading in radians, wrapped to (-pi, pi]."""
    if course is None or course is False:
        return None
    if course is True:
        return 0.0
    if isinstance(course, str):
        word = course.strip().lower()
        if word in ("", "off", "none", "false"):
            return None
        if word in ("on", "true", "birth"):
            return 0.0
    heading = float(course)
    if not math.isfinite(heading):
        raise ValueError(f"a course is a heading in radians, not {course}")
    return float(math.atan2(math.sin(heading), math.cos(heading)))


def checked_law(*, stance_pay: Any = STANCE_PAY, yaw_pay: Any = YAW_PAY, window: Any = PROGRESS_WINDOW,
                high_water: Any = HIGH_WATER, old_ground: Any = OLD_GROUND, pay_per_metre: Any = PROGRESS_PAY,
                backward_pay: Any = BACKWARD_PAY, course: Any = COURSE, drift: Any = DRIFT,
                strict: Any = STRICT) -> dict[str, Any]:
    """The law's ten settings, typed and checked; a ValueError names what is wrong. The
    drift is the world's and not the pay's, but it is set, shown, saved and counted with the
    law, so that a life whose world changed reads as the schedule it is."""
    high_water = bool(high_water)
    if not math.isfinite(float(drift)) or float(drift) < 0:
        raise ValueError(f"the drift is a speed backward in m/s, 0 or more, not {drift}")
    if not 0.0 <= float(old_ground) <= 1.0:
        raise ValueError(f"old ground is a share of the pay, 0 to 1, not {old_ground}")
    if float(old_ground) and not high_water:
        raise ValueError("old ground is paid under the high-water mark only (--high-water)")
    if float(pay_per_metre) <= 0:
        raise ValueError(f"pay per metre must be positive, not {pay_per_metre}")
    if not 0.0 <= float(backward_pay) <= 1.0:
        raise ValueError(f"backward pay is a share of the forward pay, 0 to 1, not {backward_pay}")
    if int(window) < 1:
        raise ValueError(f"window must be at least 1 moment, not {window}")
    strict = bool(strict)
    course = checked_course(course)
    if strict and course is None:
        raise ValueError("the strict course is paid under the course law only (--course)")
    return {"stance_pay": float(stance_pay), "yaw_pay": float(yaw_pay), "window": int(window),
            "high_water": high_water, "old_ground": float(old_ground),
            "pay_per_metre": float(pay_per_metre), "backward_pay": float(backward_pay),
            "course": course, "drift": float(drift), "strict": strict}


def law_text(law: dict[str, Any]) -> str:
    """The law in the page's words: ``progress`` is the plain law."""
    parts = [f"window {law['window']}" if law["window"] > 1 else "progress"]
    if law.get("course") is not None:
        parts.append(f"course {law['course']:.2f} rad" + (", strict" if law.get("strict") else ""))
    if law["pay_per_metre"] != PROGRESS_PAY:
        parts.append(f"{law['pay_per_metre']:g}/m")
    if law["backward_pay"]:
        parts.append(f"backward {law['backward_pay']:g}")
    if law["high_water"]:
        parts.append("high-water" + (f" leaking {law['old_ground']:g}" if law["old_ground"] else ""))
    if law["stance_pay"]:
        parts.append(f"stance {law['stance_pay']:g}")
    if law["yaw_pay"]:
        parts.append(f"yaw {law['yaw_pay']:g}")
    if law.get("drift"):
        parts.append(f"drift {law['drift']:g} m/s")
    return " · ".join(parts)


class Life:
    """The world, the policy and the loop. ``policy`` is ``brain``, ``random`` or ``frozen``."""

    def __init__(
        self,
        policy: str = "brain",
        seed: int = 0,
        pace: float = 20.0,
        runs: str | Path = "runs",
        genes: dict[str, Any] | None = None,
        backend: str | None = None,
        senses: Senses = DEFAULT_SENSES,
        brain_path: str | Path | None = None,
        name: str = "spider",
        stance_pay: float = STANCE_PAY,
        yaw_pay: float = YAW_PAY,
        window: int = PROGRESS_WINDOW,
        high_water: bool = HIGH_WATER,
        pay_per_metre: float = PROGRESS_PAY,
        motors: str = DEFAULT_MOTORS,
        backward_pay: float = BACKWARD_PAY,
        old_ground: float = OLD_GROUND,
        course: float | bool | None = COURSE,
        layout: str = DEFAULT_LAYOUT,
        drift: float = DRIFT,
        strict: bool = STRICT,
    ) -> None:
        self.name = name                        # the file stem under runs/
        self._take_law(checked_law(stance_pay=stance_pay, yaw_pay=yaw_pay, window=window, high_water=high_water,
                                   old_ground=old_ground, pay_per_metre=pay_per_metre, backward_pay=backward_pay,
                                   course=course, drift=drift, strict=strict))
        self.mark = 0.0                         # the farthest paid distance reached: the high-water mark
        self.law_changes: list[dict[str, Any]] = []   # every live change of the law, moment and law (a schedule)
        self.seed = int(seed)
        self.genes = genes
        self.backend = backend
        self.senses = senses
        self.runs = Path(runs)
        self.pace = float(pace)                 # moments per second; 0 = as fast as possible
        self.rate = 0.0                         # moments a second the loop actually makes, smoothed
        if motors not in MOTORS:
            raise ValueError(f"motors must be one of {MOTORS}, not {motors!r}")
        self.motors = str(motors)               # the body's motor model, the life's own; a saved life keeps its own
        if senses.setpoints and self.motors != "target":
            raise ValueError("the setpoint senses need target motors (--motors target)")
        self.body = Spider(motors=self.motors, drift=self.drift)
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")
        self.layout = str(layout)               # how the brain's answer is read, a slot per joint or per leg; the
                                                # brain's own, composed in: a saved brain keeps its layout
        self.policy = make_policy(policy, self.seed, brain_path, genes, backend, inputs=senses.inputs, layout=self.layout)
        self.layout = getattr(self.policy, "layout", self.layout)   # a loaded brain's layout is the life's
        self.reward: float | None = None        # the last executed command's outcome, not yet delivered
        self.moments = 0
        self.distance_forward = 0.0
        self.distance_course = 0.0              # metres along the course (along the birth heading, 0, without one)
        self.turned = 0.0                       # radians turned over the life, unsigned
        # the body's place in the world's frame from the point of birth, as the sum of every
        # moment's motion over the ground: a recentre moves the body, not this (the body's x, y
        # are reset at the arena's edge); its length is the honest answer to "how far did it get"
        self.displacement = [0.0, 0.0]
        self.recentres = 0
        self.switches = 0
        self.blocks: deque[dict[str, Any]] = deque(maxlen=BLOCKS_KEPT)
        self._block: list[dict[str, Any]] = []
        self.lock = threading.RLock()
        self.paused = False
        self.running = False
        self._thread: threading.Thread | None = None
        self.autosave_minutes = float(os.environ.get("AUTOSAVE_MINUTES", "5"))
        self._last_save = time.monotonic()
        self.last_save_at: float | None = None
        self.loaded_from: str | None = None
        self.notes: list[str] = []
        self.latest: dict[str, Any] = self.snapshot(None)

    # --- the law --------------------------------------------------------------------------

    def _take_law(self, law: dict[str, Any]) -> None:
        self.high_water = law["high_water"]     # the progress term pays new ground only
        self.old_ground = law["old_ground"]     # the share old ground is paid under the mark: 0 none, 1 all
        self.stance_pay = law["stance_pay"]     # 0 switches the posture hint off
        self.yaw_pay = law["yaw_pay"]           # 0 switches the course hint off
        self.pay_per_metre = law["pay_per_metre"]   # the progress term's gain; the cap stays
        self.backward_pay = law["backward_pay"]     # 0 leaves backward free
        self.course = law["course"]             # None pays along the body's own heading, a number along this one
        self.drift = law["drift"]               # the world's current, which the body applies each moment
        self.strict = law["strict"]             # under a course: the lesser of along the course and along the nose
        if hasattr(self, "body"):
            self.body.drift = self.drift
        self.window = law["window"]
        kept = list(getattr(self, "_progress", []))[-self.window:]
        self._progress: deque[float] = deque(kept, maxlen=self.window)   # the last window moments' progress

    def law(self) -> dict[str, Any]:
        """The reward law this life is paid under, the nine settings the flags set, and the
        world's drift beside them."""
        return {"stance_pay": self.stance_pay, "yaw_pay": self.yaw_pay, "window": self.window,
                "high_water": self.high_water, "old_ground": self.old_ground,
                "pay_per_metre": self.pay_per_metre, "backward_pay": self.backward_pay, "course": self.course,
                "drift": self.drift, "strict": self.strict}

    def _paid_distance(self) -> float:
        """The distance the progress term is measured on: along the course under the course law,
        else along the body's own heading."""
        return self.distance_course if self.course is not None else self.distance_forward

    def set_law(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Change the law under which this life is paid, from the next moment on. The outcome
        already earned is delivered as it was earned. A life whose law changes is a schedule,
        a different model from a life under one law: every change is noted, counted, shown
        and saved with the life, so that nobody reads it as a life under one law.

        A course of ``true`` set live is the heading the body holds now: keep going this way.
        A course that is switched on or changed starts the distance along it, and the high-water
        mark on it, at zero; switched off, the mark stands where the body stands."""
        changes = dict(changes or {})
        unknown = sorted(set(changes) - set(LAW_KEYS))
        if unknown:
            raise ValueError(f"not a setting of the law: {', '.join(unknown)} (the law is {', '.join(LAW_KEYS)})")
        with self.lock:
            if changes.get("course") is True:                    # keep going this way
                changes["course"] = self.body.heading if self.course is None else self.course
            if "course" in changes and changes["course"] is None:   # the course switched off takes the strict course with it
                changes["strict"] = False
            law = checked_law(**{**self.law(), **changes})
            if law == self.law():
                return {"law": law, "changed": False}
            if law["course"] != self.course:
                self.distance_course = 0.0
                self.mark = 0.0 if law["course"] is not None else self.distance_forward
            self._take_law(law)
            self.law_changes.append({"moment": self.moments, "law": law})
            self.notes.append(f"law changed at moment {self.moments}: {law_text(law)} (change {len(self.law_changes)}: this life is a schedule)")
            self.latest = self.snapshot(None)
        return {"law": law, "changed": True}

    # --- one moment -----------------------------------------------------------------------

    def tick(self) -> dict[str, Any]:
        with self.lock:
            observation = self.senses.observe(self.body)
            reading = self.policy.moment(observation, self.reward)
            self.reward = None
            outcome = self.body.step(reading["commands"])
            self.moments += 1
            self.distance_forward += outcome["progress"]
            # the moment's motion along one fixed heading of the world: the course, or the birth
            # heading without one, which every reading has as its line; the course law pays it
            dx, dy = outcome["moved"]
            heading = self.course if self.course is not None else 0.0
            along = dx * math.cos(heading) + dy * math.sin(heading)
            self.distance_course += along
            self.displacement[0] += dx
            self.displacement[1] += dy
            if self.course is None:
                advance = outcome["progress"]
            elif self.strict:                   # the lesser of the two measures: nose first on the course, or less
                advance = min(along, outcome["progress"])
            else:
                advance = along
            if self.high_water:
                # new ground only: a step back and the same step forward again earn nothing,
                # unless the mark leaks, when the rest of the progress is paid at the old-ground share
                new = max(0.0, self._paid_distance() - self.mark)
                gained = new + self.old_ground * (advance - new)
                self.mark = max(self.mark, self._paid_distance())
            else:
                gained = advance
            self._progress.append(gained)
            paid = sum(self._progress) / self.window    # short of a full window the missing moments count as zero
            self.reward = pay(paid, outcome["stance"], self.stance_pay,
                              outcome["turn"], self.yaw_pay, self.pay_per_metre, self.backward_pay)
            self.turned += abs(outcome["turn"])
            if outcome["recentred"]:
                self.recentres += 1
            self._block.append({**reading, "progress": outcome["progress"], "stance": outcome["stance"],
                                "turn": outcome["turn"], "reward": self.reward, "course": along})
            if len(self._block) >= BLOCK:
                self._close_block()
            self.latest = self.snapshot(reading)
            self._maybe_autosave()
            return self.latest

    def displacement_m(self) -> float:
        """Metres from the point of birth in a straight line, whatever the way there."""
        return float(math.hypot(*self.displacement))

    def _close_block(self) -> None:
        rows = self._block
        self._block = []
        if not rows:
            return
        n = len(rows)
        self.blocks.append({
            "moment": self.moments,
            "forward_m": round(sum(r["progress"] for r in rows), 4),
            "course_m": round(sum(r["course"] for r in rows), 4),
            "reward_per_moment": round(sum(r["reward"] for r in rows) / n, 4),
            "aroused_share": round(sum(1 for r in rows if r["aroused"]) / n, 3),
            "sweeps_per_moment": round(sum(r["sweeps"] for r in rows) / n, 2),
            "learning_sweeps": int(sum(r["learning_sweeps"] for r in rows)),
            "refused": int(sum(1 for r in rows if r["refused"])),
            "ms_per_moment": round(sum(r["ms"] for r in rows) / n, 2),
            "mean_stance": round(sum(r["stance"] for r in rows) / n, 2),
            "turned_rad": round(sum(abs(r["turn"]) for r in rows), 4),
            "displacement_m": round(self.displacement_m(), 4),      # from the birth point, at the block's end
            "policy": rows[-1]["policy"],
        })

    def snapshot(self, reading: dict[str, Any] | None) -> dict[str, Any]:
        reading = reading or {}
        return {
            "moments": self.moments,
            "pose": self.body.pose(),
            "speed": round(self.body.vx, 4),
            "omega": round(self.body.omega, 4),
            "distance_forward": round(self.distance_forward, 4),
            "distance_course": round(self.distance_course, 4),
            "mark": round(self.mark, 4),
            "turned": round(self.turned, 3),
            "displacement_m": round(self.displacement_m(), 4),
            "recentres": self.recentres,
            "reward": None if self.reward is None else round(self.reward, 4),
            "owed": None if getattr(self.policy, "owed", None) is None else round(float(self.policy.owed), 4),
            "pending": bool(self.policy.has_pending()),
            "policy": self.policy.policy,
            "mode": reading.get("mode", "idle"),
            "refused": bool(reading.get("refused", False)),
            "sweeps": int(reading.get("sweeps", 0)),
            "learning_sweeps": int(reading.get("learning_sweeps", 0)),
            "want": round(float(reading.get("want", 0.0)), 3),
            "level": round(float(reading.get("level", 0.0)), 3),
            "ms": round(float(reading.get("ms", 0.0)), 2),
            "commands": reading.get("commands"),
            "activity": reading.get("activity"),
            "refusals": int(getattr(self.policy, "refusals", 0)),
            "last_error": getattr(self.policy, "last_error", None),
            "paused": self.paused,
            "pace": self.pace,
            "rate": round(self.rate, 1),
            "blocks": list(self.blocks)[-12:],
            "last_save_at": self.last_save_at,
            "notes": self.notes[-5:],
        }

    # --- the loop -------------------------------------------------------------------------

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, name="spider-life", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        next_at = last = time.monotonic()
        while self.running:
            if self.paused:
                time.sleep(0.05)
                next_at = last = time.monotonic()
                continue
            self.tick()
            now = time.monotonic()
            if now > last:
                made = 1.0 / (now - last)
                self.rate = made if self.rate == 0.0 else 0.9 * self.rate + 0.1 * made
            last = now
            if self.pace > 0:
                next_at += 1.0 / self.pace
                delay = next_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -1.0:
                    next_at = time.monotonic()      # fell far behind: do not try to catch up

    # --- controls -------------------------------------------------------------------------

    def control(self, op: str, value: Any = None) -> dict[str, Any]:
        with self.lock:
            if op == "pause":
                self.paused = True
            elif op == "resume":
                self.paused = False
            elif op == "recentre":
                self.body.recentre()
                self.recentres += 1
            elif op == "save":
                return {"ok": True, **self.save()}
            elif op == "load":
                return {"ok": True, **self.load()}
            elif op == "reset":
                return {"ok": True, **self.reset(value if isinstance(value, dict) else None)}
            elif op == "law":
                if not isinstance(value, dict):
                    raise ValueError("law takes an object of the settings to change, e.g. {\"high_water\": true}")
                return {"ok": True, **self.set_law(value)}
            elif op == "policy":
                return {"ok": True, **self.switch(str(value))}
            elif op == "pace":
                self.pace = max(0.0, float(value))
            else:
                raise ValueError(f"unknown control {op!r}")
            self.latest = self.snapshot(None)
            return {"ok": True, "op": op}

    def switch(self, name: str) -> dict[str, Any]:
        """Change who commands the body. The brain keeps its custody: an outcome earned under
        the brain's last command stays owed to it, and the body's other policies are controls
        made from the same brain or a seed."""
        current = self.policy
        if name == current.policy:
            return {"policy": name, "changed": False}
        brain = self._live_brain()
        if current.policy == "brain" and self.reward is not None:
            current.owed = (current.owed or 0.0) + self.reward   # the brain's last outcome stays its
            self.reward = None
        if name == "brain":
            self.policy = brain if brain is not None else make_policy("brain", self.seed, None, self.genes, self.backend,
                                                                     inputs=self.senses.inputs, layout=self.layout)
            self._brain = None
        elif name == "random":
            self._keep_brain(brain)
            self.policy = RandomPolicy(self.seed + self.switches + 1, self.layout)
        elif name == "frozen":
            self._keep_brain(brain)
            if brain is not None:
                scratch = self.runs / "frozen-copy.npz"
                self.policy = FrozenPolicy.copy_of(brain.brain, scratch, backend_of(self.backend).get("backend"))
            else:
                self.policy = make_policy("frozen", self.seed, None, self.genes, self.backend, inputs=self.senses.inputs,
                                          layout=self.layout)
        else:
            raise ValueError(f"policy {name!r}: brain, random or frozen")
        self.switches += 1
        self.reward = None                      # the new policy has no command out yet
        self._close_block()
        return {"policy": name, "changed": True}

    _brain: SpiderBrain | None = None

    def _live_brain(self) -> SpiderBrain | None:
        if isinstance(self.policy, SpiderBrain):
            return self.policy
        return self._brain

    def _keep_brain(self, brain: SpiderBrain | None) -> None:
        self._brain = brain

    @property
    def brain(self) -> SpiderBrain | None:
        return self._live_brain()

    def spider(self) -> dict[str, Any] | None:
        """The live brain's connectome for the page's view; None when no brain is in this life."""
        brain = self._live_brain()
        return brain.connectome() if brain is not None else None

    # --- continuation ---------------------------------------------------------------------

    @property
    def brain_file(self) -> Path:
        return self.runs / f"{self.name}.npz"

    @property
    def sidecar(self) -> Path:
        return self.runs / f"{self.name}.json"

    def save(self) -> dict[str, Any]:
        """The brain with its pending outcome, and the world beside it."""
        brain = self._live_brain()
        if brain is None:
            return {"saved": False, "why": "no brain in this life"}
        with self.lock:
            _replace(self.brain_file, lambda p: brain.save(p))
            state = {
                "format": FORMAT,
                "cadence": CADENCE_VERSION,
                "seed": self.seed,
                "senses": self.senses.options(),
                "layout": self.layout,
                "genes": self.genes or {},
                # under a control, the brain's last outcome already sits in its owed
                "reward": self.reward if self.policy is brain else None,
                "owed": brain.owed,
                "moments": self.moments,
                "refusals": brain.refusals,
                "distance_forward": self.distance_forward,
                "distance_course": self.distance_course,
                "mark": self.mark,
                "law": self.law(),
                "law_changes": list(self.law_changes),
                "turned": self.turned,
                "displacement": list(self.displacement),
                "recentres": self.recentres,
                "world": self.body.state_dict(),
                "progress_window": list(self._progress),
                "blocks": list(self.blocks),
                "saved_at": time.time(),
            }
            _replace(self.sidecar, lambda p: p.write_text(json.dumps(state)))
            self._last_save = time.monotonic()
            self.last_save_at = state["saved_at"]
        return {"saved": True, "path": str(self.brain_file), "moments": self.moments}

    def load(self) -> dict[str, Any]:
        """The same brain again, with its pending outcome, and the world it left."""
        if not self.brain_file.exists():
            return {"loaded": False, "why": f"{self.brain_file} does not exist"}
        with self.lock:
            state = json.loads(self.sidecar.read_text()) if self.sidecar.exists() else None
            if state is not None:
                if state.get("format") != FORMAT:
                    raise ValueError(f"{self.sidecar}: unknown format {state.get('format')!r}")
                # the seed and the senses are the life's too (a page's birth names them): a life
                # saved before the sidecar carried them is continued with the flags', as before
                if "seed" in state and int(state["seed"]) != self.seed:
                    self.notes.append(f"saved with seed {state['seed']}, asked for {self.seed}: the saved life keeps its own")
                    self.seed = int(state["seed"])
                if "genes" in state:
                    self.genes = dict(state["genes"]) or None    # the overrides the saved brain was composed with
                if "senses" in state:
                    saved_senses = Senses(per=int(state["senses"]["per"]), setpoints=bool(state["senses"]["setpoints"]),
                                          compass=bool(state["senses"].get("compass", False)),
                                          ground=bool(state["senses"].get("ground", False)))
                    if saved_senses.options() != self.senses.options():
                        self.notes.append(f"saved with {saved_senses.inputs} inputs" + (", setpoints sensed" if saved_senses.setpoints else "")
                                          + (", a compass" if saved_senses.compass else "")
                                          + (", the ground sense" if saved_senses.ground else "")
                                          + f", asked for {self.senses.inputs}: the saved life keeps its own")
                        self.senses = saved_senses
            where = backend_of(self.backend)
            brain = fitted(SpiderBrain.load(self.brain_file, where.get("backend")), self.senses.inputs, self.brain_file)
            if brain.layout != self.layout:
                # the layout is composed into the brain: the saved life keeps its own
                self.notes.append(f"saved with a slot per {brain.layout}, asked for a slot per {self.layout}: the saved life keeps its own")
                self.layout = brain.layout
            self.policy = brain
            self._brain = None
            if state is not None:
                self.reward = state["reward"]
                brain.owed = None if state.get("owed") is None else float(state["owed"])
                brain.refusals = int(state.get("refusals", 0))
                self.moments = int(state["moments"])
                self.distance_forward = float(state["distance_forward"])
                self.distance_course = float(state.get("distance_course", 0.0))
                # a life saved before the mark existed starts it where the body stands: no back pay
                self.mark = float(state.get("mark", self.distance_forward))
                if "law" in state:
                    # the law is the life's: a life saved under one law continues under it, whatever the flags say
                    saved = checked_law(**state["law"])
                    if saved != self.law():
                        self.notes.append(f"saved under the law {law_text(saved)}, asked for {law_text(self.law())}: the saved life keeps its own")
                        self._take_law(saved)
                    self.law_changes = list(state.get("law_changes", []))
                self.turned = float(state.get("turned", 0.0))
                self.recentres = int(state.get("recentres", 0))
                self.body = Spider.restore(state["world"])
                # a life saved before the displacement existed starts it where the body stands: what
                # it walked before its recentres is lost, and the note says so
                if "displacement" in state:
                    self.displacement = [float(v) for v in state["displacement"]]
                else:
                    self.displacement = [float(self.body.x), float(self.body.y)]
                    if self.recentres:
                        self.notes.append(f"saved without a displacement after {self.recentres} recentres: "
                                          "it starts where the body stands")
                self.body.drift = self.drift    # the law's, which the sidecar's law set above
                if self.body.motors != self.motors:
                    self.notes.append(f"saved with {self.body.motors} motors, asked for {self.motors}: the saved life keeps its own")
                    self.motors = self.body.motors
                    if self.senses.setpoints and self.motors != "target":
                        raise ValueError(f"{self.brain_file}: the setpoint senses need target motors, the saved life has {self.motors}")
                self._progress = deque([float(v) for v in state.get("progress_window", [])][-self.window:], maxlen=self.window)
                self.blocks = deque(state.get("blocks", []), maxlen=BLOCKS_KEPT)
                if state.get("cadence") != CADENCE_VERSION:
                    self.notes.append(f"saved with cadence {state.get('cadence')}, running {CADENCE_VERSION}")
                custody = "kept"
            else:
                # the brain's pending command is known, its outcome is not: it reads as an
                # observed zero at the next moment, which is a break in custody, reported
                self.reward = None
                self.body = Spider(motors=self.motors, drift=self.drift)
                custody = "broken: no sidecar, the pending outcome is lost"
                self.notes.append(custody)
            self._block = []
            self.loaded_from = str(self.brain_file)
            self.latest = self.snapshot(None)
        return {"loaded": True, "path": str(self.brain_file), "moments": self.moments, "custody": custody}

    def reset(self, birth: dict[str, Any] | None = None) -> dict[str, Any]:
        """A new life in the same world: a newborn brain commands the body, the world stands at
        its origin, every counter is zero and the saved life is deleted so a restart does not
        continue the old one. The old brain is gone. This is the one deliberate end of a
        continuing life, taken only on the page's Reset after the person confirmed it.

        ``birth`` names the newborn: ``seed``, ``genes`` (overrides, a dict or the nursery's
        ``name=value`` words in one string), ``motors``, ``layout``, ``setpoints``, ``compass`` and
        ``ground``; what it leaves out stays as the life had it. Genes are composed once, at birth, which is why
        they are set here and nowhere else. The law is the life's and stays as it is; under
        the course law the distance along the course and the mark start at zero."""
        birth = dict(birth or {})
        unknown = sorted(set(birth) - {"seed", "genes", "motors", "layout", "setpoints", "compass", "ground"})
        if unknown:
            raise ValueError(f"a birth names seed, genes, motors, layout, setpoints, compass and ground, not {', '.join(unknown)}")
        seed = int(birth.get("seed", self.seed))
        genes = birth.get("genes", self.genes)
        if isinstance(genes, str):
            genes = genes_from_text(genes) or None
        elif genes is not None and not isinstance(genes, dict):
            raise ValueError("genes are name=value words or an object")
        motors = str(birth.get("motors", self.motors))
        if motors not in MOTORS:
            raise ValueError(f"motors must be one of {MOTORS}, not {motors!r}")
        layout = str(birth.get("layout", self.layout))
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")
        setpoints = bool(birth.get("setpoints", self.senses.setpoints))
        compass = bool(birth.get("compass", self.senses.compass))
        ground = bool(birth.get("ground", self.senses.ground))
        if setpoints and motors != "target":
            raise ValueError("the setpoint senses need target motors")
        senses = self.senses if (setpoints, compass, ground) == (self.senses.setpoints, self.senses.compass, self.senses.ground) \
            else Senses(per=self.senses.per, setpoints=setpoints, compass=compass, ground=ground)
        newborn = make_policy("brain", seed, None, genes, self.backend, inputs=senses.inputs, layout=layout)   # composed before the old life ends
        with self.lock:
            was = self.moments
            for path in (self.brain_file, self.sidecar, self.runs / "frozen-copy.npz"):
                if path.exists():
                    path.unlink()
            self.seed, self.genes, self.motors, self.senses, self.layout = seed, genes, motors, senses, layout
            self.policy = newborn
            self._brain = None
            self.body = Spider(motors=self.motors, drift=self.drift)
            self.reward = None
            self.moments = 0
            self.distance_forward = 0.0
            self.distance_course = 0.0
            self.mark = 0.0
            self.turned = 0.0
            self.displacement = [0.0, 0.0]
            self.recentres = 0
            self.switches = 0
            self.blocks.clear()
            self._block = []
            self._progress.clear()
            self.law_changes = []
            self._last_save = time.monotonic()
            self.last_save_at = None
            self.loaded_from = None
            born = f"seed {self.seed}, {self.motors} motors" + (", a slot per leg" if layout == "leg" else "") \
                + (", setpoints sensed" if setpoints else "") \
                + (", a compass" if compass else "") + (", the ground sense" if ground else "") \
                + (f", genes {gene_words(genes)}" if genes else ", founder genes")
            self.notes = [f"reset after {was} moments: a newborn brain of {born}"]
            self.latest = self.snapshot(None)
        return {"reset": True, "moments_before": was, "seed": self.seed, "genes": genes or {}, "motors": self.motors,
                "layout": self.layout, "setpoints": setpoints, "compass": compass, "ground": ground}

    def _maybe_autosave(self) -> None:
        if self.autosave_minutes <= 0 or not isinstance(self.policy, SpiderBrain):
            return
        if time.monotonic() - self._last_save >= self.autosave_minutes * 60.0:
            self.save()

    def describe(self) -> dict[str, Any]:
        brain = self._live_brain()
        return {
            "policy": self.policy.policy,
            "seed": self.seed,
            "pace": self.pace,
            "dt": DT,
            "motors": self.motors,
            "layout": self.layout,
            "inputs": self.senses.inputs,
            "setpoints": self.senses.setpoints,
            "compass": self.senses.compass,
            "ground": self.senses.ground,
            "input_names": self.senses.input_names(),
            "joints": JOINTS,
            "legs": LEGS,
            "hips": HIPS.tolist(),
            "height": HEIGHT,
            "reward": {**self.law(), "cap": PROGRESS_CAP, "stance_floor": STANCE_FLOOR, "mark": round(self.mark, 4)},
            "law_changes": list(self.law_changes),
            "genes": dict(self.genes or {}),
            "brain": brain.describe() if brain is not None else None,
            "control": self.policy.describe() if self.policy is not brain else None,
            "runs": str(self.runs),
            "loaded_from": self.loaded_from,
        }
