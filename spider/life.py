"""One continuing life: the body in its world, the policy that commands it, and the custody of
every outcome.

A tick observes the body, hands the previous command's reward and this moment's observation to
the policy, applies the returned commands to the body, and keeps the reward the body earned for
the next tick. The reward is the world's: progress along the body's own heading, less two hints
about how to go (feet short of four, radians turned).
Each outcome reaches the brain exactly once; a refusal keeps it owed, a policy switch keeps it
owed, a save carries it, a load delivers it.

``Life`` runs its own daemon thread. One thread owns the policy and the body; readers take the
``latest`` snapshot, which is replaced wholesale.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .body import DT, HEIGHT, HIPS, JOINTS, LEGS, Spider
from .brain import FrozenPolicy, RandomPolicy, SpiderBrain, backend_of, make_policy
from .senses import DEFAULT as DEFAULT_SENSES, Senses

PROGRESS_PAY = 40.0     # reward per metre forward: 0.4 a moment at the full stroke speed of 0.2 m/s
PROGRESS_CAP = 0.5      # the Robot Arena's cap, under the dopamine cap of 1.0
STANCE_FLOOR = LEGS // 2   # feet the world wants on the ground through every moment: half of them
STANCE_PAY = 0.1        # taken per foot short of the floor: a hint, shaping, not an outcome (0.02 was ignored)
PROGRESS_WINDOW = 1     # moments the progress term is averaged over: 1 pays each moment's own progress;
                        # a window pays the mean over the last N, which a sway that nets nothing cannot earn
                        # (README, "The rocking"); --window / PROGRESS_WINDOW set it
YAW_PAY = 0.0           # taken per radian turned in the moment: the hint to go straight, off (README: ignored at
                        # 0.1 and 1 a radian, obeyed at 5 by standing still); --yaw-pay / YAW_PAY switch it on
BLOCK = 100             # moments per statistics block
BLOCKS_KEPT = 60
FORMAT = "learning-spider/1"
CADENCE_VERSION = "0.97.1"


def pay(progress: float, stance: int = STANCE_FLOOR, stance_pay: float = STANCE_PAY,
        turn: float = 0.0, yaw_pay: float = YAW_PAY) -> float:
    """Forward progress pays; backward and standing still pay zero; every foot short of
    ``STANCE_FLOOR`` planted through the moment takes ``stance_pay``, and every radian turned
    in the moment takes ``yaw_pay``. The progress term is the outcome. The stance and yaw terms
    are hints the world gives about posture and course (reward shaping): the stance term was
    chosen after the learner under progress alone walked on ever fewer feet, the yaw term
    because progress along the body's own heading pays a circle as well as a line (README,
    "Measure it"). The first law paid backward progress negatively, and over 30,000 moments the
    learning brain lifted every foot to avoid it; forward-only pay was chosen by that reading."""
    progress_term = float(np.clip(PROGRESS_PAY * progress, 0.0, PROGRESS_CAP))
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
    ) -> None:
        self.name = name                        # the file stem under runs/
        self.stance_pay = float(stance_pay)     # 0 switches the posture hint off
        self.yaw_pay = float(yaw_pay)           # 0 switches the course hint off
        if int(window) < 1:
            raise ValueError(f"window must be at least 1 moment, not {window}")
        self.window = int(window)
        self._progress: deque[float] = deque(maxlen=self.window)   # the last window moments' progress
        self.seed = int(seed)
        self.genes = genes
        self.backend = backend
        self.senses = senses
        self.runs = Path(runs)
        self.pace = float(pace)                 # moments per second; 0 = as fast as possible
        self.spider = Spider()
        self.policy = make_policy(policy, self.seed, brain_path, genes, backend)
        self.reward: float | None = None        # the last executed command's outcome, not yet delivered
        self.moments = 0
        self.distance_forward = 0.0
        self.turned = 0.0                       # radians turned over the life, unsigned
        self.distance_net_start = (0.0, 0.0)
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

    # --- one moment -----------------------------------------------------------------------

    def tick(self) -> dict[str, Any]:
        with self.lock:
            observation = self.senses.observe(self.spider)
            reading = self.policy.moment(observation, self.reward)
            self.reward = None
            before = (self.spider.x, self.spider.y)
            outcome = self.spider.step(reading["commands"])
            self._progress.append(outcome["progress"])
            paid = sum(self._progress) / self.window    # short of a full window the missing moments count as zero
            self.reward = pay(paid, outcome["stance"], self.stance_pay,
                              outcome["turn"], self.yaw_pay)
            self.moments += 1
            self.distance_forward += outcome["progress"]
            self.turned += abs(outcome["turn"])
            if outcome["recentred"]:
                self.recentres += 1
                self.distance_net_start = (0.0, 0.0)
            self._block.append({**reading, "progress": outcome["progress"], "stance": outcome["stance"],
                                "turn": outcome["turn"], "reward": self.reward})
            if len(self._block) >= BLOCK:
                self._close_block()
            self.latest = self.snapshot(reading)
            self._maybe_autosave()
            return self.latest

    def _close_block(self) -> None:
        rows = self._block
        self._block = []
        if not rows:
            return
        n = len(rows)
        self.blocks.append({
            "moment": self.moments,
            "forward_m": round(sum(r["progress"] for r in rows), 4),
            "reward_per_moment": round(sum(r["reward"] for r in rows) / n, 4),
            "aroused_share": round(sum(1 for r in rows if r["aroused"]) / n, 3),
            "sweeps_per_moment": round(sum(r["sweeps"] for r in rows) / n, 2),
            "learning_sweeps": int(sum(r["learning_sweeps"] for r in rows)),
            "refused": int(sum(1 for r in rows if r["refused"])),
            "ms_per_moment": round(sum(r["ms"] for r in rows) / n, 2),
            "mean_stance": round(sum(r["stance"] for r in rows) / n, 2),
            "turned_rad": round(sum(abs(r["turn"]) for r in rows), 4),
            "policy": rows[-1]["policy"],
        })

    def snapshot(self, reading: dict[str, Any] | None) -> dict[str, Any]:
        reading = reading or {}
        return {
            "moments": self.moments,
            "pose": self.spider.pose(),
            "speed": round(self.spider.vx, 4),
            "omega": round(self.spider.omega, 4),
            "distance_forward": round(self.distance_forward, 4),
            "turned": round(self.turned, 3),
            "distance_net": round(float(np.hypot(self.spider.x, self.spider.y)), 4),
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
        next_at = time.monotonic()
        while self.running:
            if self.paused:
                time.sleep(0.05)
                next_at = time.monotonic()
                continue
            self.tick()
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
                self.spider.recentre()
                self.recentres += 1
            elif op == "save":
                return {"ok": True, **self.save()}
            elif op == "load":
                return {"ok": True, **self.load()}
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
            self.policy = brain if brain is not None else make_policy("brain", self.seed, None, self.genes, self.backend)
            self._brain = None
        elif name == "random":
            self._keep_brain(brain)
            self.policy = RandomPolicy(self.seed + self.switches + 1)
        elif name == "frozen":
            self._keep_brain(brain)
            if brain is not None:
                scratch = self.runs / "frozen-copy.npz"
                self.policy = FrozenPolicy.copy_of(brain.brain, scratch, backend_of(self.backend).get("backend"))
            else:
                self.policy = make_policy("frozen", self.seed, None, self.genes, self.backend)
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
                # under a control, the brain's last outcome already sits in its owed
                "reward": self.reward if self.policy is brain else None,
                "owed": brain.owed,
                "moments": self.moments,
                "refusals": brain.refusals,
                "distance_forward": self.distance_forward,
                "turned": self.turned,
                "recentres": self.recentres,
                "world": self.spider.state_dict(),
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
            where = backend_of(self.backend)
            brain = SpiderBrain.load(self.brain_file, where.get("backend"))
            self.policy = brain
            self._brain = None
            if self.sidecar.exists():
                state = json.loads(self.sidecar.read_text())
                if state.get("format") != FORMAT:
                    raise ValueError(f"{self.sidecar}: unknown format {state.get('format')!r}")
                self.reward = state["reward"]
                brain.owed = None if state.get("owed") is None else float(state["owed"])
                brain.refusals = int(state.get("refusals", 0))
                self.moments = int(state["moments"])
                self.distance_forward = float(state["distance_forward"])
                self.turned = float(state.get("turned", 0.0))
                self.recentres = int(state.get("recentres", 0))
                self.spider = Spider.restore(state["world"])
                self._progress = deque([float(v) for v in state.get("progress_window", [])][-self.window:], maxlen=self.window)
                self.blocks = deque(state.get("blocks", []), maxlen=BLOCKS_KEPT)
                if state.get("cadence") != CADENCE_VERSION:
                    self.notes.append(f"saved with cadence {state.get('cadence')}, running {CADENCE_VERSION}")
                custody = "kept"
            else:
                # the brain's pending command is known, its outcome is not: it reads as an
                # observed zero at the next moment, which is a break in custody, reported
                self.reward = None
                self.spider = Spider()
                custody = "broken: no sidecar, the pending outcome is lost"
                self.notes.append(custody)
            self._block = []
            self.loaded_from = str(self.brain_file)
            self.latest = self.snapshot(None)
        return {"loaded": True, "path": str(self.brain_file), "moments": self.moments, "custody": custody}

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
            "inputs": self.senses.inputs,
            "input_names": self.senses.input_names(),
            "joints": JOINTS,
            "legs": LEGS,
            "hips": HIPS.tolist(),
            "height": HEIGHT,
            "reward": {"pay_per_metre": PROGRESS_PAY, "cap": PROGRESS_CAP,
                       "stance_floor": STANCE_FLOOR, "stance_pay": self.stance_pay, "yaw_pay": self.yaw_pay,
                       "window": self.window},
            "brain": brain.describe() if brain is not None else None,
            "control": self.policy.describe() if self.policy is not brain else None,
            "runs": str(self.runs),
            "loaded_from": self.loaded_from,
        }
