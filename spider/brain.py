"""The spider's brain: one continuing ``Brain.live`` life wired to all sixteen joints.

``Brain.compose`` builds one neural graph: the senses drive two processing modules, the
association region exchanges signals with the motor cortex, and the working trace feeds the
association region its own recent state. The motor cortex is sixteen slots of three neurons,
one slot per joint: the settled state of the whole graph is read as one choice per joint
(decrease, hold, increase), so every joint is commanded by the same equilibrium.

The founder genes are the Robot Arena nursery's operating point (cadence-demos,
``robot-arena/arena/brain.py``) translated to cadence-net 0.97.1 names, with two choices made
here and not measured there: wider modules for the population-coded senses, and the recall
memory off. Every value is a gene; ``compose(genes=...)`` overrides any of them.

A life goes through ``SpiderBrain.moment``: the observation of this moment and the outcome of
the previous command go in, one command per joint comes out. Calm, the brain answers from one
settled state and learns nothing; surprised or in want, it samples, learns and remembers. A
refused answer is handled as the library prescribes and reported, never hidden. ``RandomPolicy``
is the baseline with the same body; ``FrozenPolicy`` is a brain that answers greedily and never
learns, the control for "kept learning".
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from cadence import Brain

from .body import JOINTS
from .senses import INPUTS

SLOTS = [3] * JOINTS
ACTIONS = sum(SLOTS)

FOUNDER: dict[str, Any] = {
    # Robot Arena's founder (its nursery grids of 2026-10-08/09), copied:
    "trace_amplitude": 0.3,
    "trace_decay": 0.1,
    "trace_scale": 12.0,            # the trace read back at scale 12, as the arena's brains were raised
    "efference_amplitude": 0.0,
    "efference_decay": 0.0,
    "episodic": True,
    "consolidation": 0.05,
    "sensory_scale": 4.0,           # the sensory projection's scale; its transfer to 152 coded inputs is unmeasured
    "lateral": -0.5,                # motor inhibition within a slot, the arena's
    "temperature": 0.3,
    "eta": 0.03,
    "eta_bias": 0.003,
    "lam": 0.6,
    "gamma": 0.95,
    "eta_critic": 5.0,
    "arousal": {
        "threshold": 0.2, "decay": 0.9, "tolerance": 2.0, "floor": 0.1,
        "fast": 0.05, "slow": 0.005, "heat": 2.0, "youth": 300,
        "value_surprise": 1.0, "need": 0.05,
    },
    # chosen here, unmeasured against the arena's values:
    "modules": [64, 32],            # the arena: [48, 24] on 32 inputs; here 152 coded inputs
    "recall": False,                # 0.97.1 composes a recall memory by default; off for cost
}


def genes_of(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """The founder genes with ``overrides`` applied; ``arousal`` merges by key."""
    genes = copy.deepcopy(FOUNDER)
    own = dict(overrides or {})
    arousal = own.pop("arousal", {})
    genes.update(own)
    genes["arousal"] = {**genes["arousal"], **arousal}
    return genes


def compose(
    seed: int = 0,
    genes: dict[str, Any] | None = None,
    *,
    inputs: int = INPUTS,
    backend: str | None = None,
    device: str | None = None,
) -> Brain:
    """A newborn brain for this body: one slot per joint, one input per sense.

    ``backend`` and ``device`` are passed only when given; otherwise the library's rule (or
    the environment variable ``CADENCE_BACKEND``) places the brain."""
    g = genes_of(genes)
    where: dict[str, Any] = {}
    if backend is not None:
        where["backend"] = backend
    if device is not None:
        where["device"] = device
    return Brain.compose(
        int(inputs),
        ACTIONS,
        modules=tuple(int(w) for w in g["modules"]),
        slots=SLOTS,
        lateral=float(g["lateral"]),
        seed=int(seed),
        recall=bool(g["recall"]),
        episodic=bool(g["episodic"]),
        consolidation=float(g["consolidation"]),
        working_memory_amplitude=float(g["trace_amplitude"]),
        working_memory_decay=float(g["trace_decay"]),
        efference_amplitude=float(g["efference_amplitude"]),
        efference_decay=float(g["efference_decay"]),
        arousal=dict(g["arousal"]),
        scales={"sensory": float(g["sensory_scale"]), "trace": float(g["trace_scale"])},
        temperature=float(g["temperature"]),
        actor_eta=float(g["eta"]),
        actor_eta_bias=float(g["eta_bias"]),
        actor_lam=float(g["lam"]),
        actor_gamma=float(g["gamma"]),
        actor_eta_critic=float(g["eta_critic"]),
        **where,
    )


def hold_commands() -> list[int]:
    """Every joint held."""
    return [1] * JOINTS


def activity(brain: Brain) -> dict[str, list[float]] | None:
    """The settled state the brain just acted from: sensory, association and motor
    activations, rounded for the page."""
    state = brain.basal_ganglia.state
    if state is None:
        return None
    act = np.asarray(state.activation)[0]
    return {
        "sensory": [round(float(v), 2) for v in act[brain.sensory_index]],
        "association": [round(float(v), 2) for v in act[brain.association_index]],
        "motor": [round(float(v), 2) for v in act[brain.motor_index]],
    }


def placement(brain: Brain) -> dict[str, Any]:
    p = getattr(brain, "placement", None) or {}
    return {k: p.get(k) for k in ("backend", "device", "precision", "chosen_by") if k in p}


def _refused(elapsed: float, policy: str) -> dict[str, Any]:
    return {
        "commands": hold_commands(), "refused": True, "mode": "refused", "aroused": False,
        "sweeps": 0, "learning_sweeps": 0, "want": 0.0, "level": 0.0, "ms": elapsed,
        "policy": policy,
    }


class SpiderBrain:
    """One continuing life. ``moment`` delivers the previous command's outcome and returns
    this moment's commands; ``save``/``load`` carry the whole life, including an outcome still
    owed after a refusal."""

    policy = "brain"
    learns = True

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self.owed: float | None = None  # an outcome kept: after a refusal, or while another policy acted
        self.refusals = 0
        self.moments = 0
        self.last_error: str | None = None

    @classmethod
    def newborn(cls, seed: int = 0, genes: dict[str, Any] | None = None, **where: Any) -> "SpiderBrain":
        return cls(compose(seed, genes, **where))

    @classmethod
    def load(cls, path: str | Path, backend: str | None = None) -> "SpiderBrain":
        kw = {"backend": backend} if backend else {}
        return cls(Brain.load(Path(path), **kw))

    def save(self, path: str | Path) -> Path:
        return self.brain.save(Path(path))

    def has_pending(self) -> bool:
        """Whether an issued command still awaits its outcome."""
        return bool(self.brain.pending_feedback)

    def moment(self, observation: np.ndarray, reward: float | None, done: bool = False) -> dict[str, Any]:
        brain = self.brain
        began = time.perf_counter()
        # the outcome to deliver: this moment's reward plus anything kept from before (a
        # refusal, or moments another policy owned the body)
        total: float | None = None
        if reward is not None:
            total = float(reward) + (self.owed or 0.0)
        elif self.owed is not None:
            total = float(self.owed)
        feedback: dict[str, Any] = {}
        if total is not None and self.has_pending():
            feedback = {"reward": [total], "done": [bool(done)]}
        elif total is not None:
            self.owed = total               # nothing to credit yet; keep it for the next command
        action = None
        try:
            action = brain.live(observation, **feedback)
            if feedback:
                self.owed = None            # delivered
        except (RuntimeError, ValueError) as error:
            self.refusals += 1
            self.last_error = f"{type(error).__name__}: {error}"[:200]
            if feedback and not self.has_pending():
                # the outcome was taken; only the answer refused: retry without it
                self.owed = None
                try:
                    action = brain.live(observation)
                except (RuntimeError, ValueError) as again:
                    self.last_error = f"{type(again).__name__}: {again}"[:200]
            elif feedback:
                self.owed = total           # still owed to the same command
        elapsed = (time.perf_counter() - began) * 1000.0
        self.moments += 1
        if action is None:
            return _refused(elapsed, self.policy)
        reading = brain.last_arousal or {}
        arousal = brain.arousal
        assert arousal is not None
        mode = reading.get("mode", arousal.mode)
        return {
            "commands": [int(c) for c in np.atleast_1d(action[0])],
            "refused": False,
            "mode": mode,
            "aroused": mode == "aroused",
            "sweeps": int(reading.get("sweeps", 0)),
            "learning_sweeps": int(reading.get("learning_sweeps", 0)),
            "want": float(arousal.want),
            "level": float(arousal.level),
            "ms": elapsed,
            "policy": self.policy,
            "activity": activity(brain),
        }

    def describe(self) -> dict[str, Any]:
        arousal = self.brain.arousal
        assert arousal is not None
        return {
            "policy": self.policy,
            "neurons": int(self.brain.connectome.n),
            "parameters": int(self.brain.parameters()),
            "age": int(arousal.age),
            "moments": {k: int(v) for k, v in arousal.moments.items()},
            "sweeps": {k: int(v) for k, v in arousal.sweeps.items()},
            "learning_sweeps": int(arousal.learning_sweeps),
            "refusals": self.refusals,
            "owed": self.owed,
            "pending": self.has_pending(),
            "placement": placement(self.brain),
            "genes": self.brain.describe().get("genes", {}),
        }


class RandomPolicy:
    """The uniform-random baseline with the same body: one draw per joint, every moment."""

    policy = "random"
    learns = False

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.moments = 0
        self.refusals = 0
        self.owed = None

    def moment(self, observation: np.ndarray, reward: float | None, done: bool = False) -> dict[str, Any]:
        self.moments += 1
        return {
            "commands": [int(self.rng.integers(size)) for size in SLOTS],
            "refused": False, "mode": "random", "aroused": False, "sweeps": 0,
            "learning_sweeps": 0, "want": 0.0, "level": 0.0, "ms": 0.0, "policy": self.policy,
        }

    def has_pending(self) -> bool:
        return False

    def save(self, path: str | Path) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"policy": self.policy, "moments": self.moments}


class FrozenPolicy:
    """A brain that answers greedily and never learns: the control for "kept learning"."""

    policy = "frozen"
    learns = False

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self.moments = 0
        self.refusals = 0
        self.owed = None

    @classmethod
    def copy_of(cls, brain: Brain, scratch: str | Path, backend: str | None = None) -> "FrozenPolicy":
        """A frozen copy of a live brain through a save/load round trip."""
        path = Path(scratch)
        brain.save(path)
        kw = {"backend": backend} if backend else {}
        return cls(Brain.load(path, **kw))

    @classmethod
    def load(cls, path: str | Path, backend: str | None = None) -> "FrozenPolicy":
        kw = {"backend": backend} if backend else {}
        return cls(Brain.load(Path(path), **kw))

    def moment(self, observation: np.ndarray, reward: float | None, done: bool = False) -> dict[str, Any]:
        self.moments += 1
        began = time.perf_counter()
        try:
            action = self.brain.act(observation, greedy=True)
        except (RuntimeError, ValueError):
            self.refusals += 1
            return _refused((time.perf_counter() - began) * 1000.0, self.policy)
        settlement = self.brain.last_settlement or {}
        return {
            "commands": [int(c) for c in np.atleast_1d(action[0])],
            "refused": False, "mode": "frozen", "aroused": False,
            "sweeps": int(settlement.get("steps", 0)), "learning_sweeps": 0,
            "want": 0.0, "level": 0.0, "ms": (time.perf_counter() - began) * 1000.0,
            "policy": self.policy, "activity": activity(self.brain),
        }

    def has_pending(self) -> bool:
        return False

    def save(self, path: str | Path) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "policy": self.policy, "moments": self.moments, "refusals": self.refusals,
            "neurons": int(self.brain.connectome.n), "placement": placement(self.brain),
        }


def backend_of(name: str | None) -> dict[str, str]:
    """``auto``/``None`` leaves the choice to the library; ``cpu``/``torch``/``mlx`` name it.
    ``torch`` on this project means CUDA when the machine has it (the library's own order)."""
    name = (name or os.environ.get("BACKEND") or "auto").strip().lower()
    if name in ("", "auto"):
        return {}
    if name in ("cpu", "torch", "mlx"):
        return {"backend": name}
    if name.startswith("torch:"):
        return {"backend": "torch", "device": name.split(":", 1)[1]}
    raise ValueError(f"backend {name!r}: auto, cpu, torch, torch:<device> or mlx")


def make_policy(
    name: str,
    seed: int = 0,
    brain_path: str | Path | None = None,
    genes: dict[str, Any] | None = None,
    backend: str | None = None,
) -> SpiderBrain | RandomPolicy | FrozenPolicy:
    where = backend_of(backend)
    if name == "random":
        return RandomPolicy(seed)
    if name == "brain":
        if brain_path is not None and Path(brain_path).exists():
            return SpiderBrain.load(brain_path, where.get("backend"))
        return SpiderBrain.newborn(seed, genes, **where)
    if name == "frozen":
        if brain_path is not None and Path(brain_path).exists():
            return FrozenPolicy.load(brain_path, where.get("backend"))
        return FrozenPolicy(compose(seed, genes, **where))
    raise ValueError(f"policy {name!r}: brain, random or frozen")
