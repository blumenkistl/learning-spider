"""The spider's brain: one continuing ``Brain.live`` life wired to all sixteen joints.

``Brain.compose`` builds one neural graph: the senses drive two processing modules, the
association region exchanges signals with the motor cortex, and the working trace feeds the
association region its own recent state. The motor cortex is sixteen slots of three neurons,
one slot per joint: the settled state of the whole graph is read as one choice per joint
(decrease, hold, increase), so every joint is commanded by the same equilibrium. That is the
``joint`` layout, the founder's; under the ``leg`` layout (``layout="leg"``) the cortex is
eight slots of nine, one per leg, each answer one of the nine pairs a leg's hip and knee can
take. The same sixteen commands reach the body either way; what differs is what the brain
chooses and is credited for: a joint, or a leg (README, "A slot per leg").

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

from .body import JOINTS, LEGS
from .senses import INPUTS

LAYOUTS = ("joint", "leg")      # how the motor cortex is read: a slot per joint (three answers) or per leg (nine)
DEFAULT_LAYOUT = "joint"        # the founder's layout; "leg" is measured in the README's "A slot per leg"
SLOTS = [3] * JOINTS            # the joint layout's slots, the founder's
ACTIONS = sum(SLOTS)


def slots_of(layout: str) -> list[int]:
    """The motor cortex's slots under a layout: ``joint`` reads sixteen slots of three, one
    joint's decrease / hold / increase each; ``leg`` reads eight slots of nine, one leg's
    pair of hip and knee commands, every pair of the same three a joint has. Both reach the
    same sixteen commands; under ``leg`` a leg is one choice, not two, so a lift with a
    swing, or a stroke, is one answer to credit (README, "A slot per leg")."""
    if layout == "joint":
        return [3] * JOINTS
    if layout == "leg":
        return [9] * LEGS
    raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")


def layout_of(brain: Brain) -> str:
    """The layout a composed brain was read by, from its slots."""
    slots = list(brain.describe()["layout"]["slots"])
    for layout in LAYOUTS:
        if slots == slots_of(layout):
            return layout
    raise ValueError(f"a brain of slots {slots} is not this body's: {LAYOUTS} are {[slots_of(l) for l in LAYOUTS]}")


def joint_commands(answer: Any, layout: str) -> list[int]:
    """The sixteen joint commands (``0`` decrease, ``1`` hold, ``2`` increase, joint order
    hip0, knee0, hip1, ...) a brain's answer names: under ``joint`` the answer itself; under
    ``leg`` each leg's choice ``c`` of nine is its hip's ``c // 3`` and its knee's ``c % 3``."""
    answer = [int(c) for c in np.atleast_1d(answer)]
    if layout == "joint":
        return answer
    if layout == "leg":
        out: list[int] = []
        for c in answer:
            out += [c // 3, c % 3]
        return out
    raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")

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
        "value_surprise": 1.0,
        "need": 0.1,                # 0.05 (the arena's) until 2026-11-13: a stand's income met it and calmed the
                                    # brain; at 0.1 three seeds of three kept walking and learning over 150,000
                                    # moments (measured with entropy 0.1 and gamma 0.99; the founder row is below)
    },
    # chosen here, unmeasured against the arena's values:
    "modules": [64, 32],            # the arena: [48, 24] on 32 inputs; here 152 coded inputs
    "recall": False,                # 0.97.1 composes a recall memory by default; off for cost
    "history_decays": [],           # no heard history: the brain hears only this moment (0.97.1 composes
                                    # one only with recall); blocks of decays (--gene history_decays=0,0.7,0.95)
                                    # would give it the moments before, a candidate gene for a gait
    # the reward learner's options (cadence docs/reward.md), each at the value that keeps the
    # composed actor to the bit; measured in the README's "The reward learner's options":
    "hold": None,                   # the actor's eligibility decay apart from the critic's: None is
                                    # gamma * lam (0.57 a moment), 0.9 a horizon of ten moments (a step),
                                    # 1 holds every act until an episode ends, which never comes here
    "entropy": 0.0,                 # the policy's entropy in the actor's step; 0.1 is the docs' dose
    "maturity": False,              # the slow use of plasticity that lowers the actor's step as it is spent
    "maturity_scale": 10.0,         # the use at which the step factor is 0.5 (with maturity only)
    "maturity_recovery": 200000,    # moments of the use's time constant (with maturity only)
    # how the brain reads its senses (cadence docs/senses.md), the founder reading the coded
    # observation itself; measured in the README's "The observation codes":
    "code": None,                   # a fixed random sparse code of that many units (at least 24) that the
                                    # cortex reads and the memories key on, in place of the 152 inputs
    "learned_code": None,           # a code region of that many units between the senses and the first
                                    # module, learned by local rules while the brain lives; costs 2-3x a moment
    "learned_code_fields": None,    # with learned_code: the widths of the observation's channels, each
                                    # read by its own units (128,16,8: the angles, speed and yaw, the flags)
}


def genes_from_text(text: str) -> dict[str, Any]:
    """Gene overrides from ``name=value`` words, the nursery's ``--gene`` syntax in one string
    (``"gamma=0.99 eta_critic=20"``): the page's way of naming a newborn's genes."""
    return genes_from([parse_gene(word) for word in str(text).split()])


def gene_words(genes: dict[str, Any] | None, prefix: str = "") -> str:
    """The overrides back in ``name=value`` words (``genes_from_text`` the other way), for notes."""
    words: list[str] = []
    for name, value in (genes or {}).items():
        if isinstance(value, dict):
            words.append(gene_words(value, f"{prefix}{name}."))
        else:
            words.append(f"{prefix}{name}={','.join(str(v) for v in value) if isinstance(value, list) else value}")
    return " ".join(w for w in words if w)


def parse_gene(text: str) -> tuple[str, Any]:
    name, sep, raw = text.partition("=")
    if not sep:
        raise ValueError(f"gene {text!r}: expected name=value")
    raw = raw.strip()
    value: Any
    if raw.lower() in ("none", "null"):
        value = None
    elif raw.lower() in ("true", "false"):
        value = raw.lower() == "true"
    elif "," in raw:
        value = [_number(v) for v in raw.split(",")]
    else:
        value = _number(raw)
    return name.strip(), value


def _number(raw: str) -> Any:
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def genes_from(items: list[tuple[str, Any]]) -> dict[str, Any]:
    genes: dict[str, Any] = {}
    for name, value in items:
        if name.startswith("arousal."):
            genes.setdefault("arousal", {})[name[len("arousal."):]] = value
        else:
            genes[name] = value
    return genes


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
    layout: str = DEFAULT_LAYOUT,
    backend: str | None = None,
    device: str | None = None,
) -> Brain:
    """A newborn brain for this body: one slot per joint (or per leg, under the ``leg``
    layout), one input per sense.

    ``backend`` and ``device`` are passed only when given; otherwise the library's rule (or
    the environment variable ``CADENCE_BACKEND``) places the brain."""
    g = genes_of(genes)
    slots = slots_of(layout)
    where: dict[str, Any] = {}
    if backend is not None:
        where["backend"] = backend
    if device is not None:
        where["device"] = device
    return Brain.compose(
        int(inputs),
        sum(slots),
        modules=tuple(int(w) for w in g["modules"]),
        slots=slots,
        lateral=float(g["lateral"]),
        seed=int(seed),
        recall=bool(g["recall"]),
        episodic=bool(g["episodic"]),
        consolidation=float(g["consolidation"]),
        working_memory_amplitude=float(g["trace_amplitude"]),
        working_memory_decay=float(g["trace_decay"]),
        efference_amplitude=float(g["efference_amplitude"]),
        efference_decay=float(g["efference_decay"]),
        history_decays=tuple(float(d) for d in np.atleast_1d(g["history_decays"])),
        arousal=dict(g["arousal"]),
        scales={"sensory": float(g["sensory_scale"]), "trace": float(g["trace_scale"])},
        temperature=float(g["temperature"]),
        actor_eta=float(g["eta"]),
        actor_eta_bias=float(g["eta_bias"]),
        actor_lam=float(g["lam"]),
        actor_gamma=float(g["gamma"]),
        actor_eta_critic=float(g["eta_critic"]),
        actor_entropy=float(g["entropy"]),
        **({"actor_hold": float(g["hold"])} if g["hold"] is not None else {}),
        **({"maturity": True, "maturity_scale": float(g["maturity_scale"]),
            "maturity_recovery": int(g["maturity_recovery"])} if g["maturity"] else {}),
        **({"code": int(g["code"])} if g["code"] is not None else {}),
        **({"learned_code": int(g["learned_code"])} if g["learned_code"] is not None else {}),
        **({"learned_code_fields": tuple(int(w) for w in np.atleast_1d(g["learned_code_fields"]))}
           if g["learned_code_fields"] is not None else {}),
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
        "all": [round(float(v), 2) for v in act],          # every neuron, in connectome order
    }


# --- the brain view: the connectome laid out for the page, and the moment's learning ----------
# A synthetic layout, not an anatomy, but shaped like one: the neurons fill an ellipsoid, the
# senses on its rear surface, the module a shell inside them, the association and prefrontal
# regions in the core, the motor cortex a cap on the front, sixteen tight triplets. A neuron's
# fibres are its incoming synapses, one curved fibre from each presynaptic soma, bulging toward
# the surface so the ball reads as a body of fibres. The page's view (static/vendor/
# brain_view.js, the connectome demo's) paints a neuron's fibres and soma green by its
# activity and flashes them gold by how much its incoming synapses changed this moment.
CONNECTOME_FORMAT = "cadence.zebrafish-skeletons/v1"   # the view's payload format, kept by name
LEARNING_GAIN = 40.0      # the view's learning formula: level = sqrt(min(1, gain * sum|dw| / sum|w|))
LEARNING_FLOOR = 1e-9
AXES = np.array([1.0, 0.8, 0.7])      # the ellipsoid's semi-axes: x forward, the longest spans 2.0
FIBRE_NODES = 3                       # segments per synapse: a quadratic curve sampled in thirds
_LABELS = {"sensory": "senses", "code": "code", "association": "association", "prefrontal": "prefrontal",
           "motor": "motor"}
_COLORS = {"sensory": [90, 209, 255], "code": [170, 150, 255], "association": [255, 214, 102],
           "prefrontal": [90, 220, 190],
           "motor": [210, 255, 120]}
_MODULE_COLOR = [103, 231, 142]
_GOLDEN = np.pi * (3.0 - np.sqrt(5.0))


def _populations(brain: Brain) -> list[tuple[str, np.ndarray]]:
    """The connectome's populations in order, without the aliases (``motor/actions``)."""
    return [(name, np.asarray(idx, dtype=np.int64))
            for name, idx in dict(brain.connectome.populations).items() if "/" not in name]


def _array(values: Any) -> np.ndarray:
    """A float64 copy of a backend's array, a tensor on a device included."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.array(values, dtype=np.float64, copy=True)


def _band(m: int, lo: float, hi: float, radius: float, centre: float = 0.0) -> np.ndarray:
    """``m`` points spread evenly over the band ``lo <= x <= hi`` of a sphere of ``radius``
    centred at ``(centre, 0, 0)``: a Fibonacci spiral along x, uniform by Archimedes."""
    x = lo + (hi - lo) * (np.arange(m) + 0.5) / m
    r = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    t = np.arange(m) * _GOLDEN
    return radius * np.stack([x, r * np.cos(t), r * np.sin(t)], axis=1) + np.array([centre, 0.0, 0.0])


def _motor_cap(m: int, slots: list[int] | None = None) -> np.ndarray:
    """The motor cortex on the front of the unit sphere: the slots in a grid on the cap, each
    slot's answers a tight ring (a triangle for a joint's three, a ring of nine for a leg's)."""
    if slots is None or sum(slots) != m:
        per = 3 if m % 3 == 0 else 1
        slots = [per] * (m // per)
    cols = int(np.ceil(np.sqrt(len(slots))))
    rows = int(np.ceil(len(slots) / cols))
    out = []
    for j, per in enumerate(slots):
        cy = -0.4 + 0.8 * (j % cols + 0.5) / cols
        cz = -0.4 + 0.8 * (j // cols + 0.5) / rows
        radius = 0.045 if per <= 3 else 0.07
        for k in range(per):
            a = 2.0 * np.pi * k / per
            y, z = cy + radius * np.cos(a) * (per > 1), cz + radius * np.sin(a) * (per > 1)
            out.append([np.sqrt(max(0.0, 1.0 - y * y - z * z)), y, z])
    return np.array(out[:m])


def connectome_layout(brain: Brain) -> np.ndarray:
    """One position per neuron, (n, 3), inside the ellipsoid ``AXES``; deterministic."""
    n = int(brain.connectome.n)
    pos = np.zeros((n, 3))
    rng = np.random.default_rng(0)
    pops = _populations(brain)
    modules = [name for name, _ in pops if name.startswith("module")]
    for name, idx in pops:
        m = len(idx)
        if name == "sensory":
            unit = _band(m, -1.0, 0.45, 1.0)                     # the rear surface
        elif name == "code":                                     # the learned code: a shell just inside the senses
            unit = _band(m, -0.95, 0.48, 0.88)
        elif name in modules:                                    # shells inside the senses
            k = modules.index(name)
            unit = _band(m, -0.9, 0.5, 0.74 - 0.12 * k)
        elif name == "association":
            unit = _band(m, -1.0, 1.0, 0.36, centre=-0.1)        # the core
        elif name == "prefrontal":
            unit = _band(m, -1.0, 1.0, 0.28, centre=0.38)        # the core, toward the front
        elif name == "motor":
            unit = _motor_cap(m, list(brain.describe()["layout"].get("slots", [])))   # the front cap
        else:
            unit = _band(m, -1.0, 1.0, 0.5)
        pos[idx] = unit * AXES + rng.normal(0.0, 0.01, (m, 3))
    return pos


def _fibres(pos: np.ndarray, pre: np.ndarray, post: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One curved fibre per synapse, (synapses, FIBRE_NODES, 3): from the target's soma to the
    source's, a quadratic curve whose control point sits outward of the chord, so fibres
    sweep along the surface rather than through the centre; the last node is the source."""
    a, b = pos[post] / AXES, pos[pre] / AXES                     # in the unit ball
    mid = 0.5 * (a + b)
    reach = np.maximum(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
    norm = np.linalg.norm(mid, axis=1) + 1e-9
    c = mid * ((1.05 * reach) / norm)[:, None] + rng.normal(0.0, 0.03, mid.shape)
    t = (np.arange(1, FIBRE_NODES + 1) / FIBRE_NODES)[None, :, None]
    curve = (1 - t) ** 2 * a[:, None] + 2 * t * (1 - t) * c[:, None] + t ** 2 * b[:, None]
    return curve * AXES


def spider_payload(brain: Brain) -> dict[str, Any]:
    """The whole connectome for the page's view: a cell per neuron, its skeleton the soma plus
    a fibre of FIBRE_NODES nodes per incoming synapse ending at the presynaptic soma;
    ``index`` is the neuron's index into the activity vector."""
    c = brain.connectome
    n = int(c.n)
    pre = np.asarray(c.pre, dtype=np.int64)
    post = np.asarray(c.post, dtype=np.int64)
    pos = connectome_layout(brain)
    fibres = np.round(_fibres(pos, pre, post, np.random.default_rng(1)), 3)
    pos = np.round(pos, 3)
    order = np.argsort(post, kind="stable")
    starts = np.searchsorted(post[order], np.arange(n + 1))
    pops = _populations(brain)
    classes = [name for name, _ in pops]
    labels = {name: _LABELS.get(name, name.replace("_", " ")) for name in classes}
    colors = {name: _COLORS.get(name, _MODULE_COLOR) for name in classes}
    counts = {name: int(len(idx)) for name, idx in pops}
    owner = np.empty(n, dtype=object)
    for name, idx in pops:
        owner[idx] = name
    cells = []
    for i in range(n):
        mine = order[starts[i]:starts[i + 1]]
        k = len(mine)
        xyz = np.concatenate([pos[i][None], fibres[mine].reshape(-1, 3)]).ravel().tolist()
        # node 0 is the soma; each fibre is a chain whose first node hangs off the soma
        parent = [-1] + [0 if j % FIBRE_NODES == 0 else j for j in range(k * FIBRE_NODES)]
        cells.append({
            "id": f"{owner[i]}:{i}", "class": str(owner[i]), "index": i, "n": 1 + k * FIBRE_NODES,
            "xyz": xyz, "parent": parent,
        })
    return {
        "format": CONNECTOME_FORMAT, "classes": classes, "labels": labels, "colors": colors,
        "counts": {"classes": counts, "neurons": n, "synapses": int(len(post))},
        "order": list(range(n)), "cells": cells,
    }


def learning_levels(post: np.ndarray, before: np.ndarray, after: np.ndarray, n: int) -> np.ndarray:
    """How much each neuron's incoming synapses changed, in [0, 1]: the view's formula, the
    change relative to the weight mass that was there."""
    change = np.bincount(post, weights=np.abs(after - before), minlength=n)
    mass = np.bincount(post, weights=np.abs(before), minlength=n)
    return np.sqrt(np.minimum(1.0, LEARNING_GAIN * change / (mass + LEARNING_FLOOR)))


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
        self.layout = layout_of(brain) if hasattr(brain, "describe") else DEFAULT_LAYOUT   # a test's stub has no layout
        self.owed: float | None = None  # an outcome kept: after a refusal, or while another policy acted
        self.refusals = 0
        self.moments = 0
        self.last_error: str | None = None
        connectome = getattr(brain, "connectome", None)                  # a test's stub has none
        self._post = None if connectome is None else np.asarray(connectome.post, dtype=np.int64)
        self._connectome: dict[str, Any] | None = None

    def connectome(self) -> dict[str, Any]:
        """The page's view payload, built once: the topology does not change in a life."""
        if self._connectome is None:
            self._connectome = spider_payload(self.brain)
        return self._connectome

    @classmethod
    def newborn(cls, seed: int = 0, genes: dict[str, Any] | None = None, inputs: int = INPUTS,
                layout: str = DEFAULT_LAYOUT, **where: Any) -> "SpiderBrain":
        return cls(compose(seed, genes, inputs=inputs, layout=layout, **where))

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
        # the synapses before this moment, for the view's learning flash (a copy: the compiled
        # brain is rebound on a learning moment, so its weights array is not revisited)
        before = _array(brain.brain.weights) if self._post is not None else None
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
        act = activity(brain)
        if act is not None and before is not None and reading.get("learned"):
            levels = learning_levels(self._post, before, _array(brain.brain.weights), int(brain.connectome.n))
            act["repair"] = [round(float(v), 2) for v in levels]
        return {
            "commands": joint_commands(action[0], self.layout),
            "answer": [int(c) for c in np.atleast_1d(action[0])],   # the brain's own answer, one choice per slot
            "refused": False,
            "mode": mode,
            "aroused": mode == "aroused",
            "sweeps": int(reading.get("sweeps", 0)),
            "learning_sweeps": int(reading.get("learning_sweeps", 0)),
            "want": float(arousal.want),
            "level": float(arousal.level),
            "ms": elapsed,
            "policy": self.policy,
            "activity": act,
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
            "layout": self.layout,
            "genes": self.brain.describe().get("genes", {}),
        }


class RandomPolicy:
    """The uniform-random baseline with the same body: one draw per slot, every moment (a
    uniform draw per leg of nine is the same distribution as a uniform draw per joint of
    three, so the baseline is the same under either layout)."""

    policy = "random"
    learns = False

    def __init__(self, seed: int = 0, layout: str = DEFAULT_LAYOUT) -> None:
        self.rng = np.random.default_rng(seed)
        self.layout = layout
        self.slots = slots_of(layout)
        self.moments = 0
        self.refusals = 0
        self.owed = None

    def moment(self, observation: np.ndarray, reward: float | None, done: bool = False) -> dict[str, Any]:
        self.moments += 1
        answer = [int(self.rng.integers(size)) for size in self.slots]
        return {
            "commands": joint_commands(answer, self.layout), "answer": answer,
            "refused": False, "mode": "random", "aroused": False, "sweeps": 0,
            "learning_sweeps": 0, "want": 0.0, "level": 0.0, "ms": 0.0, "policy": self.policy,
        }

    def has_pending(self) -> bool:
        return False

    def save(self, path: str | Path) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"policy": self.policy, "moments": self.moments, "layout": self.layout}


class FrozenPolicy:
    """A brain that answers greedily and never learns: the control for "kept learning"."""

    policy = "frozen"
    learns = False

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self.layout = layout_of(brain)
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
            "commands": joint_commands(action[0], self.layout),
            "answer": [int(c) for c in np.atleast_1d(action[0])],
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
            "layout": self.layout,
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
    inputs: int = INPUTS,
    layout: str = DEFAULT_LAYOUT,
) -> SpiderBrain | RandomPolicy | FrozenPolicy:
    """The policy by name; a newborn brain is composed for ``inputs`` senses (the life's
    senses, not the default: until 2026-11-12 ``--per`` changed the senses and not the brain,
    and every moment of such a life was refused) and read by ``layout``; a saved brain keeps
    the layout it was composed with."""
    where = backend_of(backend)
    if name == "random":
        return RandomPolicy(seed, layout)
    if name == "brain":
        if brain_path is not None and Path(brain_path).exists():
            return fitted(SpiderBrain.load(brain_path, where.get("backend")), inputs, brain_path)
        return SpiderBrain.newborn(seed, genes, inputs=inputs, layout=layout, **where)
    if name == "frozen":
        if brain_path is not None and Path(brain_path).exists():
            return fitted(FrozenPolicy.load(brain_path, where.get("backend")), inputs, brain_path)
        return FrozenPolicy(compose(seed, genes, inputs=inputs, layout=layout, **where))
    raise ValueError(f"policy {name!r}: brain, random or frozen")


def fitted(policy, inputs: int, path: str | Path):
    """The loaded policy, if its brain was composed for ``inputs`` senses; a saved brain's
    input count is fixed, and one fed another count refuses every moment. The mismatch is
    reported here, at the load, not moment by moment in a life that cannot act."""
    have = int(policy.brain.inputs)
    if have != int(inputs):
        raise ValueError(
            f"{path}: this brain was composed for {have} inputs, the senses give {inputs}; "
            f"start with the --per of the saved life, another --runs, or Reset the life"
        )
    return policy
