# Learning Spider

A spider with a body, a head and eight legs of two joints each, commanded joint by joint by one
continuing [Cadence](https://github.com/muellerberndt/cadence) brain, in a three-dimensional
world drawn in your browser. Nobody tells it how to walk. The world pays it for forward
progress, and that reward is the only teacher the brain has, while you watch and turn the
world around it. Whether it learns to walk is a measured question. The readings so far are in
[Measure it](#measure-it): over 30,000 moments the learning brain moves forward where its
controls do not, in two seeds of three, modestly; a reward hint keeps four feet on the ground
at the cost of half that distance.

Built with `cadence-net==0.97.1`.

## See it

```sh
sudo apt install python3-venv python3-pip                 # once, on Debian or Ubuntu
python3 -m venv .venv
.venv/bin/python -m pip install "cadence-net[accel]==0.97.1"   # PyTorch with CUDA, several GB
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python server.py
```

Open http://127.0.0.1:8080. Drag to turn the world horizontally, use the wheel to zoom. The
panel shows the moment, the brain's mode (aroused or routine), forward distance, the reward
just earned, arousal level and want, the sweeps of the last settle, milliseconds per moment,
and a strip of the brain's settled state: sensory inputs, association units, and for each of
the sixteen joints its three motor units with the chosen command outlined.

Buttons: pause and resume; recentre (the body returns to the origin, nothing else changes);
save and load (`runs/spider.npz` and its sidecar `runs/spider.json`); the policy that commands
the body, `brain`, `random` or a `frozen copy` of the live brain; the pace, 1× (20 moments a
second, the body's own clock), 2×, 5× or as fast as the machine goes.

The server autosaves every five minutes and at shutdown, and continues a saved life on the
next start, with the world's state and the outcome the brain had not yet received. Stopping
and starting the server is not a new spider. Delete `runs/` for a newborn.

For development against a checkout of the library: `.venv/bin/python -m pip install -e ../cadence`.

## What it demonstrates

- **One continuing brain in a body.** `Brain.compose(..., arousal=...)` once; then
  `brain.live(observation, reward=[r])` every moment for the rest of its life. Sixteen motor
  slots of three choices each command the sixteen joints directly. There is no controller,
  gait generator or inverse kinematics between the brain and the body.
- **Learning from the world's outcome only.** The reward is forward progress along the body's
  own heading, capped. No target pose, no phase signal, no shaped terms. Youth and arousal
  decide when the brain samples and learns; the application never gates learning itself.
  This is the lifecycle as built; the measurements say how far it has worked.
- **A world that needs a gait.** Only feet planted at both the previous and the current moment
  push the body. A foot in the air pushes nothing; a planted foot that does not move anchors
  the body; a closed cycle of joint angles with the same feet planted integrates to zero. To
  move, legs must lift, swing, re-plant and stroke in some pattern.
- **Exactly-once custody of every outcome.** A refusal keeps the outcome owed; switching to a
  control policy keeps it for the brain; a save writes it to the sidecar; a load delivers it.
  Tests prove each path.
- **Honest controls.** `random` is the baseline. `frozen` is a brain that answers greedily and
  never learns: a newborn, or a save/load copy of the live brain raised so far.

## How it is built

```
browser (three.js, z up)  <-- WebSocket, 30 frames/s at most --  aiohttp server
                          --> POST /api/control (pause, recentre, save, load, policy, pace)
                                            |
                              Life: one thread, 20 moments/s
                       observe -> policy.moment(x, reward) -> spider.step(commands) -> reward
```

Each moment (`DT = 0.05 s`) the life observes the body, hands the observation and the reward
of the previous command to the brain, applies the sixteen commands to the joints, moves the
body by what its planted feet did, and keeps the resulting reward for the next moment.

### The body (`spider/body.py`)

Ellipsoid body (semi-axes 0.15 × 0.10 × 0.06 m) at a fixed standing height of 0.10 m, a head
sphere in front, eight legs: hip swing in ±0.6 rad at 1.5 rad/s, knee lift in −0.5 to 1 rad at
2.5 rad/s, femur 0.10 m at a fixed rise, tibia 0.14 m. At knee 0 the foot is on the ground;
below 0 the leg presses into it and touches it where the tibia crosses the ground plane, so a
third of the knee's range keeps the foot planted. With at least two feet planted at both
moments, the body's velocity relaxes toward the negated rigid two-dimensional fit (translation
and yaw) of those feet's motion; with fewer, it decays.

The first world had the foot planted only at knee exactly 0 and needed three feet. Over 3000
moments the brain and the random policy both averaged under two planted feet and moved nothing
(three seeds each, forward −0.03 to +0.04 m, random −0.07 to +0.03 m): the reward almost never
appeared. The pressing range and the two-foot minimum are the smallest change that made the
reward reachable; they were chosen by that probe, not by a physical argument. Beyond a radius of 5 m the body is recentred, which is a change of place, not a
new life. No pitch or roll, no leg collisions, no friction model: the contact rule is a model
chosen so that sliding and swinging do not pay and a gait does.

### The senses (`spider/senses.py`)

Eighteen continuous readings (sixteen joint angles, forward speed, yaw rate) through a
`PopulationCode` with eight tuned inputs each, plus one raw flag per foot that says whether
it touches the ground: 152 inputs in `[0, 1]`. No constant drive input, no target.

### The reward (`spider/life.py`)

`reward = clip(40 × progress, 0, 0.5) − 0.1 × max(0, 4 − stance) − 0.1 × |turn|`, clipped
to ±0.5, where progress is the metres moved this moment along the heading held at the start of
the moment, stance is the number of feet planted through the moment (at its start and its end)
and turn is the radians the heading changed. Backward, sideways and standing still pay nothing.
`done` is never true.

The progress term is the outcome. The other two are hints, which is reward shaping: the world
is paying for a posture and a course, not for an outcome. The stance term was added after the
learner under progress alone walked on ever fewer feet (three to one on average over 30,000
moments) and stalled once calm. At 0.02 a foot the learner ignored it; at 0.1 it keeps four
feet down and walks less far. The yaw term was added because progress along the body's own
heading pays a circle as well as a straight line. Four planted feet and a straight course cost
nothing, so the only positive reward is still forward progress. The readings for each are below.
`--stance-pay 0 --yaw-pay 0` in the nursery restores the progress-only law.

The first law paid backward progress negatively (`clip(…, −0.5, 0.5)`). Under it the learning
brain lifted every foot within 10,000 moments and stood still, since a push could pay negative
while a foot in the air paid exactly zero; the readings are kept below. The forward-only law
was chosen by that comparison.

### The brain (`spider/brain.py`)

| Gene | Value | Origin |
| --- | --- | --- |
| inputs / motors | 152 / 48 (`slots=[3]*16`) | this body |
| modules | 64, 32 | chosen here; the Robot Arena uses 48, 24 on 32 inputs |
| working trace | amplitude 0.3, decay 0.1, scale 12 | Robot Arena founder |
| efference | off | Robot Arena founder |
| episodic memory | on, consolidation 0.05 | Robot Arena founder |
| recall memory | off | chosen here (a 0.97.1 default, off for cost) |
| sensory scale | 4 | Robot Arena founder, unmeasured on 152 coded inputs |
| lateral (motor inhibition) | −0.5 | Robot Arena founder |
| temperature | 0.3 | Robot Arena founder |
| actor eta / bias / lam / gamma / critic | 0.03 / 0.003 / 0.6 / 0.95 / 5 | Robot Arena founder |
| arousal | threshold 0.2, decay 0.9, tolerance 2, floor 0.1, fast 0.05, slow 0.005, heat 2, youth 300, value surprise 1, need 0.05 | Robot Arena founder |

328 neurons, 22,041 parameters. Candidate genes are set on the nursery with
`--gene name=value` (`--gene modules=48,24`, `--gene arousal.need=0`, `--gene recall=true`,
`--gene efference_amplitude=3`) and `--per N` for the population code.

### Backend

The brain is placed by the library's default rule, which keeps a brain of this size on the
compiled CPU path (NumPy float64, repeatable to the bit). CUDA is taken by name:
`BACKEND=torch .venv/bin/python server.py`, or `--backend torch` on the nursery. On this
machine (RTX 5070 Ti) the CPU path is faster for this one-row brain; see the readings below.
A brain saved on CUDA loads on the CPU.

## Measure it

```sh
.venv/bin/python -m spider.nursery --moments 3000 --policy brain --seed 0 --save runs/s0.npz
.venv/bin/python -m spider.nursery --moments 3000 --policy random --seed 0
.venv/bin/python -m spider.nursery --moments 3000 --policy frozen --seed 0            # a newborn, never learns
.venv/bin/python -m spider.nursery --moments 3000 --policy frozen --brain runs/s0.npz # the raised brain, frozen
.venv/bin/python -m spider.nursery --moments 3000 --policy brain --brain runs/s0.npz  # continue the saved life
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --stance-pay 0 --yaw-pay 0  # progress only
```

One line per block of 100 moments (forward metres, reward per moment, aroused share, sweeps
and learning sweeps per moment, refusals, ms per moment, mean feet planted, radians turned) and a JSON
summary (`--json path` writes it).

### Readings

All readings are from this machine (one CPU core per run, NumPy float64), 2026-11-11, with
the world and reward law this README describes. One run per seed; the random policy and the
frozen newborn do not read the reward, so their rows come from the same world under the
earlier symmetric law. No run refused a moment.

Forward metres over 3000 moments (150 s of the body's time):

| Policy | Seed 0 | Seed 1 | Seed 2 | Mean planted feet | ms / moment |
| --- | --- | --- | --- | --- | --- |
| brain (learns) | -0.029 | +0.287 | +0.362 | 3.6, 1.1, 2.9 | 4.4–5.2 |
| brain, actor rates 0 (samples, never learns) | -0.143 | +0.155 | -0.085 | 4.2, 1.9, 4.3 | 4.6–5.3 |
| random | -0.002 | +0.007 | -0.223 | 3.0 | 0.1 |
| frozen newborn (greedy) | -0.007 | +0.012 | -0.005 | 4.3 | 0.5 |
| frozen copy of the 3000-moment brain (greedy) | -0.005 | -0.019 | -0.013 | 4.0, 2.0, 3.0 | 0.5 |

At 3000 moments nothing is established: two seeds of the learning brain move a few
decimetres, one does not, and the spread between seeds is as large as the difference between
policies. The brain was aroused in 81–94% of these moments (youth 300, then sustained
surprise), so it learned at nearly every moment.

Over 30,000 moments (25 minutes of the body's time) the learning brain separates from its
controls:

| Policy, 30,000 moments | Seed 0 | Seed 1 | Seed 2 | Planted feet, first 5000 → last 5000 |
| --- | --- | --- | --- | --- |
| brain (learns) | +2.425 | +0.846 | +2.123 | 3.3 → 1.1, 1.1 → 0.6, 2.8 → 1.1 |
| brain, actor rates 0 | +0.375 | +1.016 | +0.160 | 4.1 → 4.1, 1.8 → 1.9, 4.3 → 4.5 |
| random | -0.288 | +0.259 | -0.078 | 2.8 → 2.9, 3.2 → 3.4, 3.0 → 3.1 |
| frozen copy of the 30,000-moment brain, 3000 moments | +0.014 | 0.000 | +0.056 | 1.2, 0.6, 1.2 |

The learning brain keeps moving through the whole life: seed 0 per 5000 moments +0.271,
+0.674, +0.293, +0.354, +0.412, +0.420; seed 2 +0.542, +0.439, +0.262, +0.432, +0.231,
+0.218; seed 1 +0.335, +0.003, +0.219, +0.173, +0.074, +0.043. It walks on fewer feet as
it ages (three to one planted on average) and slows in seeds 1 and 2. The brain that never
learns walks on four feet and drifts: seed 0 per 5000 moments -0.093, +0.119, +0.107,
+0.064, +0.128, +0.050. In two seeds of three the learner covers five to thirteen times the
distance of its no-learning twin; in seed 1 the twin covers more. Frozen greedy copies of the
raised brains hold one posture and stay put: what the raised brain does is in its sampling,
not in its greedy answer.

Continuing seed 0's saved brain for 6000 more moments (30,000 to 36,000) moved +0.096 m in
the first 1000 and 0.000 m after that. Its arousal fell to routine in that first block and
stayed there: a calm brain answers greedily, learns nothing, and its greedy answer here is a
posture that does not move; a reward of zero is then no surprise. The `live` law has no
boredom, so nothing wakes it. That is a reading about this founder's arousal genes meeting a
reward that is zero at rest, and the first open question for this project.

**Reading:** under the forward-only reward, reward learning produces forward motion that the
sampling of the same brain without learning, random commands and greedy frozen copies do not
produce, in two seeds of three over 30,000 moments; the gain is modest (two metres in 25
minutes, with a 0.2 m/s body), one seed does not show it, and the raised brain stalls once it
calms. It is a learning reading, not a demonstration of a gait.

The first reward law paid backward progress negatively and forward positively. Under it the
learning brain lifted its feet and stood still: over 30,000 moments seeds 0–2 moved +0.487,
+0.178 and +0.437 m, all of it in the first 5000–10,000 moments, while their planted feet fell
from 2.2, 0.8 and 1.6 to 0.6, 0.2 and 0.4; the no-learning twin moved +0.999, +1.285 and
+1.249 m on two to four feet. A push can pay negative while a foot in the air pays exactly
zero, and not touching the ground is the policy the actor converged on. The forward-only law
was chosen by that reading, with the same controls.

**The stance term.** The same comparison with the stance term, 30,000 moments, progress-only
rows repeated from the table above:

| 30,000 moments, forward metres (planted feet, first 5000 → last 5000) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| brain, progress only | +2.425 (3.3 → 1.1) | +0.846 (1.1 → 0.6) | +2.123 (2.8 → 1.1) |
| brain, 0.02 a missing foot | +2.279 (4.1 → 1.4) | +1.252 (1.1 → 1.0) | +1.476 (3.1 → 1.0) |
| brain, 0.1 a missing foot (shipped) | +0.600 (4.2 → 4.5) | +0.688 (2.4 → 4.0) | +1.024 (3.7 → 3.0) |
| actor rates 0, progress only | +0.375 (4.1 → 4.1) | +1.016 (1.8 → 1.9) | +0.160 (4.3 → 4.5) |
| actor rates 0, 0.02 | +0.178 (4.2 → 3.9) | +1.217 (1.9 → 1.9) | +0.715 (4.5 → 4.2) |
| actor rates 0, 0.1 | -0.271 (3.8 → 3.9) | +3.014 (2.0 → 2.0) | +0.518 (3.7 → 4.1) |

At 0.02 a foot the learning brain lifts its feet exactly as it did with no hint, from four
planted to about one, and walks about as far. At 0.1 a foot the hint takes: seeds 0 and 1 end
on four planted feet, seed 2 on three, and seed 1 learned the posture during its life (2.4 in
its first 5000 moments, 4.0 in its last). What it costs is distance: 0.6 to 1.0 m against 0.8
to 2.4 m with no hint. The term keeps the brain aroused in every moment (its reward stays
below zero while a foot is up), so the calm stall of the progress-only brain has not been
seen under it; that is not a measurement of the stall, which would need a continuation of a
brain that has reached four feet. The no-learning twin's rows are the same sampling under a
different reward and show the seed spread: seed 1's twin walks 3 m on two feet with no
learning at all. Frozen greedy copies of the three 0.1 brains over 3000 moments stand on five
to six feet and move +0.18, +0.03 and +0.06 m.

**Reading:** the posture hint is obeyed at 0.1 a foot and ignored at 0.02, and obeying it
halves the distance walked. The project's rule stands: a shaped term buys the posture it pays
for, and the brain pays for it with the outcome.

What a next step would measure, each against the same controls: a continuation of a four-foot
brain, for the calm stall; a way out of that stall
(`--gene arousal.need=` or a floor on the arousal level, read under `live`); a lower actor eta
or a shorter eligibility (`--gene lam=`); module sizes (`--gene modules=48,24`); and the
population code's resolution (`--per 16`).

CUDA (`--backend torch`, float32) ran the same brain at 21.9 ms per moment against
4.3 ms on the CPU, as the library's placement rule predicts for a one-row brain of
328 neurons; the default rule keeps it on the CPU. The first CUDA moment pays about 1.6 s of
warm-up.

Summaries of every run are in the `--json` files the nursery wrote; the commands above
reproduce them with the same seeds.

## Limits

- The contact rule is a model: no pitch or roll, no body mass, no leg collisions, no slipping
  except the decay with fewer than two feet planted. A gait that moves this body may not
  move a physical one.
- The reward is forward progress only; the brain is free to drift sideways and turn, and does.
- The readings above are one machine and three seeds; they do not establish a learning rate,
  an efficiency, or a property of the library.
- The population code's resolution (`per=8`), the module sizes and the sensory scale were
  chosen, not measured against alternatives.

## Folders

| Path | What |
| --- | --- |
| `spider/` | body, senses, brain, life, nursery, server |
| `static/` | the page; `static/vendor/three` is three.js r160 (MIT, licence included) |
| `tests/` | `python -m pytest tests -q` |
| `runs/` | saved lives (gitignored) |

GPL-3.0, as the library.
