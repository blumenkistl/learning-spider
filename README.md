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
a learning bar, and a strip of the brain's settled state: sensory inputs, association units,
and for each of the sixteen joints its three motor units with the chosen command outlined.

The learning bar runs from "Brain Stall" (0) to "Learning" (100). It is the share of the last
1000 moments the brain spent aroused, which under the arousal law is exactly when it learns;
a calm brain answers greedily and changes nothing. It reads the arousal law, not the gait: a
brain can be fully aroused and still walk no better. Random and frozen never learn, and the
bar says so.

Buttons: pause and resume; recentre (the body returns to the origin, nothing else changes);
save and load (`runs/spider.npz` and its sidecar `runs/spider.json`); the policy that commands
the body, `brain`, `random` or a `frozen copy` of the live brain; the pace, 1× (20 moments a
second, the body's own clock), 2×, 5× or as fast as the machine goes.

The server autosaves every five minutes and at shutdown, and continues a saved life on the
next start, with the world's state and the outcome the brain had not yet received. Stopping
and starting the server is not a new spider. Delete `runs/` for a newborn.

Environment variables for the server: `PORT` (8080), `POLICY` (`brain`), `SEED` (0), `PACE`
(20), `BACKEND` (auto), `RUNS` (`runs`), `STANCE_PAY` (0.1), `YAW_PAY` (0) and
`PROGRESS_WINDOW` (1), the reward's two hints and its window described under
[the reward](#the-reward-spiderlifepy).

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

Ellipsoid body (semi-axes 0.15 × 0.10 × 0.06 m) at a fixed standing height of 0.10 m, eight
legs: hip swing in ±0.6 rad at 1.5 rad/s, knee lift in −0.5 to 1 rad at 2.5 rad/s, femur 0.16 m
at a fixed rise, tibia 0.20 m, so a foot reaches 0.28 m out from its hip. Each leg swings
about its own resting yaw (`HIP_SPLAY`: the front pair 0.3 rad forward, the second pair 0.1,
the third 0.1 back, the rear pair 0.3 back), so the spider stands as a tarantula does rather
than as a star; a splayed leg's stroke pushes forward by the cosine of its yaw, 0.96 for
the front and rear pairs. A splay of 0.6 rad was measured first and cost the brain its
walking (the readings below); 0.3 is the measured compromise. The page draws that
body as a tarantula (front part, abdomen, coxae, bristles); only the hips, knees and feet are the
world's. Until the tarantula body the femur was 0.10 m and the tibia 0.14 m; the readings below
say which legs they were measured with. At knee 0 the foot is on the ground;
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

`reward = clip(40 × progress, 0, 0.5) − 0.1 × max(0, 4 − stance) − yaw_pay × |turn|`,
clipped to ±0.5, where progress is the metres moved this moment along the heading held at the
start of the moment, stance is the number of feet planted through the moment (at its start and
its end) and turn is the radians the heading changed. Backward, sideways and standing still pay
nothing. `done` is never true. `--window N` (`PROGRESS_WINDOW` for the server, default 1) replaces
progress with the mean progress of the last N moments, so a sway that moves the body forward and
back earns nothing over the window; the window is saved with the life. Its readings are under
"The windowed reward" below; the default stays 1.

The progress term is the outcome. The other two are hints, which is reward shaping: the world
is paying for a posture and a course, not for an outcome. The stance term was added after the
learner under progress alone walked on ever fewer feet (three to one on average over 30,000
moments) and stalled once calm. At 0.02 a foot the learner ignored it; at 0.1 it keeps four
feet down and walks less far. The yaw term exists because progress along the body's own
heading pays a circle as well as a straight line, but it ships switched off (`yaw_pay = 0`):
at 0.1 and 1 a radian the learner ignored it, and at 5 it obeyed by standing still. Four
planted feet cost nothing, so the only positive reward is still forward progress. The readings
for each are below. `--stance-pay` and `--yaw-pay` in the nursery, or `STANCE_PAY` and
`YAW_PAY` in the server's environment, set the doses; both at 0 is the progress-only law.

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
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --stance-pay 0  # progress only
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --yaw-pay 5     # charge turning too
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --window 20      # pay the mean progress of 20 moments
```

One line per block of 100 moments (forward metres, reward per moment, aroused share, sweeps
and learning sweeps per moment, refusals, ms per moment, mean feet planted, radians turned) and a JSON
summary (`--json path` writes it).

### Readings

All readings are from this machine (one CPU core per run, NumPy float64), 2026-11-11, with
the world and reward law this README describes. The tables up to the yaw term were measured
with the earlier short legs (femur 0.10 m, tibia 0.14 m); the long legs have their own table
at the end. One run per seed; the random policy and the frozen newborn do not read the reward,
so their rows come from the same world under the earlier symmetric law. No run refused a
moment.

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
posture that does not move. That reading took the stalled brain's reward to be zero; a later
inspection of a stalled brain ("The rocking", below) found it rocking in place for a steady
income instead, and this run's income was not inspected. Either way the `live` law has no
boredom, so nothing wakes a brain whose income meets its need.

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

**The yaw term.** The nursery's `turned` column is the unsigned sum of the radians the heading
changed each moment, so it counts the jitter of shuffling feet as well as circling. Every policy
turns a great deal by that measure: random turns 600, 510 and 580 radians in 30,000 moments,
the no-learning twin 460, 700 and 360, the brain under the stance term alone 360, 250 and 450,
about 0.013 radians a moment. The same comparison with the yaw charge added to the shipped
stance term, 30,000 moments, forward metres with the radians turned (first 5000 → last 5000):

| 30,000 moments | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| brain, no yaw charge | +0.60 (61 → 50) | +0.69 (88 → 27) | +1.02 (100 → 77) |
| brain, 0.1 a radian | +0.68 (75 → 82) | +0.67 (99 → 89) | +1.25 (77 → 63) |
| brain, 1 a radian | +0.85 (70 → 87) | +0.83 (94 → 85) | +1.26 (62 → 60) |
| brain, 5 a radian | +0.07 (54 → 55) | +0.49 (104 → 18) | +0.88 (68 → 35) |
| actor rates 0, 5 a radian | -0.09 (104 → 110) | +2.48 (116 → 88) | +0.85 (68 → 72) |

At 0.1 a radian the charge costs the brain about 0.001 a moment, a hundredth of the stance
term, and it is ignored: turning and distance are the same as without it. At 1 a radian it is
still ignored. At 5 a radian the brain obeys in seeds 1 and 2, turning a third as much in its
last 5000 moments as in its first, and it obeys by stopping: those seeds walk +0.28 and +0.49 m
in their first 5000 moments and +0.01 and +0.06 m in their last. Standing still turns nothing,
and under this charge it is the cheapest course. The term therefore ships switched off. A
charge on the unsigned turn cannot separate a circle from the jitter every gait has; a next
version would charge the net heading drift over a window of moments, which a straight jittering
gait does not pay.

**The long legs.** The tarantula body lengthened the femur from 0.10 to 0.16 m and the tibia
from 0.14 to 0.20 m, so a stroke of the same hip angle carries the body almost twice as far.
Fresh brains in that world, the shipped reward (stance 0.1, yaw off), 30,000 moments, against
random:

| 30,000 moments, forward metres (planted feet, first 5000 → last 5000) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| brain, long legs | +2.352 (3.8 → 1.9) | +3.205 (1.4 → 1.6) | +2.410 (3.3 → 2.4) |
| random, long legs | -0.705 (2.8 → 2.9) | +0.470 (3.2 → 3.4) | -0.149 (3.0 → 3.1) |

The learning brain walks two to three times as far as with the short legs under the same
stance term, and it does so by giving up feet again: it ends with two planted, not four. With
the long legs a stroke earns more than a missing foot costs, so the 0.1 dose that held four
feet down on the short legs no longer does; a four-foot posture on this body would need a
higher dose, which has not been measured. Random turns 720–830 radians in these runs, the brain
460–680. 3.5 ms a moment on the CPU. A brain saved with the short legs loads into this body,
since its senses are joint angles and not positions, but its habits were learned with short
legs; the readings above are from fresh brains.

**Four legs.** Is the eight-legged body too much for this brain? A four-legged comparison
body (two legs a side, eight joints, eight motor slots, 80 inputs; the stance floor half the
feet, two) was built as a `SPIDER_LEGS` option, measured against the eight-legged one in the
same world (long legs, no splay), fresh brains, 30,000 moments, and removed again after the
reading below; this table is its receipt. The no-learning twin is the same brain with every learning rate
zero (`--gene eta=0 --gene eta_bias=0 --gene eta_critic=0`): it samples its motors at the
same temperature and never changes. "Last 5000" is the distance in the final 5000 moments.

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| 4 legs, brain, progress only | +6.759 (+1.17; 1.2 → 2.0) | +0.715 (0.00; 1.2 → 2.0) | +0.095 (0.00; 1.7 → 2.0) |
| 4 legs, brain, stance 0.1 (floor two feet) | +0.339 (+0.02; 1.6 → 2.0) | +11.789 (+2.38; 1.6 → 1.9) | +0.867 (0.00; 1.0 → 2.0) |
| 4 legs, no-learning twin, progress only | +3.353 (+1.52; 0.9 → 1.8) | +4.654 (+1.16; 1.8 → 1.8) | +0.608 (0.00; 1.9 → 3.0) |
| 4 legs, random | -0.537 (1.4 → 1.6) | -0.850 (1.7 → 1.6) | +0.040 (1.6 → 1.7) |
| 8 legs, brain, progress only | +34.074 (+6.49; 3.4 → 3.4) | +0.817 (0.00; 0.8 → 2.0) | +1.304 (0.00; 3.2 → 3.1) |
| 8 legs, random | -0.705 (2.8 → 2.9) | +0.470 (3.2 → 3.4) | -0.149 (3.0 → 3.1) |

The reading is that four legs do not help. The four-legged learner's spread, 0.1 to 11.8 m,
is the same lottery as the eight-legged one, and in four of its six runs it is beaten by its
own no-learning twin, which walked 3.4 and 4.7 m in two seeds by sampling alone; the twin's
success is the body being easy to move by chance, not learning. What ends learning in both
bodies is the calm stall: in seven of the nine learning runs the last 5000 moments moved
nothing, the aroused share was zero and the planted count sat at exactly 2.00. "The rocking"
below says what that pose is doing. The one run that kept moving on
eight legs, seed 0 with progress only, walked 34 m and was still accelerating (6.5 m in its
last 5000, three to four feet down without a stance term): the best reading in this project
and ten times anything before it, from one seed of three. The eight-legged body is not the
limit; the stall is. 2 ms a moment for the four-legged brain, 2–3 ms for the eight-legged one.

**The splayed legs.** The resting yaw (`HIP_SPLAY`, front legs forward, rear legs back) was
measured at two sizes right after it was added, eight legs, fresh brains, 30,000 moments,
against the same runs on the star-shaped body above. "0.6" splays the front and rear pairs
0.6 rad and the middle pairs 0.2; "0.3" splays them 0.3 and 0.1:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| brain, progress only, splay 0.6 | +0.125 (+0.01; 4.0 → 4.0) | +0.161 (0.00; 2.5 → 2.5) | +0.627 (0.00; 3.2 → 3.0) |
| brain, progress only, splay 0.3 | +0.503 (0.00; 3.1 → 3.0) | +0.044 (0.00; 2.5 → 2.5) | +0.709 (0.00; 2.3 → 2.0) |
| brain, progress only, star (table above) | +34.074 | +0.817 | +1.304 |
| brain, stance 0.1, splay 0.6 | +0.492 (+0.01; 3.7 → 3.8) | +1.049 (+0.09; 1.7 → 3.4) | +1.341 (+0.27; 3.3 → 1.9) |
| brain, stance 0.1, splay 0.3 | +2.736 (+0.23; 3.4 → 2.9) | +3.339 (+0.59; 2.3 → 1.4) | +1.938 (+0.20; 2.8 → 2.9) |
| brain, stance 0.1, star (long legs table) | +2.352 | +3.205 | +2.410 |
| random, splay 0.6 | -0.394 (2.8 → 2.9) | -0.151 (3.2 → 3.4) | -0.446 (3.0 → 3.1) |
| random, splay 0.3 | -0.583 (2.8 → 2.9) | +0.172 (3.2 → 3.4) | -0.290 (3.0 → 3.1) |

At 0.6 all six learning runs fall below the matching run on the star body, by two to five
times under the stance term; random is unchanged. The plausible mechanism is geometric: a
splayed leg's stroke pushes partly sideways and partly into a turn, and those parts cancel
only when the planted set is mirrored, so the same uncoordinated strokes that moved the star
body forward now move the splayed one sideways or round. At 0.3 the stance-term runs are back
at the star body's level (1.9–3.3 m against 2.4–3.2) and the progress-only runs match the
star body's two ordinary seeds (0.5–0.7 against 0.8–1.3); the star body's 34 m seed has no
counterpart in either splay, and one seed cannot say whether that is the splay or the
lottery. The progress-only runs stall calm on every body; the stance runs stay aroused the
whole 30,000 moments (aroused share 1.00). Three seeds each, so readings. 0.3 is the body's
default; the readings before this paragraph were taken on the star body. The star body is
every `HIP_SPLAY` entry at zero.

**The rocking.** A stalled brain was inspected (seed 1, progress only, splay 0.3, replayed
to 16,000 moments). It is not resting: in its last 200 moments every moment's progress has
the opposite sign of the one before, with equal size, so the net is zero while one knee
toggles between pressing and resting each moment. Pressing moves the point where the tibia
meets the ground, the rigid fit reads that as the foot moving, and the body sways. The
forward half of each sway pays forty times its progress and the backward half pays nothing,
so the sway earns 0.058 a moment with no displacement. That income is above the founder need
of 0.05, the want is 0.01, and the brain stays calm. The calm stall is a reward exploit of
the forward-only law, not an idle brain; it is what the earlier "stalled" rows most likely
hold too, though only this one was inspected.

**A raised need.** The founder need is 0.05 reward a moment. Raising it keeps a rocking brain
wanting, since rocking pays less than the need. Eight legs, splay 0.3, fresh brains, 30,000
moments; "r/moment" is the reward over the last 5000:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| progress only, need 0.05 (table above) | +0.503 (0.00; 3.1 → 3.0) | +0.044 (0.00; 2.5 → 2.5) | +0.709 (0.00; 2.3 → 2.0) |
| progress only, need 0.1 | +1.207 (+0.23; 2.5 → 1.0; 0.016) | +1.355 (+0.07; 1.3 → 1.0; 0.001) | +2.203 (+0.11; 1.7 → 0.7; 0.010) |
| progress only, need 0.2 | +2.332 (−0.01; 2.2 → 0.5; 0.003) | +1.251 (0.00; 0.7 → 0.3; 0.000) | +1.687 (+0.16; 2.2 → 0.5; 0.004) |
| stance 0.1, need 0.05 (table above) | +2.736 (+0.23; 3.4 → 2.9) | +3.339 (+0.59; 2.3 → 1.4) | +1.938 (+0.20; 2.8 → 2.9) |
| stance 0.1, need 0.1 | +1.292 (+0.09; 3.3 → 3.0; −0.081) | +1.778 (+0.07; 2.3 → 2.6; −0.121) | +1.858 (+0.03; 3.6 → 3.2; −0.064) |

The raised need does what it says: every run stays aroused for all 30,000 moments, the
rocking income is gone (0.00–0.016 a moment at the end) and the progress-only distance
doubles or triples. It does not produce a walker. Under permanent arousal the brain drifts
to lifting its feet: it ends on one planted foot at need 0.1 and on half a foot at 0.2, where
nothing pushes, with the last 5000 moments moving 0 to 0.2 m. Under the stance term the
raised need walks less than the founder (1.3–1.9 m against 1.9–3.3). One zero-progress end
state, calm rocking, is exchanged for another, aroused standing on air. 4–5 ms a moment,
since every moment learns. The need stays at 0.05; the reward law is the open question, and
the next measurement is a law that pays net displacement over a window, which rocking cannot
earn and standing cannot either. That measurement follows.

**The windowed reward.** `--window N` pays the mean progress of the last N moments instead
of this moment's, so a sway of equal halves earns nothing once the window holds a whole
cycle (a test, `test_windowed_progress_pays_a_walk_and_not_a_sway`, checks that a steady walk
earns the same and a sway earns nothing). Eight legs, splay 0.3, fresh brains, 30,000 moments;
"aroused" is the share of aroused moments over the last 5000:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| progress only, window 1 (table above) | +0.503 (0.00; 3.1 → 3.0; 0.00) | +0.044 (0.00; 2.5 → 2.5; 0.00) | +0.709 (0.00; 2.3 → 2.0; 0.00) |
| progress only, window 20 | +1.661 (+0.63; 3.8 → 2.1; 0.91; 0.018) | +0.952 (+0.20; 1.3 → 1.1; 0.96; 0.011) | +1.766 (+0.58; 3.9 → 1.8; 0.91; 0.019) |
| progress only, window 50 | −0.269 (+0.24; 3.8 → 3.3; 0.92; 0.015) | +0.930 (+0.16; 1.6 → 2.0; 0.99; 0.010) | +1.334 (+0.78; 3.9 → 3.1; 0.90; 0.017) |
| stance 0.1, window 1 (table above) | +2.736 (+0.23; 3.4 → 2.9; 1.00) | +3.339 (+0.59; 2.3 → 1.4; 1.00) | +1.938 (+0.20; 2.8 → 2.9; 1.00) |
| stance 0.1, window 20 | +0.771 (+0.06; 4.3 → 4.9; 1.00; 0.004) | +0.231 (−0.02; 4.1 → 4.7; 1.00; −0.014) | +0.893 (+0.11; 4.6 → 4.6; 0.99; −0.001) |
| random (table above) | −0.583 | +0.172 | −0.290 |

The window removes the calm stall: no progress-only run under a window goes calm, every one
is still aroused and still moving in its last 5000 moments, and the progress-only distance
grows two to twenty times over window 1 (0.0–0.7 m to 1.0–1.8 m at window 20). It does not
produce a better walker than the stance law. With the stance term the window halves to
quarters the distance (0.2–0.9 m against 1.9–3.3): the brain keeps four to five feet down,
its reward sits at zero or below, and the window's late, averaged credit is a weaker teacher
than the moment's own progress. Window 50 is no better than 20 and its seed 0 ends behind
its start. The windowed law trades the rocking exploit for slow, aroused learning, which is
a different stall, not a gait. The default stays window 1 with stance 0.1, since that is the
law that walked farthest; `--window` stays as an opt-in with this receipt. 4–5 ms a moment,
since every moment learns.

What a next step would measure, each against the same controls and on the eight-legged body:
a longer life under window 20 progress only, the one law whose runs were still moving at
30,000 moments; a continuation of the 34 m brain;
a course hint on the net heading drift over a window rather than the unsigned turn; a lower
actor eta or a shorter eligibility (`--gene lam=`); module sizes (`--gene modules=48,24`); and
the population code's resolution (`--per 16`).

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
