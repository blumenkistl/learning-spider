# Learning Spider

A spider with a body, a head and eight legs of two joints each, commanded joint by joint by one
continuing [Cadence](https://github.com/muellerberndt/cadence) brain, in a three-dimensional
world drawn in your browser. Nobody tells it how to walk. The world pays it for forward
progress, and that reward is the only teacher the brain has, while you watch and turn the
world around it. Whether it learns to walk is a measured question. The readings so far are in
[Measure it](#measure-it): over 30,000 moments the learning brain moves forward where its
controls do not, in two seeds of three, modestly; a reward hint keeps four feet on the ground
at the cost of half that distance.

Built with `cadence-net==0.99.0` (0.97.1 until 2026-11-13; 0.98 and 0.99 change no default, and a brain acts, learns and saves to the bit as on 0.97.1).

## See it

```sh
sudo apt install python3-venv python3-pip                 # once, on Debian or Ubuntu
python3 -m venv .venv
.venv/bin/python -m pip install "cadence-net[fast]==0.99.0"    # numba and scipy: the compiled CPU path
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python server.py
```

Currently best results with:

```sh
.venv/bin/python server.py --gene gamma=0.99 --gene entropy=0.1 --gene arousal.need=0.1
```

(Genes are set at birth: these take effect on a newborn, not on a life continued from `runs/`;
on the page, the same words in the "next birth" gene field and Reset. Measured 2026-11-13:
4–8 forward metres over 30,000 moments on seeds 0–2, all three still walking and learning at
150,000, against 0.2–1.5 m for the founder genes; the walk circles.)

Open http://127.0.0.1:8080.

**On a new computer**, from nothing to the page, with the life carried over:

```sh
git clone https://github.com/blumenkistl/learning-spider.git ~/git/github/learning-spider
cd ~/git/github/learning-spider
sudo apt install python3-venv python3-pip                 # Debian or Ubuntu; WSL2 too
python3 -m venv .venv
.venv/bin/python -m pip install "cadence-net[fast]==0.99.0"    # numba and scipy: the compiled CPU path
.venv/bin/python -m pip install -r requirements.txt             # aiohttp for the server, pytest for the tests
.venv/bin/python -m pytest tests -q                             # 76 pass; one CUDA test is skipped without a card
.venv/bin/python server.py
```

`ModuleNotFoundError: No module named 'aiohttp'` on `server.py` means the second `pip install`
was skipped: the library's package brings only the brain, the server and the tests come from
`requirements.txt`. `[fast]` brings numba and scipy for the compiled CPU path, the default
backend; `cadence-net[accel]==0.99.0` adds PyTorch with CUDA (several GB) for `--backend torch`,
and a computer without an NVIDIA card leaves it out and loses nothing. To continue the life rather than start a newborn, copy the old computer's
`runs/` (gitignored; `spider.npz` and `spider.json`, or the stems a `--runs` names) into the
new checkout before the first start: the server loads what it finds there and the sidecar's
law outranks the flags. A life saved on CUDA loads on the CPU. The nursery's saves and
summaries live wherever `--save` and `--json` put them and travel the same way. Drag to turn the world horizontally, use the wheel to zoom. The
panel shows the moment, the brain's mode (aroused or routine), forward distance, the speed with
a bar under it (the moment's speed filled from the centre on a scale of a full stroke each way, the
last block's mean as a tick), the reward
just earned, the law it is paid under (progress or a window, the course with the metres along it, the high-water mark with its
current mark, any shaping term), arousal level and want, the sweeps of the last settle, milliseconds per moment,
a learning bar, and a strip of the brain's settled state: sensory inputs, association units,
and for each of the sixteen joints its three motor units with the chosen command outlined.

On the right, the whole brain in three dimensions: every one of its 328 neurons and 18,016
synapses, drawn by the connectome demo's view (`static/vendor/brain_view.js`, copied unchanged
from cadence-demos). The layout is not an anatomy, only shaped like one: the neurons fill an
ellipsoid, the senses on its rear surface, the module a shell inside them, the association and
prefrontal regions in the core, the motor cortex a cap on the front as sixteen triplets; each
neuron's fibres are its incoming synapses, one curved fibre from every presynaptic soma. Blue
is the settled state of the moment, the same vector the strip shows (green until 2026-11-13). Gold is learning: on a
moment the brain learned, each neuron flashes by how much its incoming synapses changed,
relative to the weight that was there (the demo's formula, gain 40), and the flash fades over
the following frames. A calm brain is blue only; the random control leaves the view at the
last brain state. Drag to orbit, wheel to zoom, `fit` returns to the resting view, `spin`
toggles the slow turn. The payload (`/api/brainview`, about 1.5 MB) is built once per brain;
the per-moment cost of the flash is a copy of the weights before `live` and one difference
after a learning moment, about 0.1 ms on this machine, inside the ms/moment the panel shows.
Under it, "The world": the colours of the ground disc and of the grid, the light on the body
and the ground (sky and sun together, 0.3× to 3× the page's own), with `default` for the page's
own. They are how the world is drawn, not a law of it (the brain senses joints, feet and
motion, never a colour or a light), and the choices are kept in the browser, not with the life.

The learning bar runs from "Brain Stall" (0) to "Learning" (100). It is the share of the last
1000 moments the brain spent aroused, which under the arousal law is exactly when it learns;
a calm brain answers greedily and changes nothing. It reads the arousal law, not the gait: a
brain can be fully aroused and still walk no better. Random and frozen never learn, and the
bar says so.

Buttons: pause and resume; recentre (the body returns to the origin, nothing else changes);
save and load (`runs/spider.npz` and its sidecar `runs/spider.json`); reset, which asks for
confirmation and then ends the life: a newborn brain takes the body, the world
returns to the origin, every counter is zero and the saved life is deleted; the policy that commands
the body, `brain`, `random` or a `frozen copy` of the live brain; the pace, 1× (20 moments a
second, the body's own clock), 2×, 5× or as fast as the machine goes; the HUD's pace line
shows the pace asked for and the moments a second the loop actually makes.

Every reading, button and setting on the panel explains itself on mouse-over, with the measured dose where one exists. Under the buttons, a collapsed `settings` line opens two columns. **law, live** holds the
reward's nine settings (the stance and yaw hints, the window, the high-water mark and its leak,
the pay per metre, the backward charge, the course and the strict course) and, beside them, the world's drift (a
current the body loses ground to; "The drift" below), and `Apply law` changes them from the next moment on (a number typed past a field's range is set back to its end, 40 in backward pay to 1, before it is read);
the outcome already earned is delivered as it was earned, and the same law is no change. A life
whose law changed is not a life under one law: it is a schedule, and the page says so; the
`drift` button under the pace switches the current on at the dose the field names (0.002 m/s
when it names none) or off, and counts the same way. Every
change is noted ("law changed at moment N: …, this life is a schedule"), counted on the law
line ("a schedule: changed 2×"), reported by `/api/describe` as `law_changes` and saved with
the life, so a continued life keeps the law it was last paid under, whatever flags the server
was started with (a note says which). **next birth, on Reset** holds what Reset gives birth
to: the seed, the motor model, the motor layout (a slot per joint or per leg; measured in
"A slot per leg"), whether the servos' setpoints are sensed (target motors only),
the reward learner's three options from cadence's docs/reward.md as fields of their own (the
actor's `hold`, its `entropy`, `maturity`; measured in "The reward learner's options"), the two
observation codes of cadence's docs/senses.md as fields too (a fixed sparse `code`, a
`learned code` region; measured in "The observation codes") and
the newborn's other genes in the nursery's words (`gamma=0.99 eta_critic=20 arousal.need=0
modules=48,24 recall=true`; empty is the founder). Genes are composed once, at birth, which is
why they change nowhere else; the live brain's genes and seed are shown under the field. The
confirmation names the birth, and the page reloads on the new life. The sidecar carries the
seed, the senses, the layout and the gene overrides of such a birth, so the server continues it on the next
start whatever its flags say; a brain file without its sidecar still needs the senses it was
composed for.

The server autosaves every five minutes and at shutdown, and continues a saved life on the
next start, with the world's state and the outcome the brain had not yet received. Stopping
and starting the server is not a new spider. Delete `runs/` for a newborn.

The server takes the nursery's flags (`.venv/bin/python server.py --help`): `--port` (8080),
`--policy` (`brain`), `--seed` (0), `--pace` (20), `--backend` (auto), `--runs` (`runs`),
`--autosave-minutes` (5), `--gene` and `--per` for a newborn brain, and the reward's settings
`--stance-pay` (0), `--yaw-pay` (0), `--window` (1), `--high-water` (off), `--old-ground` (0),
`--pay-per-metre` (40), `--backward-pay` (0), `--course` (off) and `--strict` (off), the two hints, the window, the high-water
mark and its leak, the gain, the backward charge, the course and the strict course
described under [the reward](#the-reward-spiderlifepy), `--drift` (0) for the world's current,
set and saved with the law, `--motors`
(`target` since 2026-11-12, `rate` before) for a newborn's body, `--layout` (`joint`; `leg` reads
a newborn brain's answer a leg at a time, [the brain](#the-brain-spiderbrainpy)), `--setpoints` (off) to sense the servos' setpoints too
and `--compass` (off) to sense the heading, `--ground` (off) to sense the velocity over the
ground. An environment variable of the same name (`PORT`, `POLICY`, `SEED`, `PACE`, `BACKEND`, `RUNS`,
`AUTOSAVE_MINUTES`, `STANCE_PAY`, `YAW_PAY`, `PROGRESS_WINDOW`, `HIGH_WATER`, `OLD_GROUND`, `PAY_PER_METRE`,
`BACKWARD_PAY`, `COURSE`, `MOTORS`, `LAYOUT`, `SETPOINTS`, `COMPASS`, `GROUND`) sets a flag's
default. A saved life is continued with its own genes, seed, senses, motors, layout and law, each
noted when a flag asked for another (a life saved before the sidecar carried the law, the seed
and the senses takes them from the flags, as before), and the high-water mark of a life saved
without one starts where the body stands. A brain file without its sidecar has fixed senses:
a start with another `--per` over it is refused with a message naming both input counts (not
a life that refuses every moment), and the file is left as it was; Reset on the page, or
another `--runs`, gives the new senses a newborn.

For development against a checkout of the library: `.venv/bin/python -m pip install -e ../cadence`.

## What it demonstrates

- **One continuing brain in a body.** `Brain.compose(..., arousal=...)` once; then
  `brain.live(observation, reward=[r])` every moment for the rest of its life. Sixteen motor
  slots of three choices each command the sixteen joints directly (or, under `--layout leg`,
  eight slots of nine, one per leg, the pairs of its two joints' choices: the same sixteen
  commands, chosen a leg at a time). There is no controller,
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
below 0 the knee presses the foot against it and the foot stays where a resting knee puts it,
so a third of the knee's range keeps the foot planted and a press moves nothing. (Until
2026-11-12 the pressed leg touched the ground where the tibia crossed the plane, a point that
slid in under the knee as the press deepened; "The held foot" below has the readings of
the change.) The feet planted at both moments carry the body when they support it: at least
one on each side and three in all (`supported` in `spider/body.py`); then the body's velocity
relaxes toward the negated rigid two-dimensional fit (translation and yaw) of those feet's
motion. Standing on one side's feet alone, or on fewer than three, the body is down: its
velocity decays and nothing pushes it. (Until 2026-11-12 any two planted feet carried it,
so a body on two left feet stood and pushed; "The support rule" below has the readings of
the change.) Neighbouring legs of a side cannot cross: once the leg behind has turned 0.2 rad
further forward than the leg in front of it, the two stop against each other, each hip
pushed back half the excess (`separate` in `spider/body.py`; `LEG_CONVERGE`). At that
angle the two legs' lines meet 0.40 m from the hips and a foot reaches at most 0.35 m, so
the stop sits where the legs would touch, with the lifted and the planted leg bound alike.
The split keeps both hips in their range (two legs converging past the stop have a mean
swing inside ±0.4, and the split puts them 0.2 either side of it); a few passes settle a side. (Until 2026-11-12 the hips were independent and the legs turned
through each other; "The separated legs" below has the readings of the change.)

The first world had the foot planted only at knee exactly 0 and needed three feet. Over 3000
moments the brain and the random policy both averaged under two planted feet and moved nothing
(three seeds each, forward −0.03 to +0.04 m, random −0.07 to +0.03 m): the reward almost never
appeared. The pressing range and the two-foot minimum were the smallest change that made the
reward reachable; they were chosen by that probe, not by a physical argument. The minimum is
three feet across both sides again since 2026-11-12, this time by the physical argument that
one side's feet cannot hold a body up, with the pressing range keeping it reachable. Beyond a radius of 5 m the body is recentred, which is a change of place, not a
new life. No pitch or roll, no leg thickness beyond the stop, no friction model: the contact rule is a model
chosen so that sliding and swinging do not pay and a gait does.

The joints take the brain's three answers in one of two motor models, the body's for its life
(`--motors`, `MOTORS` for the server; a saved life keeps its own and a flag that differs is
noted, not applied). Under `rate`, the founder until 2026-11-12, `decrease` and `increase` move the joint by its
rate for the moment and `hold` leaves it where it is: a full swing needs the same answer
sixteen moments running. Under `target` each joint is a position servo: `decrease` sets its
setpoint to the low end of the range, `increase` to the high end, `hold` keeps the setpoint,
and the joint travels toward it at the same rate, so one answer followed by holds drives the
swing. The servo chooses nothing and knows no gait, but it is a different model from the
founder's, named in every reading made under it; "The position commands" below has them.
Since 2026-11-12 `target` is the default of a newborn body (`DEFAULT_MOTORS` in `spider/body.py`,
the nursery's and the server's `--motors`, the page's next birth), because it is the model under
which every life that walked farther than a metre was raised; `--motors rate` names the
founder's. The readings above that moment are under `rate` unless they say otherwise, and
every reading since names its model. The page's `motors` row shows which model the body has.

### The senses (`spider/senses.py`)

Eighteen continuous readings (sixteen joint angles, forward speed, yaw rate) through a
`PopulationCode` with eight tuned inputs each, plus one raw flag per foot that says whether
it touches the ground: 152 inputs in `[0, 1]`. No constant drive input, no target.

Under the position commands (`--motors target`) the body carries sixteen setpoints the brain
cannot see in the angles alone: a joint still on its way reads the same as a joint that was
never sent. `--setpoints` adds them as sixteen more readings through the same code (280
inputs): the brain senses what it last told each servo, which is its own state, not the
world's. It is an option of the senses, measured under [the setpoint senses](#the-setpoint-senses),
and a saved brain's senses are fixed as with `--per`.

The brain senses its yaw rate and nothing of where it is pointed, which is all the plain law
asks of it. `--compass` adds the heading as a point on the circle, its cosine and sine, through
the same code (168 inputs): the world's heading, 0 the heading at birth, not the course. It is
the sense the course law can be read by (a brain without it is paid for a line it cannot
sense), an option of the senses like the setpoints, named in every reading made under it and
measured under [the course](#the-course); the page's "next birth" has it as `compass`.

`--ground` adds the ground sense: the body's velocity over the ground in the world's frame,
x and y in m/s, through the same code (168 inputs; 184 with the compass). The forward speed
and the compass give the brain its speed along its heading and the heading, and what the
course law pays, the motion along a fixed world heading, is their product, which a code of
tuned inputs read by one layer cannot take; the ground sense reads that motion from the
world as two readings, and the course's pay is a fixed weighting of them, its cosine and
sine. It reads the world and not the law: a body without a course has the same two
readings. It is an option of the senses like the compass, named in every reading made under
it and measured under [the ground sense](#the-ground-sense); the page's "next birth" has it
as `ground`.

### The reward (`spider/life.py`)

`reward = clip(40 × progress, 0, 0.5) − stance_pay × max(0, 4 − stance) − yaw_pay × |turn|`,
clipped to ±0.5, where progress is the metres moved this moment along the heading held at the
start of the moment, stance is the number of feet planted through the moment (at its start and
its end) and turn is the radians the heading changed. Backward, sideways and standing still pay
nothing. `done` is never true. `--window N` (`PROGRESS_WINDOW` for the server, default 1) replaces
progress with the mean progress of the last N moments, so a sway that moves the body forward and
back earns nothing over the window; the window is saved with the life. Its readings are under
"The windowed reward" below; the default stays 1. `--high-water` (`HIGH_WATER=1` for the
server, default off) pays new ground only: the progress term is the metres the body's forward
distance has gained beyond the farthest it ever reached, so a sway earns on its forward half
once and never again, while every step onto new ground pays at once. The mark is the life's,
saved in the sidecar and loaded with it (`mark` in the summary and the state); a save made
before the mark existed starts it where the body stands. Its readings are under "The
high-water mark" below. `--old-ground S` (`OLD_GROUND`, default 0) lets the mark leak: under
`--high-water`, progress on ground already reached is paid the share S of what the plain law
would pay, new ground all of it; 0 is the mark, 1 the plain law, and the readings between are
under "The leaking mark" below. `--pay-per-metre G` (`PAY_PER_METRE` for the server, default 40) sets
the progress term's gain under the same cap: at 40 one planted stroke moment (0.01 m) pays 0.4,
all but the cap, so four pushing feet pay what one does; at 10 the cap is four such moments
away. Its readings are under "The pay per metre" below; the default stays 40.
`--backward-pay S` (`BACKWARD_PAY` for the server, default 0) charges a metre backward the
share S of what a metre forward pays, under the same cap: at 1 a jitter nets nothing and a
stroke pays, at 0 backward is free. The first law charged backward in full and was left
because the brain lifted every foot to avoid it, on a body where that was free; "The backward
pay" below has it measured again on this body. The default stays 0.
`--course H` (`COURSE` for the server, default off) is the course law: the progress term is
the moment's motion along one fixed heading of the world, H radians (0, or `birth`, is the
heading at birth), instead of along the body's own heading. It is not a hint: the same outcome,
the body's motion, measured along one line instead of a turning one, so that a circle nets
nothing over a round and only a line pays; backward along the course is free or charged as
`--backward-pay` says, and the window and the high-water mark apply to the distance along the
course. The page's law panel sets it live as the heading the body holds at that moment ("keep
going this way"), which starts the metres along it and the mark at zero; the heading is shown
on the law line with the metres along it, saved and loaded with the life. The compass above is
the sense that heading can be read from. Its readings are under "The course" below; the default
stays off.

`--strict` (`STRICT` for the server, default off) is the strict course: under the course law,
the progress term is the lesser of the moment's motion along the course and along the body's
own heading. The course law alone pays the motion along its line whichever way the body
points, and a raised brain on the page learned to go down its course tail first (760,000
moments, 136 m from birth, the forward metres falling while the course metres rose); under
the strict course that is backward along the nose, free or charged as `--backward-pay` says,
nose first off the course pays little, and only nose first on the course pays in full. It is
not a hint: the same outcome measured twice, the lesser paid. It needs a course (set without
one it is refused; a course switched off switches it off) and is set live on the page's law
panel like the rest, shown on the law line as `strict`, saved and loaded with the life. The
default stays off.

`--drift D` (`DRIFT` for the server, default 0) is not a setting of the pay but of the world:
a current of D metres a second that the body loses backward along its own heading every
moment, planted or not, on top of what its feet did (`drift` in `spider/body.py`). The
reward is still the body's motion, which now includes the ground lost: under the plain law
backward is free, so the current costs a standing or swaying body nothing in pay and takes
its ground, and a stroke pays what it made less the current's share of the moment. It is the
world supplying a problem, not a hint, but a life on a current is a different world from a
life on still ground, so the drift is set, shown, saved and counted with the law (a change
live is a law change, and the life a schedule) and named in every reading made under it. The
measured dose is 0.002 m/s, 3 m of ground over 30,000 moments, below the speed of the lives
that walked above; its readings are under "The drift" below; the default stays 0.

The progress term is the outcome. The other two are hints, which is reward shaping: the world
is paying for a posture and a course, not for an outcome. Both ship switched off
(`stance_pay = 0`, `yaw_pay = 0`), so the shipped law is progress only. The stance term was
added after the learner under progress alone walked on ever fewer feet (three to one on
average over 30,000 moments) and stalled once calm. At 0.02 a foot the learner ignored it; at
0.1 it keeps four feet down on the star body and walks less far there, while on the splayed
body it is the law that walked farthest (readings below). It was the default until the
project chose the unshaped law as the one to ship, with the dose kept as an option. The yaw
term exists because progress along the body's own heading pays a circle as well as a straight
line: at 0.1 and 1 a radian the learner ignored it, and at 5 it obeyed by standing still.
Four planted feet cost nothing under either hint, so the only positive reward is forward
progress. The readings for each are below. `--stance-pay` and `--yaw-pay` in the nursery, or
`STANCE_PAY` and `YAW_PAY` in the server's environment, set the doses; 0.1 is the measured
stance dose.

The first law paid backward progress negatively (`clip(…, −0.5, 0.5)`). Under it the learning
brain lifted every foot within 10,000 moments and stood still, since a push could pay negative
while a foot in the air paid exactly zero; the readings are kept below. The forward-only law
was chosen by that comparison.

### The brain (`spider/brain.py`)

| Gene | Value | Origin |
| --- | --- | --- |
| inputs / motors | 152 / 48 (`slots=[3]*16`; 72, `[9]*8`, under `--layout leg`) | this body |
| modules | 64, 32 | chosen here; the Robot Arena uses 48, 24 on 32 inputs |
| working trace | amplitude 0.3, decay 0.1, scale 12 | Robot Arena founder |
| efference | off | Robot Arena founder |
| heard history | none (the brain hears this moment only) | 0.97.1's default without recall |
| episodic memory | on, consolidation 0.05 | Robot Arena founder |
| recall memory | off | chosen here (a 0.97.1 default, off for cost; measured in "The recall memory and the temperature") |
| sensory scale | 4 | Robot Arena founder, unmeasured on 152 coded inputs |
| lateral (motor inhibition) | −0.5 | Robot Arena founder |
| temperature | 0.3 | Robot Arena founder |
| actor eta / bias / lam / gamma / critic | 0.03 / 0.003 / 0.6 / 0.95 / 5 | Robot Arena founder |
| actor hold / entropy | off (`hold=none`) / 0 | cadence docs/reward.md, the library's founder values; measured in "The reward learner's options" |
| maturity (scale, recovery) | off (10, 200,000) | cadence docs/reward.md, the library's founder values; measured in "The reward learner's options" |
| observation code / learned code (fields) | off / off (none): the cortex reads the coded senses themselves | cadence docs/senses.md, the library's founder values; measured in "The observation codes" |
| arousal | threshold 0.2, decay 0.9, tolerance 2, floor 0.1, fast 0.05, slow 0.005, heat 2, youth 300, value surprise 1, need 0.1 (0.05, the arena's, until 2026-11-13) | Robot Arena founder; the need raised on a reading of 2026-11-13 (a stand's income met 0.05 and calmed the brain; at 0.1 the lives kept walking and learning) |

328 neurons, 22,041 parameters. The motor cortex is read by a layout, composed into the
brain at birth and kept for its life (`--layout`, `LAYOUT`): `joint`, the founder, sixteen
slots of three, one joint's decrease / hold / increase each; or `leg`, eight slots of nine,
one per leg, each answer one of the nine pairs the leg's hip and knee commands can make
(hip `c // 3`, knee `c % 3`). Both reach the body as the same sixteen commands and the body
cannot tell them apart; what differs is the brain's choice and its credit: under `leg` a leg
is one decision, so a lift with a swing, or a stroke, is one answer the outcome credits
instead of two that must agree. It is not a controller (every pair stays reachable, nothing
is chosen for the brain), but it is a different model, named in every reading made under it
and measured in "A slot per leg". Candidate genes are set on the nursery with
`--gene name=value` (`--gene modules=48,24`, `--gene arousal.need=0`, `--gene recall=true`,
`--gene efference_amplitude=3`, `--gene history_decays=0,0.7,0.95`, `--gene hold=0.9`,
`--gene entropy=0.1`, `--gene maturity=true`, `--gene code=64`, `--gene learned_code=64`,
`--gene learned_code_fields=128,16,8`), `--per N` for the
population code and `--setpoints` for the servo setpoints as readings.

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
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --stance-pay 0.1  # the posture hint, measured dose
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --yaw-pay 5     # charge turning too
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --window 20      # pay the mean progress of 20 moments
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --high-water     # pay new ground only
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --high-water --old-ground 0.2  # and old ground a fifth
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --pay-per-metre 10  # a quarter of the gain, the same cap
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --backward-pay 1   # charge backward as forward pays
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --motors rate      # the founder's rate motors (the default until 2026-11-12)
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --setpoints        # the servos' setpoints sensed too
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --course birth --compass  # pay the line of the birth heading, sensed by a compass
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --course birth --compass --ground  # and the velocity over the ground sensed too
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --layout leg      # a slot per leg: the brain chooses a leg's pair of commands at once
.venv/bin/python -m spider.nursery --moments 30000 --policy brain --seed 0 --drift 0.002     # on a current: the body loses 0.002 m/s backward, planted or not
```

One line per block of 100 moments (forward metres, reward per moment, aroused share, sweeps
and learning sweeps per moment, refusals, ms per moment, mean feet planted, radians turned,
metres along the course or, without one, along the birth heading, and `from_birth`: the
straight-line metres from the point of birth at the block's end) and a JSON
summary (`--json path` writes it). The summary's `displacement_m` is that straight line at
the end and `run_displacement_m` the straight line over this run alone (the same on a
newborn). Forward metres are the sum of every moment's progress along the body's own heading,
so a circle walked collects them; the displacement is the sum of every moment's motion over
the ground, taken as one vector in the world's frame, so a circle collects nothing. It belongs
to the life, is saved and loaded with it, and a recentre at the arena's edge (5 m out, where
the body is put back at the origin) does not touch it. A life saved before 2026-11-13 has
none; it starts where the body stands on the load.

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
| brain, 0.1 a missing foot | +0.600 (4.2 → 4.5) | +0.688 (2.4 → 4.0) | +1.024 (3.7 → 3.0) |
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
about 0.013 radians a moment. The same comparison with the yaw charge added to the then shipped
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
Fresh brains in that world, the reward shipped at the time (stance 0.1, yaw off), 30,000 moments, against
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
a different stall, not a gait. Window 1 with stance 0.1 is the law that walked farthest;
the shipped default is window 1 progress only, chosen for the unshaped reward, with the
stance dose and `--window` as opt-ins with these receipts. 4–5 ms a moment,
since every moment learns.

**The held foot.** The rocking above rode on a property of the contact model: a pressed
leg touched the ground where the tibia crossed the plane, so a deepening press slid the
contact point in under the knee and the rigid fit read that as a stroke. From 2026-11-12 a
pressed foot stays where a resting knee puts it; only the hip swing moves a planted foot
(`kinematics` in `spider/body.py`; the test `test_rocking_on_a_knee_moves_nothing`). Fresh
brains on that body, 30,000 moments, splay 0.3, the same seeds; the rows above this
paragraph were measured on the sliding foot:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| brain, progress only, sliding foot (table above) | +0.503 (0.00; 3.1 → 3.0; 0.00) | +0.044 (0.00; 2.5 → 2.5; 0.00) | +0.709 (0.00; 2.3 → 2.0; 0.00) |
| brain, progress only, held foot | +0.206 (+0.02; 3.9 → 4.0; 0.00; 0.073) | +0.289 (+0.04; 2.8 → 3.1; 0.95; 0.030) | +0.432 (0.00; 2.8 → 3.0; 0.00; 0.044) |
| brain, stance 0.1, sliding foot (table above) | +2.736 (+0.23; 3.4 → 2.9; 1.00) | +3.339 (+0.59; 2.3 → 1.4; 1.00) | +1.938 (+0.20; 2.8 → 2.9; 1.00) |
| brain, stance 0.1, held foot | +0.354 (+0.05; 4.3 → 5.8; 1.00; 0.019) | −0.125 (−0.02; 3.3 → 5.0; 1.00; 0.018) | +0.275 (+0.05; 4.5 → 5.1; 1.00; 0.016) |
| random, held foot | −0.931 (+0.39; 2.8 → 2.8; 0.070) | +0.182 (+0.51; 3.2 → 3.4; 0.075) | −0.258 (−0.54; 3.0 → 3.1; 0.066) |
| frozen newborn, held foot, 3000 moments | +0.004 (4.0) | +0.042 (2.0) | −0.025 (6.0) |
| frozen copy of the progress-only brain, 3000 moments | +0.090 (4.0) | +0.152 (4.0) | +0.073 (4.0) |
| frozen copy of the stance brain, 3000 moments | +0.152 (8.0) | +0.149 (6.0) | +0.155 (6.5) |

Two readings. First, the held foot closes the knee's lever and not the law's leak: seeds 0
and 2 under progress only go calm within the first thousand moments and stay there on four
feet with an income of 0.04–0.07 a moment and no displacement, and the replayed seed 0 is
rocking on a hip (leg 5's hip swings back for +0.1 to +0.2 a moment and forward for
nothing). Random earns the same 0.07 a moment for the same reason. Any back-and-forth
movement of a planted foot is such an income under the forward-only law, and the body has
sixteen of them. Second, the knee slide was also the stroke the stance-term walkers used:
with it gone they keep five to six feet down, earn 0.02 a moment and walk 0.3 m where they
walked 2–3 m, and the hip swing, now the only stroke, has not become a gait in 30,000
moments. Their frozen greedy copies move 0.15 m in their first block and then stand on six
to eight feet: 0.15 m is one sweep of every hip to its back limit (the body makes 0.152 m
from rest that way), which is the greedy answer of every raised brain in this table and the
next, and not a gait. The held foot is the physical rule
and the shipped one; the reading says the forward-only law needs a term that a sway cannot
earn before any body can show a gait, and that a knee that can stroke was doing work the
hip has not learned to do. 3–5 ms a moment.

**The rhythm genes.** Two genes give the brain a sense of its own timing without a controller:
`efference_amplitude=3` (an efference copy of its last command, 48 neurons; the Robot Arena's
operating point has it at 0, this brain's founder too) and `history_decays=0,0.7,0.95` (three
echoes of the senses at three decays, 456 neurons; the founder hears this moment only). Fresh
brains, held foot, progress only, 30,000 moments, the same seeds; the progress-only row is the
one from the table above:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| founder genes (table above) | +0.206 (+0.02; 3.9 → 4.0; 0.00; 0.073) | +0.289 (+0.04; 2.8 → 3.1; 0.95; 0.030) | +0.432 (0.00; 2.8 → 3.0; 0.00; 0.044) |
| efference 3 | +0.626 (+0.10; 4.6 → 3.4; 0.92; 0.033) | +0.276 (+0.18; 2.1 → 1.9; 0.94; 0.023) | +0.376 (−0.01; 3.1 → 4.0; 0.98; 0.029) |
| history 0, 0.7, 0.95 | +0.588 (+0.04; 3.8 → 3.6; 0.54; 0.042) | +0.218 (+0.08; 3.2 → 3.5; 0.96; 0.028) | +0.328 (+0.03; 4.0 → 3.8; 1.00; 0.024) |
| both | +0.722 (0.00; 3.7 → 3.1; 0.93; 0.031) | +0.126 (−0.08; 4.5 → 2.6; 0.89; 0.031) | +0.169 (0.00; 3.3 → 3.0; 0.00; 0.068) |
| frozen copies, 3000 moments | 0.154 / 0.089 / 0.152 | 0.147 / 0.148 / 0.160 | 0.154 / 0.154 / 0.154 |

**Reading:** either gene keeps the brain aroused where the founder went calm (five of the six
single-gene runs are over 0.9 aroused in their last 5000 moments, the sixth 0.54), and on seed
0 it triples the life's distance (0.2 to 0.6 m); on seeds 1 and 2 the distance is the founder's
or less, and no run makes more than 0.2 m in its last 5000 moments. Both genes together send
seed 2 calm at 0.068 a moment, the rocking's income. The frozen copies, one hip sweep each, are
the same answer as the founder's. A sense of timing did not make the hip swing a gait and did
not close the law's leak; the genes stay options (`--gene`), the founder stays at 0 and none.
4.8 ms a moment with the efference copy, 5.6 with the history, 5.7 with both.

**The high-water mark.** `--high-water` pays new ground only: the progress term is the
metres gained beyond the farthest the body's forward distance has ever reached, so the
rocking's income is gone by construction (the tests
`test_high_water_pays_new_ground_and_not_a_sway` and `test_save_carries_the_high_water_mark`).
Random under it earns 0.000 a moment where it earned 0.07 under the plain law. Fresh brains,
held foot, 30,000 moments, the same seeds; the plain-law rows are the ones from "The held
foot":

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| brain, progress only (table above) | +0.206 (+0.02; 3.9 → 4.0; 0.00; 0.073) | +0.289 (+0.04; 2.8 → 3.1; 0.95; 0.030) | +0.432 (0.00; 2.8 → 3.0; 0.00; 0.044) |
| brain, progress only, high-water | +0.019 (−0.03; 3.7 → 3.9; 1.00; 0.000) | −0.303 (−0.45; 1.9 → 2.2; 1.00; 0.000) | +0.394 (+0.28; 3.9 → 3.1; 0.99; 0.001) |
| brain, stance 0.1 (table above) | +0.354 (+0.05; 4.3 → 5.8; 1.00; 0.019) | −0.125 (−0.02; 3.3 → 5.0; 1.00; 0.018) | +0.275 (+0.05; 4.5 → 5.1; 1.00; 0.016) |
| brain, stance 0.1, high-water | +0.490 (+0.09; 4.2 → 4.8; 1.00; 0.000) | +0.015 (+0.04; 3.4 → 5.0; 1.00; 0.000) | −0.169 (−0.04; 4.6 → 5.2; 1.00; 0.000) |
| random, high-water | −0.931 (+0.39; 2.8 → 2.8; 0.000) | +0.182 (+0.51; 3.2 → 3.4; 0.000) | −0.258 (−0.54; 3.0 → 3.1; 0.000) |
| frozen copy of the high-water brain, 3000 moments | −0.028 (5.0) | +0.143 (2.0) | +0.078 (6.0) |
| frozen copy of the high-water stance brain, 3000 moments | +0.079 (6.0) | +0.147 (5.0) | +0.051 (8.0) |

**Reading:** the mark closes the leak and with it nearly all the income. The progress-only
brains reach a mark of 0.41–0.52 m, earn 17–21 reward in a whole life, in 9–15 of 300
blocks, which is what random earns (12–25, in 6–16 blocks); no run goes calm, every one is
fully aroused to the end, and the body wanders at 0.3 m forward or back per 5000 moments
with no trend and ends 0.0 to 0.4 m from where it started. Under the stance term the pay is
negative almost throughout (−0.02 to −0.10 a moment at the start), the brain plants five feet
and the mark is passed in 0–14 blocks. The frozen copies are the one hip sweep again. A
reward a sway cannot earn is also, on this body, a reward that nothing the brain stumbles on
earns: new ground needs a lift, a swing and a re-plant in order, and the search has not
produced that sequence in 30,000 moments under any law. The mark is the honest accounting
and stays an option; the shipped law stays the plain one. 4–5 ms a moment.

**The support rule.** Until 2026-11-12 any two feet planted through the moment carried the
body, so it stood and pushed on two left feet. Now the planted feet carry it only when they
support it: at least one on each side and three in all (`supported` in `spider/body.py`; the
test `test_one_side_cannot_hold_the_body_up`). Otherwise the body is down, its velocity
decays and nothing pushes it, as before with fewer than two feet. Fresh brains, held foot,
30,000 moments, the same seeds; the rows marked "two feet" are the ones from "The held foot":

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| brain, progress only, two feet (table above) | +0.206 (+0.02; 3.9 → 4.0; 0.00; 0.073) | +0.289 (+0.04; 2.8 → 3.1; 0.95; 0.030) | +0.432 (0.00; 2.8 → 3.0; 0.00; 0.044) |
| brain, progress only, supported | +0.316 (−0.05; 4.5 → 3.9; 1.00; 0.024) | +0.167 (+0.05; 2.6 → 3.2; 1.00; 0.021) | +0.380 (0.00; 3.4 → 3.5; 1.00; 0.021) |
| brain, stance 0.1, two feet (table above) | +0.354 (+0.05; 4.3 → 5.8; 1.00; 0.019) | −0.125 (−0.02; 3.3 → 5.0; 1.00; 0.018) | +0.275 (+0.05; 4.5 → 5.1; 1.00; 0.016) |
| brain, stance 0.1, supported | +0.092 (−0.01; 4.0 → 4.6; 1.00; 0.021) | +0.222 (−0.01; 3.1 → 4.9; 1.00; 0.017) | **+24.942** (+4.49; 4.9 → 4.9; 0.00; 0.044) |
| random, supported | +0.558 (+0.47; 2.8 → 2.8; 0.044) | +0.136 (+0.41; 3.2 → 3.4; 0.047) | +0.190 (−0.27; 3.0 → 3.1; 0.044) |
| frozen newborn, supported, 3000 moments | +0.004 (4.0) | +0.011 (2.0) | −0.025 (6.0) |
| frozen copy of the progress-only brain, 3000 moments | +0.155 (6.0) | +0.145 (4.0) | +0.153 (6.0) |
| frozen copy of the stance brain, 3000 moments | +0.158 (6.0) | +0.149 (6.0) | **+1.664** (6.2) |

**Reading:** one gait. Seed 2 under the stance term found a walk within its first 5000
moments (2.5 m in them) and then repeats it: 4.49 m in every later 5000, 0.9 m per thousand
moments, calm from the fifth thousand on, 4.9 feet planted, at 1.8 ms a moment since a calm
brain learns nothing. Its frozen greedy copy walks too, 0.053 m in every block of 100 on 6.2
feet, a third of the learner's speed and the first copy in these tables that does more than
one hip sweep. Its reward is 0.044 a moment, which is what random earns on this body, and
that is why it is calm: the walk earns the brain's need and the law sees no surprise left.
The other five brains are where they were: 0.1–0.4 m, fully aroused, no trend. Random earns
0.044 a moment here where it earned 0.07 on two feet, since a one-sided rock no longer pushes.
Whether the gait is the support rule's or seed 2's luck, three seeds cannot say; the rule is
the physical one and ships. 4 ms a moment aroused.

**The separated legs.** Until 2026-11-12 the hips were independent and neighbouring legs of a
side turned through each other: both hips at their limits put the leg behind 0.4 rad past
the leg in front, their lines meeting 0.2 m out, well inside a foot's reach. The 24.9 m gait
above walked that way: its frozen copy replayed on the old body has the legs crossed in
2995 of 3000 moments. Now neighbouring legs stop against each other at 0.2 rad of
convergence, where their lines meet 0.40 m out and no foot reaches (`separate` in
`spider/body.py`, `LEG_CONVERGE`; the test `test_neighbouring_legs_stop_against_each_other`).
Fresh brains, held foot, supported, 30,000 moments, the same seeds; the rows marked
"crossing" are the ones from "The support rule":

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| brain, progress only, crossing (table above) | +0.316 (−0.05; 4.5 → 3.9; 1.00; 0.024) | +0.167 (+0.05; 2.6 → 3.2; 1.00; 0.021) | +0.380 (0.00; 3.4 → 3.5; 1.00; 0.021) |
| brain, progress only, separated | +0.211 (−0.10; 3.7 → 3.2; 0.99; 0.018) | +0.286 (+0.05; 2.5 → 3.1; 1.00; 0.019) | +0.078 (−0.08; 3.5 → 4.4; 1.00; 0.020) |
| brain, stance 0.1, crossing (table above) | +0.092 (−0.01; 4.0 → 4.6; 1.00; 0.021) | +0.222 (−0.01; 3.1 → 4.9; 1.00; 0.017) | **+24.942** (+4.49; 4.9 → 4.9; 0.00; 0.044) |
| brain, stance 0.1, separated | +0.202 (+0.08; 4.1 → 4.4; 1.00; 0.009) | +0.200 (0.00; 2.7 → 5.0; 1.00; 0.018) | +0.290 (+0.03; 4.4 → 5.3; 1.00; 0.018) |
| random, separated | +0.894 (+0.42; 2.8 → 2.8; 0.039) | +0.064 (+0.43; 3.2 → 3.4; 0.055) | +0.413 (−0.29; 3.0 → 3.1; 0.042) |
| frozen newborn, separated, 3000 moments | +0.012 (4.0) | +0.011 (2.0) | −0.194 (6.8) |
| frozen copy of the progress-only brain, 3000 moments | +0.158 (6.0) | +0.152 (4.0) | +0.152 (5.0) |
| frozen copy of the stance brain, 3000 moments | +0.158 (6.0) | +0.149 (6.0) | +0.151 (7.0) |
| frozen copy of the 24.9 m crossing brain on the separated body, 3000 moments | | | +0.088 (6.0) |

**Reading:** the gait was a crossed-leg gait and the stop takes it away: the 24.9 m brain's
copy on the separated body makes 0.09 m in its first hundred moments and then stands on six
feet, and seed 2 under the stance term, raised on the separated body, is where the other
seeds are, 0.29 m, fully aroused, its planted feet climbing from 4.4 to 5.3 without a step.
The six learners sit at 0.1–0.3 m, the random policy at 0.06–0.9 m, which is the sway
of a leg sweep and not a direction (its last 5000 go either way), and every frozen copy
makes its one hip sweep and stands. Random earns 0.04–0.055 a moment, as on the crossing
body. Nothing walks on this body yet; the stop is the physical rule and ships, and the walk
has to be found again under it. 4 ms a moment.

**The eligibility and the window.** A step is a chain of unpaid moments: lift, 8–16 moments
of swinging forward in the air, plant, then the stroke that pays, and one moment of a
planted stroke (0.015 m) already pays the cap. The stroke's length is not the problem; the
credit has to reach back over the swing. Two ways of lengthening it, on the separated body,
progress only, 30,000 moments, the same seeds: a longer actor eligibility
(`--gene lam=0.9` for 0.6, a trace of about ten moments for two or three) and the reward
window (`--window 20`, each moment paid the mean progress of the last twenty), and both at
once; the plain-law rows are the ones from "The separated legs":

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| brain, lam 0.6, window 1 (table above) | +0.211 (−0.10; 3.7 → 3.2; 0.99; 0.018) | +0.286 (+0.05; 2.5 → 3.1; 1.00; 0.019) | +0.078 (−0.08; 3.5 → 4.4; 1.00; 0.020) |
| brain, lam 0.9 | +0.405 (+0.16; 4.3 → 4.5; 0.86; 0.034) | +0.149 (+0.06; 3.7 → 3.7; 0.83; 0.036) | −0.166 (0.00; 3.3 → 4.5; 0.95; 0.031) |
| brain, window 20 | +0.002 (+0.20; 3.5 → 3.1; 1.00; 0.008) | −0.311 (+0.01; 2.0 → 2.4; 0.99; 0.009) | −0.510 (−0.15; 4.0 → 3.7; 1.00; 0.009) |
| brain, lam 0.9 and window 20 | +0.092 (−0.14; 3.9 → 4.1; 1.00; 0.005) | +0.414 (+0.05; 2.6 → 4.3; 1.00; 0.003) | +0.184 (−0.03; 4.8 → 4.3; 1.00; 0.003) |
| random, window 1 (table above) | +0.894 (+0.42; 2.8 → 2.8; 0.039) | +0.064 (+0.43; 3.2 → 3.4; 0.055) | +0.413 (−0.29; 3.0 → 3.1; 0.042) |
| random, window 20 | +0.894 (+0.42; 2.8 → 2.8; 0.017) | +0.064 (+0.43; 3.2 → 3.4; 0.023) | +0.413 (−0.29; 3.0 → 3.1; 0.017) |
| frozen copy of the lam 0.9 brain, 3000 moments | +0.138 (6.0) | +0.150 (4.0) | +0.153 (7.0) |
| frozen copy of the window 20 brain, 3000 moments | +0.128 (5.0) | +0.132 (5.0) | +0.136 (6.2) |
| frozen copy of the lam 0.9 + window 20 brain, 3000 moments | +0.138 (7.0) | +0.149 (5.0) | +0.156 (6.0) |

**Reading:** no gait from either. The longer trace earns more a moment, 0.031–0.036 for
0.018–0.020, and is a little calmer (0.83–0.95 aroused), but the metres are the same sway,
−0.2 to +0.4 with no trend across the six blocks of 5000. The window earns less, 0.008–0.009
a moment, since a sway's twenty-moment mean is near zero, and the brains under it drift
backward as readily as forward; random under it earns 0.017–0.023 for the same reason. (The
1.0–1.8 m under window 20 in "The windowed reward" above were on the body whose pressed foot
slid; that slide is closed since the held foot.) Both
together earn least of all, 0.003–0.005. Every frozen copy is the one hip sweep. Neither
gene becomes a founder; the credit over a swing is not what a longer trace alone supplies,
and the window takes the per-moment signal away without replacing it. 4.4–5.3 ms a moment.

**The faster hip.** The other way of shortening the unpaid swing is a faster hip:
`HIP_RATE` 3.0 for 1.5, a full swing in 8 moments for 16, the paid stroke still at the cap.
Separated body, 30,000 moments, the same seeds, the full batch; the rows at 1.5 are the ones
from "The separated legs":

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| brain, progress only, hip 1.5 (table above) | +0.211 (−0.10; 3.7 → 3.2; 0.99; 0.018) | +0.286 (+0.05; 2.5 → 3.1; 1.00; 0.019) | +0.078 (−0.08; 3.5 → 4.4; 1.00; 0.020) |
| brain, progress only, hip 3.0 | +0.476 (0.00; 4.2 → 4.0; 0.00; 0.043) | +0.732 (+0.09; 2.6 → 3.2; 0.73; 0.034) | +0.463 (0.00; 2.8 → 3.0; 0.00; 0.045) |
| brain, stance 0.1, hip 1.5 (table above) | +0.202 (+0.08; 4.1 → 4.4; 1.00; 0.009) | +0.200 (0.00; 2.7 → 5.0; 1.00; 0.018) | +0.290 (+0.03; 4.4 → 5.3; 1.00; 0.018) |
| brain, stance 0.1, hip 3.0 | +0.472 (−0.02; 4.6 → 5.1; 0.90; 0.030) | +0.065 (−0.04; 3.5 → 5.9; 0.84; 0.037) | +0.110 (+0.08; 4.3 → 5.1; 0.85; 0.031) |
| random, hip 1.5 (table above) | +0.894 (+0.42; 2.8 → 2.8; 0.039) | +0.064 (+0.43; 3.2 → 3.4; 0.055) | +0.413 (−0.29; 3.0 → 3.1; 0.042) |
| random, hip 3.0 | +1.516 (+0.88; 2.8 → 2.8; 0.069) | −0.263 (+0.39; 3.2 → 3.4; 0.096) | +0.132 (−0.37; 3.0 → 3.1; 0.075) |
| frozen newborn, hip 3.0, 3000 moments | +0.022 (4.0) | +0.035 (2.0) | −1.085 (6.8) |
| frozen copy of the progress-only brain, 3000 moments | +0.152 (5.0) | +0.151 (5.0) | +0.136 (5.0) |
| frozen copy of the stance brain, 3000 moments | +0.151 (7.0) | +0.145 (7.0) | +0.155 (7.0) |

**Reading:** the faster hip doubles the sway's income and the learners settle on it. Random
earns 0.07–0.10 a moment for 0.04–0.055, since a faster rock makes more metres a moment
each way and the law pays each moment on its own. Two of the three progress-only brains
(seeds 0 and 2) go calm within 10,000 moments and never move again, earning 0.043–0.045 a
moment standing still on four and three feet: the sway's income reaches the brain's need
and the law sees nothing left to learn, the calm stall of "The rocking" back under a plain
law. The third stays three-quarters aroused and sways 0.7 m. Under the stance term the
brains sway too, 0.07–0.47 m, no trend. Every frozen copy is the one hip sweep; the newborn
of seed 2 sweeps 1.1 m backward in its first moments and stands. No gait, and the stall is
back, so the hip stays at 1.5 rad/s. 1.9–3.6 ms a moment (the calm ones cheapest).

**Time and seeds.** Two questions the tables above leave open: whether a brain that has not
walked in 30,000 moments walks later, and whether the one gait under the stance term was
that law's or seed 2's. Separated body, plain law, lives of 200,000 moments under the stance
term and progress only (seeds 0–2), and six more seeds of 30,000 under the stance term with
random at the same seeds:

| 200,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| brain, stance 0.1 | +0.200 (−0.02; 4.1 → 4.7; 0.98; 0.028) | +0.192 (−0.01; 2.7 → 4.5; 1.00; 0.022) | +0.338 (−0.01; 4.4 → 6.4; 1.00; 0.015) |
| brain, progress only | +0.844 (0.00; 3.7 → 5.3; 1.00; 0.021) | +0.283 (−0.05; 2.5 → 3.7; 0.99; 0.026) | +0.014 (+0.01; 3.5 → 5.1; 1.00; 0.020) |
| frozen copy of the stance brain, 3000 moments | +0.157 (5.0) | +0.147 (5.0) | +0.151 (7.0) |
| frozen copy of the progress-only brain, 3000 moments | +0.157 (5.0) | +0.152 (4.0) | +0.154 (6.0) |

| 30,000 moments, stance 0.1, forward metres (last 5000; planted feet, first 5000 → last 5000; r/moment) | Seed 3 | Seed 4 | Seed 5 | Seed 6 | Seed 7 | Seed 8 |
| --- | --- | --- | --- | --- | --- | --- |
| brain | +0.370 (+0.01; 4.1 → 5.9; 0.019) | +0.375 (−0.03; 4.2 → 5.3; 0.019) | +0.258 (−0.04; 4.1 → 5.1; 0.020) | +0.219 (0.00; 5.4 → 5.8; 0.018) | +0.023 (−0.07; 4.3 → 5.4; 0.017) | +0.276 (−0.02; 3.7 → 4.9; 0.019) |
| random | +1.012 (0.049) | −1.386 (0.046) | +1.034 (0.039) | +0.408 (0.041) | −1.055 (0.040) | −0.517 (0.047) |
| frozen copy of the brain, 3000 moments | +0.153 (7.0) | +0.152 (8.0) | +0.153 (5.0) | +0.155 (7.0) | +0.155 (7.0) | +0.149 (6.0) |

**Reading:** time does not help, and the gait was seed 2's on the crossing body. Every one
of the six long lives makes its 0.1–0.4 m in the first 20,000 moments and nothing in the
next 180,000, aroused the whole way (0.98–1.00), its planted count climbing to 4.5–6.4
feet; under the stance term it ends standing on more feet than it started on, since standing
is what that term pays. The six new seeds under the stance term do the same within 30,000:
all their metres fall in the first 5000, every later block is within ±0.1 m of zero, the feet
climb to 5–6, and random moves more in either direction (±1 m) than any learner. Every
frozen copy, long or short, is the one hip sweep. Nine seeds and 200,000 moments say the
same thing: on the honest body no brain has found a gait, and the learner's state after
30,000 moments is where it stays. 3.6–4.1 ms a moment.

**The genes.** The founder genes that had never been measured on this body, one at a time
against the founders, separated body, plain law unless said, 30,000 moments, seeds 0–2: the
arousal need at 0.1 and 0.02 (founder 0.05), the actor eta at 0.01 (0.03), the arena's module
sizes 48,24 (64,32), the population code at 16 tuned inputs per reading (`--per 16`, 296
inputs for 152), and the eligibility 0.9 under the stance term. (The first `--per 16` runs
were void: the brain was composed for the default senses and refused every moment of a
296-input life, 0.000 m on eight feet; since 2026-11-12 a life composes its newborn for its
own senses, `make_policy(..., inputs=)`, and a test keeps it so. The rows below are the rerun.)
The founder rows are the ones from "The separated legs":

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| founders, progress only (table above) | +0.211 (−0.10; 3.7 → 3.2; 0.99; 0.018) | +0.286 (+0.05; 2.5 → 3.1; 1.00; 0.019) | +0.078 (−0.08; 3.5 → 4.4; 1.00; 0.020) |
| need 0.1 | +0.229 (+0.12; 4.2 → 4.0; 1.00; 0.024) | −0.062 (−0.20; 1.8 → 3.2; 1.00; 0.022) | +0.343 (−0.01; 4.2 → 4.2; 1.00; 0.024) |
| need 0.02 | +0.108 (0.00; 3.2 → 3.0; 0.00; 0.021) | −0.048 (−0.07; 2.4 → 2.8; 0.73; 0.012) | +0.232 (0.00; 4.1 → 3.0; 0.00; 0.019) |
| eta 0.01 | +0.484 (+0.02; 4.1 → 4.4; 0.99; 0.028) | +0.302 (−0.10; 1.8 → 3.8; 1.00; 0.026) | +0.275 (+0.01; 4.1 → 4.0; 0.00; 0.057) |
| modules 48,24 | +0.189 (0.00; 3.0 → 3.0; 0.00; 0.046) | +0.113 (−0.05; 4.9 → 3.9; 1.00; 0.021) | −0.200 (−0.04; 4.5 → 3.5; 1.00; 0.018) |
| per 16 | +0.475 (0.00; 3.4 → 4.5; 1.00; 0.025) | +0.528 (−0.09; 3.6 → 4.1; 1.00; 0.023) | +0.849 (+0.09; 2.7 → 3.2; 1.00; 0.021) |
| founders, stance 0.1 (table above) | +0.202 (+0.08; 4.1 → 4.4; 1.00; 0.009) | +0.200 (0.00; 2.7 → 5.0; 1.00; 0.018) | +0.290 (+0.03; 4.4 → 5.3; 1.00; 0.018) |
| stance 0.1, lam 0.9 | +0.474 (+0.05; 5.1 → 5.5; 0.99; 0.029) | +0.249 (0.00; 4.4 → 5.5; 1.00; 0.025) | +0.226 (+0.06; 4.7 → 5.3; 0.98; 0.030) |
| frozen newborn, per 16, 3000 moments | −0.075 (5.0) | +0.053 (3.1) | −0.048 (3.0) |
| frozen copies, 3000 moments (need 0.1 / 0.02 / eta / modules / per 16 / lam) | 0.156 / 0.066 / 0.156 / −0.022 / 0.138 / 0.153 | 0.143 / 0.056 / 0.150 / 0.147 / 0.147 / 0.151 | 0.154 / 0.152 / 0.064 / 0.154 / 0.152 / 0.153 |

**Reading:** no gene finds a gait, and two bring the calm stall back. The lower need
(0.02) is met by a sway's income: seeds 0 and 2 go calm within 10,000 moments and never move
again, as under the faster hip. The lower eta calms seed 2 (0.057 a moment, standing) and the
smaller modules calm seed 0. The raised need keeps every brain aroused and changes nothing
else. The finer code is the one gene with a visible effect: all three brains make 0.53–0.55 m
in their first 5000 moments, twice to three times the founders' first block (0.17–0.27), and
then stop like the rest, every later block within ±0.2 m of zero. The longer trace under the
stance term earns more a moment (0.025–0.030) on more feet (5.3–5.5) and goes nowhere. Every
frozen copy is the one hip sweep, or less. The founders stay. 4.3–5.2 ms a moment aroused,
2 ms calm.

**The rhythm genes and the heat, on this body.** Every reading above has the brain aroused
through its whole life (a sway's income stays under the need), and aroused means every joint
is drawn afresh each moment at heat 2, while a swing is sixteen moments on several legs at
once; the chain that a gait needs is not found by uncorrelated exploration and is paid only at
its end. The two genes that bear on this, measured in "The rhythm genes" on the body before
the support rule and the stop, again here: the efference copy (`efference_amplitude=3`), the
heard history (`history_decays=0,0.7,0.95`), both; and the exploration heat lowered to 1 and
0.5 (founder 2), alone and with the efference copy. Separated body, plain law, 30,000 moments,
seeds 0–2; the founder row is the one above:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| founders, progress only (table above) | +0.211 (−0.10; 3.7 → 3.2; 0.99; 0.018) | +0.286 (+0.05; 2.5 → 3.1; 1.00; 0.019) | +0.078 (−0.08; 3.5 → 4.4; 1.00; 0.020) |
| efference 3 | +0.345 (−0.01; 3.3 → 3.6; 0.99; 0.027) | +0.497 (−0.02; 2.6 → 3.7; 1.00; 0.025) | +0.311 (−0.04; 4.7 → 4.0; 1.00; 0.022) |
| history 0, 0.7, 0.95 | +0.571 (+0.16; 3.6 → 3.8; 0.99; 0.025) | +0.438 (+0.08; 3.4 → 3.3; 0.99; 0.023) | +0.437 (−0.02; 4.1 → 3.1; 1.00; 0.015) |
| both | +0.393 (+0.07; 5.3 → 4.2; 1.00; 0.025) | +0.401 (−0.09; 3.8 → 3.8; 1.00; 0.024) | +0.214 (−0.05; 3.7 → 4.7; 1.00; 0.023) |
| heat 1 | +0.170 (+0.01; 4.0 → 4.0; 0.00; 0.049) | +0.051 (−0.01; 2.5 → 3.2; 1.00; 0.019) | +0.103 (+0.02; 3.9 → 3.6; 1.00; 0.021) |
| heat 0.5 | −0.035 (+0.01; 3.6 → 4.1; 0.99; 0.023) | +0.385 (+0.08; 2.6 → 3.3; 1.00; 0.020) | +0.054 (−0.02; 4.2 → 3.4; 1.00; 0.018) |
| efference 3, heat 1 | +0.101 (−0.02; 3.6 → 4.2; 1.00; 0.023) | −0.090 (+0.05; 2.6 → 2.9; 0.99; 0.020) | +0.579 (0.00; 3.2 → 5.0; 1.00; 0.022) |
| frozen copies, 3000 moments (efference / history / both / heat 1 / heat 0.5 / efference, heat 1) | 0.152 / 0.142 / 0.153 / 0.095 / 0.160 / 0.145 | 0.139 / 0.160 / 0.147 / 0.143 / 0.152 / 0.161 | 0.154 / 0.155 / 0.161 / 0.155 / 0.154 / 0.149 |

**Reading:** no gait. The heard history is the best of the batch, 0.44–0.57 m over the life on
all three seeds against the founders' 0.08–0.29, and its seed 0 is the one run of the batch
whose last 5000 moments make more than 0.1 m (+0.16); the efference copy gives 0.31–0.50 m
and nothing in its last block; both together are no better than either. The lower heat does
not make exploration coherent, it makes it poorer: heat 1 and 0.5 lie at or below the founders
on four of six seeds, and heat 1 calms seed 0 at the sway's 0.049 a moment, the stall again.
Every frozen copy is the one hip sweep. A sense of the moments before raises what a life
gathers by the first blocks and does not make a chain of them; the genes stay options, the
founder stays at 0, none and heat 2. 6.3 ms a moment with the efference copy, 7.1 with the
history, 7.3 with both, 2.6 calm.

**The pay per metre.** At the founder's 40 a metre one planted stroke moment pays 0.4 of the
cap's 0.5, so a moment on which four feet push pays what a moment on which one foot pushes
does, and the law cannot tell a stroke from a stronger one. `--pay-per-metre` lowers the gain
under the same cap: at 20 a stroke moment pays 0.2, at 10 it pays 0.1 and the cap is four
strokes away, at 5 eight. Separated body, progress only, 30,000 moments, seeds 0–2, the gain
alone and, at 10 and 20, with the heard history (the best gene of the batch above); random
under the gain of 10 for the income a jitter earns; the founder and random rows are the ones
from "The separated legs":

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| founders, 40 a metre (table above) | +0.211 (−0.10; 3.7 → 3.2; 0.99; 0.018) | +0.286 (+0.05; 2.5 → 3.1; 1.00; 0.019) | +0.078 (−0.08; 3.5 → 4.4; 1.00; 0.020) |
| 20 a metre | +0.862 (+0.10; 3.6 → 4.2; 1.00; 0.013) | +0.083 (−0.07; 1.8 → 2.9; 1.00; 0.011) | +0.018 (+0.01; 3.8 → 3.3; 1.00; 0.010) |
| 10 a metre | −0.008 (−0.12; 3.8 → 4.3; 1.00; 0.008) | −0.444 (+0.08; 1.9 → 3.3; 1.00; 0.006) | −0.285 (−0.04; 3.8 → 4.1; 1.00; 0.007) |
| 5 a metre | +0.272 (0.00; 3.8 → 4.6; 1.00; 0.005) | +0.121 (+0.03; 1.9 → 2.7; 1.00; 0.003) | +0.163 (0.00; 4.0 → 4.0; 1.00; 0.004) |
| 20 a metre, history 0, 0.7, 0.95 | +0.184 (+0.13; 3.6 → 4.2; 1.00; 0.013) | −0.130 (+0.03; 1.5 → 3.5; 1.00; 0.013) | +0.191 (−0.03; 4.2 → 4.3; 1.00; 0.011) |
| 10 a metre, history 0, 0.7, 0.95 | +0.747 (+0.06; 3.5 → 3.7; 1.00; 0.007) | −0.070 (−0.14; 1.2 → 3.3; 1.00; 0.006) | +0.162 (+0.04; 4.0 → 4.2; 1.00; 0.007) |
| random, 40 a metre, 30,000 moments (table above) | +0.894 (0.043) | +0.064 (0.046) | +0.413 (0.044) |
| random, 10 a metre, 3000 moments | +0.221 (0.009) | −0.396 (0.011) | +0.063 (0.011) |
| frozen copies, 3000 moments (20 / 10 / 5 / 20 history / 10 history) | 0.158 / 0.126 / 0.156 / 0.152 / 0.135 | 0.140 / 0.145 / 0.000 / 0.142 / 0.145 | 0.154 / 0.154 / 0.136 / 0.154 / 0.154 |

**Reading:** no gait at any gain, and the gain is not what holds one back. The distances are
the founders' scatter: 20 a metre gives the longest life on this body so far on seed 0
(0.86 m) and nothing on seeds 1 and 2, 10 a metre goes backward on all three, 5 stands. The
brain stays aroused at every gain, as it must: at 10 it earns 0.006–0.008 a moment, at 5 under
0.005. The one thing the gain shows is the leak in another unit: random under 10 a metre earns
0.009–0.011 a moment and the learner 0.006–0.008, under 40 random earns 0.043–0.046 and the
learner 0.018–0.020. A random jitter is paid on its forward half and charged nothing on its
back half, and the learning brain, which jitters less, is paid less than chance at every
gain. The frozen copies are the hip sweep; the copy of 5 a metre on seed 1 stands. The gain
stays 40, the flag stays an option. 6.7 ms a moment, 7.4 with the history.

**The position commands.** The chain a gait needs under the founder's rate motors is long:
a swing is the same answer sixteen moments running, on several legs, paid only at its end.
`--motors target` makes each joint a position servo (the body section above): one answer sets
the setpoint to an end of the range, `hold` keeps it, and the joint travels there on its own,
so a swing is one decision. This is a different model, not a gene of the founder's, and it is
measured as one: a newborn body under it against random and frozen newborn under it, seeds
0–2, 30,000 moments, the law plain and with the heard history, the high-water mark and the
stance term:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| founders, rate motors (table above) | +0.211 (−0.10; 3.7 → 3.2; 0.99; 0.018) | +0.286 (+0.05; 2.5 → 3.1; 1.00; 0.019) | +0.078 (−0.08; 3.5 → 4.4; 1.00; 0.020) |
| target motors | +0.205 (+0.08; 4.7 → 2.9; 0.99; 0.017) | +0.360 (+0.03; 3.1 → 4.2; 1.00; 0.021) | +1.503 (−0.10; 3.7 → 3.9; 0.99; 0.023) |
| target, history 0, 0.7, 0.95 | +0.182 (+0.02; 4.2 → 4.0; 0.99; 0.026) | +0.795 (−0.09; 2.6 → 3.7; 0.99; 0.023) | +0.356 (+0.01; 4.7 → 4.3; 1.00; 0.022) |
| target, high-water | −0.462 (+0.50; 4.1 → 4.0; 1.00; 0.000) | +0.396 (+0.18; 2.2 → 2.0; 1.00; 0.000) | +1.549 (+0.16; 3.9 → 3.4; 0.99; 0.003) |
| target, stance 0.1 | +0.618 (0.00; 4.1 → 6.6; 1.00; 0.017) | +0.525 (−0.04; 3.2 → 5.9; 1.00; 0.018) | +0.539 (+0.03; 4.7 → 5.4; 1.00; 0.021) |
| random, target motors | −1.131 (+0.41; 2.9 → 3.1; 0.060) | +0.508 (−0.17; 3.1 → 3.2; 0.062) | −0.120 (−0.95; 3.1 → 3.3; 0.060) |
| random, rate motors (table above) | +0.894 (+0.42; 2.8 → 2.8; 0.039) | +0.064 (+0.43; 3.2 → 3.4; 0.055) | +0.413 (−0.29; 3.0 → 3.1; 0.042) |
| frozen newborn, target motors, 3000 moments | −0.106 (5.0) | 0.000 (2.0) | +0.035 (6.0) |
| frozen copies, 3000 moments (plain / history / high-water / stance) | 0.155 / 0.156 / 0.048 / 0.152 | 0.149 / 0.158 / 0.083 / 0.151 | 0.153 / 0.153 / 0.124 / 0.155 |

**Reading:** the servos lengthen the first phase and find no gait. Seed 2 under the plain
law makes 1.17 m in its first 9000 moments (0.39, 0.34, 0.44 m per 3000, on 3.6–3.7 feet),
the longest sustained stretch on this body so far, then settles to the founders' ±0.1 m per
3000 moments and its frozen copy at the end is the one hip sweep; whatever it did it did not
keep. The stance term gives 0.52–0.62 m on all three seeds, the one row that agrees across
seeds, and nothing in any last block. Under the high-water mark the brain earns nothing
(0.000–0.003 a moment) and the body wanders as random does, ±0.5 m per 3000 moments; the
+0.50 m of seed 0's last block is one such swing, between a −0.68 and a −0.29 before it, not
a walk, and the high-water copies are short of a sweep. Random under servos moves more than
random under rate motors (1.1 m against 0.9 at most, 0.060 a moment against 0.043) because a
random setpoint holds a joint's motion for several moments where a random rate reverses it,
and the learning brain is again paid less than chance at every law. The model stays `rate`,
`target` stays an option named in its readings. 4.9 ms a moment.

**The backward pay and the horizon.** Two readings aimed at the leak and at the chain.
`--backward-pay 1` charges a metre backward what a metre forward pays, so a jitter nets
nothing and only a stroke that the body keeps pays; this was the first law, left on the
body before the support rule because the brain lifted every foot to avoid it. And
`--gene gamma=0.99` lengthens the value horizon from twenty moments to a hundred, where a
lift, swing, plant and stroke span some forty. Separated body, rate motors, 30,000 moments,
seeds 0–2: the full charge, half of it, the horizon, both, and the charge with the stance
term; random under the full charge for the income a jitter nets:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| founders (table above) | +0.211 (−0.10; 3.7 → 3.2; 0.99; 0.018) | +0.286 (+0.05; 2.5 → 3.1; 1.00; 0.019) | +0.078 (−0.08; 3.5 → 4.4; 1.00; 0.020) |
| backward 1 | +0.055 (+0.03; 4.0 → 3.2; 1.00; 0.000) | +0.123 (−0.01; 1.7 → 1.2; 1.00; 0.000) | +0.413 (−0.01; 3.8 → 3.6; 1.00; 0.000) |
| backward 0.5 | +0.367 (0.00; 4.4 → 4.4; 1.00; 0.007) | −0.176 (−0.01; 1.3 → 1.6; 1.00; 0.002) | −0.053 (+0.07; 3.3 → 3.6; 1.00; 0.007) |
| gamma 0.99 | +0.168 (0.00; 3.2 → 3.0; 0.00; 0.064) | +0.431 (+0.08; 3.4 → 3.8; 1.00; 0.023) | +0.208 (−0.01; 4.3 → 4.0; 1.00; 0.023) |
| backward 1, gamma 0.99 | +0.417 (−0.03; 2.8 → 3.1; 1.00; 0.000) | +0.301 (+0.09; 1.5 → 2.7; 1.00; 0.001) | +0.104 (+0.01; 3.1 → 2.2; 1.00; 0.000) |
| backward 1, stance 0.1 | +0.180 (0.00; 4.4 → 5.1; 1.00; −0.001) | +0.200 (0.00; 3.6 → 4.9; 1.00; 0.000) | +0.338 (0.00; 3.9 → 4.2; 1.00; −0.028) |
| random, backward 1 | +0.894 (+0.42; 2.8 → 2.8; 0.003) | +0.064 (+0.43; 3.2 → 3.4; 0.003) | +0.413 (−0.29; 3.0 → 3.1; −0.002) |
| frozen copies, 3000 moments (backward 1 / 0.5 / gamma / both / stance) | 0.158 / 0.155 / 0.043 / 0.160 / 0.155 | 0.000 / 0.140 / 0.149 / 0.000 / 0.149 | 0.152 / 0.154 / 0.134 / 0.152 / 0.151 |

**Reading:** the charge closes the leak as the high-water mark does, and with the same
result. Chance now earns nothing (random nets ±0.003 a moment), and so does the learner:
under the full charge every brain's income is 0.000 from its second block on, and what it
learns is to net nothing. Seed 1 does it the first law's way, on one or two feet, lifted
(1.0–1.6 planted through the life, its frozen copy standing at 0.000 m); seeds 0 and 2 keep
three to four feet down and sway, each block within ±0.06 m. The half charge leaves 0.002 to
0.007 a moment and the founders' scatter. The longer horizon calms seed 0 at the sway's
0.064 a moment within 3000 moments, the stall again and sooner, and leaves seeds 1 and 2 at
the founders'. With both, with the stance term: zero income, no gait, the copies a hip sweep
or a stand. Two laws now pay a walk and nothing else, the mark and the charge, and under
either the learner finds nothing to be paid for: the law is not what holds the gait back.
The defaults stay (backward free, gamma 0.95). 4.7 ms a moment.

**The servo brain of seed 2, looked at.** The one stretch on this body that lasted was the
servo brain of seed 2: 1.17 m in its first 9000 moments, spread over nine blocks of a
thousand, then 0.33 m in the 21,000 after. Was it a gait the brain then unlearned, or a
drift? The same run again (deterministic: its ninety blocks match the reading above), saved
at 9000 moments, a frozen copy of the save, and a trace of every moment's commands, joint
angles and planted feet through `Life.tick`, for the 9000 learning moments, for 1000 more of
the saved brain learning on, and for random under servos:

| per 1000 moments, seed 2 under servos, learning from birth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forward m | +0.31 | +0.09 | −0.01 | +0.22 | −0.01 | +0.13 | +0.06 | +0.28 | +0.10 |
| mean hip angle (range −0.6 to +0.6) | −0.27 | −0.45 | −0.49 | −0.49 | −0.50 | −0.52 | −0.54 | −0.53 | −0.55 |
| hip excursion (std over the block, mean of eight) | 0.25 | 0.16 | 0.13 | 0.13 | 0.12 | 0.11 | 0.08 | 0.09 | 0.07 |
| hip commands: forward / back share | 0.26 / 0.51 | 0.18 / 0.61 | 0.17 / 0.63 | 0.16 / 0.66 | 0.15 / 0.66 | 0.15 / 0.67 | 0.13 / 0.72 | 0.14 / 0.72 | 0.13 / 0.72 |
| stroke moments (over 4 mm forward) | 137 | 103 | 89 | 86 | 72 | 75 | 58 | 84 | 53 |

The brain learning on from the save, 1000 moments: +0.02 m, every hip held at −0.52 to
−0.59 with an excursion of 0.02 to 0.12, the back command on 60–88 % of hip moments, the
knees lifting and planting (17–36 lifts per leg, two legs never moving), 37 stroke moments
against 48 backward ones. Its frozen copy, 3000 moments: 0.153 m, all of it in the first
hundred moments, every hip commanded back on every moment and every knee held; then a stand
with six feet down. Random under servos, 1000 moments: hips about 0 with an excursion of
0.33, 180 stroke moments, +0.37 m.

**Reading:** it was not a gait. It was the stroke, learned, fed by what exploration left:
the brain learned early that a hip moving back on a planted leg pays (the stroke moments
carry a planted-hip change of −0.030 a moment, the rest +0.002), and commands it more and
more; the forward swing it half learned too, in the air twice as often as planted (9874
forward hip moves in the air against 4888 planted), but as a drawn exploration, not a
policy. Every excursion the drawn commands pushed a hip forward, the learned stroke took
it back and was paid; as the policy closed on the back command the excursions dried up (0.25
to 0.07), the strokes with them (137 to 53 a thousand), and the hips came to rest against the
back stop, where the back command pays nothing and costs nothing and the forward command on
a planted leg is backward progress, free under the plain law and worth nothing to the
learner. The ratchet was never a cycle: no leg's air time correlates with another's (every
pair within ±0.2), no joint has a period. The frozen copy says the same as every frozen copy
before it: the greedy policy is the one hip sweep, the stroke without its return. What held
the gait back here is the return swing, the half of the chain that pays only through the
stroke after it: the policy keeps the half that pays at once. A longer eligibility (`lam`
0.9, measured above under rate motors) and a longer horizon (`gamma` 0.99) are the brain's
own handles on that credit; under servos neither is measured. 4.9 ms a moment.

**Credit for the return swing, under servos.** The reading above says the policy keeps the
stroke and loses the swing because the swing pays only through the stroke after it. The
brain's own handles on that credit are the actor's eligibility (`--gene lam=0.9` for 0.6,
a trace of ten moments for two or three) and the value horizon (`--gene gamma=0.99` for
0.95, a hundred moments for twenty). Under servos, where a swing is one command held, 30,000
moments, seeds 0–2, each, both, frozen copies; and the trace above for every run's first
9000 moments:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| servos, founders (table above) | +0.205 (+0.08; 4.7 → 2.9; 0.99; 0.017) | +0.360 (+0.03; 3.1 → 4.2; 1.00; 0.021) | +1.503 (−0.10; 3.7 → 3.9; 0.99; 0.023) |
| servos, lam 0.9 | +0.532 (−0.04; 4.0 → 4.3; 0.77; 0.036) | −0.446 (−0.23; 3.2 → 4.0; 0.92; 0.031) | +0.533 (−0.11; 3.2 → 4.7; 0.89; 0.031) |
| servos, gamma 0.99 | +1.689 (−0.01; 3.8 → 5.0; 0.00; 0.049) | +0.939 (+0.03; 3.1 → 3.9; 0.98; 0.025) | +0.674 (+0.08; 3.8 → 4.8; 0.99; 0.025) |
| servos, lam 0.9 and gamma 0.99 | +0.935 (+0.17; 5.0 → 4.1; 0.77; 0.035) | +0.806 (−0.06; 3.4 → 4.9; 0.88; 0.034) | −0.028 (0.00; 3.8 → 5.0; 0.00; 0.042) |
| frozen copies, 3000 moments (lam / gamma / both) | 0.132 / 0.138 / 0.154 | 0.146 / 0.149 / 0.152 | 0.155 / 0.154 / 0.128 |

**Reading:** the horizon is the handle, and it bought the longest stretch on this body.
Gamma 0.99 lies above the servo founders on two seeds and above every other reading on
seed 0: +0.44, +0.35, +0.40, +0.45 m in its first four blocks of 5000, 1.64 m in 20,000
moments at a steady rate, before it stopped. The trace of those moments is the first in
which the chain appears: from 10,000 moments on, six hips rest at the back stop (std 0.02 to
0.05) and two legs of the left side, 2 and 3, lift about thirty times a thousand each, move
the hip forward in the air and back on the ground (+0.0008 and +0.0012 a moment in the air,
−0.0006 and −0.0043 planted), with the knee doing the lifting (std 0.47 to 0.49). Those two
legs' stroke moments carry +3.46 m over 10,000 moments and every other moment −2.61, net
+0.85: a crawl of two legs on one side, which is why the body turns (150 rad over the life).
It is a chain, not a cycle: the hip advances in only a fifth to a third of the air runs and
retreats in a quarter of the ground runs, no planted flag has a period, and no leg's air time
follows another's. And it was undone by the law, not forgotten: from 20,000 moments the
brain calmed (aroused 0.31, then 0.00), five feet came down, the crawl stopped, and its
income rose from 0.032–0.035 a moment to 0.047–0.049, the sway's. The plain law pays the sway
more than the crawl. The same half chain shows in the 5000–9000 window of six other runs on
one leg (3, 4–5, 6, 3, 5, and 5 under the founders), none of them held. The longer eligibility
alone lies at or below the founders on two seeds and brings the sway's income (0.031–0.036)
sooner; with the horizon it is no better than the horizon alone and stalls seed 2 at 0.042.
Every frozen copy is the one hip sweep, 0.13–0.16 m: the greedy policy remains the stroke.
The founders stay (lam 0.6, gamma 0.95, rate motors); `gamma=0.99` under servos is the
reading to build on. 4.0 ms a moment.

**The crawl under the laws that pay nothing to a sway.** The crawl above was undone because
the plain law pays a sway more. The two laws under which a sway nets nothing are the
high-water mark and the full backward charge; under either, the crawl's +0.08 m a thousand
would be the only income. Servos, gamma 0.99, seeds 0–2, 30,000 moments under each law,
random under each law for the income chance nets, frozen copies, and the trace over the
first 20,000 moments:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| servos, gamma 0.99 (table above) | +1.689 (−0.01; 3.8 → 5.0; 0.00; 0.049) | +0.939 (+0.03; 3.1 → 3.9; 0.98; 0.025) | +0.674 (+0.08; 3.8 → 4.8; 0.99; 0.025) |
| servos, gamma 0.99, high-water | −1.673 (+0.25; 3.7 → 3.6; 1.00; 0.000) | −0.276 (+0.05; 2.7 → 2.5; 1.00; 0.000) | −0.115 (−0.38; 3.8 → 4.2; 1.00; 0.000) |
| servos, gamma 0.99, backward 1 | +0.577 (−0.01; 4.3 → 4.0; 1.00; 0.000) | +0.346 (0.00; 2.5 → 2.0; 1.00; 0.000) | +0.125 (−0.01; 2.7 → 1.7; 1.00; 0.000) |
| random, servos, high-water | −1.131 (+0.41; 2.9 → 3.1; 0.000) | +0.508 (−0.17; 3.1 → 3.2; 0.000) | −0.120 (−0.95; 3.1 → 3.3; 0.000) |
| random, servos, backward 1 | −1.131 (+0.41; 2.9 → 3.1; 0.003) | +0.508 (−0.17; 3.1 → 3.2; −0.002) | −0.120 (−0.95; 3.1 → 3.3; −0.007) |
| frozen copies, 3000 moments (high-water / backward 1) | 0.032 / 0.154 | 0.095 / 0.155 | 0.113 / 0.156 |

**Reading:** neither law finds the crawl, and each loses it a different way. Under the
mark the brain's income is 0.000 from its second block of 5000 on, and so is everything
else: the hips never settle (mean +0.1 to −0.3, excursion 0.13–0.30 through 20,000
moments), every leg lifts 150–200 times a block, the stroke moments carry +3.5 m a block and
the rest −3.5, and the body wanders as random does (seed 0 −1.67 m, turning 55 rad a block).
Not even the stroke is learned, since a stroke pays only on new ground, and after the first
sweep there is none; the frozen copies are short of a sweep. Under the charge the first
block pays as the founders' does (+0.46, +0.28, +0.30 m, the strokes of the fresh body),
then the hips reach the back stop (excursion 0.02–0.09), the income is 0.000, seeds 1 and
2 lift feet (2.0 and 1.9 planted) and the stroke moments shrink to a quarter; the copies
are the sweep. The chain flag of the trace lights on one leg in some blocks and goes out in
the next, under either law. What the plain law did for the crawl was to pay the jitter
around it, and the jitter's credit carried the half chain for 20,000 moments; a law that
pays the jitter nothing carries nothing. The charge short of full cannot help either: the
crawl as found is +3.46 m of strokes against −2.61 of the rest per 10,000 moments, so at any
charge below one the sway, whose halves are equal, still nets more. The defaults stay; the
crawl of gamma 0.99 under the plain law remains the longest stretch and the law its end.
The moment costs of this batch were disturbed by the traces running beside it and are not
read.

**The longer horizons, under servos.** Gamma 0.99 bought the crawl and the plain law ended
it; the next lengths of the horizon, 0.995 (two hundred moments) and 0.998 (five hundred),
asked whether more credit for the return swing holds it. Servos, seeds 0–2, 30,000 moments,
frozen copies, and the trace over the first 20,000 moments:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| servos, founders (table above) | +0.205 (+0.08; 4.7 → 2.9; 0.99; 0.017) | +0.360 (+0.03; 3.1 → 4.2; 1.00; 0.021) | +1.503 (−0.10; 3.7 → 3.9; 0.99; 0.023) |
| servos, gamma 0.99 (table above) | +1.689 (−0.01; 3.8 → 5.0; 0.00; 0.049) | +0.939 (+0.03; 3.1 → 3.9; 0.98; 0.025) | +0.674 (+0.08; 3.8 → 4.8; 0.99; 0.025) |
| servos, gamma 0.995 | +0.373 (0.00; 3.9 → 4.0; 0.00; 0.052) | −0.017 (−0.01; 3.2 → 4.8; 0.91; 0.027) | +1.362 (+0.01; 3.2 → 3.4; 0.98; 0.024) |
| servos, gamma 0.998 | +0.225 (+0.06; 4.2 → 4.9; 0.99; 0.026) | +0.293 (−0.03; 2.6 → 4.4; 1.00; 0.021) | +0.995 (+0.11; 3.4 → 3.7; 0.95; 0.028) |
| frozen copies, 3000 moments (0.995 / 0.998) | 0.075 / 0.130 | 0.148 / 0.152 | 0.152 / 0.122 |

**Reading:** the crawl comes again and ends the same way. Under 0.995 seed 2 walks on
legs 5 and 6 for 10,000 moments (+0.46 and +0.49 m per 5000, the strokes +3.15 against
−2.69 of the rest, 3.3 feet down), then the chain flag goes out and the blocks fall to
+0.28, +0.04, +0.08, +0.01: the same crawl gamma 0.99 found on seed 0, on another seed,
not longer. Seed 0 under 0.995 is the stall at its purest: after one block of strokes
(+0.37 m) every foot stays down, no leg lifts again through 15,000 traced moments, the hips
hold at −0.28 within 0.02, and the body sways at 0.052 a moment, the highest income of any
run, calm (aroused 0.00) from its second block on. Seed 1 is the founders' scatter. Under
0.998 no chain lights for more than one window on any seed; seed 2 keeps +0.66 m of its
first block and then earns +0.03 to +0.13 per 5000 from one or two legs that lift 150–200
times a block, the rest of the feet planted (3.7 at the end), and seeds 0 and 1 are the
founders'. The copies are the hip sweep, and seed 0's under 0.995 is short of one. A longer
horizon lengthens nothing: the credit that reaches the return swing at 0.99 reaches it at
0.998 too, and so does the sway's, which is why the longest stretch of this reading is the
same 10,000 moments. The founders stay (gamma 0.95); 0.99 remains the reading to build on.
4.8–5.1 ms a moment, 2.2 for the stalled seed 0, which hardly learns.

**The setpoint senses.** Under servos a joint on its way to an end reads, in its angle,
as a joint that was never sent, and the brain cannot tell its own last command from the
world unless it recalls it. `--setpoints` adds the sixteen setpoints to the readings (280
inputs), so the brain's state holds what it told each servo; a swing then has a sensed
beginning the credit can reach. Servos, seeds 0–2, 30,000 moments, under the founders and
under gamma 0.99, the frozen newborn with the setpoints, frozen copies, and the trace over
the first 20,000 moments; random does not read the senses and is the servo row above:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| servos, founders (table above) | +0.205 (+0.08; 4.7 → 2.9; 0.99; 0.017) | +0.360 (+0.03; 3.1 → 4.2; 1.00; 0.021) | +1.503 (−0.10; 3.7 → 3.9; 0.99; 0.023) |
| servos, setpoints | +1.207 (+0.02; 3.1 → 4.1; 0.97; 0.029) | +0.758 (+0.02; 4.1 → 5.2; 1.00; 0.022) | +0.596 (−0.07; 3.0 → 4.9; 0.99; 0.022) |
| servos, gamma 0.99 (table above) | +1.689 (−0.01; 3.8 → 5.0; 0.00; 0.049) | +0.939 (+0.03; 3.1 → 3.9; 0.98; 0.025) | +0.674 (+0.08; 3.8 → 4.8; 0.99; 0.025) |
| servos, setpoints, gamma 0.99 | +0.850 (+0.20; 3.1 → 3.7; 0.92; 0.030) | +0.549 (+0.04; 4.3 → 5.6; 1.00; 0.020) | +1.428 (+0.28; 3.7 → 4.1; 0.93; 0.029) |
| frozen newborn, setpoints, 3000 moments | +0.053 (2.0) | −0.062 (5.0) | +0.099 (4.0) |
| frozen copies, 3000 moments (founders / gamma 0.99) | 0.157 / 0.155 | 0.149 / 0.152 | 0.151 / 0.112 |

**Reading:** the setpoints change the first block and not the life. Every setpoint run
earns its first 5000 moments more than the founders' body (+0.93, +0.54, +0.62 m under the
founders; +0.37, +0.39, +0.56 under gamma 0.99), with three legs in the chain for seed 0's
first window (legs 0, 1 and 5, the strokes +3.23 against −2.30), and then the hips reach
the back stop (mean −0.57 to −0.59, within 0.03–0.06) and every block after is within
±0.15 m, the income 0.020–0.030 a moment: the stroke, then the sway, as under every law
before. No run stalls calm the way seed 0 of gamma 0.99 and 0.995 did; the sway under the
setpoints is a lifting one (one to four legs lifting 100–240 times a block, 3.6–5.8 feet
down). The one run whose late blocks grow, seed 2 under gamma 0.99 (+0.11, +0.15, +0.17,
+0.28 per 5000), is a ratchet on legs 1 and 4, not a chain: continued for 10,000 traced
moments it earns +0.09 and +0.05, the chain flag on leg 4 in one window and out in the
next. The copies are the hip sweep. What the setpoints give the brain is what its working
memory already held of its own commands; the credit's reach, not the state, is what the
return swing lacks. The senses stay without them; `--setpoints` is an option named in its
readings. 5.8–6.0 ms a moment for 280 inputs against 4.8–5.1 for 152.

**The leaking mark.** The mark starved the crawl and the plain law fed the sway; between
them lies a mark that leaks: `--old-ground S` pays new ground whole and progress on ground
already reached the share S of what the plain law pays (0 the mark, 1 the plain law). At
0.2 a sway keeps a fifth of its forward half, a crawl keeps its new ground; the arithmetic
of the crawl as found (+3.46 m of strokes against −2.61 of the rest per 10,000 moments,
the strokes mostly on old ground) says it nets about what the sway does, so the reading
asked whether the brain finds a cleaner crawl when the sway pays less. Servos, gamma 0.99,
the mark leaking 0.2 and 0.5, seeds 0–2, 30,000 moments; random under each leak for the
income chance nets; frozen copies; the trace over the first 20,000 moments:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| servos, gamma 0.99 (table above) | +1.689 (−0.01; 3.8 → 5.0; 0.00; 0.049) | +0.939 (+0.03; 3.1 → 3.9; 0.98; 0.025) | +0.674 (+0.08; 3.8 → 4.8; 0.99; 0.025) |
| servos, gamma 0.99, high-water (table above) | −1.673 (+0.25; 3.7 → 3.6; 1.00; 0.000) | −0.276 (+0.05; 2.7 → 2.5; 1.00; 0.000) | −0.115 (−0.38; 3.8 → 4.2; 1.00; 0.000) |
| servos, gamma 0.99, mark leaking 0.2 | −0.212 (−0.05; 4.3 → 4.5; 1.00; 0.006) | −0.215 (+0.10; 3.0 → 4.1; 1.00; 0.006) | +0.446 (−0.11; 3.7 → 4.3; 1.00; 0.007) |
| servos, gamma 0.99, mark leaking 0.5 | −0.079 (0.00; 3.9 → 4.6; 1.00; 0.012) | +0.271 (−0.07; 3.1 → 4.5; 1.00; 0.014) | +0.230 (−0.09; 3.7 → 3.7; 1.00; 0.012) |
| random, servos, mark leaking 0.2 | −1.131 (+0.41; 2.9 → 3.1; 0.012) | +0.508 (−0.17; 3.1 → 3.2; 0.013) | −0.120 (−0.95; 3.1 → 3.3; 0.012) |
| random, servos, mark leaking 0.5 | −1.131 (+0.41; 2.9 → 3.1; 0.030) | +0.508 (−0.17; 3.1 → 3.2; 0.031) | −0.120 (−0.95; 3.1 → 3.3; 0.030) |
| frozen copies, 3000 moments (0.2 / 0.5) | 0.155 / 0.151 | 0.079 / 0.144 | 0.128 / 0.154 |

**Reading:** the leak pays chance its share and the learner less. Random's income is the
leak's share of its plain income (0.012 and 0.030 a moment for 0.060), as the law says.
The brain's income falls through the life under either leak, from 0.017 a moment in the
first 5000 to 0.006 under 0.2 and from 0.033 to 0.012 under 0.5, the end below random's at
both doses: what it learns is, again, to net less than chance earns, the way it did under
the mark and the charge. No crawl forms. The chain flag lights on one to three legs in a
first window (legs 2, 3 and 5 for seed 2 under 0.2, +0.57 m) and is out in the next; from
10,000 moments on the hips sit at the back stop (−0.52 to −0.58 within 0.05–0.08), five to
seven legs lift 100–220 times a block, and the blocks are within ±0.3 m, the body
wandering as random does, 120–200 rad turned in the life. Seed 1 under 0.2 stays below its
start for 30,000 moments. The copies are the hip sweep, two of them short of it. The leak
does what the arithmetic said: at 0.2 the sway's jitter pays a fifth, which is too little
to carry the half chain the plain law carried for 20,000 moments, and the crawl that would
beat the sway under this law is one the brain never found under any law; a leak large
enough to carry the jitter is the plain law. The defaults stay (the mark off, the leak 0).
4.9–5.2 ms a moment.

**The found crawl under the leak-free laws.** Every law above was measured from birth, where
what the law does to a crawl is tangled with what it does to the search for one. The
diagnosis that separates them: take a crawl the plain law found, and continue it under the
laws that starved the newborn, to see whether they starve a crawl that exists or only the
finding of it. Two specimens, one of each: seed 0 under servos and gamma 0.99, saved at
15,000 moments in its crawl on legs 2–3 (+1.19 m, +0.40 in its last 5000, 4.0 feet planted);
seed 2 under servos and gamma 0.995, saved at 10,000 moments in its crawl on legs 5–6
(+0.95 m, +0.49 in its last 5000, 3.3 planted). Each save is continued, with its world and
its pending outcome, under the plain law, the mark, the charge (backward pay 1) and the mark
leaking 0.2: 15,000 moments for seed 0, 20,000 for seed 2, and 50,000 for the two that held
longest and the plain law beside them; frozen copies of the saves, 3000 moments, for the hip
sweep's floor. Each number is one specimen under one law. A life that changes its law is a
schedule, a different model to be named as such; these continuations measure what such a
schedule would do, they are not readings of any law from birth.

| continued, forward metres per 5000 moments (r/moment; planted feet; aroused, first → last window) | seed 0 from 15,000, legs 2–3 | seed 2 from 10,000, legs 5–6 |
| --- | --- | --- |
| plain law | +0.45 +0.06 −0.01 (0.034 → 0.049; 4.0 → 5.0; 0.81 → 0.00) | +0.28 +0.04 +0.08 +0.01 (0.025; 3.3; 0.97) |
| high-water mark | +0.26 +0.14 +0.10 (0.003 → 0.002; 3.9 → 4.1; 1.00) | +0.06 −0.13 +0.01 −0.14 (0.002 → 0.000; 3.3 → 3.7; 1.00) |
| charge, backward pay 1 | +0.10 −0.10 −0.04 (0.001 → 0.000; 3.9; 1.00) | +0.25 −0.05 −0.05 −0.05 (0.002 → 0.000; 3.2 → 3.3; 1.00) |
| mark leaking 0.2 | +0.48 +0.06 +0.05 (0.013 → 0.008; 4.0; 1.00) | +0.17 +0.32 +0.17 +0.21 (0.008 → 0.009; 3.2 → 3.8; 1.00) |
| frozen copy of the save, 3000 moments | 0.098 | 0.152 |

| 50,000 moments on, forward metres per 5000 (total; rad turned) | |
| --- | --- |
| seed 0 under the mark | +0.26 +0.14 +0.10 −0.07 +0.11 −0.15 −0.07 −0.07 −0.05 +0.30 (+0.50; 458) |
| seed 2 under the mark leaking 0.2 | +0.17 +0.32 +0.17 +0.21 +0.04 −0.10 +0.16 +0.00 +0.02 +0.07 (+1.07; 203) |
| seed 2 under the plain law | +0.28 +0.04 +0.08 +0.01 −0.03 +0.04 +0.07 −0.13 −0.04 +0.02 (+0.35; 166) |

**Reading:** every law loses the crawl it was handed; they differ in how. Under the plain law
seed 0 does what its newborn did (the gamma 0.99 row above): one more block of its crawl
(+0.45, the chain flag on legs 2–3), then the lifts fall from 130–156 a leg a block to 43–59
and to none, five feet stay planted, and the life ends calm at 0.049 a moment with no learning
sweep, the sway that pays better than the crawl. Seed 2 keeps its three lifting legs (0, 5, 6
at 90–150 lifts a block) and loses the ground they made: +0.28 in the first block, within
±0.1 in every block after, the strokes +1.8 m against −1.75 of the rest, the income 0.025 for
50,000 moments. Under the mark seed 2 loses the crawl in one block and seed 0 in three: the
lifts rise to 160–200 a leg, the hips leave the stop (−0.44 within 0.15), the strokes pay +4.8
against −4.5, and the life wanders within ±0.3 a block for 45,000 moments at an income of
0.000, 458 rad turned, the newborn's end under the mark. Under the charge both stop: the same
legs keep lifting but the stroke and the return shrink together (seed 0 from +2.5 / −2.1 to
+0.9 / −1.0 per 5000, seed 2 from +1.3 / −1.0 to +0.75 / −0.8), the income 0.000; the
charge teaches to move less, not to move clean. Under the leak seed 0 holds one block
(+0.48, legs 2–3) and then nets what the plain law nets, awake; seed 2 is the one specimen
that holds: +0.87 over 20,000 moments, the chain flag on leg 6 in three windows of four, the
strokes rising from +2.6 to +3.1 against −2.4 to −2.9 of the rest. From 20,000 on it fades
as leg 1 joins the lifting (0 → 155–209 a block), the planted feet rise to 4.0 and the
strokes come to +2.4 against −2.4: +1.07 in 50,000 moments, 0.021 a thousand, under the
0.051 a thousand the frozen copy's sweep makes, and the +0.87 it held was 0.044 a thousand,
the sweep's pace. So the laws that starved the newborn starve the found crawl too, the leak
slowest, and the plain law keeps a crawl no better than it finds one: the loss is in the
credit, which pays the stroke at once and the return never, under every law built, and a
schedule of these laws is not a next step. 3.7–4.0 ms a moment.

**The recall memory and the temperature.** Two genes of the brain that no reading above had
moved, each aimed at the credit from the brain's side rather than the law's. The recall
memory (`--gene recall=true`; off in the founders for cost, and named in `AGENTS.md` as a
gene to measure, not a default to flip) gives the brain a memory of situations met before,
the place where a cycle might be seen as a cycle. The temperature (`--gene temperature=T`,
founder 0.3) is the policy's sampling noise: if the sway earns on the forward half of a
jitter, a colder brain jitters less and the sway pays less without a change of law; 0.15 is
that reading and 0.5 its converse. Servos, gamma 0.99, seeds 0–2, 30,000 moments; frozen
copies; the trace over the first 20,000 moments:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| servos, gamma 0.99 (table above) | +1.689 (−0.01; 3.8 → 5.0; 0.00; 0.049) | +0.939 (+0.03; 3.1 → 3.9; 0.98; 0.025) | +0.674 (+0.08; 3.8 → 4.8; 0.99; 0.025) |
| servos, gamma 0.99, recall on | +0.415 (−0.04; 4.1 → 5.4; 1.00; 0.020) | +0.360 (+0.41; 1.6 → 3.8; 0.86; 0.032) | +0.499 (+0.02; 4.6 → 4.2; 1.00; 0.025) |
| servos, gamma 0.99, temperature 0.15 | +0.939 (+0.28; 4.9 → 4.1; 0.85; 0.034) | +0.051 (+0.08; 3.8 → 3.5; 0.91; 0.032) | +0.580 (+0.05; 5.2 → 4.4; 0.93; 0.031) |
| servos, gamma 0.99, temperature 0.5 | +0.015 (−0.17; 3.4 → 3.4; 0.96; 0.024) | +0.252 (−0.03; 3.1 → 4.8; 0.98; 0.030) | +0.258 (+0.07; 3.3 → 3.3; 0.96; 0.028) |
| random, servos (the tables above) | −1.131 (+0.41; 2.9 → 3.1; 0.060) | +0.508 (−0.17; 3.1 → 3.2; 0.060) | −0.120 (−0.95; 3.1 → 3.3; 0.060) |
| frozen copies, 3000 moments (recall / 0.15 / 0.5) | 0.152 / 0.153 / 0.155 | 0.116 / 0.141 / 0.152 | 0.153 / 0.154 / 0.152 |

**Reading:** neither gene changes what happens after the first block, and both lose the
stretch gamma 0.99 bought on seed 0. With recall the first blocks are +0.30, +0.10 and +0.53
and every block after within ±0.2 but seed 1's last (+0.41, a brain that began on 1.6 feet
and spent the life getting up, its income rising from 0.013 to 0.032); by 10,000 moments
seeds 0 and 2 sit at the back stop (−0.53 to −0.58 within 0.04–0.08) with five to six legs
lifting 140–190 times a block and the strokes paying what the rest costs (+2.5 / −2.5,
+1.8 / −1.8), the sway, at twice the cost: 9.7–10.4 ms a moment for 4.9–5.2 without the
memory. The cold brain is not the one the arithmetic wanted. Its income is not lower but
higher (0.031–0.034 a moment at the end against 0.025 at the founder's 0.3): the jitter the
sway earns on is not the sampling noise but a learned rhythm, hips at the stop within
0.02–0.07 and four to five legs lifting 130–200 times a block, which a colder policy
executes more cleanly. Seed 0 under 0.15 keeps one block (+0.31) and seed 2 two (+0.24,
+0.23, the chain flag on leg 4 and then 5), and seed 1 never begins: +0.03 and within ±0.07
for the life, three legs lifting and the hips at −0.29 within 0.03 from the second window
on, a sway from birth with no stroke to lose. The warm brain has the largest first block of
the three (seed 0, +0.74, the chain flag on leg 1, the hips still moving) and then loses it
(−0.04, −0.34, −0.12, …) as the hips go to the stop; its other seeds are within ±0.2
throughout. The copies are the sweep. The defaults stay (recall off, temperature 0.3): the
memory finds the sway at twice the price, and the temperature, in either direction, moves
only the first block.

**The critic's rate.** The third handle the brain has on its credit, after the eligibility
and the horizon above: the critic's learning rate (`--gene eta_critic=R`, founder 5, the
multiple of the actor's rate at which the value of a situation is learned). A slower critic
(1) averages a situation's value over more of the cycle, a faster one (20) follows the
stroke. Servos, gamma 0.99, seeds 0–2, 30,000 moments; frozen copies; the trace over the
first 20,000; and the one life that held, continued 30,000 moments more under the same law:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| servos, gamma 0.99 (table above) | +1.689 (−0.01; 3.8 → 5.0; 0.00; 0.049) | +0.939 (+0.03; 3.1 → 3.9; 0.98; 0.025) | +0.674 (+0.08; 3.8 → 4.8; 0.99; 0.025) |
| servos, gamma 0.99, critic rate 1 | +0.412 (0.00; 3.3 → 3.0; 0.00; 0.043) | +0.899 (+0.16; 2.0 → 3.2; 0.99; 0.020) | +0.369 (0.00; 3.6 → 3.2; 0.99; 0.020) |
| servos, gamma 0.99, critic rate 20 | +0.425 (−0.04; 3.9 → 3.7; 0.96; 0.027) | +1.771 (+0.13; 2.8 → 4.2; 0.79; 0.033) | +0.713 (+0.01; 3.7 → 3.7; 0.99; 0.024) |
| frozen copies, 3000 moments (1 / 20) | 0.059 / 0.142 | 0.149 / 0.111 | 0.155 / 0.152 |
| critic rate 20, seed 1, per 5000 moments, 0–30,000 | +0.31 +0.23 +0.67 +0.17 +0.26 +0.13 | | |
| the same life continued, 30,000–60,000 | +0.10 +0.21 +0.06 +0.20 +0.18 +0.01 (+0.757; 0.023–0.026 a moment; 4.1 planted; aroused 0.96) | | |

**Reading:** the fast critic holds a stretch on one seed, the slow one stalls one. Critic 1
on seed 0 is the rocking of the rate body back on the servos: one block (+0.41), then not
one lift in 25,000 moments, three feet planted, the hips at −0.14 within 0.00, an income of
0.043 a moment from a body that rocks in place and never wakes (aroused 0.00, 2.0 ms a
moment with nothing to learn); its copy walks 0.059, under the sweep. Seed 1 under critic 1
nets +0.2 a block for four blocks with the chain flag on leg 5, at an income of 0.020–0.031,
and seed 2 ends in the sway. Critic 20 on seed 1 is the longest-held forward motion in this
project on this body: +1.771 m in 30,000 moments with every block positive, +0.67 in the
third (the strokes +2.27 against −1.60 of the rest, the chain flag on legs 4 and 5), and
continued 30,000 moments more it still moves, +0.757, eleven blocks positive of twelve,
+2.53 m in 60,000 moments. What holds is not the chain crawl: from 20,000 on the chain flag
is out, the hips sit at the back stop (−0.59 within 0.04), five legs (1, 2, 4, 6, 7) lift
150–190 times a block and the strokes net a twelfth of what they pay (+1.65 against −1.45
per 5000). It is a rhythm that keeps a little of each stroke, 0.025 m a thousand moments,
half the pace of a frozen copy's hip sweep and a sixth of the crawl gamma 0.99 found; its
own copy walks 0.111, the walking being in the sampling, not the greedy answer. Seeds 0 and
2 under critic 20 lose their first blocks (+0.33, +0.65) to the sway within 15,000 moments.
One seed of three, and below the founder's reading on seed 0 by 1.3 m: the founder stays
at 5. The critic's rate is the first gene whose specimen did not lose its ground in 60,000
moments, and the one reading above worth more seeds. 4.2–4.6 ms a moment.

**The reward learner's options.** The library's reward learner has three settings beyond
the five above, documented in cadence's docs/reward.md and off in the founder: the actor's
own eligibility decay (`--gene hold=H`; the founder, none, lets an act's credit fade at gamma
times lam a moment, 0.57 here with gamma 0.99; 0.9 remembers about 10 moments, 0.97 about
33), the policy's entropy in the actor's step (`--gene entropy=E`; 0.1 is the docs' dose; it
learns on the cpu backend only and costs about a third more a moment) and maturity
(`--gene maturity=true`; the slow use of plasticity that lowers the actor's step as it is
spent, scale 10 and recovery 200,000 moments the founder's). Servos, gamma 0.99, seeds 0–2,
30,000 moments; frozen copies; the trace of the entropy lives; and the one life that walked
farthest, continued 30,000 moments more under the same law:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| servos, gamma 0.99 (table above) | +1.689 (−0.01; 3.8 → 5.0; 0.00; 0.049) | +0.939 (+0.03; 3.1 → 3.9; 0.98; 0.025) | +0.674 (+0.08; 3.8 → 4.8; 0.99; 0.025) |
| servos, gamma 0.99, entropy 0.1 | +0.564 (+0.01; 3.3 → 3.0; 0.00; 0.070) | +5.218 (+0.63; 3.1 → 4.6; 0.46; 0.053) | +4.263 (+0.69; 4.9 → 4.9; 0.00; 0.052) |
| servos, gamma 0.99, hold 0.9 | +1.119 (+0.30; 5.2 → 4.2; 0.64; 0.040) | +0.225 (+0.14; 3.6 → 4.6; 0.72; 0.038) | −0.563 (+0.15; 4.5 → 4.8; 0.80; 0.035) |
| servos, gamma 0.99, hold 0.97 | +0.837 (−0.03; 4.5 → 4.9; 0.65; 0.042) | +0.056 (0.00; 2.4 → 3.0; 0.00; 0.074) | +0.847 (0.00; 3.9 → 5.0; 0.00; 0.043) |
| servos, gamma 0.99, maturity | +1.527 (+0.15; 4.2 → 4.5; 0.91; 0.031) | +0.213 (0.00; 3.1 → 3.8; 0.75; 0.038) | −0.213 (−0.05; 3.5 → 3.8; 0.99; 0.022) |
| frozen copies, 3000 moments (entropy / hold 0.9 / hold 0.97 / maturity) | 0.066 / 0.128 / 0.086 / 0.157 | 0.119 / 0.151 / 0.038 / 0.112 | 0.122 / 0.155 / 0.057 / 0.154 |
| entropy 0.1, per 5000 moments, 0–30,000 | +0.58 −0.02 +0.01 +0.01 −0.02 +0.01 | +1.35 +1.61 +0.70 +0.36 +0.58 +0.63 | +0.84 +0.69 +0.69 +0.69 +0.69 +0.69 |
| entropy 0.1, seed 1, continued 30,000–60,000 | | +1.30 +1.28 +1.30 +1.27 +1.30 +1.28 (+7.717; 0.046 a moment; 5.4 planted; aroused 0.00; every block positive) | |

**Reading:** the entropy term is the one option that moved a reading, and by more than any
gene above: two seeds of three walked 4 to 5 m in 30,000 moments, three times the previous
best on this body, and seed 1 continued is the first life of this project that walks every
block, +12.9 m in 60,000 moments. What walks is not a gait of the legs: on both seeds the
brain fell calm into a cycle it repeats to the bit. Seed 2 from moment 1300 (aroused 0.00,
not one learning sweep after, 0.0137 m per 100 moments for 28,700 moments): legs 3, 4 and
5 in the air for good, leg 2 tapping (2943 lifts of one moment), the knee of leg 2 and the
hips of legs 3 and 7 flipping end to end on a period of ten, a jitter that nets 0.14 m a
thousand moments across the arena (3.9 m from the origin, its heading turned 2.7 rad in
all). Seed 1 learned for 29,000 moments (aroused 0.46, 177 blocks of 300 positive, all
eight legs lifting 500–800 times each) and then fell calm into a one-legged row: six legs stand,
legs 5 and 7 hang in the air, and leg 1 alone sweeps its hip back on the ground (−0.021 rad
a moment) and forward in the air (+0.016) on a period of sixteen, 188 strokes per 3000
moments, 0.26 m a thousand moments at 1.6 ms a moment, and it turns with it: −0.79 rad per
3000, a circle of about a metre's radius, so the body that walked 7.7 m forward in the
continuation is 1.2 m from where it started it. The law pays forward along the heading and
never asked for a straight line (the yaw term above is the hint for that, and it stopped
every walk it was dosed on; the course law, "The course" below, is the law that asks for one). Seed 0 under entropy fell calm by 5000 moments into the rock
(0.070 a moment, three feet planted, aroused 0.00 for 25,000 moments). The copies walk 0.07–0.12, under
the sweep: both cycles are the sampled policy's, not the greedy answer's. The hold at 0.9
and 0.97 and maturity change nothing a seed holds: hold 0.9 gains +0.30 in its last block
on seed 0 and loses 0.56 on seed 2, hold 0.97 stalls two seeds calm, maturity holds +0.15 a
block on seed 0 only and sways the others. The founder stays as it is: entropy 0.1 is not a
walk of the legs but a calm cycle that happens to go forward, found twice in three seeds,
and it is the one gene whose next reading is worth seeds 3–5, a dose of 0.03 and 0.3, and
the stance term with it (the calm cycles stand on five feet already). Entropy 0.1 costs
3.7 ms a moment aroused, 2.0 calm; hold 3.3–3.8; maturity 4.3.

**The observation codes.** Cadence's docs/senses.md offers a vector brain two ways to read
its senses other than the observation itself, both off in the founder and genes here:
`--gene code=U`, a fixed random sparse code of U units (one in ten active, 24 of 64) that the
cortex reads in place of the 152 coded inputs and the memories key on; and
`--gene learned_code=U`, a code region of U units between the senses and the first module,
learned by local rules while the brain lives, on which the cortex computes and the memories
key (joined to the observation's own identity), with `--gene learned_code_fields=128,16,8`
giving the joint angles, the speed and yaw rate, and the planted flags units of their own.
(The doc's image inputs, learned receptive fields and ports are for a screen; this body has
none.) Servos, gamma 0.99, seeds 0–2, 30,000 moments, 64 units each (the docs' width);
frozen copies:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| servos, gamma 0.99 (table above) | +1.689 (−0.01; 3.8 → 5.0; 0.00; 0.049) | +0.939 (+0.03; 3.1 → 3.9; 0.98; 0.025) | +0.674 (+0.08; 3.8 → 4.8; 0.99; 0.025) |
| servos, gamma 0.99, code 64 | +0.129 (+0.02; 3.8 → 3.9; 0.98; 0.026) | +0.841 (+0.11; 3.3 → 3.7; 0.99; 0.020) | −0.130 (+0.14; 3.9 → 3.9; 0.99; 0.020) |
| servos, gamma 0.99, learned code 64 | +0.056 (−0.14; 3.2 → 4.4; 1.00; 0.019) | +0.863 (0.00; 2.7 → 3.5; 0.99; 0.024) | +0.779 (0.00; 3.3 → 4.7; 0.79; 0.034) |
| servos, gamma 0.99, learned code 64, fields 128,16,8 | −0.474 (+0.03; 3.2 → 1.8; 1.00; 0.009) | +0.335 (+0.11; 2.8 → 2.8; 0.99; 0.015) | +0.424 (+0.09; 3.6 → 4.0; 0.69; 0.040) |
| frozen copies, 3000 moments (code / learned / learned with fields) | 0.155 / 0.151 / 0.145 | 0.153 / 0.152 / 0.149 | 0.151 / 0.151 / 0.151 |
| learned code 64, per 5000 moments, 0–30,000 | +0.15 +0.11 0.00 −0.01 −0.05 −0.14 | +0.69 −0.05 +0.17 0.00 +0.05 0.00 | +0.63 +0.21 +0.04 −0.05 −0.05 0.00 |

**Reading:** neither code walks, and both read less than the senses themselves: every life
under a code is below its seed's baseline (code 64 +0.13 / +0.84 / −0.13, the learned code
+0.06 / +0.86 / +0.78 against +1.69 / +0.94 / +0.67), the fields make it worse (−0.47 / +0.34
/ +0.42), and the one thing the codes change is the arousal: the baseline's seed 0 fell calm
into its posture by 5000 moments, where under every code every seed stays aroused for the
whole life (0.98–1.00 in the last 5000, the learned code's seed 2 at 0.69–0.79) and sways,
about half the blocks positive (141–175 of 300) and 75–183 rad turned. The fixed code's
first 5000 moments are the baseline's first block or less; the learned code's are its best
(+0.69 and +0.63 on seeds 1 and 2, the baseline's +0.22 and +0.33 there), and then it stops:
the region keeps learning (20,000–29,000 steps on 30,000 moments) and the code it settles on
is dense on the life's own path (20–37 of 64 units lit, a newborn's 17–26 on the same poses)
and dark off it (its thresholds rose from 0.1 to 3.3–4.2, so a pose the life never took lights
0–5 units), the docs' behaviour, and what it learned to tell apart is not what the reward
asked for. The copies walk 0.15, the greedy answer's. The fixed code halves the parameters
(12,097 against 22,041) at 3.9–4.3 ms a moment; the learned code costs 8.6–10.0 ms a moment
(24,025 parameters, 392 neurons) against 3.3 for the founder. The founder stays: the senses
themselves, read by the cortex at scale 4. What a code would be for, a world whose
observation is wider than the cortex should read or whose members share contexts the lesson
should carry across, is not this world, whose 152 inputs are already a code (the population
code, "The senses").

**The course.** The course law (`--course`, "The reward") measures the same outcome, the
moment's motion, along one fixed heading of the world instead of along the body's own, so
that a circle nets nothing; the compass (`--compass`, "The senses") gives the brain that
heading's cosine and sine, 168 inputs. Servos, gamma 0.99, seeds 0–2, 30,000 moments, the
course the birth heading (0), backward pay 0 and window 1 as in the founder; random under the
course law; frozen copies at 3000 moments. The first number of a cell is the metres along the
course, every life's, the plain-law lives' along their birth heading:

| 30,000 moments, metres along the course (forward metres; course metres in the last 5000; rad turned; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| servos, gamma 0.99 (the founder, plain law) | +0.876 (+1.69; +0.04; 150; 3.8 → 5.0; 0.00; 0.049) | +0.110 (+0.94; +0.03; 113; 3.1 → 3.9; 0.98; 0.025) | +0.841 (+0.67; +0.06; 107; 3.8 → 4.8; 0.99; 0.025) |
| course 0 | +0.503 (+0.40; +0.06; 211; 3.9 → 4.3; 0.92; 0.028) | −0.291 (−0.07; −0.15; 167; 2.2 → 4.2; 0.92; 0.026) | +1.188 (+1.37; +0.16; 205; 3.6 → 3.3; 0.63; 0.043) |
| compass, plain law | −0.174 (−0.11; −0.09; 111; 4.4 → 4.9; 1.00; 0.023) | +0.796 (+1.06; +0.01; 152; 4.2 → 3.6; 0.90; 0.025) | +0.291 (+0.36; +0.03; 118; 5.3 → 4.0; 0.96; 0.029) |
| course 0, compass | +0.312 (+0.42; −0.02; 183; 4.9 → 5.0; 0.00; 0.041) | +0.594 (+0.21; +0.38; 220; 3.3 → 4.0; 0.82; 0.033) | +0.403 (+0.35; −0.02; 113; 5.4 → 4.8; 0.99; 0.022) |
| course 0, compass, entropy 0.1 | +0.824 (+0.90; −0.01; 178; 3.8 → 4.0; 0.00; 0.050) | +1.075 (+1.30; +0.01; 241; 3.8 → 3.0; 0.00; 0.046) | +2.890 (+3.67; +0.14; 208; 3.8 → 3.9; 0.72; 0.038) |
| random, course 0 | −0.880 (−1.13; +0.04; 260; 2.9 → 3.1; 0.033) | +0.934 (+0.51; +0.11; 266; 3.1 → 3.2; 0.037) | −0.458 (−0.12; −0.01; 256; 3.1 → 3.3; 0.046) |
| frozen newborn with the compass, 3000 moments, course metres | +0.018 | +0.131 | −0.078 |
| frozen copies, 3000 moments, course metres (course / compass / both / both with entropy) | 0.093 / 0.149 / 0.094 / 0.112 | 0.118 / 0.110 / −0.018 / 0.132 | 0.073 / 0.154 / 0.152 / 0.151 |
| course 0, compass, entropy 0.1, per 5000 moments, course / forward | +0.84 0.00 −0.01 0.00 0.00 −0.01 / +0.90 0.00 −0.01 0.00 0.00 −0.01 | +0.32 +0.73 +0.01 +0.01 +0.01 +0.01 / +0.54 +0.77 0.00 0.00 0.00 0.00 | +0.90 +0.68 +0.04 +0.45 +0.68 +0.14 / +0.72 +0.92 +0.25 +0.48 +0.86 +0.43 |

**Reading:** the course law closes the circle and not the sway. Under the plain law the
founder's seed 0 walked a round: +1.69 m forward and +0.88 along its birth heading, with one
window of 5000 moments in which it went +0.45 forward and −0.24 along the heading; under the
course law no life nets that way, since a round pays nothing. But a sway along the course
pays as well as it does under the plain law, because backward along the course is free
(backward pay 0) and the window is one moment: three of the nine course lives fell calm into
one (course with compass seed 0 at 900 moments, with entropy seeds 0 and 1 at 2800 and 6500)
and earn 0.041–0.050 a moment from it, the founder's calm seed 0's 0.049, netting −0.02 to
+0.01 m per 5000 moments along the course. The compass by itself changes nothing (−0.17 /
+0.80 / +0.29 against +0.88 / +0.11 / +0.84, aroused to the end on every seed); the course
alone, with and without the compass, is within the baseline's spread (+0.50 / −0.29 / +1.19
and +0.31 / +0.59 / +0.40), and every course life turns more than the plain-law lives
(167–241 rad against 107–152: a heading is not a sense of the plain law's body, and under the
course law a turn is what the brain can do to change what the same stroke pays). The one
life that goes the farthest along a line of this project is the course with the compass and
the entropy term at 0.1 on seed 2: +2.89 m along the birth heading and +3.67 forward, still
aroused (0.72) and still learning at 30,000 moments, every window positive along the course
(+0.90 +0.68 +0.04 +0.45 +0.68 +0.14), 0.79 of its forward metres along the course against the
founder's seed 0's 0.52, 208 rad turned. One seed is a life, not a dose; the entropy term is
the gene that walked the founder's seeds too ("The entropy term"), and whether the course adds
to it is seeds 3–5. The copies walk 0.09–0.15, the greedy answer's. Random under the course law
is −0.88 / +0.93 / −0.46: a random walk's course metres are as wide as its forward ones. The
course costs 4.8–5.4 ms a moment learning and 2.3–2.7 calm, the founder's. Both stay off:
they are named in every reading, and what the sway shows is that a line is the course with
the backward pay on or a window, the next thing to measure.

**The ground sense.** `--ground` ("The senses") gives the brain its velocity over the ground
in the world's frame, two readings through the code: the motion the course law pays, read
from the world instead of left to be computed as the product of the forward speed and the
compass. Servos, gamma 0.99, seeds 0–2, 30,000 moments; with the course (0, the birth heading)
and the compass, and with the entropy term at 0.1 beside them, the dose that went farthest
under the course above; random under the course is the row above; frozen newborn and copies at
3000 moments:

| 30,000 moments, metres along the course (forward metres; course metres in the last 5000; rad turned; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| servos, gamma 0.99 (the founder, plain law; table above) | +0.876 (+1.69; +0.04; 150; 3.8 → 5.0; 0.00; 0.049) | +0.110 (+0.94; +0.03; 113; 3.1 → 3.9; 0.98; 0.025) | +0.841 (+0.67; +0.06; 107; 3.8 → 4.8; 0.99; 0.025) |
| ground, plain law | +1.003 (+1.19; 0.00; 147; 4.2 → 3.4; 0.87; 0.032) | +3.004 (+2.74; −0.03; 133; 3.5 → 4.3; 0.96; 0.029) | +0.269 (+0.27; −0.06; 109; 4.7 → 4.4; 0.99; 0.024) |
| course 0, ground | +0.741 (+1.06; +0.22; 154; 2.7 → 3.4; 0.92; 0.028) | +0.342 (+0.37; +0.03; 196; 2.7 → 3.4; 0.83; 0.029) | +0.594 (+0.66; −0.02; 114; 5.3 → 5.0; 0.99; 0.024) |
| course 0, compass, ground | +0.370 (+0.64; −0.07; 134; 2.5 → 3.7; 0.97; 0.028) | +0.980 (+0.39; 0.00; 161; 3.9 → 3.0; 0.00; 0.046) | +0.760 (+0.90; 0.00; 134; 3.8 → 4.7; 0.97; 0.025) |
| course 0, compass, ground, entropy 0.1 | +0.745 (+0.74; −0.34; 237; 3.5 → 4.0; 0.58; 0.047) | +7.942 (+6.49; +0.99; 203; 3.8 → 4.8; 0.39; 0.063) | +2.650 (+0.76; −0.08; 228; 3.5 → 6.0; 0.00; 0.048) |
| course 0, compass, entropy 0.1, without the ground sense (table above) | +0.824 | +1.075 | +2.890 |
| frozen newborn with compass and ground, 3000 moments, course metres | −0.083 | +0.024 | −0.066 |
| frozen copies, 3000 moments, course metres (ground / course and ground / with compass / with compass and entropy) | 0.121 / 0.152 / 0.156 / −0.110 | 0.151 / 0.046 / −0.012 / −0.036 | 0.157 / 0.152 / 0.152 / −0.039 |
| course 0, compass, ground, entropy 0.1, per 5000 moments, course / forward | +1.02 +0.43 +0.01 +0.17 −0.55 −0.34 / +1.06 +0.12 0.00 −0.15 −0.86 +0.56 | +1.73 +1.85 +1.19 +1.14 +1.04 +0.99 / +1.68 +2.01 +1.44 +1.25 +1.14 −1.04 | +0.78 +1.09 +0.96 −0.06 −0.06 −0.08 / +0.85 +0.45 −0.55 +0.02 +0.02 −0.02 |

**Reading:** one life walks the line for its whole life, and it is one life. The course with
the compass, the ground sense and the entropy term on seed 1 went 7.94 m along its birth
heading in 30,000 moments, 6.49 forward, between +0.99 and +1.85 m along the course in every
window of 5000, earning 0.049–0.063 a moment, half aroused (0.39–0.58) to the end, planting
3.4–4.8 feet, turning 31–41 rad a window as the lives that go nowhere do: the farthest any
life of this body has gone in 30,000 moments, and farther along a line than along its own
heading, because in its last window it faced back (heading 2.58 rad at the end) and went on
along the course pushing backward in its own frame, −1.04 m forward and +0.99 m along the
line, which the course law pays and the plain law would have charged. Its frozen copy walks
−0.04 m: the greedy answer of every entropy brain here goes backward (−0.11 / −0.04 / −0.04),
so the walk is in the sampled policy, not in its mode, which is what the entropy term asks
for. The other two seeds under the same genes are the course's usual: seed 2 went +2.65 m and
fell calm at 15,100 moments into a sway (−0.06 to −0.08 a window), seed 0 wandered (+0.75).
Without the entropy term the ground sense changes nothing the seeds do not: with the course
and the compass +0.37 / +0.98 / +0.76 (seed 1 calm at 9200 into a sway), with the course alone
+0.74 / +0.34 / +0.59, and under the plain law +1.00 / +3.00 / +0.27 along the birth heading,
where seed 1 walked 2.7 m forward in its first 15,000 moments and then stalled, aroused to the
end. Against the same three genes without the ground sense (+0.82 / +1.08 / +2.89) the sense
moved one seed from 1 m to 8 m and one from 2.9 m to 2.7 m: one seed is a life, not a dose,
and whether the sense reads what the course pays is seeds 3–5 of this row, the next thing to
measure, with the course's backward pay beside it. The copies without entropy walk
0.12–0.16, the hip sweep; the newborn with both senses −0.08 to +0.02. 3.3–4.3 ms a moment
learning, 2.3–2.6 calm. The sense stays off: it is an option named in every reading, and no
default has a dose yet.

**The sizes and the rates.** The founder genes that the readings above had still not moved
on this body: the module widths (`--gene modules=128,64`, 456 neurons and 42,681 parameters,
and `32,16`, 264 and 14,025, against the founder's 64,32, 328 and 22,041), the sensory scale
(`--gene sensory_scale=2` and `8` against the founder's 4) and the actor's rates together
(`--gene eta=0.01 --gene eta_bias=0.001`, a third of the founder's 0.03 and 0.003). Servos,
gamma 0.99, seeds 0–2, 30,000 moments; frozen newborns of the two widths and frozen copies at
3000 moments:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| servos, gamma 0.99 (the founder; table above) | +1.689 (−0.01; 3.8 → 5.0; 0.00; 0.049) | +0.939 (+0.03; 3.1 → 3.9; 0.98; 0.025) | +0.674 (+0.08; 3.8 → 4.8; 0.99; 0.025) |
| modules 128,64 | +0.855 (+0.13; 4.2 → 4.4; 0.94; 0.029) | +1.824 (+0.08; 3.5 → 4.2; 0.95; 0.031) | +0.497 (+0.20; 3.8 → 3.4; 0.97; 0.025) |
| modules 32,16 | +0.317 (0.00; 3.2 → 3.0; 0.00; 0.054) | −1.523 (−0.29; 4.7 → 5.0; 0.00; 0.052) | +0.218 (0.00; 5.3 → 6.0; 0.00; 0.042) |
| sensory scale 2 | +1.310 (0.00; 3.7 → 4.5; 1.00; 0.022) | +0.657 (+0.24; 3.0 → 4.1; 0.89; 0.028) | +0.310 (−0.01; 3.2 → 3.0; 0.00; 0.062) |
| sensory scale 8 | +1.088 (−0.08; 4.0 → 4.1; 0.97; 0.026) | +0.285 (−0.06; 3.4 → 4.4; 0.76; 0.036) | +0.806 (+0.05; 3.9 → 3.9; 0.88; 0.032) |
| eta 0.01, eta bias 0.001 | +0.419 (+0.01; 4.2 → 5.0; 0.00; 0.067) | +0.350 (−0.01; 2.8 → 4.1; 0.90; 0.030) | +1.304 (+0.08; 4.1 → 4.3; 0.90; 0.031) |
| frozen newborn, 3000 moments (128,64 / 32,16) | −0.050 / +0.018 | +0.107 / +0.007 | +0.002 / −0.031 |
| frozen copies, 3000 moments (128,64 / 32,16 / scale 2 / scale 8 / eta 0.01) | 0.151 / 0.061 / 0.156 / 0.155 / 0.114 | 0.149 / 0.108 / 0.125 / 0.154 / 0.140 | 0.153 / 0.089 / 0.124 / 0.154 / 0.151 |
| modules 128,64, per 5000 moments | +0.45 0.00 +0.08 +0.05 +0.13 +0.13 | +1.11 +0.26 +0.37 −0.08 +0.09 +0.08 | +0.19 +0.07 +0.01 +0.01 +0.02 +0.20 |
| modules 32,16, per 5000 moments | +0.32 0.00 0.00 0.00 0.00 0.00 | +0.02 −0.32 −0.30 −0.30 −0.33 −0.29 | +0.22 0.00 0.00 0.00 0.00 0.00 |

**Reading:** no size and no rate finds a gait, and the small brain brings the calm stall
back on every seed. The 32,16 brain fell calm at 1000, 2700 and 1600 moments and never
learned again: two seeds into the rock (0.00 a window, 0.042–0.054 a moment, five to six
feet planted) and seed 1 into a sway that goes backward 0.3 m a window for 27,000 moments
while earning 0.052 a moment, twice what the founder's aroused seeds earn going forward,
the leak of "The leaking mark" in one row. The 128,64 brain stays aroused to the end on
every seed (0.94–0.97) and walks the founder's distances, +0.86 / +1.82 / +0.50, its best
window seed 1's first (+1.11) and 0.1 a window after, at 5.5–5.9 ms a moment with fifteen
lives sharing the machine (the small brain 2.1); a wider cortex is more to sweep and nothing
more to find. The sensory scale
at 2 and 8 and the slow rates stay inside the founder's spread (+0.31 to +1.31), scale 2 and
the slow rate each stalling one seed calm (seed 2 at 3500, seed 0 at 4000) into the rock the
founder stalls into, the slow rate's seed 2 the one life of the batch that gained in every
window (+0.46 +0.19 +0.27 +0.23 +0.08 +0.08, aroused 0.90 to the end, 4.1–4.3 feet
planted). The copies walk the hip sweep at 0.11–0.16 except the small brain's, 0.06–0.11;
the newborns −0.05 to +0.11. The founder stays at 64,32, scale 4 and 0.03: the size of the
cortex and the speed of the actor are not where the walk is.

**A slot per leg.** `--layout leg` ("The brain") composes the motor cortex as eight slots of
nine, one per leg, each answer one of the nine pairs the leg's hip and knee commands can
make, so that a leg is one choice to credit instead of two that must agree; the body reads
the same sixteen commands. Servos, gamma 0.99, seeds 0–2, 30,000 moments, under the plain law,
with the entropy term at 0.1, and in the row that went farthest above (the course at the
birth heading, the compass, the ground sense, entropy 0.1); random and frozen newborns under
the layout and frozen copies at 3000 moments; the joint layout's rows from the tables above
for comparison:

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| a slot per joint (the founder; table above) | +1.689 (−0.01; 3.8 → 5.0; 0.00; 0.049) | +0.939 (+0.03; 3.1 → 3.9; 0.98; 0.025) | +0.674 (+0.08; 3.8 → 4.8; 0.99; 0.025) |
| a slot per leg | +0.267 (+0.04; 3.7 → 4.1; 0.90; 0.030) | +0.601 (0.00; 3.3 → 3.0; 0.00; 0.052) | +0.149 (0.00; 4.8 → 5.0; 0.00; 0.042) |
| a slot per joint, entropy 0.1 (table above) | +0.564 (+0.01; 3.3 → 3.0; 0.00; 0.070) | +5.218 (+0.63; 3.1 → 4.6; 0.46; 0.053) | +4.263 (+0.69; 4.9 → 4.9; 0.00; 0.052) |
| a slot per leg, entropy 0.1 | +0.351 (−0.02; 3.2 → 3.0; 0.00; 0.075) | +1.833 (0.00; 3.0 → 6.1; 0.00; 0.048) | +0.893 (−0.01; 4.2 → 4.0; 0.00; 0.058) |
| random, a slot per leg, 3000 moments | +0.329 (3.1) | +0.594 (3.2) | +0.271 (3.1) |
| frozen newborn, a slot per leg, 3000 moments | +0.099 (5.0) | −0.029 (2.0) | −0.070 (7.0) |
| frozen copies, 3000 moments (plain / entropy) | 0.132 / 0.089 | 0.080 / 0.162 | 0.108 / 0.100 |
| a slot per leg, per 5000 moments (plain / entropy) | +0.01 −0.09 +0.13 −0.15 +0.33 +0.04 / +0.21 +0.16 −0.02 +0.01 +0.01 −0.02 | +0.60 −0.01 0.00 0.00 −0.01 0.00 / +1.06 +0.73 +0.05 −0.01 0.00 0.00 | +0.15 0.00 0.00 0.00 0.00 0.00 / +0.76 +0.14 −0.01 0.00 +0.01 −0.01 |

| 30,000 moments, metres along the course (forward metres; course metres in the last 5000; rad turned; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| a slot per joint: course 0, compass, ground, entropy 0.1 (table above) | +0.745 (+0.74; −0.34; 237; 3.5 → 4.0; 0.58; 0.047) | +7.942 (+6.49; +0.99; 203; 3.8 → 4.8; 0.39; 0.063) | +2.650 (+0.76; −0.08; 228; 3.5 → 6.0; 0.00; 0.048) |
| a slot per leg: course 0, compass, ground, entropy 0.1 | +1.151 (+0.12; −0.06; 258; 3.8 → 5.0; 0.00; 0.061) | +2.705 (+1.97; +0.66; 231; 3.8 → 3.8; 0.24; 0.061) | +0.533 (−0.02; +0.08; 310; 3.0 → 3.0; 0.00; 0.047) |
| frozen newborn, a slot per leg, compass and ground, 3000 moments, course metres | +0.070 | −0.033 | +0.006 |
| frozen copies, 3000 moments, course metres | +0.124 | +0.125 | −0.141 |
| a slot per leg, per 5000 moments, course / forward | +0.15 +0.31 +0.43 +0.23 +0.09 −0.06 / +0.29 +0.06 −0.22 −0.06 +0.10 −0.04 | +0.59 +0.11 +0.11 +0.30 +0.94 +0.66 / +0.58 −0.36 +0.04 +0.45 +0.62 +0.64 | +0.08 +0.08 +0.10 +0.08 +0.10 +0.08 / −0.01 −0.01 +0.02 −0.02 +0.02 −0.01 |

**Reading:** a leg as one choice finds no gait and loses what the joints found. Under the
plain law the leg layout walks +0.27 / +0.60 / +0.15 against the founder's +1.69 / +0.94 /
+0.67, and two seeds fall calm by 4800 moments (seed 1 into a three-footed sway at 0.052 a
moment, seed 2 into the rock at 1400); seed 0 stays aroused (0.90) and no window of its life
exceeds +0.33. With the entropy term, the one gene that walked the joints 4–5 m on two seeds,
the leg layout walks +0.35 / +1.83 / +0.89 and every seed falls calm (5300, 12,300, 6600
moments): seed 1 gained +1.06 and +0.73 in its first two windows and then stood on six feet
(0.048 a moment) for 18,000 moments. In the row that walked the joints 7.9 m along a line,
the legs go +1.15 / +2.71 / +0.53 along the course: seed 1 gained along the course in every
window (+0.11 to +0.94), 1.97 m forward, and learned until 27,300 moments, longer than any
other life of the layout, then fell calm; seed 0 fell
calm at 18,900 and seed 2 at 900, into a sway that nets +0.08 a window along the course.
Eight lives of nine are calm before the end, against four of nine under the joint layout
in the same rows; a nine-way answer whose outcome credits one choice is, on this body, a
policy that settles sooner, not one that finds the step. The random control under the
layout walks the usual sway (+0.27 to +0.59 in 3000 moments; the distribution is the
founder's, nine pairs at a ninth each as two joints at a third each), the newborns −0.07 to
+0.10, the copies the hip sweep at 0.08–0.16 (the course copy of seed 2 −0.14, turning). The
founder keeps a slot per joint; the layout stays a birth option named in every reading, and
no row of it is worth seeds 3–5 before the joint rows above are. 3.7–3.9 ms a moment for
the two plain-law lives that ran with twelve others; the rest of the batch shared the
machine with the previous batch's controls as well, and their costs are not a reading.

**The drift.** The world's current (`--drift`; "The reward" above): 0.002 m/s backward along
the body's own heading every moment, planted or not, 3 m of ground over 30,000 moments (0.5 m
a window of 5000), below the speed of the lives that walked above; and 0.005 m/s, 7.5 m (1.25
a window). Servos, gamma 0.99, seeds 0–2, 30,000 moments, without and with the entropy term
at 0.1; random, frozen newborns and frozen copies at 3000 moments under the same current.
Forward metres are net of the ground lost: what the feet made is the row plus 0.5 a window at
0.002 and 1.25 at 0.005 (0.3 and 0.75 over 3000 moments):

| 30,000 moments, forward metres (last 5000; planted feet, first 5000 → last 5000; aroused; r/moment) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| servos, gamma 0.99, still ground (table above) | +1.689 (−0.01; 3.8 → 5.0; 0.00; 0.049) | +0.939 (+0.03; 3.1 → 3.9; 0.98; 0.025) | +0.674 (+0.08; 3.8 → 4.8; 0.99; 0.025) |
| drift 0.002 | −1.185 (−0.45; 3.4 → 4.2; 1.00; 0.023) | −2.646 (−0.34; 2.6 → 4.8; 0.87; 0.028) | −2.393 (−0.40; 3.4 → 3.7; 0.98; 0.024) |
| servos, gamma 0.99, entropy 0.1, still ground (table above) | +0.564 (+0.01; 3.3 → 3.0; 0.00; 0.070) | +5.218 (+0.63; 3.1 → 4.6; 0.46; 0.053) | +4.263 (+0.69; 4.9 → 4.9; 0.00; 0.052) |
| drift 0.002, entropy 0.1 | +5.128 (+0.45; 4.4 → 4.7; 0.00; 0.041) | +0.079 (−0.49; 3.6 → 3.0; 0.00; 0.083) | +0.181 (−0.50; 3.7 → 4.0; 0.00; 0.053) |
| drift 0.005 | −6.377 (−1.25; 3.6 → 3.0; 0.00; 0.068) | −7.005 (−1.40; 2.7 → 4.1; 1.00; 0.018) | −7.109 (−1.17; 3.7 → 4.1; 1.00; 0.019) |
| random, drift 0.002, 3000 moments | −0.524 (2.8) | −0.331 (3.1) | −0.756 (3.1) |
| frozen newborn, drift 0.002, 3000 moments | −0.406 (5.0) | −0.300 (2.0) | −0.265 (6.0) |
| frozen copies of the drift 0.002 brains, 3000 moments | −0.145 (7.0) | −0.177 (5.0) | −0.146 (6.0) |
| frozen copies of the drift 0.002, entropy brains, 3000 moments | −0.058 (5.0) | −0.181 (2.0) | −0.224 (3.0) |
| frozen copies of the drift 0.005 brains, 3000 moments, drift 0.005 | −0.731 (8.0) | −0.601 (5.0) | −0.596 (6.0) |
| drift 0.002, per 5000 moments | +1.03 −0.04 −0.36 −0.96 −0.40 −0.45 | −0.49 −0.54 −0.44 −0.36 −0.47 −0.34 | −0.31 −0.62 −0.31 −0.43 −0.32 −0.40 |
| drift 0.002, entropy 0.1, per 5000 moments | +1.05 +1.78 +0.87 +0.54 +0.45 +0.45 | +0.40 +1.05 +0.11 −0.50 −0.50 −0.49 | +0.86 +0.93 −0.10 −0.50 −0.50 −0.50 |
| drift 0.005, per 5000 moments | −0.12 −1.27 −1.25 −1.23 −1.27 −1.25 | −1.02 −1.07 −0.91 −1.30 −1.32 −1.40 | −1.39 −1.08 −1.17 −1.12 −1.19 −1.17 |

**Reading:** the current takes ground, not income. Under the plain law backward is free, so a
body carried backward pays nothing for it; only the forward moments pay, each 40·D·DT less
than on still ground (0.004 at 0.002, 0.01 at 0.005), and a moment slower than the current
pays nothing. The rows show it. Two of the entropy lives on the 0.002 current (seeds 1 and 2)
walked +0.4 to +1.05 a window net for 10,000 moments (0.9 to 1.55 made) and then fell calm
(11,400 and 13,400 moments) into a stand on three and four feet that loses exactly the
window's drift, −0.50, the feet making nothing, at 0.083 and 0.053 a moment: more than the
one walker earns, and what the same stands earn on still ground (0.070 and 0.052 in the row
above). A stand on a current earns what it earns on still ground, so the current does not
push a calm body to walk. Seed 0 with the entropy term is the one life that walks against the
current: +5.13 net, 8.1 made, calm from 16,400 moments, and in each of its last two windows
+0.45 net, 0.95 made, 0.0038 m/s on 4.7 feet at 0.041 a moment, the first calm cycle above
that nets ground against a current (the same seed stood still on still ground at +0.56; the
current changes nothing the brain senses, only the pay of each moment, and that is a
different life, not a dose). Without the entropy term the three lives stay aroused to the
end (0.87–1.00) in sways, seed 0 after walking +1.03 net (1.53 made) in its first window;
the feet made 1.8 / 0.35 / 0.6 m against the founder's 1.69 / 0.94 / 0.67 on still ground.
At 0.005 the lives made 1.1 / 0.5 / 0.4 m of the 7.5 lost; seed 0 fell calm at 4400 into a
three-footed sway that loses the full 1.25 a window at 0.068 a moment, the standing income
again, and seeds 1 and 2 stayed aroused at 0.018–0.019. The controls: random made −0.22 /
−0.03 / −0.46 m beyond the 0.3 m the current took; the frozen newborn loses the current and
nothing else (seed 0's −0.406 is its −0.106 under target motors above less 0.300); the
frozen copies make the hip sweep beyond the current, 0.12–0.24 m in 3000 moments at 0.002
and 0.02–0.15 at 0.005. The drift stays off in the founder. For the current to bite, backward
must cost: under the backward pay or the high-water mark a carried body earns nothing until
it regains the ground, and that pairing is the next step, not a stronger current. 2.6–2.9
ms a moment for the entropy lives, 4.0–4.2 for the aroused plain lives, twelve lives sharing
the machine.

What a next step would measure, each against the same controls and on the eight-legged body:
the finer code with the stance term over a longer life, the one gene that moved the first
block; the course with the compass, the ground sense and the entropy term at 0.1 on seeds
3–5, the row whose seed 1 walked 7.9 m along its line, so that the row is a dose; and the
course law with the backward pay at 1 or a window, so that the sway it leaves pays nothing
along the line, with the same senses. The current at 0.002 with the backward pay at 1 or the
high-water mark, so that a body carried backward earns nothing until it regains the ground,
on the entropy row whose seed 0 walked 0.95 m a window against it and whose seeds 1 and 2
stood and were carried at a stand's income. The horizon past 0.99, the setpoints as senses, the
leaking mark, the recall memory, the temperature, the actor's hold and maturity, and the
observation codes (fixed and learned) are measured above and change nothing after the first
block, the codes losing it; the critic's rate at 20 held one seed's slow
rhythm for 60,000 moments and is worth seeds 3–5 and a dose between 5 and 20; the entropy
term at 0.1 walked two seeds of three 4–5 m and one of them every block of 60,000 moments,
in calm cycles the brain repeats to the bit (a jitter, a one-legged row on a circle), and is
the one gene above worth seeds 3–5, a dose of 0.03 and 0.3, and the stance term alongside;
the module widths, the sensory scale and the actor's rates are measured above and change
nothing, the small cortex stalling every seed calm; a slot per leg, the one change to how
the brain's answer meets the body, loses metres on every row and falls calm sooner, and
stays a named option;
the 34 m brain's world (contact model, support rule, leg stop) has
changed under it since, so its continuation is no longer one life. What the
crawl needs is a law that pays it more than
the sway without paying the jitter nothing, and none of the laws built does, the leaking
mark included, and the continuations above show that no law built keeps a crawl once it is
found either, so a schedule of them (the plain law to find it, another to keep it) is not a
next step. A smaller body, four legs of two
joints where the support rule lets one leg swing at a time, is a different model to be named
as the servos are; it does not meet the collapse above, which is in the credit and the law,
not in the number of legs.

CUDA (`--backend torch`, float32) ran the same brain at 21.9 ms per moment against
4.3 ms on the CPU, as the library's placement rule predicts for a one-row brain of
328 neurons; the default rule keeps it on the CPU. The first CUDA moment pays about 1.6 s of
warm-up.

Summaries of every run are in the `--json` files the nursery wrote; the commands above
reproduce them with the same seeds.

## Limits

- The contact rule is a model: no pitch or roll, no body mass, no leg thickness beyond the
  stop that keeps a side's legs from crossing, no slipping
  except the decay while the body is down (fewer than three feet planted, or none on one
  side). A down body does not fall: its lifted legs stay where they are and it stands again
  the moment its feet support it. A gait that moves this body may not move a physical one.
- The reward is forward progress only; the brain is free to drift sideways and turn, and does.
- The readings above are one machine and three seeds; they do not establish a learning rate,
  an efficiency, or a property of the library.
- The population code's resolution (`per=8`), the module sizes and the sensory scale were
  chosen, not measured against alternatives.

## Folders

| Path | What |
| --- | --- |
| `spider/` | body, senses, brain, life, nursery, server |
| `static/` | the page; `static/vendor/three` is three.js r160 (MIT, licence included); `static/vendor/brain_view.js` is the connectome demo's brain view (cadence-demos, GPL-3.0) |
| `tests/` | `python -m pytest tests -q` |
| `runs/` | saved lives (gitignored) |

GPL-3.0, as the library.
