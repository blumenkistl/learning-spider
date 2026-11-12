# Building on the learning spider

This is an application of [Cadence](https://github.com/muellerberndt/cadence). Its rules
apply here unchanged: read `~/git/github/cadence/AGENTS.md` (or the copy in the installed
release's repository) before changing how the brain is composed, taught or measured, and the
[README](README.md) of this project for what is built and what was measured.

## What this project is

One continuing Cadence brain commands an eight-legged body of sixteen joints in a simulated
world and learns, from the world's reward alone, to move forward. The web page is a window and a
set of controls on that one life; it is not the model. The headless nursery is the same life
without a page, for measurement.

## Rules this project keeps

- **One continuing brain.** The brain is composed once (`Brain.compose(..., arousal=...)`) and
  lives through `brain.live`. It is never recreated per moment, per episode or per page load.
  A saved life is continued, with its world and its pending outcome, not replayed. The page's
  Reset is the one deliberate end of a life: after confirmation it deletes the save and a
  newborn brain takes the body, of the seed, genes, motor model and senses the page's "next
  birth" names (the life's own when it names nothing); nothing else recreates a brain, and
  genes are set at birth only.
- **No controller between brain and motors.** Each motor slot is one joint's
  `decrease / hold / increase`. There is no gait generator, oscillator, inverse kinematics or
  smoothing between the brain's answer and the joints. Adding one makes a different model;
  label it as a control. The body's own stops (the hip and knee ranges, the stop that keeps a
  side's neighbouring legs from crossing) are the world's, not a controller: they bound what
  the brain's answer can do, they never choose it. The body has two motor models
  (`--motors`, `MOTORS`): `rate`, the founder until 2026-11-12, where a command moves the joint for one moment;
  and `target`, where each joint is a position servo whose setpoint a command sets to an end
  of the joint's range and `hold` keeps. The servo chooses nothing and knows no gait, but it
  is a different model, named in every reading made under it (the README's "The position
  commands"); the model is the life's, saved and loaded with it, never switched on a saved life.
- **The world supplies problems, never answers.** The reward is the body's forward progress
  along its own heading, capped. No target pose, no phase, no "move leg 3 now". Shaping terms
  (lateral drift, yaw, posture) are not added without a measured reason and a README note.
  Two such terms exist in `spider/life.py`: every foot short of four planted through the
  moment takes `STANCE_PAY` (off by default; 0.1 is the measured dose), and every radian turned takes `YAW_PAY`
  (off by default: no measured dose straightened the course without stopping the walk).
  Both are labelled as shaping, their readings are in the README, and `--stance-pay` and
  `--yaw-pay` in the nursery and the server, or the same names in the server's environment,
  set them. The page's "law, live" changes the seven settings on the living brain
  (`Life.set_law`, control `law`); a life whose law changed is a schedule, a different model
  from a life under one law, so every change is noted, counted on the page's law line,
  reported as `law_changes` and saved with the life, and the sidecar's law outranks the flags
  on a load. A reading of a life with a `law_changes` entry is a reading of a schedule.
  The high-water mark (`--high-water`, `HIGH_WATER`; off by default) is not a hint: it is the
  same outcome counted once, progress beyond the farthest point reached, so a sway earns
  nothing. The mark belongs to the life and is saved and loaded with it.
- **Every outcome is delivered exactly once.** `Life.reward` is the last executed command's
  outcome not yet delivered. A refusal keeps it in the policy's `owed`; a policy switch moves it
  to the brain's `owed`; a save writes it to the sidecar; a load delivers it. Keep the tests in
  `tests/test_life.py` and `tests/test_brain.py` that prove this.
- **Controls are named honestly.** `random` is the baseline; `frozen` is a brain that answers
  greedily and never learns, either a newborn or a save/load copy of the live brain. Neither
  saves. The page shows which policy commands the body at every moment.
- **Memory writes and reads are the brain's.** Episodic and working memory are composed in;
  the application declares no memory of its own, keys nothing on raw observations and does not
  reach into the brain's memory to help it. Recall memory is off by name (`recall=False`);
  turning it on is a gene to measure, not a default to flip.
- **Readings are readings.** A block's forward metres, reward per moment, aroused share and
  sweeps are what the nursery measured on one machine with one seed each; they are not a claim
  about the library. A found gait is reported as what it is. Change a founder gene only with a
  nursery comparison against the previous value, random and frozen, at matched moments and
  seeds, written into the README's measurement section.
- **Residuals are not energy.** `ms/moment` and sweep counts are costs of this implementation
  on this machine; they establish nothing about efficiency in general.

## Where things are

| Part | File |
| --- | --- |
| Body, kinematics, contact model | `spider/body.py` |
| Senses (population code + planted flags) | `spider/senses.py` |
| Founder genes, the brain wrapper, the controls | `spider/brain.py` |
| The life: reward, blocks, custody, save/load, the loop thread | `spider/life.py` |
| Headless measurement | `spider/nursery.py` |
| The web server | `spider/server.py`, `server.py` |
| The page | `static/index.html`, `static/app.js`, `static/style.css`; the 3D brain view is `static/vendor/brain_view.js` (the connectome demo's, unchanged) fed by `spider_payload` in `spider/brain.py` |
| Tests | `tests/` (`python -m pytest tests -q`) |
| Saved lives | `runs/` (gitignored) |

## Checks before a change lands

1. `.venv/bin/python -m pytest tests -q` passes.
2. A change to the body, senses, reward or founder genes comes with nursery readings
   (3000 moments, seeds 0–2, brain / random / frozen newborn / frozen raised copy) and the README's
   measurement section updated with them.
3. A change to custody, save or load keeps `tests/test_life.py::test_save_and_load_continue_the_same_life` and the
   policy-switch test passing and adds a test for the new path.
4. The page still works at `http://127.0.0.1:8080` after a restart from `runs/`.
