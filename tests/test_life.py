import json
import math

import numpy as np
import pytest

from spider.body import JOINTS, Spider
from spider.brain import SpiderBrain
from spider.life import law_text, BLOCK, PROGRESS_PAY, Life, pay
from spider.senses import Senses


class RecordingPolicy:
    policy = "record"
    learns = False

    def __init__(self):
        self.rewards = []
        self.moments = 0
        self.refusals = 0
        self.owed = 0.0

    def moment(self, observation, reward, done=False):
        self.rewards.append(reward)
        self.moments += 1
        cmd = [1] * JOINTS
        for i in range(8):
            cmd[2 * i] = 0          # all hips sweep back: forward progress while planted
        return {"commands": cmd, "refused": False, "mode": "record", "aroused": False, "sweeps": 0,
                "learning_sweeps": 0, "want": 0.0, "level": 0.0, "ms": 0.0, "policy": "record"}

    def has_pending(self):
        return False

    def describe(self):
        return {"policy": "record"}


def test_each_outcome_reaches_the_policy_once_and_in_order(tmp_path):
    life = Life("random", seed=0, pace=0, runs=tmp_path)
    life.policy = RecordingPolicy()
    earned = []
    for _ in range(8):
        life.tick()
        earned.append(life.reward)
    got = life.policy.rewards
    assert got[0] is None
    assert got[1:] == earned[:-1]
    assert earned[0] > 0.0
    assert pay(1.0) == 0.5 and pay(-1.0) == 0.0 and pay(0.001) == pytest.approx(0.04)


def test_feet_short_of_four_take_pay():
    assert pay(0.0, 0) == 0.0                            # off by default: progress-only pay
    hint = dict(stance_pay=0.1)                          # the measured dose, switched on by hand
    assert pay(0.0, 4, **hint) == 0.0 and pay(0.0, 8, **hint) == 0.0
    assert pay(0.0, 3, **hint) == pytest.approx(-0.1) and pay(0.0, 0, **hint) == pytest.approx(-0.4)
    assert pay(0.001, 2, **hint) == pytest.approx(0.04 - 0.2)
    assert pay(1.0, 0, **hint) == pytest.approx(0.5 - 0.4)
    assert pay(-1.0, 0, **hint) == pytest.approx(-0.4)   # backward still pays zero progress


def test_the_pay_per_metre_sets_the_gain_under_the_same_cap():
    assert pay(0.01) == pytest.approx(0.4)               # the founder: 40 a metre, a stroke moment all but at the cap
    assert pay(0.01, per_metre=10.0) == pytest.approx(0.1)
    assert pay(0.04, per_metre=10.0) == pytest.approx(0.4)
    assert pay(1.0, per_metre=10.0) == 0.5               # the cap stays
    assert pay(-0.01, per_metre=10.0) == 0.0             # backward still pays zero
    life = Life("random", seed=0, pace=0, runs="/nonexistent", pay_per_metre=10.0)
    assert life.describe()["reward"]["pay_per_metre"] == 10.0
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs="/nonexistent", pay_per_metre=0.0)


def test_backward_pay_charges_a_share_of_the_forward_pay():
    assert pay(-0.01) == 0.0                                      # off by default: backward is free
    assert pay(-0.01, backward_pay=1.0) == pytest.approx(-0.4)    # charged as forward pays
    assert pay(-0.01, backward_pay=0.5) == pytest.approx(-0.2)
    assert pay(0.01, backward_pay=1.0) == pytest.approx(0.4)      # forward unchanged
    assert pay(-1.0, backward_pay=1.0) == -0.5                    # the cap holds
    assert pay(-0.01, 0, stance_pay=0.1, backward_pay=1.0) == -0.5  # with the hint, still the cap
    life = Life("random", seed=0, pace=0, runs="/nonexistent", backward_pay=0.5)
    assert life.describe()["reward"]["backward_pay"] == 0.5
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs="/nonexistent", backward_pay=2.0)


def test_turning_takes_pay_when_switched_on():
    assert pay(0.0, 4, turn=0.5) == 0.0                  # off by default
    assert pay(0.0, 4, turn=0.1, yaw_pay=0.1) == pytest.approx(-0.01)
    assert pay(0.0, 4, turn=-0.1, yaw_pay=0.1) == pytest.approx(-0.01)
    assert pay(0.01, 4, turn=0.5, yaw_pay=0.1) == pytest.approx(0.4 - 0.05)
    assert pay(0.0, 0, turn=3.0, yaw_pay=5.0) == -0.5    # the cap holds both hints


def test_blocks_close_every_hundred_moments(tmp_path):
    life = Life("random", seed=0, pace=0, runs=tmp_path)
    for _ in range(2 * BLOCK + 5):
        life.tick()
    assert len(life.blocks) == 2
    block = life.blocks[-1]
    assert block["moment"] == 2 * BLOCK and block["policy"] == "random"
    assert set(block) >= {"forward_m", "reward_per_moment", "aroused_share", "sweeps_per_moment",
                          "learning_sweeps", "refused", "ms_per_moment", "mean_stance", "turned_rad"}
    assert life.latest["moments"] == 2 * BLOCK + 5
    assert len(life.latest["pose"]["joints"]) == JOINTS


def test_save_and_load_continue_the_same_life(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(12):
        life.tick()
    assert life.reward is not None and life.brain.has_pending()
    saved = life.save()
    assert saved["saved"] and (tmp_path / "spider.npz").exists()
    sidecar = json.loads((tmp_path / "spider.json").read_text())
    assert sidecar["format"] == "learning-spider/1" and sidecar["moments"] == 12
    assert sidecar["reward"] == life.reward

    again = Life("brain", seed=99, pace=0, runs=tmp_path, backend="cpu")
    out = again.load()
    assert out["loaded"] and out["custody"] == "kept"
    assert again.moments == 12 and again.reward == life.reward
    assert again.turned == life.turned
    assert again.body.state_dict() == life.body.state_dict()
    for _ in range(6):
        a = life.tick()
        b = again.tick()
        assert a["commands"] == b["commands"]
        assert a["pose"] == b["pose"]
        assert a["reward"] == b["reward"]
    assert again.moments == 18 and again.brain.describe()["age"] == life.brain.describe()["age"]


def test_load_without_sidecar_reports_the_break(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(3):
        life.tick()
    life.brain.save(tmp_path / "spider.npz")
    again = Life("random", seed=0, pace=0, runs=tmp_path, backend="cpu")
    out = again.load()
    assert out["loaded"] and out["custody"].startswith("broken")
    assert again.policy.policy == "brain" and again.reward is None
    assert again.notes


def test_policy_switch_delivers_the_owed_outcome_exactly_once(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(4):
        life.tick()
    brain = life.brain
    owed = life.reward
    assert owed is not None
    assert brain.owed is None
    delivered = []
    real_live = brain.brain.live

    def spy(observation, **feedback):
        delivered.append(dict(feedback))
        return real_live(observation, **feedback)

    brain.brain.live = spy
    assert life.control("policy", "random")["changed"]
    assert life.policy.policy == "random" and brain.owed == owed and life.reward is None
    for _ in range(5):
        life.tick()
    assert delivered == []                              # the brain learned nothing meanwhile
    life.control("policy", "frozen")
    assert life.policy.policy == "frozen"
    for _ in range(3):
        life.tick()
    assert delivered == []
    life.control("policy", "brain")
    assert life.policy is brain and life.reward is None
    life.tick()
    assert delivered[0] == {"reward": [owed], "done": [False]}
    assert brain.owed is None
    fresh = life.reward
    life.tick()
    assert len(delivered) == 2 and delivered[1] == {"reward": [fresh], "done": [False]}
    assert life.brain is brain and life.policy.describe()["refusals"] == 0


def test_controls_and_random_life(tmp_path):
    life = Life("random", seed=0, pace=0, runs=tmp_path)
    for _ in range(200):
        life.tick()
    assert life.moments == 200
    assert life.control("pause")["ok"] and life.paused
    assert life.control("resume")["ok"] and not life.paused
    life.body.x = 1.0
    assert life.control("recentre")["ok"] and life.body.x == 0.0
    assert life.control("pace", 0)["ok"] and life.pace == 0.0
    assert life.control("save")["saved"] is False            # a random life has no brain to save
    with pytest.raises(ValueError):
        life.control("teleport")
    d = life.describe()
    assert d["policy"] == "random" and d["brain"] is None and len(d["input_names"]) == d["inputs"]


def test_thread_runs_and_stops(tmp_path):
    import time
    life = Life("random", seed=0, pace=200.0, runs=tmp_path)
    life.start()
    time.sleep(0.3)
    life.stop()
    assert 10 <= life.moments <= 80


class SwayPolicy(RecordingPolicy):
    """All hips back one moment, all hips forward the next: the body sways and nets nothing."""

    def moment(self, observation, reward, done=False):
        out = super().moment(observation, reward, done)
        if self.moments % 2 == 0:
            out["commands"] = [2 if i % 2 == 0 else 1 for i in range(JOINTS)]
        return out


class HalfStrokePolicy(RecordingPolicy):
    """Four legs stroke back while four anchor: half the speed, so the pay stays under the cap."""

    def moment(self, observation, reward, done=False):
        out = super().moment(observation, reward, done)
        out["commands"] = [0 if i in (0, 2, 8, 10) else 1 for i in range(JOINTS)]
        return out


def test_windowed_progress_pays_a_walk_and_not_a_sway(tmp_path):
    walk = Life("random", seed=0, pace=0, runs=tmp_path, stance_pay=0.0, window=20)
    walk.policy = HalfStrokePolicy()
    each = Life("random", seed=0, pace=0, runs=tmp_path, stance_pay=0.0)
    each.policy = HalfStrokePolicy()
    earned_walk, earned_each = 0.0, 0.0
    for _ in range(40):                                      # the stroke ends at the hip limit after 8 moments
        earned_walk += walk.tick()["reward"]; earned_each += each.tick()["reward"]
    assert earned_walk == pytest.approx(earned_each, abs=1e-3)   # the window spreads the same pay over 20 moments
    assert walk.distance_forward == pytest.approx(each.distance_forward) and earned_walk > 0.05

    sway = Life("random", seed=0, pace=0, runs=tmp_path, stance_pay=0.0, window=20)
    sway.policy = SwayPolicy()
    per_moment = Life("random", seed=0, pace=0, runs=tmp_path, stance_pay=0.0)
    per_moment.policy = SwayPolicy()
    for _ in range(60):                                      # the sway settles into its alternation
        sway.tick(); per_moment.tick()
    earned_sway, earned_each = 0.0, 0.0
    for _ in range(60):
        sway.tick(); per_moment.tick()
        earned_sway += sway.reward; earned_each += per_moment.reward
    assert abs(sway.distance_forward) < 0.01                 # it went nowhere
    assert earned_each > 0.2                                 # the per-moment law pays the forward half of each sway
    assert earned_sway < 0.02                                # the windowed law does not
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs=tmp_path, window=0)


def test_save_carries_the_progress_window(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", window=20)
    for _ in range(12):
        life.tick()
    life.save()
    assert len(json.loads((tmp_path / "spider.json").read_text())["progress_window"]) == 12
    again = Life("brain", seed=99, pace=0, runs=tmp_path, backend="cpu", window=20)
    again.load()
    assert list(again._progress) == list(life._progress)
    for _ in range(6):
        assert life.tick()["reward"] == again.tick()["reward"]


def test_high_water_pays_new_ground_and_not_a_sway(tmp_path):
    walk = Life("random", seed=0, pace=0, runs=tmp_path, stance_pay=0.0, high_water=True)
    walk.policy = HalfStrokePolicy()
    plain = Life("random", seed=0, pace=0, runs=tmp_path, stance_pay=0.0)
    plain.policy = HalfStrokePolicy()
    earned_walk, earned_plain = 0.0, 0.0
    for _ in range(40):
        earned_walk += walk.tick()["reward"]; earned_plain += plain.tick()["reward"]
    assert earned_walk == pytest.approx(earned_plain) and earned_walk > 0.05   # every step is new ground
    assert walk.mark == pytest.approx(walk.distance_forward) and plain.mark == 0.0

    sway = Life("random", seed=0, pace=0, runs=tmp_path, stance_pay=0.0, high_water=True)
    sway.policy = SwayPolicy()
    first, later = 0.0, 0.0
    for i in range(120):
        sway.tick()
        if i < 60:
            first += sway.reward
        else:
            later += sway.reward
    assert abs(sway.distance_forward) < 0.01                 # it went nowhere
    assert 0 < sway.mark < 0.02                              # the farthest it ever got, in its first moments
    assert first <= PROGRESS_PAY * sway.mark + 1e-9          # and that is all the law ever paid
    assert later == 0.0                                      # once settled, no sway reaches the mark


def test_save_carries_the_high_water_mark(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", high_water=True)
    for _ in range(30):
        life.tick()
    assert life.mark >= life.distance_forward and life.describe()["reward"]["high_water"] is True
    life.save()
    state = json.loads((tmp_path / "spider.json").read_text())
    assert state["mark"] == pytest.approx(life.mark)

    # a mark ahead of the body: the ground up to it is paid for already
    state["mark"] = life.distance_forward + 0.03
    (tmp_path / "spider.json").write_text(json.dumps(state))
    again = Life("brain", seed=99, pace=0, runs=tmp_path, backend="cpu", high_water=True)
    again.load()
    assert again.mark == pytest.approx(life.distance_forward + 0.03)
    assert again.distance_forward == pytest.approx(life.distance_forward)
    again.policy = HalfStrokePolicy()
    paid = [again.tick()["reward"] for _ in range(8)]   # the stroke makes under 0.01 m a moment
    assert paid[0] == paid[1] == 0.0 and sum(paid) > 0.0
    assert sum(paid) <= PROGRESS_PAY * (again.distance_forward - life.distance_forward - 0.03) + 1e-3   # rounded pay
    assert again.mark == pytest.approx(again.distance_forward)

    del state["mark"]
    (tmp_path / "spider.json").write_text(json.dumps(state))
    older = Life("brain", seed=99, pace=0, runs=tmp_path, backend="cpu", high_water=True)
    older.load()
    assert older.mark == pytest.approx(older.distance_forward)   # a save without a mark starts it here


def test_reset_starts_a_new_life_and_removes_the_save(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(BLOCK + 5):
        life.tick()
    old = life.policy
    assert life.control("save")["saved"] and life.brain_file.exists() and life.sidecar.exists()
    assert life.control("policy", "random")["changed"]
    life.body.x = 1.0
    r = life.control("reset")
    assert r["ok"] and r["reset"] and r["moments_before"] == BLOCK + 5
    assert isinstance(life.policy, SpiderBrain) and life.policy is not old and life.policy.moments == 0
    assert life.moments == 0 and life.distance_forward == 0.0 and life.body.x == 0.0
    assert life.reward is None and not life.blocks and life.last_save_at is None
    assert not life.brain_file.exists() and not life.sidecar.exists()
    assert life.latest["moments"] == 0 and life.latest["policy"] == "brain"
    assert life.control("load")["loaded"] is False            # nothing left to continue
    life.tick()
    assert life.moments == 1


def test_a_saved_life_keeps_its_motor_model(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", motors="target")
    assert life.body.motors == "target" and life.describe()["motors"] == "target"
    for _ in range(12):
        life.tick()
    life.save()
    again = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", motors="rate")   # asked for rate motors
    assert again.load()["custody"] == "kept"
    assert again.body.motors == "target" and again.motors == "target"          # the saved life keeps its own
    assert any("target motors" in n for n in again.notes)
    assert np.array_equal(again.body.targets, life.body.targets)
    for _ in range(6):
        assert life.tick()["pose"] == again.tick()["pose"]
    again.reset()
    assert again.body.motors == "target"                                        # a reset keeps the life's model
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs="/nonexistent", motors="servo")


def test_the_setpoint_senses_need_target_motors_and_are_continued(tmp_path):
    from spider.senses import Senses

    with pytest.raises(ValueError):
        Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", motors="rate", senses=Senses(setpoints=True))
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", motors="target", senses=Senses(setpoints=True))
    assert life.describe()["setpoints"] is True and life.describe()["inputs"] == 8 * 34 + 8
    for _ in range(12):
        life.tick()
    life.save()
    again = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", motors="target", senses=Senses(setpoints=True))
    assert again.load()["custody"] == "kept"
    for _ in range(6):
        assert life.tick()["pose"] == again.tick()["pose"]
    plain = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", motors="target")
    assert plain.load()["custody"] == "kept" and plain.senses.setpoints is True   # the sidecar names the senses


def test_the_leaking_mark_pays_old_ground_a_share(tmp_path):
    def earned(policy, **law):
        life = Life("random", seed=0, pace=0, runs=tmp_path, stance_pay=0.0, **law)
        life.policy = policy()
        return sum((life.tick(), life.reward)[1] for _ in range(120)), life        # unrounded

    plain, _ = earned(SwayPolicy)
    sealed, _ = earned(SwayPolicy, high_water=True)
    leaking, life = earned(SwayPolicy, high_water=True, old_ground=0.2)
    whole, _ = earned(SwayPolicy, high_water=True, old_ground=1.0)
    assert plain > 0.1 and 0 <= sealed < 0.02 * PROGRESS_PAY         # a sway: the plain law pays it, the mark not
    assert leaking == pytest.approx(sealed + 0.2 * (plain - sealed))  # the leak pays the old ground a fifth
    assert whole == pytest.approx(plain)                              # and all of it is the plain law again
    assert life.describe()["reward"]["old_ground"] == 0.2
    walk_plain, _ = earned(HalfStrokePolicy)
    walk_leak, _ = earned(HalfStrokePolicy, high_water=True, old_ground=0.2)
    assert walk_leak == pytest.approx(walk_plain)                     # every step is new ground: no leak to pay
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs=tmp_path, old_ground=0.2)             # old ground needs the mark
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs=tmp_path, high_water=True, old_ground=1.5)


def test_the_law_changes_live_and_is_noted_counted_and_saved(tmp_path):
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    for _ in range(3):
        life.tick()
    owed_before = life.reward
    report = life.control("law", {"high_water": True, "window": 4})
    assert report["ok"] and report["changed"] and report["law"]["high_water"] is True
    assert life.reward == owed_before                       # the outcome earned stays as earned
    assert len(life.law_changes) == 1 and life.law_changes[0]["moment"] == 3
    assert "law changed at moment 3" in life.notes[-1] and "schedule" in life.notes[-1]
    assert life.describe()["law_changes"] == life.law_changes
    assert life.control("law", {"window": 4})["changed"] is False   # the same law is no change
    for _ in range(6):
        life.tick()                                              # the window of 4 averages the last four moments
    assert len(life._progress) == 4
    with pytest.raises(ValueError):
        life.control("law", {"old_ground": 0.5, "high_water": False})
    with pytest.raises(ValueError):
        life.control("law", {"horizon": 2})
    with pytest.raises(ValueError):
        life.control("law", "high_water")
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")   # the flags' plain law
    again.load()
    assert again.law() == life.law() and again.law_changes == life.law_changes
    assert any("keeps its own" in note for note in again.notes)


def test_reset_takes_a_birth(tmp_path):
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", motors="rate")
    life.tick()
    with pytest.raises(ValueError):
        life.reset({"setpoints": True})                          # setpoints need target motors
    with pytest.raises(ValueError):
        life.reset({"genes": "gamma"})
    with pytest.raises(ValueError):
        life.reset({"law": {}})
    report = life.control("reset", {"seed": 3, "genes": "eta_critic=20 gamma=0.99", "motors": "target", "setpoints": True})
    assert report["seed"] == 3 and report["genes"] == {"eta_critic": 20, "gamma": 0.99}
    described = life.describe()
    assert described["seed"] == 3 and described["motors"] == "target" and described["setpoints"] is True
    assert described["brain"]["genes"]["actor_eta_critic"] == 20 and described["brain"]["genes"]["actor_gamma"] == 0.99
    assert described["genes"] == {"eta_critic": 20, "gamma": 0.99}
    assert life.moments == 0 and "seed 3" in life.notes[0] and "setpoints sensed" in life.notes[0]
    life.tick()
    assert life.reward is not None
    plain = life.reset()                                         # the same birth again
    assert plain["seed"] == 3 and life.senses.setpoints is True


def test_a_saved_life_keeps_its_seed_and_senses(tmp_path):
    from spider.senses import Senses

    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", motors="rate")
    life.control("reset", {"seed": 3, "motors": "target", "setpoints": True, "genes": "gamma=0.99"})
    for _ in range(4):
        life.tick()
    life.save()
    plain = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", motors="rate")   # the flags name another birth
    assert plain.load()["custody"] == "kept"
    assert plain.seed == 3 and plain.motors == "target" and plain.senses.setpoints is True and plain.genes == {"gamma": 0.99}
    assert plain.describe()["inputs"] == life.describe()["inputs"]
    assert sum("keeps its own" in note for note in plain.notes) == 3        # seed, senses, motors
    for _ in range(3):
        assert life.tick()["pose"] == plain.tick()["pose"]


def test_the_course_law_pays_a_line_and_not_a_circle(tmp_path):
    """Under ``course`` the progress term is the moment's motion along one fixed heading of the
    world: the same strokes pay in full along it, nothing across it, and backward against it;
    along the body's own heading they pay the same whichever way the body points."""
    import math

    def earned(heading, **law):
        life = Life("random", seed=0, pace=0, runs=tmp_path, **law)
        life.policy = RecordingPolicy()                      # every hip sweeps back: forward progress
        life.body.heading = heading
        for _ in range(40):
            life.tick()
        return sum(r for r in life.policy.rewards if r is not None), life

    plain_0, plain = earned(0.0)
    plain_side, _ = earned(math.pi / 2)
    assert plain_0 > 0.5 and plain_side == pytest.approx(plain_0)   # the plain law does not care where the body points
    along, life = earned(0.0, course=0.0)
    across, _ = earned(math.pi / 2, course=0.0)
    against, _ = earned(math.pi, course=0.0)
    assert along == pytest.approx(plain_0) and life.distance_course == pytest.approx(life.distance_forward)
    assert across == pytest.approx(0.0, abs=0.02) and against == pytest.approx(0.0)  # backward is free under the course too
    charged, _ = earned(math.pi, course=0.0, backward_pay=1.0)
    assert charged < -0.5
    assert life.describe()["reward"]["course"] == 0.0
    assert sum(r["course"] for r in life._block) == pytest.approx(life.distance_course)
    assert plain.distance_course == pytest.approx(plain.distance_forward)   # without a course: along the birth heading
    assert Life("random", seed=0, pace=0, runs=tmp_path, course=True).course == 0.0        # true is the birth heading
    assert Life("random", seed=0, pace=0, runs=tmp_path, course=7.0).course == pytest.approx(7.0 - 2 * math.pi)
    assert Life("random", seed=0, pace=0, runs=tmp_path).course is None
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs=tmp_path, course=float("nan"))


def test_the_course_is_set_live_as_the_heading_held_and_is_saved(tmp_path):
    from spider.life import checked_course

    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", senses=Senses(compass=True))
    assert life.describe()["compass"] is True and life.describe()["inputs"] == 8 * 20 + 8
    for _ in range(5):
        life.tick()
    life.body.heading = 0.7
    life.distance_course = 9.0                               # nothing was along a course yet: must start at zero
    report = life.control("law", {"course": True})
    assert report["changed"] and report["law"]["course"] == pytest.approx(0.7)
    assert life.course == pytest.approx(0.7) and life.distance_course == 0.0 and life.mark == 0.0
    assert "course 0.70 rad" in life.notes[-1] and len(life.law_changes) == 1
    assert life.control("law", {"course": True})["changed"] is False          # ticked again: the same course
    for _ in range(3):
        life.tick()
    assert life.latest["distance_course"] == pytest.approx(life.distance_course)
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")    # plain law, plain senses asked
    assert again.load()["custody"] == "kept"
    assert again.course == pytest.approx(0.7) and again.distance_course == pytest.approx(life.distance_course)
    assert again.senses.compass is True and again.policy.brain.inputs == 8 * 20 + 8
    assert any("a compass" in note for note in again.notes)
    for _ in range(3):
        assert life.tick()["pose"] == again.tick()["pose"]
    off = life.control("law", {"course": None})
    assert off["changed"] and life.course is None and life.mark == pytest.approx(life.distance_forward)
    assert checked_course("birth") == 0.0 and checked_course("off") is None and checked_course("1.5") == 1.5
    born = life.control("reset", {"compass": False})
    assert born["compass"] is False and life.senses.compass is False and life.distance_course == 0.0
    with pytest.raises(ValueError):
        life.reset({"heading": 1.0})


def test_the_ground_sense_is_a_birth_option_saved_with_the_life(tmp_path):
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", senses=Senses(compass=True, ground=True),
                course=0.0)
    assert life.describe()["ground"] is True and life.describe()["inputs"] == 8 * 22 + 8
    for _ in range(5):
        life.tick()
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")    # plain senses asked for
    assert again.load()["custody"] == "kept"
    assert again.senses.ground is True and again.senses.compass is True and again.policy.brain.inputs == 8 * 22 + 8
    assert any("the ground sense" in note for note in again.notes)
    for _ in range(3):
        assert life.tick()["pose"] == again.tick()["pose"]
    born = life.control("reset", {"ground": False})
    assert born["ground"] is False and born["compass"] is True
    assert life.senses.ground is False and life.senses.compass is True and life.policy.brain.inputs == 8 * 20 + 8
    assert "a compass" in life.notes[0] and "ground" not in life.notes[0]
    born = life.control("reset", {"ground": True, "compass": False})
    assert life.senses.options() == {"per": 8, "setpoints": False, "compass": False, "ground": True}
    assert "the ground sense" in life.notes[0] and life.policy.brain.inputs == 8 * 20 + 8


def test_a_slot_per_leg_is_a_birth_option_saved_with_the_life(tmp_path):
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", layout="leg")
    assert life.describe()["layout"] == "leg" and life.policy.layout == "leg" and len(life.policy.brain.motor_index) == 72
    for _ in range(5):
        out = life.tick()
        assert len(out["commands"]) == 16 and all(c in (0, 1, 2) for c in out["commands"])
    life.save()
    assert json.loads(life.sidecar.read_text())["layout"] == "leg"
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")    # a slot per joint asked for
    assert again.load()["custody"] == "kept"
    assert again.layout == "leg" and again.policy.layout == "leg"
    assert any("a slot per leg" in note for note in again.notes)
    for _ in range(3):
        assert life.tick()["pose"] == again.tick()["pose"]
    # the controls take the life's layout
    life.switch("random")
    assert life.policy.layout == "leg" and len(life.tick()["commands"]) == 16
    life.switch("frozen")
    assert life.policy.layout == "leg" and len(life.tick()["commands"]) == 16
    life.switch("brain")
    # a birth names it; what it leaves out stays
    born = life.control("reset", {"layout": "joint"})
    assert born["layout"] == "joint" and life.layout == "joint" and len(life.policy.brain.motor_index) == 48
    assert "slot per leg" not in life.notes[0]
    born = life.control("reset", {"layout": "leg", "compass": True})
    assert born["layout"] == "leg" and life.layout == "leg" and len(life.policy.brain.motor_index) == 72
    assert "a slot per leg" in life.notes[0] and "a compass" in life.notes[0]
    with pytest.raises(ValueError):
        life.control("reset", {"layout": "wheel"})
    with pytest.raises(ValueError):
        Life("brain", seed=0, pace=0.0, runs=tmp_path / "x", backend="cpu", layout="wheel")


def test_the_drift_is_the_worlds_and_is_set_saved_and_counted_with_the_law(tmp_path):
    from spider.body import DT

    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0.0, runs=tmp_path, drift=-1)
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", drift=0.002)
    assert life.describe()["reward"]["drift"] == 0.002 and life.body.drift == 0.002
    from spider.life import law_text
    assert law_text(life.law()) == "progress · drift 0.002 m/s"
    for _ in range(3):
        life.tick()
    # switched off live: counted as a law change, the body follows from the next moment on
    report = life.control("law", {"drift": 0})
    assert report["changed"] and life.body.drift == 0.0 and len(life.law_changes) == 1
    assert life.control("law", {"drift": 0})["changed"] is False
    with pytest.raises(ValueError):
        life.control("law", {"drift": -0.002})
    report = life.control("law", {"drift": 0.005})
    assert report["changed"] and life.body.drift == 0.005 and len(life.law_changes) == 2
    # saved and loaded with the life, whatever the flags say; the loaded world runs under it
    for _ in range(3):
        life.tick()
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")       # still ground asked for
    assert again.load()["custody"] == "kept"
    assert again.drift == 0.005 and again.body.drift == 0.005 and again.law() == life.law()
    assert any("keeps its own" in note and "drift 0.005" in note for note in again.notes)
    for _ in range(4):
        assert again.tick()["pose"] == life.tick()["pose"]
    # on a current a standing body loses ground: the random control and a held body go backward
    held = Life("random", seed=0, pace=0.0, runs=tmp_path, drift=0.01)
    held.policy.moment = lambda observation, reward: {"commands": [1] * JOINTS, "answer": [1] * JOINTS, "aroused": False,
                                                       "sweeps": 0, "learning_sweeps": 0, "refused": False}
    for _ in range(10):
        held.tick()
    assert held.distance_forward == pytest.approx(-10 * 0.01 * DT)
    assert held.reward == 0.0                                       # backward is free under the plain law
    # a reset body takes the life's drift
    life.control("reset", {"seed": 1})
    assert life.body.drift == 0.005 and life.drift == 0.005


def test_the_displacement_is_from_the_birth_point_and_survives_recentres_and_saves(tmp_path):
    life = Life("random", seed=0, pace=0, runs=tmp_path)
    for _ in range(50):
        life.tick()
    walked = list(life.displacement)
    assert life.displacement_m() == pytest.approx(math.hypot(*walked))
    assert life.latest["displacement_m"] == pytest.approx(life.displacement_m(), abs=1e-4)
    # the body at the arena's edge: the next moment recentres it, the displacement keeps counting
    life.body.x, life.body.y = 5.0, 0.3        # past the edge whatever the moment moves
    before = life.displacement_m()
    life.tick()
    assert life.recentres == 1 and life.body.x == 0.0 and life.body.y == 0.0
    assert life.displacement_m() != pytest.approx(0.0) and abs(life.displacement_m() - before) < 0.01
    # a recentre by hand neither
    life.body.x = 1.0
    life.control("recentre")
    assert life.recentres == 2 and life.displacement_m() == pytest.approx(life.displacement_m())
    # a straight walk's displacement is its forward metres; a circle's is not
    straight = Life("random", seed=0, pace=0, runs=tmp_path)
    straight.body.vx = 0.5
    for _ in range(100):
        straight.body.omega = 0.0
        straight.tick()
    assert straight.displacement_m() == pytest.approx(straight.distance_forward, abs=1e-3)
    # saved and loaded with the life; reset starts it at zero
    brain = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(20):
        brain.tick()
    brain.save()
    again = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    again.load()
    assert again.displacement == pytest.approx(brain.displacement) and again.displacement_m() > 0.0
    again.control("reset")
    assert again.displacement == [0.0, 0.0] and again.displacement_m() == 0.0
    # a save from before the displacement existed starts it where the body stands
    brain.save()
    state = json.loads(brain.sidecar.read_text())
    del state["displacement"]
    state["recentres"] = 1
    brain.sidecar.write_text(json.dumps(state))
    older = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    older.load()
    assert older.displacement == pytest.approx([brain.body.x, brain.body.y])
    assert any("without a displacement" in n for n in older.notes)


def test_the_strict_course_pays_nose_first_on_the_course_only(tmp_path):
    """Under ``strict`` the progress term is the lesser of the motion along the course and along
    the body's own heading: nose first on the course pays as the course law does, tail first
    down the course is backward along the nose (free, or charged as backward pay says), nose
    first across the course pays nothing."""
    import math

    def earned(heading, **law):
        life = Life("random", seed=0, pace=0, runs=tmp_path, **law)
        life.policy = RecordingPolicy()                      # every hip sweeps back: forward progress
        life.body.heading = heading
        for _ in range(40):
            life.tick()
        return sum(r for r in life.policy.rewards if r is not None), life

    along, life = earned(0.0, course=0.0, strict=True)
    course_only, _ = earned(0.0, course=0.0)
    assert along == pytest.approx(course_only) and along > 0.5
    across, _ = earned(math.pi / 2, course=0.0, strict=True)
    assert across == pytest.approx(0.0, abs=0.02)
    # tail first down the course: the body points against the course and is driven backward
    life = Life("random", seed=0, pace=0, runs=tmp_path, course=0.0, strict=True, backward_pay=1.0)
    life.policy = RecordingPolicy()
    life.body.heading = math.pi
    for _ in range(40):
        life.body.vx = -0.2                                   # pushed tail first: along the course, against the nose
        life.tick()
    assert life.distance_course > 0.05 and life.distance_forward < -0.05
    assert sum(r for r in life.policy.rewards if r is not None) < -0.5
    loose = Life("random", seed=0, pace=0, runs=tmp_path, course=0.0, backward_pay=1.0)
    loose.policy = RecordingPolicy()
    loose.body.heading = math.pi
    for _ in range(40):
        loose.body.vx = -0.2
        loose.tick()
    assert sum(r for r in loose.policy.rewards if r is not None) > 0.5   # the course law alone pays tail first
    assert "strict" in law_text(life.law()) and life.describe()["reward"]["strict"] is True
    # it needs a course; a course switched off switches it off; it is set live, counted and saved
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs=tmp_path, strict=True)
    plain = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    with pytest.raises(ValueError):
        plain.set_law({"strict": True})
    assert plain.set_law({"course": True, "strict": True})["changed"] and plain.strict and plain.course == 0.0
    assert plain.set_law({"course": None})["changed"] and plain.strict is False and len(plain.law_changes) == 2
    plain.set_law({"course": True, "strict": True})
    for _ in range(5):
        plain.tick()
    plain.save()
    again = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    again.load()
    assert again.strict is True and again.course == 0.0 and len(again.law_changes) == 3


def test_the_strict_course_takes_the_lesser_over_the_window_not_each_moment(tmp_path):
    """A stride sways: the body's speed along its nose and along the course alternate in sign
    within one step while both are positive over it. The lesser of the two each moment is
    negative on average; the lesser of the two window means is not. With window 1 the orders
    are the same."""
    import math

    def earned(window):
        life = Life("random", seed=0, pace=0, runs=tmp_path, course=0.0, strict=True, backward_pay=1.0, window=window)
        life.policy = RecordingPolicy()
        life.body.heading = 1.0                              # nose first, well off the course
        for i in range(40):
            # a sway: forward with a lurch to one side, back with a lurch to the other; net forward
            # along the nose and net along the course, but the two measures' signs never agree
            life.body.vx, life.body.vy = (0.2, 0.3) if i % 2 == 0 else (-0.1, -0.3)
            life.tick()
        return sum(r for r in life.policy.rewards if r is not None), life

    each, life = earned(1)
    over, _ = earned(2)
    assert life.distance_forward > 0 and life.distance_course > 0   # net forward and net along the course
    assert each < 0 < over                                   # the moment's lesser charges a net-forward stride; the window's pays it
    life.save()
    again = Life("random", seed=0, pace=0, runs=tmp_path, course=0.0, strict=True, backward_pay=1.0, window=2)
    assert len(again._nose) == 0

