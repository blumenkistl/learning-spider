import json

import numpy as np
import pytest

from spider.body import JOINTS, Spider
from spider.brain import SpiderBrain
from spider.life import BLOCK, PROGRESS_PAY, Life, pay


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
