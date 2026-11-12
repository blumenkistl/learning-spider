import json

import numpy as np
import pytest

from spider.body import JOINTS, Spider
from spider.brain import SpiderBrain
from spider.life import BLOCK, Life, pay


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
    assert pay(0.0, 4) == 0.0 and pay(0.0, 8) == 0.0
    assert pay(0.0, 3) == pytest.approx(-0.1) and pay(0.0, 0) == pytest.approx(-0.4)
    assert pay(0.001, 2) == pytest.approx(0.04 - 0.2)
    assert pay(1.0, 0) == pytest.approx(0.5 - 0.4)
    assert pay(0.0, 0, stance_pay=0.0) == 0.0            # progress-only pay, the earlier law
    assert pay(-1.0, 0) == pytest.approx(-0.4)           # backward still pays zero progress


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
    assert again.spider.state_dict() == life.spider.state_dict()
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
    life.spider.x = 1.0
    assert life.control("recentre")["ok"] and life.spider.x == 0.0
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


def test_reset_starts_a_new_life_and_removes_the_save(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(BLOCK + 5):
        life.tick()
    old = life.policy
    assert life.control("save")["saved"] and life.brain_file.exists() and life.sidecar.exists()
    assert life.control("policy", "random")["changed"]
    life.spider.x = 1.0
    r = life.control("reset")
    assert r["ok"] and r["reset"] and r["moments_before"] == BLOCK + 5
    assert isinstance(life.policy, SpiderBrain) and life.policy is not old and life.policy.moments == 0
    assert life.moments == 0 and life.distance_forward == 0.0 and life.spider.x == 0.0
    assert life.reward is None and not life.blocks and life.last_save_at is None
    assert not life.brain_file.exists() and not life.sidecar.exists()
    assert life.latest["moments"] == 0 and life.latest["policy"] == "brain"
    assert life.control("load")["loaded"] is False            # nothing left to continue
    life.tick()
    assert life.moments == 1
