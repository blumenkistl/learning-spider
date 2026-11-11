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
