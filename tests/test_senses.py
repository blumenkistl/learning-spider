import numpy as np

from spider.body import JOINTS, LEGS, Spider
from spider.senses import INPUTS, Senses, input_names, observe


def test_shape_and_range():
    x = observe(Spider())
    assert x.shape == (1, INPUTS)
    assert INPUTS == 8 * 18 + LEGS
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    assert len(input_names()) == INPUTS


def test_a_reading_lights_at_most_five_inputs():
    spider = Spider()
    x = observe(spider)[0]
    coded = x[: INPUTS - LEGS].reshape(18, 8)
    assert np.all((coded > 0).sum(axis=1) <= 5)
    assert np.all((coded > 0).sum(axis=1) >= 1)


def test_lifted_foot_flag_is_zero():
    spider = Spider()
    cmd = [1] * JOINTS
    cmd[2 * 3 + 1] = 2
    for _ in range(10):
        spider.step(cmd)
    flags = observe(spider)[0, INPUTS - LEGS:]
    assert flags[3] == 0.0
    assert flags.sum() == LEGS - 1


def test_per_is_a_gene():
    senses = Senses(per=16)
    assert senses.inputs == 16 * 18 + LEGS
    assert observe(Spider(), senses).shape == (1, senses.inputs)
    assert len(input_names(senses)) == senses.inputs


def test_a_life_composes_its_brain_for_its_senses(tmp_path):
    """A life with per 16 senses gets a brain of 296 inputs, and so refuses nothing (until
    2026-11-12 the brain was composed for the default senses and refused every moment)."""
    from spider.life import Life
    for policy in ("brain", "frozen"):
        life = Life(policy, seed=0, pace=0, runs=tmp_path, senses=Senses(per=16), backend="cpu")
        assert life.policy.brain.inputs == 16 * 18 + LEGS
        for _ in range(5):
            life.tick()
        assert life.policy.refusals == 0
    life.switch("brain")
    assert life.policy.brain.inputs == 16 * 18 + LEGS


def test_a_saved_brain_for_other_senses_is_refused_at_the_load(tmp_path):
    """A life saved with the default senses asked for per 16 senses: the sidecar names the
    saved senses and the load continues the life with them, noted, instead of a life that
    refuses every moment (which is what ``server.py --per 16`` over an old runs/ did until
    2026-11-12). A brain file without its sidecar still reports the mismatch."""
    import pytest
    from spider.life import Life
    from spider.brain import make_policy
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(3):
        life.tick()
    life.save()
    wide = Life("brain", seed=0, pace=0, runs=tmp_path, senses=Senses(per=16), backend="cpu")
    assert wide.load()["custody"] == "kept" and wide.moments == 3
    assert wide.senses.per == 8 and wide.policy.brain.inputs == 8 * 18 + LEGS
    assert any("saved with 152 inputs, asked for 296" in note for note in wide.notes)
    for name in ("brain", "frozen"):
        with pytest.raises(ValueError, match="composed for 152 inputs"):
            make_policy(name, 0, tmp_path / "spider.npz", None, "cpu", inputs=16 * 18 + LEGS)
    (tmp_path / "spider.json").unlink()                                      # the brain file alone
    alone = Life("brain", seed=0, pace=0, runs=tmp_path, senses=Senses(per=16), backend="cpu")
    with pytest.raises(ValueError, match="composed for 152 inputs, the senses give 296"):
        alone.load()
    assert alone.moments == 0 and (tmp_path / "spider.npz").exists()


def test_the_setpoint_senses_are_an_option_of_target_muscles():
    import pytest

    senses = Senses(setpoints=True)
    assert senses.inputs == 8 * (18 + 2 * LEGS) + LEGS and len(senses.input_names()) == senses.inputs
    assert any("hip3_set" in n for n in senses.input_names())
    spider = Spider(muscles="target")
    x = senses.observe(spider)
    assert x.shape == (1, senses.inputs) and np.all(x >= 0.0) and np.all(x <= 1.0)
    readings = senses.readings(spider)
    assert readings.shape == (18 + 2 * LEGS,)
    assert np.array_equal(readings[18:18 + LEGS], spider.targets[:, 0])
    assert np.array_equal(readings[18 + LEGS:], spider.targets[:, 1])
    spider.step([2] * LEGS * 2)                                   # every joint sent to its high end
    assert np.all(senses.readings(spider)[18:] == spider.targets.reshape(-1, order="F"))
    with pytest.raises(ValueError):
        senses.readings(Spider(muscles="rate"))
    assert not Senses().setpoints and Senses().inputs == INPUTS


def test_the_compass_senses_the_heading():
    import math

    senses = Senses(compass=True)
    assert senses.inputs == 8 * 20 + LEGS and len(senses.input_names()) == senses.inputs
    assert senses.input_names()[8 * 18] == "heading_cos@-1.00" and senses.options()["compass"] is True
    spider = Spider()
    assert np.allclose(senses.readings(spider)[18:], [1.0, 0.0])          # born facing the world's zero heading
    spider.heading = math.pi / 2
    assert np.allclose(senses.readings(spider)[18:], [0.0, 1.0])
    spider.heading = -math.pi + 1e-9                                         # no seam at ±pi: the same point
    assert np.allclose(senses.readings(spider)[18:], [-1.0, 0.0])
    x = senses.observe(spider)
    assert x.shape == (1, senses.inputs) and np.all(x >= 0.0) and np.all(x <= 1.0)
    both = Senses(setpoints=True, compass=True)
    assert both.inputs == 8 * 36 + LEGS and both.readings(Spider(muscles="target")).shape == (36,)
    assert not Senses().compass and Senses().options() == {"per": 8, "setpoints": False, "compass": False, "ground": False}


def test_the_ground_sense_reads_the_velocity_over_the_ground():
    import math

    senses = Senses(ground=True)
    assert senses.inputs == 8 * 20 + LEGS and senses.input_names()[8 * 18] == "ground_vx@-0.30"
    assert senses.options()["ground"] is True
    spider = Spider()
    assert np.allclose(senses.readings(spider)[18:], [0.0, 0.0])            # at rest
    spider.vx, spider.vy = 0.2, 0.05                                         # moving forward and a little left
    assert np.allclose(senses.readings(spider)[18:], [0.2, 0.05])           # facing the world's zero heading
    spider.heading = math.pi / 2                                             # the same motion, facing world y
    assert np.allclose(senses.readings(spider)[18:], [-0.05, 0.2])
    assert np.allclose(spider.ground_velocity(), [-0.05, 0.2])
    spider.heading = math.pi                                                 # facing back: forward is world -x
    assert np.allclose(senses.readings(spider)[18:], [-0.2, -0.05])
    x = senses.observe(spider)
    assert x.shape == (1, senses.inputs) and np.all(x >= 0.0) and np.all(x <= 1.0)
    everything = Senses(setpoints=True, compass=True, ground=True)
    assert everything.inputs == 8 * 38 + LEGS and everything.readings(Spider(muscles="target")).shape == (38,)
    assert everything.input_names()[8 * 34] == "heading_cos@-1.00" and everything.input_names()[8 * 37] == "ground_vy@-0.30"
    # what a step adds to the world position is the ground velocity times DT
    from spider.body import DT
    spider = Spider()
    spider.vx, spider.vy, spider.heading = 0.1, 0.0, 0.3
    before = (spider.x, spider.y)
    out = spider.step([1] * 16)
    vx, vy = spider.ground_velocity()
    assert abs(out["moved"][0] - (spider.x - before[0])) < 1e-12 and abs(out["moved"][1] - (spider.y - before[1])) < 1e-12
    assert abs(vx * DT - out["moved"][0]) < 0.05 * abs(out["moved"][0]) + 1e-9   # the velocity after the step, the same motion within the moment's decay
