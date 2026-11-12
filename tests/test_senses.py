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
    """A life saved with the default senses cannot be continued with per 16 senses: the
    load reports the mismatch instead of a life that refuses every moment (which is what
    ``server.py --per 16`` over an old runs/ did until 2026-11-12). The saved life stays."""
    import pytest
    from spider.life import Life
    from spider.brain import make_policy
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(3):
        life.tick()
    life.save()
    wide = Life("brain", seed=0, pace=0, runs=tmp_path, senses=Senses(per=16), backend="cpu")
    with pytest.raises(ValueError, match="composed for 152 inputs, the senses give 296"):
        wide.load()
    assert wide.moments == 0 and wide.policy.brain.inputs == 16 * 18 + LEGS
    assert (tmp_path / "spider.npz").exists() and (tmp_path / "spider.json").exists()
    for name in ("brain", "frozen"):
        with pytest.raises(ValueError, match="composed for 152 inputs"):
            make_policy(name, 0, tmp_path / "spider.npz", None, "cpu", inputs=16 * 18 + LEGS)
    # the matching senses still continue it
    same = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    assert same.load()["custody"] == "kept" and same.moments == 3
