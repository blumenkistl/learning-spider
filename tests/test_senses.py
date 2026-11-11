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
