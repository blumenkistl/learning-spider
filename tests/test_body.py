import math

import numpy as np
import pytest

from spider.body import (
    HIPS, LEG_CONVERGE, SPLAY, separate, supported,
    ARENA_RADIUS, CONTACT_Z, HEIGHT, HIP_RANGE, JOINTS, KNEE_RANGE, LEGS, Spider, kinematics,
)


def hold():
    return [1] * JOINTS


def command(hips=1, knees=1, legs=range(LEGS)):
    cmd = hold()
    for i in legs:
        cmd[2 * i] = hips
        cmd[2 * i + 1] = knees
    return cmd


def test_rest_pose_is_planted_and_mirrored():
    spider = Spider()
    knee, foot = kinematics(spider.joints)
    assert np.all(np.abs(HEIGHT + foot[:, 2]) < 1e-9)
    assert spider.planted().all()
    assert spider.stance_count == LEGS
    # left legs 0-3 mirror right legs 4-7 in y
    assert np.allclose(foot[:4, 0], foot[4:, 0])
    assert np.allclose(foot[:4, 1], -foot[4:, 1])
    assert np.allclose(knee[:4, 1], -knee[4:, 1])
    assert spider.z == HEIGHT


def test_joints_clip_at_their_ranges():
    spider = Spider()
    for _ in range(100):
        spider.step(command(hips=2, knees=2))
    assert np.allclose(spider.joints[:, 0], HIP_RANGE[1])
    assert np.allclose(spider.joints[:, 1], KNEE_RANGE[1])
    for _ in range(100):
        spider.step(command(hips=0, knees=0))
    assert np.allclose(spider.joints[:, 0], HIP_RANGE[0])
    assert np.allclose(spider.joints[:, 1], KNEE_RANGE[0])


def test_bad_commands_refused():
    with pytest.raises(ValueError):
        Spider().step([3] * JOINTS)


def test_all_hips_sweeping_back_moves_forward_without_turning():
    spider = Spider()
    progress = 0.0
    for _ in range(6):
        progress += spider.step(command(hips=0))["progress"]
    assert spider.x > 0.01
    assert progress > 0.01
    assert abs(spider.heading) < 1e-9
    assert abs(spider.y) < 1e-9


def test_one_side_cannot_hold_the_body_up():
    """Feet on one side alone, however many, are no support; one and two across the body are."""
    assert supported([1, 1, 1, 1, 0, 0, 0, 0]) is False     # the whole left side
    assert supported([1, 1, 0, 0, 0, 0, 0, 0]) is False
    assert supported([1, 0, 0, 0, 1, 0, 0, 0]) is False     # one each side: two feet
    assert supported([1, 0, 0, 0, 1, 1, 0, 0]) is True      # one left, two right
    assert supported([0, 1, 1, 0, 0, 0, 0, 1]) is True
    assert supported([1] * LEGS) is True

    # the left feet stroke back while every right foot is lifted: the body is down and stays
    spider = Spider()
    cmd = hold()
    for i in range(4, LEGS):
        cmd[2 * i + 1] = 2                 # lift the right legs
    for _ in range(10):
        spider.step(cmd)
    assert spider.stance_count == 4 and not supported(spider.planted()) and spider.z < HEIGHT
    cmd = hold()
    for i in range(4):
        cmd[2 * i] = 0                     # stroke the left hips back
    out = [spider.step(cmd) for _ in range(6)]
    assert all(o["progress"] == 0.0 and o["stance"] == 4 for o in out)
    assert spider.x == 0.0 and spider.heading == 0.0

    # one right foot down too: the stroke pushes
    spider = Spider()
    cmd = hold()
    for i in range(5, LEGS):
        cmd[2 * i + 1] = 2                 # lift three right legs, leg 4 stays down
    for _ in range(10):
        spider.step(cmd)
    assert supported(spider.planted()) and spider.z == HEIGHT
    cmd = hold()
    for i in range(4):
        cmd[2 * i] = 0
    for _ in range(6):
        spider.step(cmd)
    assert spider.x > 0.01 and spider.heading < 0   # the left stroke turns it too


def legs_cross(spider, i, j):
    """Whether the hip-to-foot lines of legs i and j cross in the ground plane."""
    _, foot = spider.feet()
    p, r = HIPS[i, :2], foot[i, :2] - HIPS[i, :2]
    q, u = HIPS[j, :2], foot[j, :2] - HIPS[j, :2]
    den = r[0] * u[1] - r[1] * u[0]
    if abs(den) < 1e-12:
        return False
    t = ((q[0] - p[0]) * u[1] - (q[1] - p[1]) * u[0]) / den
    w = ((q[0] - p[0]) * r[1] - (q[1] - p[1]) * r[0]) / den
    return 0.0 <= t <= 1.0 and 0.0 <= w <= 1.0


def test_neighbouring_legs_stop_against_each_other():
    """Leg 1 swings forward and leg 0 back as far as they can: they meet at the stop, never cross."""
    spider = Spider(motors="rate")                              # a command a moment, as the test steps it
    cmd = hold()
    cmd[0] = 0                             # leg 0's hip back
    cmd[2] = 2                             # leg 1's hip forward
    for _ in range(30):
        spider.step(cmd)
        yaw = SPLAY + spider.joints[:, 0]
        assert yaw[1] - yaw[0] <= LEG_CONVERGE + 1e-9
        assert not legs_cross(spider, 0, 1)
    assert yaw[1] - yaw[0] == pytest.approx(LEG_CONVERGE)
    assert spider.joints[0, 0] > HIP_RANGE[0] and spider.joints[1, 0] < HIP_RANGE[1]   # neither reached its range

    # the stop is a stop, not a wall: the leg behind may turn forward until it meets the one in front
    spider = Spider(motors="rate")
    cmd = hold()
    cmd[2] = 2
    for _ in range(30):
        spider.step(cmd)
    assert spider.joints[1, 0] == pytest.approx(HIP_RANGE[1])   # from rest, leg 1 reaches its full swing
    assert not legs_cross(spider, 0, 1) and not legs_cross(spider, 1, 2)

    # without the stop, the same commands cross the legs
    crossed = Spider()
    crossed.joints[0, 0], crossed.joints[1, 0] = HIP_RANGE[0], HIP_RANGE[1]
    assert legs_cross(crossed, 0, 1)
    assert not legs_cross(Spider(joints=separate(crossed.joints)), 0, 1)

    # a whole side pressed together settles in order, every leg in range
    j = np.zeros((LEGS, 2))
    j[:4, 0] = [HIP_RANGE[0], HIP_RANGE[1], HIP_RANGE[1], HIP_RANGE[1]]
    out = separate(j)
    yaw = SPLAY + out[:, 0]
    assert np.all(np.diff(yaw[:4]) <= LEG_CONVERGE + 1e-9)
    assert np.all(out[:, 0] >= HIP_RANGE[0] - 1e-9) and np.all(out[:, 0] <= HIP_RANGE[1] + 1e-9)
    assert np.array_equal(out[4:], j[4:])                     # the other side is untouched


def test_lifted_legs_push_nothing():
    spider = Spider()
    for _ in range(10):
        spider.step(command(knees=2))
    assert spider.stance_count == 0
    for _ in range(6):
        spider.step(command(hips=0))
    assert abs(spider.x) < 1e-9
    assert abs(spider.heading) < 1e-9


def test_step_reports_the_turn():
    spider = Spider()
    assert spider.step(hold())["turn"] == 0.0
    turned = 0.0
    for _ in range(6):
        turned += spider.step(command(hips=0, legs=range(4)))["turn"]
    assert turned < 0.0
    assert turned == pytest.approx(spider.heading, abs=1e-9)   # the moments' turns sum to the heading


def test_left_stroke_turns_right_and_moves_forward():
    spider = Spider()
    for _ in range(6):
        spider.step(command(hips=0, legs=range(4)))
    assert spider.x > 0.0
    assert spider.heading < 0.0


def test_closed_cycle_with_all_feet_planted_nets_nothing():
    spider = Spider(motors="rate")                              # the cycle is commanded moment by moment
    for _ in range(4):
        spider.step(command(hips=0))
    for _ in range(4):
        spider.step(command(hips=2))
    # let the grip filter and the decay settle
    for _ in range(40):
        spider.step(hold())
    assert abs(spider.x) < 2e-3
    assert abs(spider.heading) < 2e-3


def test_a_gait_moves_the_body():
    """Two groups of legs alternating stance and swing walk the body forward."""
    spider = Spider()
    a, b = [0, 2, 5, 7], [1, 3, 4, 6]
    for _ in range(100):
        for group, other in ((a, b), (b, a)):
            cmd = hold()
            for i in other:
                cmd[2 * i + 1] = 2      # lift
            for _ in range(2):
                spider.step(cmd)
            cmd = hold()
            for i in group:
                cmd[2 * i] = 0          # stroke back, planted
            for i in other:
                cmd[2 * i] = 2          # swing forward, lifted
            for _ in range(3):
                spider.step(cmd)
            cmd = hold()
            for i in other:
                cmd[2 * i + 1] = 0      # plant
            for _ in range(2):
                spider.step(cmd)
    assert spider.x > 0.3


def test_recentre_keeps_joints_and_heading():
    spider = Spider()
    for _ in range(3):
        spider.step(command(hips=0, legs=range(4)))
    twin = Spider.restore(spider.state_dict())
    spider.x, spider.y = ARENA_RADIUS + 1.0, 0.0
    out = spider.step(hold())
    twin.step(hold())
    assert out["recentred"]
    assert spider.x == 0.0 and spider.y == 0.0
    assert np.allclose(spider.joints, twin.joints)
    assert spider.heading == twin.heading       # the moment's own turn still happens
    assert spider.recentres == 1


def test_state_round_trip():
    spider = Spider()
    for i in range(7):
        spider.step(command(hips=i % 3, knees=(i + 1) % 3, legs=[0, 3, 5]))
    twin = Spider.restore(spider.state_dict())
    for _ in range(5):
        a = spider.step(command(hips=0))
        b = twin.step(command(hips=0))
        assert a == b
    assert twin.state_dict() == spider.state_dict()


def test_pose_shape():
    pose = Spider().pose()
    assert len(pose["legs"]) == LEGS and len(pose["legs"][0]) == 6
    assert len(pose["joints"]) == JOINTS and len(pose["planted"]) == LEGS
    assert all(abs(HEIGHT + leg[5]) <= CONTACT_Z for leg in pose["legs"])


def test_pressing_knee_holds_the_foot_where_it_is():
    from spider.body import CONTACT_Z, HEIGHT, kinematics
    rest = np.zeros((LEGS, 2))
    pressed = rest.copy()
    pressed[:, 1] = -0.3
    _, foot_rest = kinematics(rest)
    _, foot = kinematics(pressed)
    assert np.allclose(foot, foot_rest)                           # the press moves the foot nowhere
    assert np.all(HEIGHT + foot[:, 2] <= CONTACT_Z)               # planted
    assert Spider(joints=pressed).stance_count == LEGS


def test_rocking_on_a_knee_moves_nothing():
    # the stalled brain's exploit: one knee pressing and resting in turn, every other joint held
    spider = Spider()
    for moment in range(40):
        commands = hold()
        commands[1] = 0 if moment % 2 == 0 else 2                 # leg 0's knee: press, rest, press, ...
        out = spider.step(commands)
        assert out["progress"] == 0.0 and out["stance"] == LEGS
    assert spider.x == 0.0 and spider.y == 0.0 and spider.heading == 0.0



def test_target_motors_are_servos_with_a_kept_setpoint():
    """Under target motors one command sets a joint's setpoint to an end of its range and
    hold keeps it: the joint then travels there at its rate over the following moments,
    where the founder's rate motors need the command every moment."""
    spider = Spider(motors="target")
    cmd = hold()
    cmd[0] = cmd[2] = 2                     # legs 0 and 1: hip setpoints to the front end
    spider.step(cmd)
    assert spider.targets[1, 0] == HIP_RANGE[1]
    assert spider.joints[1, 0] == pytest.approx(1.5 * 0.05)   # one moment at the hip's rate
    for _ in range(20):
        spider.step(hold())                 # hold keeps the setpoint, the joint keeps going
    assert spider.joints[1, 0] == pytest.approx(HIP_RANGE[1])
    assert spider.targets[1, 0] == HIP_RANGE[1]
    # a servo holds its setpoint against the stop: leg 1 sent forward alone against a leg 0
    # held at rest settles at the stop short of its range, where under rate motors the
    # split pushed the held leg 0 along
    alone = Spider(motors="target")
    cmd = hold(); cmd[2] = 2
    alone.step(cmd)
    for _ in range(30):
        alone.step(hold())
    assert alone.joints[1, 0] < HIP_RANGE[1] and alone.joints[0, 0] > 0
    assert (SPLAY + alone.joints[:, 0])[1] - (SPLAY + alone.joints[:, 0])[0] == pytest.approx(LEG_CONVERGE)
    cmd[2] = 0                              # back end
    spider.step(cmd)
    assert spider.targets[1, 0] == HIP_RANGE[0] and spider.joints[1, 0] < HIP_RANGE[1]
    # the knee the same way: up, and down to the pressing end
    cmd = hold(); cmd[1] = 2
    for _ in range(25):
        spider.step(cmd if _ == 0 else hold())
    assert spider.joints[0, 1] == pytest.approx(KNEE_RANGE[1]) and not spider.planted()[0]
    cmd[1] = 0
    for _ in range(40):
        spider.step(cmd if _ == 0 else hold())
    assert spider.joints[0, 1] == pytest.approx(KNEE_RANGE[0]) and spider.planted()[0]
    # the leg stop holds under servos too: a setpoint past the stop is not reached
    spider = Spider(motors="target")
    cmd = hold(); cmd[0] = 0; cmd[2] = 2
    spider.step(cmd)
    for _ in range(30):
        spider.step(hold())
        assert not legs_cross(spider, 0, 1)
    yaw = SPLAY + spider.joints[:, 0]
    assert yaw[1] - yaw[0] == pytest.approx(LEG_CONVERGE)

    # the founder's rate motors (the default until 2026-11-12) are unchanged by the model's existence
    rate = Spider(motors="rate")
    assert Spider().motors == "target"                           # the default since
    cmd = hold(); cmd[2] = 2
    rate.step(cmd)
    rate.step(hold())
    assert rate.joints[1, 0] == pytest.approx(1.5 * 0.05)    # hold holds the joint, not a setpoint
    assert rate.motors == "rate"
    with pytest.raises(ValueError):
        Spider(motors="servo")


def test_the_motor_model_and_its_setpoints_survive_the_world_state():
    spider = Spider(motors="target")
    cmd = hold(); cmd[2] = 2; cmd[5] = 0
    spider.step(cmd)
    again = Spider.restore(spider.state_dict())
    assert again.motors == "target" and np.array_equal(again.targets, spider.targets)
    for _ in range(5):
        a = spider.step(hold()); b = again.step(hold())
        assert a == b and np.array_equal(spider.joints, again.joints)
    # a world saved before the model existed is a rate body
    old = Spider().state_dict(); del old["motors"]; del old["targets"]
    assert Spider.restore(old).motors == "rate"
