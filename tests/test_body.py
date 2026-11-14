import math

import numpy as np
import pytest

from spider.body import (
    HIPS, LEG_CONVERGE, SPLAY, separate, supported,
    ARENA_RADIUS, CONTACT_Z, DT, HEIGHT, HIP_RANGE, JOINTS, KNEE_RANGE, LEGS, Spider, kinematics,
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
    spider = Spider(muscles="rate")                              # a command a moment, as the test steps it
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
    spider = Spider(muscles="rate")
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
    spider = Spider(muscles="rate")                              # the cycle is commanded moment by moment
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



def test_target_muscles_are_servos_with_a_kept_setpoint():
    """Under target muscles one command sets a joint's setpoint to an end of its range and
    hold keeps it: the joint then travels there at its rate over the following moments,
    where the founder's rate muscles need the command every moment."""
    spider = Spider(muscles="target")
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
    # held at rest settles at the stop short of its range, where under rate muscles the
    # split pushed the held leg 0 along
    alone = Spider(muscles="target")
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
    spider = Spider(muscles="target")
    cmd = hold(); cmd[0] = 0; cmd[2] = 2
    spider.step(cmd)
    for _ in range(30):
        spider.step(hold())
        assert not legs_cross(spider, 0, 1)
    yaw = SPLAY + spider.joints[:, 0]
    assert yaw[1] - yaw[0] == pytest.approx(LEG_CONVERGE)

    # the founder's rate muscles (the default until 2026-11-12) are unchanged by the model's existence
    rate = Spider(muscles="rate")
    assert Spider().muscles == "target"                           # the default since
    cmd = hold(); cmd[2] = 2
    rate.step(cmd)
    rate.step(hold())
    assert rate.joints[1, 0] == pytest.approx(1.5 * 0.05)    # hold holds the joint, not a setpoint
    assert rate.muscles == "rate"
    with pytest.raises(ValueError):
        Spider(muscles="servo")


def test_the_muscle_model_and_its_setpoints_survive_the_world_state():
    spider = Spider(muscles="target")
    cmd = hold(); cmd[2] = 2; cmd[5] = 0
    spider.step(cmd)
    again = Spider.restore(spider.state_dict())
    assert again.muscles == "target" and np.array_equal(again.targets, spider.targets)
    for _ in range(5):
        a = spider.step(hold()); b = again.step(hold())
        assert a == b and np.array_equal(spider.joints, again.joints)
    # a world saved before the model existed is a rate body
    old = Spider().state_dict(); del old["muscles"]; del old["targets"]
    assert Spider.restore(old).muscles == "rate"


def test_the_drift_takes_ground_every_moment_planted_or_not():
    from spider.body import DT

    still = Spider()
    current = Spider(drift=0.01)
    for _ in range(5):
        a = still.step(hold())
        b = current.step(hold())
        assert a["progress"] == 0.0                                   # a standing body on still ground stays
        assert b["progress"] == pytest.approx(-0.01 * DT)             # on a current it loses ground backward
        assert b["moved"][0] == pytest.approx(-0.01 * DT) and b["moved"][1] == pytest.approx(0.0)
        assert b["turn"] == 0.0 and b["stance"] == a["stance"]
    assert current.x == pytest.approx(-5 * 0.01 * DT) and still.x == 0.0
    assert current.ground_velocity() == pytest.approx((-0.01, 0.0))
    for _ in range(3):
        current.step(command(knees=2))                                 # every foot lifted: the current still takes its ground
    assert current.x == pytest.approx(-8 * 0.01 * DT)
    # a stroke nets what the feet made minus the current, and the same stroke on still ground more
    still, current = Spider(), Spider(drift=0.01)
    a = sum(still.step(command(hips=0))["progress"] for _ in range(4))
    b = sum(current.step(command(hips=0))["progress"] for _ in range(4))
    assert a > 0 and b == pytest.approx(a - 4 * 0.01 * DT)
    # the current turns with the body: backward along its own heading, whatever it faces
    turned = Spider(drift=0.01, heading=math.pi / 2)
    out = turned.step(hold())
    assert out["progress"] == pytest.approx(-0.01 * DT)
    assert out["moved"][0] == pytest.approx(0.0, abs=1e-12) and out["moved"][1] == pytest.approx(-0.01 * DT)
    # it is part of the world's state, and a world saved before it existed has none
    assert Spider.restore(current.state_dict()).drift == 0.01
    old = current.state_dict()
    del old["drift"]
    assert Spider.restore(old).drift == 0.0
    with pytest.raises(ValueError):
        Spider(drift=-0.01)


def test_the_muscle_rate_scales_how_far_a_joint_runs_in_a_moment():
    import numpy as np
    from spider.body import DT, HIGH, RATES, Spider

    slow, fast = Spider(muscles="target"), Spider(muscles="target", muscle_rate=2.0)
    start = slow.joints.copy()
    increase = [2] * 16                                    # every joint toward the high end of its range
    slow.step(increase)
    fast.step(increase)
    ran_slow = np.abs(slow.joints - start)
    ran_fast = np.abs(fast.joints - start)
    unclamped = (fast.joints < HIGH - 1e-9) & (ran_slow > 0)
    assert np.allclose(ran_fast[unclamped], 2.0 * ran_slow[unclamped], atol=1e-9)
    assert np.all(ran_slow <= np.broadcast_to(RATES * DT, ran_slow.shape) + 1e-9)
    rate = Spider(muscles="rate", muscle_rate=0.5)
    rate.step(increase)
    assert np.allclose(np.abs(rate.joints - start)[unclamped], 0.5 * ran_slow[unclamped], atol=1e-9)
    assert Spider.restore(fast.state_dict()).muscle_rate == 2.0
    older = fast.state_dict()
    del older["muscle_rate"]                                # a world saved before the rate existed ran at 1
    assert Spider.restore(older).muscle_rate == 1.0
    renamed = Spider(muscles="target", muscle_rate=3.0).state_dict()   # saved before 2026-11-14
    renamed["motors"], renamed["motor_rate"] = renamed.pop("muscles"), renamed.pop("muscle_rate")
    assert Spider.restore(renamed).muscles == "target" and Spider.restore(renamed).muscle_rate == 3.0
    with pytest.raises(ValueError):
        Spider(muscle_rate=0.0)


def test_the_hills_raise_the_body_and_plant_the_feet_on_the_terrain():
    from spider.body import ARENAS, ground_height

    from spider.body import HILLS, STEEP

    assert ARENAS == ("flat", "hills", "steep")
    # the steep hills are the same waves half as high again: the same ground, scaled
    xs0, ys0 = np.linspace(-5, 5, 37), np.linspace(5, -5, 37)
    assert np.allclose(ground_height("steep", xs0, ys0), 1.5 * ground_height("hills", xs0, ys0), atol=1e-3)
    assert [w[:2] + w[3:] for w in STEEP] == [w[:2] + w[3:] for w in HILLS]
    steep = Spider(arena="steep")
    assert all(steep.pose()["planted"]) and Spider.restore(steep.state_dict()).arena == "steep"
    with pytest.raises(ValueError):
        Spider(arena="moon")
    flat, hills = Spider(), Spider(arena="hills")
    assert np.all(ground_height("flat", np.arange(5.0), np.arange(5.0)) == 0.0)
    # the terrain rolls a few centimetres either way and is fixed: the same height at the same point
    xs, ys = np.meshgrid(np.linspace(-5, 5, 101), np.linspace(-5, 5, 101))
    h = ground_height("hills", xs, ys)
    assert 0.03 < h.max() < 0.1 and -0.1 < h.min() < -0.03
    assert np.array_equal(h, ground_height("hills", xs, ys))
    # at rest on the hills every foot stands on the ground under it, the body rides on the lowest
    pose = hills.pose()
    under = np.array(pose["ground"])
    assert np.allclose(under, ground_height("hills", *_feet_world(hills)), atol=1e-4)
    assert abs(pose["z"] - (under.min() + HEIGHT)) < 1e-4 and all(pose["planted"])
    # the same walk on both arenas: the flat one is the founder's, the hills one moves too
    for _ in range(40):
        a, b = flat.step(command(hips=0)), hills.step(command(hips=0))
    assert flat.pose()["ground"] == [0.0] * LEGS and flat.pose()["z"] == HEIGHT
    assert hills.x != 0.0 or hills.y != 0.0
    twin = Spider.restore(hills.state_dict())
    assert twin.arena == "hills" and twin.state_dict() == hills.state_dict()
    old = hills.state_dict()
    del old["arena"]
    assert Spider.restore(old).arena == "flat"                # a save from before the hills stood on flat ground


def _feet_world(spider):
    pose = spider.pose()
    c, s = math.cos(pose["heading"]), math.sin(pose["heading"])
    fx = np.array([leg[3] for leg in pose["legs"]])
    fy = np.array([leg[4] for leg in pose["legs"]])
    return pose["x"] + c * fx - s * fy, pose["y"] + s * fx + c * fy


def test_the_slope_pull_creeps_downhill_on_the_terrains_and_nowhere_else():
    from spider.body import ground_slope

    with pytest.raises(ValueError):
        Spider(arena="steep", slope_pull=-1)
    # the gradient is the height's: a small step along it rises by its length times the slope
    gx, gy = ground_slope("steep", 0.7, -1.3)
    from spider.body import ground_height
    e = 1e-5
    assert abs(gx - (ground_height("steep", 0.7 + e, -1.3) - ground_height("steep", 0.7 - e, -1.3)) / (2 * e)) < 1e-6
    assert abs(gy - (ground_height("steep", 0.7, -1.3 + e) - ground_height("steep", 0.7, -1.3 - e)) / (2 * e)) < 1e-6
    assert ground_slope("flat", 0.7, -1.3) == (0.0, 0.0)
    # a body holding still on a slope creeps down it and loses height; on flat ground nothing pulls
    still = Spider(arena="steep", slope_pull=0.5, x=0.7, y=-1.3)
    free = Spider(arena="steep", x=0.7, y=-1.3)
    flat = Spider(slope_pull=0.5, x=0.7, y=-1.3)
    for _ in range(40):
        still.step(hold()), free.step(hold()), flat.step(hold())
    assert (free.x, free.y) == (0.7, -1.3) and (flat.x, flat.y) == (0.7, -1.3)
    assert ground_height("steep", still.x, still.y) < ground_height("steep", 0.7, -1.3)
    # the pull is the world's: the progress along the heading and the ground velocity carry it
    out = Spider(arena="steep", slope_pull=0.5, x=0.7, y=-1.3).step(hold())
    assert out["progress"] == pytest.approx(-0.5 * gx * DT, rel=1e-6)   # heading 0: along x
    assert out["moved"] == pytest.approx((-0.5 * gx * DT, -0.5 * gy * DT), rel=1e-6)
    assert Spider.restore(still.state_dict()).slope_pull == 0.5
    old = still.state_dict()
    del old["slope_pull"]
    assert Spider.restore(old).slope_pull == 0.0
