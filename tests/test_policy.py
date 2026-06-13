import mujoco
import numpy as np
import pytest
import yaml

from locomotion.policy import G1WalkPolicy, POLICY_JOINTS, get_gravity_orientation
from sim.loader import load_scene

with open("configs/config.yaml") as f:
    CFG = yaml.safe_load(f)


def test_gravity_orientation_upright():
    g = get_gravity_orientation(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(g, [0, 0, -1], atol=1e-6)


def test_policy_joint_list_has_29():
    assert len(POLICY_JOINTS) == 29


def _make():
    model, _ = load_scene(CFG["sim"])
    d = mujoco.MjData(model)
    pol = G1WalkPolicy(model, d, CFG["policy"])
    pol.reset(spawn=CFG["sim"]["spawn_pos"])
    return model, d, pol


def test_obs_dim_is_96():
    m, d, pol = _make()
    obs = pol._build_obs()
    assert obs.shape == (96,)


def test_obs_indexed_by_robot_joint_names():
    """Scene with many objects: obs must pick the correct robot DOFs by name, not slice globally."""
    m, d, pol = _make()
    # left_knee is policy index 9 -> obs qpos block starts at index 9
    knee_adr = m.joint("left_knee_joint").qposadr[0]
    d.qpos[knee_adr] += 0.123
    mujoco.mj_forward(m, d)
    from locomotion.policy import DEFAULT_POS
    obs = pol._build_obs()
    # qpos block is obs[9:38]; left_knee at policy index 9 -> obs[18] = q - default
    assert abs(obs[9 + 9] - (d.qpos[knee_adr] - DEFAULT_POS[9])) < 1e-5, \
        "obs qpos block index 9 must reflect left_knee by joint name"
    # base address must be the robot's freejoint by name
    assert pol.base_qadr == m.joint("floating_base_joint").qposadr[0]


@pytest.mark.slow
def test_stands_still_with_zero_command():
    m, d, pol = _make()
    pol.set_command(0.0, 0.0, 0.0)
    for _ in range(int(3.0 / m.opt.timestep)):
        pol.step()
        mujoco.mj_step(m, d)
    assert not pol.fallen
    assert d.qpos[pol.base_qadr + 2] > 0.5, "robot must stand stable with zero command"


@pytest.mark.slow
def test_walks_forward():
    m, d, pol = _make()
    pol.set_command(0.5, 0.0, 0.0)
    x0 = d.qpos[pol.base_qadr]
    for _ in range(int(5.0 / m.opt.timestep)):
        pol.step()
        mujoco.mj_step(m, d)
    assert not pol.fallen, "robot fell while walking straight"
    assert d.qpos[pol.base_qadr] - x0 > 1.0, "must travel >1m in 5s"


@pytest.mark.slow
def test_turns_in_place():
    """policy_29dof.pt turns in place: a pure yaw command rotates the base with
    negligible translation."""
    m, d, pol = _make()
    x0, y0 = d.qpos[pol.base_qadr], d.qpos[pol.base_qadr + 1]
    pol.set_command(0.0, 0.0, 1.0)
    for _ in range(int(4.0 / m.opt.timestep)):
        pol.step()
        mujoco.mj_step(m, d)
    quat = d.qpos[pol.base_qadr + 3 : pol.base_qadr + 7]
    yaw = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                     1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
    drift = np.hypot(d.qpos[pol.base_qadr] - x0, d.qpos[pol.base_qadr + 1] - y0)
    assert not pol.fallen
    assert abs(yaw) > 0.8, f"must turn >0.8rad in 4s with wz=1.0, yaw={yaw:.2f}"
    assert drift < 0.7, f"turn-in-place should barely translate, drift={drift:.2f}m"
