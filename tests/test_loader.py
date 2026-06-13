import hashlib
from pathlib import Path

import mujoco
import numpy as np
import yaml

from sim.loader import load_scene

with open("configs/config.yaml") as f:
    CFG = yaml.safe_load(f)


def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def test_loader_does_not_touch_scene_files():
    before = {p: _md5(p) for p in ("scene/scene.xml", "scene/robot/robot.xml")}
    load_scene(CFG["sim"])
    after = {p: _md5(p) for p in before}
    assert before == after, "loader must not modify scene files"


def test_loader_injects_camera_and_keeps_robot():
    model, meta = load_scene(CFG["sim"])
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam") >= 0
    # robot: freejoint base BY NAME + all joints (do NOT count globally —
    # the scene may add dynamic objects later)
    base = model.joint("floating_base_joint")
    assert model.jnt_type[base.id] == mujoco.mjtJoint.mjJNT_FREE
    # check representatives of the joint groups (all 29 joints checked thoroughly in Task 4 per POLICY_JOINTS)
    for n in ("left_knee_joint", "waist_yaw_joint", "right_wrist_yaw_joint",
              "left_ankle_roll_joint", "right_shoulder_pitch_joint"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0, f"missing {n}"
    assert model.nu >= 29, f"must have >=29 actuators, got {model.nu}"
    # room still present
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_north") >= 0
    # meta returns a valid lidar mount body
    assert meta["lidar_body_id"] >= 0
    assert meta["lidar_body_id"] == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, CFG["sim"]["lidar_mount_body"]
    )


def test_loaded_model_steps():
    model, _ = load_scene(CFG["sim"])
    d = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, d)
    assert np.all(np.isfinite(d.qpos))
