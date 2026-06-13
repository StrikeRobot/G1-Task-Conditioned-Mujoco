"""Load scene.xml via MjSpec and inject the head_cam camera at runtime.

DO NOT write/modify any file in scene/ — all changes exist only
in the compiled model in RAM.
"""
import os

import mujoco
import numpy as np


def _camera_quat(pitch_down_deg: float):
    """Quat for a camera mounted on the body (x forward, z up): looks toward +x, tilted down.

    MuJoCo camera frame: looks along -z, x right, y up.
    """
    p = np.radians(pitch_down_deg)
    view = np.array([np.cos(p), 0.0, -np.sin(p)])   # view direction in body frame
    zc = -view                                       # camera z axis = -view direction
    xc = np.array([0.0, -1.0, 0.0])                  # camera x = robot's right side
    yc = np.cross(zc, xc)
    R = np.column_stack([xc, yc, zc])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return quat


def load_scene(sim_cfg):
    """Return (model, meta). meta = {'lidar_body_id', 'cam_body_name'}."""
    # mujoco 3.8: relative paths break asset resolution in included files
    # -> always use absolute paths (still READ-only, nothing written to scene/)
    spec = mujoco.MjSpec.from_file(os.path.abspath(sim_cfg["scene_xml"]))

    # find the camera-mount body: prefer cam_body, fallback to pelvis
    body = None
    for name in (sim_cfg["cam_body"], "pelvis"):
        body = spec.body(name)
        if body is not None:
            break
    if body is None:
        raise ValueError(
            f"cannot find camera-mount body: tried {sim_cfg['cam_body']!r} then 'pelvis'")

    cam = body.add_camera()
    cam.name = sim_cfg["cam_name"]
    cam.pos = sim_cfg["cam_pos"]
    cam.fovy = sim_cfg["cam_fovy"]
    cam.quat = _camera_quat(sim_cfg["cam_pitch_down_deg"])

    model = spec.compile()

    lidar_name = sim_cfg.get("lidar_mount_body") or body.name
    lidar_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, lidar_name)
    if lidar_body_id == -1:
        raise ValueError(f"cannot find lidar-mount body: {lidar_name!r}")

    meta = {
        "cam_body_name": body.name,
        "lidar_body_id": lidar_body_id,
    }
    return model, meta
